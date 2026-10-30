"""
Инструменты агента: смотреть, прогнозировать, подать, отменить, шагнуть.
Прогноз не двигает часы. Отмена не трогает уже выполненную команду.
Наблюдение не содержит причину отказа.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from space_env.mission.awareness import SensorModel, SensorStream
from space_env.mission.session import MissionSession, RevisionError
from space_env.mission.tasking import request_lifecycle
from space_env.simulation.commands import MissionCommand


class ToolBudgetError(RuntimeError):
    """The declared per-session call budget has been exhausted."""


_KINDS = {"wait", "imaging", "comms"}
__all__ = ["MissionTools", "RevisionError", "ToolBudgetError", "choose_request", "run_baseline"]


class MissionTools:
    """Один сеанс и один канал измерения."""

    def __init__(self, scenario: str | Path, *, seed: int = 0, call_budget: int = 0) -> None:
        self._configure(MissionSession(scenario), seed=seed, call_budget=call_budget)

    @classmethod
    def from_session(cls, session: MissionSession, *, seed: int = 0, call_budget: int = 0) -> MissionTools:
        tools = cls.__new__(cls)
        tools._configure(session, seed=seed, call_budget=call_budget)
        return tools

    def _configure(self, session: MissionSession, *, seed: int, call_budget: int) -> None:
        if type(call_budget) is not int or call_budget < 0:
            raise ValueError("call_budget must be a nonnegative integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("sensor seed must be a nonnegative integer")
        self.stream = SensorStream(SensorModel(), np.random.default_rng(seed))
        self.session = session
        self.session.world.note_truth()
        self._pending: list[MissionCommand] = []
        self.call_budget = call_budget
        self.calls = 0
        self._sensor_seed = seed
        self._channels: dict[str, SensorStream] = {}

    def _spend(self) -> None:
        self.calls += 1
        if self.call_budget > 0 and self.calls > self.call_budget:
            raise ToolBudgetError("бюджет вызовов исчерпан")

    @property
    def revision(self) -> int:
        return self.session.revision

    def observe(self, actor: str | None = None) -> dict:
        self._spend()
        world = self.session.world
        actors = {}
        for name in world.engines:
            if actor is not None and name != actor:
                continue
            first = next(iter(world.engines))
            if name == first:
                channel = self.stream
            else:
                if name not in self._channels:
                    index = list(world.engines).index(name)
                    self._channels[name] = SensorStream(self.stream.model, np.random.default_rng(
                        np.random.SeedSequence([self._sensor_seed, index])))
                channel = self._channels[name]
            reading = world.measure(name, channel)
            actors[name] = reading
        receipts = [item for engine in world.engines.values() for item in engine.receipts]
        now = world.now_sec()
        requests = []
        for item in self.session.requests:
            status = request_lifecycle(item, receipts, now)
            requests.append(
                {
                    "id": item.request_id,
                    "release_sec": item.release_sec,
                    "deadline_sec": item.deadline_sec,
                    "priority": item.priority,
                    "status": status,
                    "released": now + 1e-9 >= item.release_sec and status not in {"delivered", "expired"},
                }
            )
        masks = {}
        for name, engine in world.engines.items():
            snap = engine.get_state_snapshot()
            masks[name] = {
                "imaging": [row["id"] for row in requests if row["released"]],
                "comms": bool(snap.in_contact),
            }
        return {
            "revision": self.revision,
            "now_sec": now,
            "actors": actors,
            "requests": requests,
            "masks": masks,
            "pending": [item.command_id for item in self._pending],
        }

    def forecast(self, actor: str, dt_sec: float) -> dict:
        self._spend()
        ahead = self.session.world.forecast(actor, dt_sec)
        ahead["revision"] = self.revision
        ahead["now_sec"] = self.session.world.now_sec()
        return ahead

    def forecast_geometry(self, actor: str, dt_sec: float) -> dict:
        """Nominal ephemeris forecast; no future hidden faults or subsystem truth."""
        from space_env.core.frames import ecef_to_geodetic, eci_to_ecef

        self._spend()
        if not math.isfinite(dt_sec) or dt_sec <= 0:
            raise ValueError("forecast duration must be positive and finite")
        world = self.session.world
        engine = world.engines[actor]
        position = engine.preview_positions(dt_sec)[-1][1]
        lat, lon, altitude = ecef_to_geodetic(eci_to_ecef(position, engine.now() + dt_sec))
        return {"kind": "nominal_geometry_forecast", "actor": actor, "revision": self.revision,
                "now_sec": world.now_sec(), "forecast_sec": dt_sec,
                "lat_deg": lat, "lon_deg": lon, "altitude_km": altitude / 1000.}

    def submit(self, commands: list[dict]) -> dict:
        self._spend()
        built = [self._command(item) for item in commands]
        for command in built:
            self.session.world.commands.check(command)
        actors: set[str] = set()
        for command in [*self._pending, *built]:
            if command.actor_id not in self.session.world.engines:
                raise KeyError(f"нет аппарата {command.actor_id}")
            if command.actor_id in actors:
                raise ValueError(f"аппарат {command.actor_id} повторяется в пачке")
            actors.add(command.actor_id)
        self._pending.extend(built)
        return {"pending": [item.command_id for item in self._pending], "revision": self.revision}

    def cancel(self, command_id: str) -> dict:
        self._spend()
        kept = [item for item in self._pending if item.command_id != command_id]
        if len(kept) != len(self._pending):
            self._pending = kept
            return {"command_id": command_id, "status": "dropped", "revision": self.revision}
        record = self.session.world.cancel_operation(command_id)
        return {"command_id": command_id, "status": record["status"], "revision": self.revision}

    def advance(self, dt_sec: float, *, expected_revision: int) -> dict:
        self._spend()
        report = self.session.advance(dt_sec, list(self._pending), expected_revision=expected_revision)
        self._pending = []
        return report

    def relay(self, source: str, destination: str, dt_sec: float, *, expected_revision: int) -> dict:
        self._spend()
        if self._pending:
            raise ValueError("advance or cancel pending commands before relay")
        return self.session.relay(source, destination, dt_sec, expected_revision=expected_revision)

    def capture(self) -> dict:
        """Capture pending commands, call accounting and sensor/session identity."""
        from space_env.simulation.snapshot import digest, encode

        index = next((i for i, stream in enumerate(self.session.world._sensor_streams)
                      if stream is self.stream), None)
        channels = {name: {"state": stream.capture(), "index": next(
            (i for i, row in enumerate(self.session.world._sensor_streams) if row is stream), None)}
            for name, stream in self._channels.items()}
        body = encode({"schema": "space-env.tools/1", "session": self.session.capture(),
                       "channels": channels, "sensor_seed": self._sensor_seed,
                       "stream": self.stream.capture(), "stream_index": index,
                       "pending": [asdict(command) for command in self._pending],
                       "call_budget": self.call_budget, "calls": self.calls})
        return {**body, "sha256": digest(body)}

    @classmethod
    def from_snapshot(cls, payload: dict) -> MissionTools:
        from space_env.simulation.snapshot import decode, digest, encode

        body = {key: value for key, value in payload.items() if key != "sha256"}
        if payload.get("schema") != "space-env.tools/1" or payload.get("sha256") != digest(body):
            raise ValueError("unsupported or corrupt tools snapshot")
        for name in ("calls", "call_budget", "sensor_seed"):
            if type(payload[name]) is not int or payload[name] < 0:
                raise ValueError("tool call counters must be nonnegative integers")
        tools = cls.__new__(cls)
        tools.session = MissionSession.from_snapshot(payload["session"])
        state = decode(payload["stream"])
        index = payload["stream_index"]
        if index is None:
            tools.stream = SensorStream.from_snapshot(state)
        else:
            if type(index) is not int or not 0 <= index < len(tools.session.world._sensor_streams):
                raise ValueError("invalid sensor stream identity")
            tools.stream = tools.session.world._sensor_streams[index]
            if encode(tools.stream.capture()) != payload["stream"]:
                raise ValueError("tool and world sensor states differ")
        tools._pending = [MissionCommand(**row) for row in payload["pending"]]
        tools.call_budget, tools.calls = payload["call_budget"], payload["calls"]
        tools._sensor_seed = payload["sensor_seed"]
        tools._channels = {}
        for name, channel in payload["channels"].items():
            if name not in tools.session.world.engines:
                raise ValueError("unknown actor sensor channel")
            index = channel["index"]
            if index is None:
                stream = SensorStream.from_snapshot(decode(channel["state"]))
            else:
                if type(index) is not int or not 0 <= index < len(tools.session.world._sensor_streams):
                    raise ValueError("invalid actor sensor stream identity")
                stream = tools.session.world._sensor_streams[index]
                if encode(stream.capture()) != channel["state"]:
                    raise ValueError("actor and world sensor states differ")
            tools._channels[name] = stream
        return tools

    def _command(self, item: dict) -> MissionCommand:
        kind = str(item.get("kind", "wait"))
        if kind not in _KINDS:
            raise ValueError(f"команда {kind} не поддерживается")
        return MissionCommand(
            str(item["command_id"]),
            str(item["actor_id"]),
            kind,
            str(item.get("request_id", "")),
            str(item.get("destination_id", "")),
            float(item.get("duration_sec", 0.0)),
        )


def choose_request(
    requests: list[dict],
    policy: str,
    *,
    now_sec: float = 0.0,
    horizon_sec: float = 0.0,
) -> dict | None:
    """Одна открытая заявка. fifo — ранний выпуск, deadline — ранний срок, priority — больший приоритет.

    rolling внутри горизонта берёт более узкое окно выпуска и срока. Горизонт 0 смотрит все открытые.
    """
    open_rows = [item for item in requests if item.get("released")]
    if not open_rows:
        return None
    if policy == "fifo":
        return min(open_rows, key=lambda item: (float(item["release_sec"]), str(item["id"])))
    if policy == "deadline":
        return min(open_rows, key=lambda item: (float(item["deadline_sec"]), -int(item["priority"]), str(item["id"])))
    if policy == "priority":
        return min(open_rows, key=lambda item: (-int(item["priority"]), float(item["deadline_sec"]), str(item["id"])))
    if policy == "rolling":
        horizon = float(horizon_sec)
        now = float(now_sec)
        inside = [item for item in open_rows if float(item["deadline_sec"]) <= now + horizon] if horizon > 0.0 else []
        pool = inside or open_rows
        return min(
            pool,
            key=lambda item: (
                float(item["deadline_sec"]) - float(item["release_sec"]),
                -int(item["priority"]),
                str(item["id"]),
            ),
        )
    raise ValueError(f"политика {policy} не поддерживается")


def run_baseline(scenario: str | Path, policy: str, *, step_sec: float, steps: int, actor: str) -> dict:
    """Базовая политика ходит теми же инструментами, что и агент."""
    tools = MissionTools(scenario, seed=0)
    for index in range(steps):
        view = tools.observe()
        chosen = choose_request(
            view["requests"],
            policy,
            now_sec=float(view["now_sec"]),
            horizon_sec=600.0,
        )
        if chosen is None or not view["masks"].get(actor, {}).get("imaging"):
            tools.advance(step_sec, expected_revision=tools.revision)
            continue
        tools.submit(
            [
                {
                    "command_id": f"{policy}-{index}",
                    "actor_id": actor,
                    "kind": "imaging",
                    "request_id": chosen["id"],
                }
            ]
        )
        tools.advance(step_sec, expected_revision=tools.revision)
    return tools.session.report()
