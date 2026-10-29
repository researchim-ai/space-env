"""
Инструменты агента: смотреть, прогнозировать, подать, отменить, шагнуть.
Прогноз не двигает часы. Отмена не трогает уже выполненную команду.
Наблюдение не содержит причину отказа.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from space_env.mission.awareness import SensorModel, SensorStream
from space_env.mission.session import MissionSession, RevisionError
from space_env.mission.tasking import request_lifecycle
from space_env.simulation.commands import MissionCommand

_KINDS = {"wait", "imaging", "comms"}


class MissionTools:
    """Один сеанс и один канал измерения."""

    def __init__(self, scenario: str | Path, *, seed: int = 0, call_budget: int = 0) -> None:
        self.session = MissionSession(scenario)
        self.session.world.note_truth()
        self.stream = SensorStream(SensorModel(), np.random.default_rng(seed))
        self._pending: list[MissionCommand] = []
        self.call_budget = int(call_budget)
        self.calls = 0

    def _spend(self) -> None:
        self.calls += 1
        if self.call_budget > 0 and self.calls > self.call_budget:
            raise RuntimeError("бюджет вызовов исчерпан")

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
            reading = world.measure(name, self.stream)
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

    def submit(self, commands: list[dict]) -> dict:
        self._spend()
        built = [self._command(item) for item in commands]
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
        record = self.session.world.commands.cancel(command_id)
        return {"command_id": command_id, "status": record["status"], "revision": self.revision}

    def advance(self, dt_sec: float, *, expected_revision: int) -> dict:
        self._spend()
        report = self.session.advance(dt_sec, list(self._pending), expected_revision=expected_revision)
        self._pending = []
        return report

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
