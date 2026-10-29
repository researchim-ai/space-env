"""Общие часы нескольких аппаратов и снимок, с которого прогон продолжается."""
from __future__ import annotations

import math

import numpy as np

from space_env.core.ground import GroundStation
from space_env.mission.scenarios import GroundTarget
from space_env.simulation.commands import CommandBook, MissionCommand
from space_env.simulation.engine import SimulationEngine


class MissionWorld:
    """Один шаг двигает каждый аппарат на один и тот же интервал."""

    def __init__(
        self,
        engines: dict[str, SimulationEngine],
        *,
        epoch: str,
        horizon_sec: float,
        seed: int = 0,
    ) -> None:
        if len(engines) < 1:
            raise ValueError("в мире нужен хотя бы один аппарат")
        if len(set(engines)) != len(engines):
            raise ValueError("идентификаторы аппаратов повторяются")
        self.engines = dict(engines)
        self.epoch = epoch
        self.horizon_sec = float(horizon_sec)
        self.rng = np.random.default_rng(seed)
        self.outages: list[tuple[str, float, float]] = []
        self.commands = CommandBook()
        self.relay_used: dict[str, list] = {name: [] for name in self.engines}
        self.task_requests: dict = {}
        self._batch: list[str] = []
        self._truth_log: dict[str, list[dict]] = {name: [] for name in self.engines}
        self._book_on = False
        self._book: dict[str, list[tuple[str, float, float]]] = {}
        self._reject_reason: dict[str, str] = {}
        self._require_common_clock()

    def now_sec(self) -> float:
        """Общие часы. Разъезд аппаратов — ошибка, а не среднее."""
        self._require_common_clock()
        return float(next(iter(self.engines.values())).time_sec)

    def _require_common_clock(self) -> None:
        times = [float(engine.time_sec) for engine in self.engines.values()]
        if max(times) - min(times) > 1e-6:
            raise RuntimeError("часы аппаратов разъехались")
        epochs = {round(float(engine.orbit.epoch.jd), 8) for engine in self.engines.values()}
        if len(epochs) != 1:
            raise RuntimeError("эпохи аппаратов различаются")

    def accept(self, commands: list[MissionCommand]) -> dict[str, str]:
        """Проверяет всю пачку до записи. Конфликт внутри пачки книгу не меняет."""
        seen_ids: dict[str, tuple] = {}
        seen_actors: set[str] = set()
        for command in commands:
            if command.actor_id not in self.engines:
                raise KeyError(f"нет аппарата {command.actor_id}")
            if command.actor_id in seen_actors:
                raise ValueError(f"аппарат {command.actor_id} повторяется в пачке")
            seen_actors.add(command.actor_id)
            if command.actor_id in self.commands.running_actors():
                raise ValueError("операция ещё идёт")
            body = (
                command.actor_id,
                command.kind,
                command.request_id,
                command.destination_id,
                float(command.duration_sec),
            )
            previous = seen_ids.get(command.command_id)
            if previous is not None and previous != body:
                raise ValueError(f"команда {command.command_id} уже записана с другим телом")
            seen_ids[command.command_id] = body
            self.commands.check(command)
        orders: dict[str, str] = {}
        self._batch = []
        for command in commands:
            record = self.commands.submit(command)
            if record["status"] != "accepted":
                continue
            engine = self.engines[command.actor_id]
            if command.kind == "imaging" and command.request_id and not self._request_ready(engine, command.request_id):
                self.commands.reject(command.command_id, "неизвестная заявка")
                orders[command.actor_id] = "wait"
                continue
            station = engine.get_state_snapshot().contact_station
            if command.kind == "comms" and command.destination_id and station and command.destination_id != station:
                self.commands.reject(command.command_id, "другая станция")
                orders[command.actor_id] = "wait"
                continue
            self._batch.append(command.command_id)
            self.commands.arm_window(command.command_id, self.now_sec())
            orders[command.actor_id] = command.kind
        return orders

    def _request_ready(self, engine: SimulationEngine, request_id: str) -> bool:
        if any(target.name == request_id for target in engine.scenario.ground_targets):
            return True
        request = self.task_requests.get(request_id)
        if request is None:
            return False
        engine.scenario.ground_targets.append(
            GroundTarget(name=request.request_id, lat_deg=float(request.lat_deg), lon_deg=float(request.lon_deg))
        )
        return True

    def settle(self) -> None:
        """Принятая команда становится completed или rejected по исходу шага."""
        rejected = {item["actor_id"]: item["reason"] for item in getattr(self, "last_rejected", [])}
        pending = list(self._batch)
        for command_id in self.commands.running_ids():
            if command_id not in pending:
                pending.append(command_id)
        for command_id in pending:
            record = self.commands.get(command_id)
            if record["status"] not in {"accepted", "running"}:
                continue
            if record["actor_id"] in rejected:
                self.commands.reject(command_id, rejected[record["actor_id"]])
            elif record["kind"] == "imaging" and not self.engines[record["actor_id"]].last_command_ok:
                self.commands.fail(command_id, "съёмка не выполнена")
            else:
                until = record.get("until_sec")
                actor = record["actor_id"]
                if until is not None and self.engines[actor].time_sec + 1e-9 < float(until):
                    self.commands.mark_running(command_id)
                else:
                    self.commands.complete(command_id)
                    if float(record.get("duration_sec") or 0.0) > 0.0:
                        engine = self.engines[actor]
                        if record["kind"] == "comms":
                            engine._downlink_armed = False
                        elif record["kind"] == "imaging":
                            engine.cancel_imaging()
        self._batch = []

    def cancel_operation(self, command_id: str) -> dict:
        """Снимает ещё идущую команду. Уже сданные биты остаются."""
        record = self.commands.cancel(command_id)
        engine = self.engines[record["actor_id"]]
        if record["kind"] == "comms":
            engine._downlink_armed = False
        elif record["kind"] == "imaging":
            engine.cancel_imaging()
        return record

    def set_outages(self, intervals: list[tuple[str, float, float]]) -> None:
        """Станция закрыта на [start, end) для любого вызывающего, не только для политики."""
        self.outages = [(str(name), float(start), float(end)) for name, start, end in intervals]
        for engine in self.engines.values():
            engine.set_link_outages(self.outages)

    def step(self, dt_sec: float, commands: dict[str, str] | None = None) -> None:
        if not math.isfinite(dt_sec) or dt_sec <= 0.0:
            raise ValueError("шаг должен быть положительным и конечным")
        self._require_common_clock()
        if self.now_sec() + dt_sec > self.horizon_sec + 1e-9:
            raise ValueError("шаг выходит за горизонт")
        orders = dict(commands or {})
        for actor, kind in self.commands.running_orders().items():
            orders.setdefault(actor, kind)
        unknown = set(orders) - set(self.engines)
        if unknown:
            raise KeyError(f"нет аппарата {sorted(unknown)[0]}")
        snapshot = self.snapshot()
        try:
            adjusted = self._one_antenna(orders)
            self._limit_overlapping_contacts(adjusted, dt_sec)
            self.last_rejected = [
                {
                    "actor_id": name,
                    "kind": kind,
                    "reason": self._reject_reason.get(name, "станция занята"),
                }
                for name, kind in orders.items()
                if kind not in {None, "wait"} and adjusted.get(name) != kind
            ]
            for engine in self.engines.values():
                engine.set_link_outages(self.outages)
            for name, engine in self.engines.items():
                command = adjusted.get(name)
                engine.step(dt_sec, None if command in {None, "wait"} else command)
            self._record_truth()
        except Exception:
            self.restore(snapshot)
            self.last_rejected = []
            raise

    def use_book(self, booked: dict) -> None:
        """Принятые окна становятся действующей книгой. Снять её может только clear_book."""
        self._book = {}
        for item in booked.get("accepted", []):
            self._book.setdefault(str(item["vehicle"]), []).append(
                (str(item["station"]), float(item["start_sec"]), float(item["end_sec"]))
            )
        self._book_on = True

    def clear_book(self) -> None:
        self._book_on = False
        self._book = {}

    def _one_antenna(self, orders: dict[str, str]) -> dict[str, str]:
        """Станция отдаёт линию не больше чем на число терминалов. Лишние — по большему имени."""
        self._reject_reason = {}
        for engine in self.engines.values():
            engine._rate_share = 1.0
        groups: dict[str, list[str]] = {}
        for name, engine in self.engines.items():
            command = orders.get(name)
            snap = engine.get_state_snapshot()
            transmits = command == "comms" or bool(engine._downlink_armed)
            if transmits and snap.in_contact and snap.contact_station:
                groups.setdefault(snap.contact_station, []).append(name)
        adjusted = dict(orders)
        for station, names in groups.items():
            slots = self._terminals(station, names)
            winners = set(sorted(names)[:slots])
            for name in names:
                if name in winners:
                    continue
                adjusted[name] = "wait"
                self.engines[name]._downlink_armed = False
                self._reject_reason[name] = "станция занята"
        if self._book_on:
            now = self.now_sec()
            for name, kind in list(adjusted.items()):
                if kind != "comms" and not self.engines[name]._downlink_armed:
                    continue
                snap = self.engines[name].get_state_snapshot()
                windows = self._book.get(name, [])
                inside = any(item[0] == snap.contact_station and item[1] <= now < item[2] for item in windows)
                if inside:
                    continue
                adjusted[name] = "wait"
                self.engines[name]._downlink_armed = False
                self._reject_reason[name] = "вне окна"
        for station, names in groups.items():
            live = [
                name
                for name in names
                if adjusted.get(name) == "comms" or self.engines[name]._downlink_armed
            ]
            station_row = self._station_row(station, names)
            if station_row is not None and station_row.share_bandwidth and len(live) > 1:
                share = 1.0 / len(live)
                for name in live:
                    self.engines[name]._rate_share = share
        return adjusted

    def _limit_overlapping_contacts(self, orders: dict[str, str], dt_sec: float) -> None:
        """Если внутри шага два аппарата претендуют на один терминал, лишний интервал вырезается."""
        from space_env.simulation.terminals import assign_terminal_intervals

        for engine in self.engines.values():
            engine._downlink_mask = None
        claims: list[tuple[str, str, float, float]] = []
        for name, engine in self.engines.items():
            command = orders.get(name)
            if command != "comms" and not engine._downlink_armed:
                continue
            claims.extend(
                (station, name, start, end) for station, start, end in engine.preview_contact_intervals(dt_sec)
            )
        if len(claims) < 2:
            return
        slots = {station: self._terminals(station, [name]) for station, name, _start, _end in claims}
        allowed = assign_terminal_intervals(claims, slots)
        claimed: dict[str, list[tuple[float, float]]] = {}
        for _station, name, start, end in claims:
            claimed.setdefault(name, []).append((start, end))
        for name, pieces in claimed.items():
            grant = allowed.get(name, [])
            if _same_cover(grant, pieces):
                continue
            self.engines[name]._downlink_mask = grant
            if not grant and orders.get(name) == "comms":
                orders[name] = "wait"
                self.engines[name]._downlink_armed = False
                self._reject_reason.setdefault(name, "станция занята")

    def _station_row(self, station: str, names: list[str]):
        engine = self.engines[names[0]]
        for item in engine.stations:
            if item.name == station:
                return item
        return None

    def _terminals(self, station: str, names: list[str]) -> int:
        engine = self.engines[names[0]]
        for item in engine.stations:
            if item.name == station:
                return max(1, int(item.terminals))
        return 1

    def snapshot(self) -> dict:
        return {
            "engines": {name: engine.capture() for name, engine in self.engines.items()},
            "rng": self.rng.bit_generator.state,
            "outages": list(self.outages),
            "commands": self.commands.export(),
            "relay_used": {name: list(intervals) for name, intervals in self.relay_used.items()},
            "stations": {
                name: [
                    {
                        "name": station.name,
                        "lat_deg": station.lat_deg,
                        "lon_deg": station.lon_deg,
                        "alt_m": station.alt_m,
                        "min_elevation_deg": station.min_elevation_deg,
                        "terminals": station.terminals,
                        "band_hz": station.band_hz,
                        "share_bandwidth": station.share_bandwidth,
                    }
                    for station in engine.stations
                ]
                for name, engine in self.engines.items()
            },
            "book_on": self._book_on,
            "book": {
                name: [list(item) for item in windows] for name, windows in self._book.items()
            },
            "targets": {
                name: [_target_row(target) for target in engine.scenario.ground_targets]
                for name, engine in self.engines.items()
            },
        }

    def restore(self, snapshot: dict) -> None:
        if "engines" in snapshot and "rng" in snapshot:
            for name, payload in snapshot["engines"].items():
                self.engines[name].restore(payload)
            self.rng.bit_generator.state = snapshot["rng"]
            self.outages = list(snapshot.get("outages", []))
            if "commands" in snapshot:
                self.commands.load(snapshot["commands"])
            if "relay_used" in snapshot:
                self.relay_used = {
                    str(name): [tuple(item) for item in intervals]
                    for name, intervals in snapshot["relay_used"].items()
                }
            if "stations" in snapshot:
                for name, rows in snapshot["stations"].items():
                    engine = self.engines[name]
                    engine.stations = [GroundStation(**row) for row in rows]
                    engine.scenario.ground_stations = list(engine.stations)
                    engine._refresh_environment()
            for engine in self.engines.values():
                engine.set_link_outages(self.outages)
            if "book_on" in snapshot:
                self._book_on = bool(snapshot["book_on"])
                self._book = {
                    str(name): [(str(item[0]), float(item[1]), float(item[2])) for item in windows]
                    for name, windows in snapshot.get("book", {}).items()
                }
            if "targets" in snapshot:
                for name, rows in snapshot["targets"].items():
                    self.engines[name].scenario.ground_targets = [GroundTarget(**row) for row in rows]
            return
        for name, payload in snapshot.items():
            self.engines[name].restore(payload)

    def observe(self, name: str, *, noise: float = 0.0, rng: np.random.Generator | None = None) -> dict:
        """Измерение для агента. Правильной команды здесь нет. noise > 0 портит заряд и объём."""
        engine = self.engines[name]
        snap = engine.get_state_snapshot()
        soc = float(snap.battery_soc)
        stored = float(engine.stored_bits)
        if noise > 0.0 and rng is not None:
            soc = float(np.clip(soc + float(rng.normal(0.0, noise)), 0.0, 1.0))
            stored = max(0.0, stored * (1.0 + float(rng.normal(0.0, noise))))
        return {
            "time_sec": float(engine.time_sec),
            "soc": soc,
            "stored_bits": stored,
            "in_contact": bool(snap.in_contact),
            "station": snap.contact_station,
        }

    def truth(self, name: str) -> dict:
        """Состояние без шума и без правильной команды."""
        reading = self.observe(name)
        reading["kind"] = "truth"
        return reading

    def estimate(self, name: str, *, noise: float = 0.0, rng: np.random.Generator | None = None) -> dict:
        """Шумное измерение. Свой поток seed не совпадает с истиной, когда noise > 0."""
        reading = self.observe(name, noise=noise, rng=self.rng if rng is None else rng)
        reading["kind"] = "estimate"
        return reading

    def forecast(self, name: str, dt_sec: float) -> dict:
        """Коуэлл на dt вперёд тем же шагом. После вызова часы мира стоят на прежнем месте."""
        engine = self.engines[name]
        snap = engine.capture()
        try:
            engine.step(dt_sec, "wait")
            ahead = self.truth(name)
        finally:
            engine.restore(snap)
        ahead["forecast_sec"] = float(dt_sec)
        ahead["kind"] = "forecast"
        return ahead

    def note_truth(self) -> None:
        self._record_truth()

    def _record_truth(self) -> None:
        for name in self.engines:
            self._truth_log.setdefault(name, []).append(self.truth(name))

    def measure(self, name: str, stream) -> dict:
        """Измерение по ленте истины. Причины отказов в пакет не входят."""
        return stream.read(self.now_sec(), self._truth_log.get(name, []))

    def safety_events(self) -> int:
        """Сколько аппаратов сейчас ниже порога заряда. Это не счётчик сданных заявок."""
        count = 0
        for engine in self.engines.values():
            if engine.spacecraft.state.power.battery_soc < engine.constraints.battery_soc_min:
                count += 1
        return count


def _same_cover(grant: list[tuple[float, float]], claimed: list[tuple[float, float]]) -> bool:
    def length(pieces: list[tuple[float, float]]) -> float:
        return sum(max(0.0, end - start) for start, end in pieces)

    if abs(length(grant) - length(claimed)) > 1e-6:
        return False
    for start, end in claimed:
        covered = 0.0
        for left, right in grant:
            covered += max(0.0, min(end, right) - max(start, left))
        if covered + 1e-6 < end - start:
            return False
    return True


def _target_row(target) -> dict:
    return {
        "name": target.name,
        "lat_deg": float(target.lat_deg),
        "lon_deg": float(target.lon_deg),
        "cloud_fraction": float(getattr(target, "cloud_fraction", 0.0)),
        "cloud_sigma": float(getattr(target, "cloud_sigma", 0.0)),
        "geometry": str(getattr(target, "geometry", "point")),
        "length_km": float(getattr(target, "length_km", 0.0)),
        "vertices": list(getattr(target, "vertices", []) or []),
        "looks": int(getattr(target, "looks", 1)),
        "gap_sec": float(getattr(target, "gap_sec", 0.0)),
        "min_stereo_deg": float(getattr(target, "min_stereo_deg", 0.0)),
        "max_stereo_deg": float(getattr(target, "max_stereo_deg", 0.0)),
        "depends_on": str(getattr(target, "depends_on", "")),
    }
