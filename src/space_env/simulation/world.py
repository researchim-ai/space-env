"""Общие часы нескольких аппаратов и снимок, с которого прогон продолжается."""
from __future__ import annotations

import math

import numpy as np

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

    def set_outages(self, intervals: list[tuple[str, float, float]]) -> None:
        """Станция закрыта на [start, end) для любого вызывающего, не только для политики."""
        self.outages = [(str(name), float(start), float(end)) for name, start, end in intervals]
        for engine in self.engines.values():
            engine.set_link_outages(self.outages)

    def step(self, dt_sec: float, commands: dict[str, str] | None = None) -> None:
        if not math.isfinite(dt_sec) or dt_sec <= 0.0:
            raise ValueError("шаг должен быть положительным и конечным")
        self._require_common_clock()
        orders = dict(commands or {})
        unknown = set(orders) - set(self.engines)
        if unknown:
            raise KeyError(f"нет аппарата {sorted(unknown)[0]}")
        adjusted = self._one_antenna(orders)
        self.last_rejected = [
            {"actor_id": name, "kind": kind, "reason": "станция занята"}
            for name, kind in orders.items()
            if kind not in {None, "wait"} and adjusted.get(name) != kind
        ]
        orders = adjusted
        for engine in self.engines.values():
            engine.set_link_outages(self.outages)
        for name, engine in self.engines.items():
            command = orders.get(name)
            engine.step(dt_sec, None if command in {None, "wait"} else command)

    def _one_antenna(self, orders: dict[str, str]) -> dict[str, str]:
        """Одна станция в один момент отдаёт линию одному аппарату. Победитель — меньшее имя."""
        groups: dict[str, list[str]] = {}
        for name, command in orders.items():
            if command != "comms":
                continue
            snap = self.engines[name].get_state_snapshot()
            if snap.in_contact and snap.contact_station:
                groups.setdefault(snap.contact_station, []).append(name)
        adjusted = dict(orders)
        for names in groups.values():
            if len(names) <= 1:
                continue
            winner = min(names)
            for name in names:
                if name == winner:
                    continue
                adjusted[name] = "wait"
                self.engines[name]._downlink_armed = False
        return adjusted

    def snapshot(self) -> dict:
        return {
            "engines": {name: engine.capture() for name, engine in self.engines.items()},
            "rng": self.rng.bit_generator.state,
            "outages": list(self.outages),
        }

    def restore(self, snapshot: dict) -> None:
        if "engines" in snapshot and "rng" in snapshot:
            for name, payload in snapshot["engines"].items():
                self.engines[name].restore(payload)
            self.rng.bit_generator.state = snapshot["rng"]
            self.outages = list(snapshot.get("outages", []))
            for engine in self.engines.values():
                engine.set_link_outages(self.outages)
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
