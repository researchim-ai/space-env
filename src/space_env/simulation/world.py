"""Общие часы нескольких аппаратов и снимок, с которого прогон продолжается."""
from __future__ import annotations

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
        self.engines = dict(engines)
        self.epoch = epoch
        self.horizon_sec = horizon_sec
        self.rng = np.random.default_rng(seed)

    def step(self, dt_sec: float, commands: dict[str, str] | None = None) -> None:
        orders = commands or {}
        for name, engine in self.engines.items():
            command = orders.get(name)
            engine.step(dt_sec, None if command in {None, "wait"} else command)

    def snapshot(self) -> dict:
        return {name: engine.capture() for name, engine in self.engines.items()}

    def restore(self, snapshot: dict) -> None:
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
        logged = len(engine.log)
        engine.step(dt_sec, "wait")
        ahead = self.truth(name)
        engine.restore(snap)
        extra = len(engine.log) - logged
        if extra > 0:
            del engine._log[-extra:]
        ahead["forecast_sec"] = float(dt_sec)
        ahead["kind"] = "forecast"
        return ahead
