"""Совместный шаг нескольких аппаратов тем же миром, что и одиночная команда."""
from __future__ import annotations


class JointExecutor:
    """Один вызов двигает всех названных аппаратов одним world.step."""

    def __init__(self, world, dt_sec: float = 1.0) -> None:
        self.world = world
        self.dt_sec = float(dt_sec)

    def step(self, orders: dict[str, str]) -> dict[str, float]:
        self.world.step(self.dt_sec, orders)
        return {name: float(engine.delivered_bits) for name, engine in self.world.engines.items()}
