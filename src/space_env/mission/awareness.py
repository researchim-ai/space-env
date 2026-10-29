"""
Истина, измерение и оценка не смешиваются.
Измерение не содержит причину отказа. Одинаковый seed даёт тот же поток.
Безопасность считается отдельно от сдачи.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorModel:
    """Параметры канала. Ноль периода, пропуска и задержки — отсчёт на каждый вызов."""

    bias_soc: float = 0.0
    sigma_soc: float = 0.0
    sigma_stored: float = 0.0
    period_sec: float = 0.0
    dropout: float = 0.0
    delay_sec: float = 0.0


class SensorStream:
    """Состояние канала: последний отсчёт и генератор. Лента истины приходит снаружи."""

    def __init__(self, model: SensorModel, rng: np.random.Generator) -> None:
        self.model = model
        self.rng = rng
        self._last: dict | None = None
        self._emitted_at = -1.0e30

    def read(self, now_sec: float, tape: list[dict]) -> dict:
        target = float(now_sec) - float(self.model.delay_sec)
        chosen: dict | None = None
        for item in tape:
            if float(item["time_sec"]) <= target + 1e-9:
                chosen = item
        if chosen is None:
            return {"kind": "measurement", "missing": True, "time_sec": float(now_sec), "reason": "нет отсчёта"}
        period = float(self.model.period_sec)
        if self._last is not None and period > 0.0 and float(now_sec) + 1e-9 < self._emitted_at + period:
            stale = dict(self._last)
            stale["stale"] = True
            return stale
        self._emitted_at = float(now_sec)
        if self.model.dropout > 0.0 and float(self.rng.random()) < float(self.model.dropout):
            missed = {
                "kind": "measurement",
                "missing": True,
                "time_sec": float(chosen["time_sec"]),
                "reason": "пропуск",
                "stale": False,
            }
            self._last = missed
            return missed
        soc = float(np.clip(float(chosen["soc"]) + self.model.bias_soc + self.model.sigma_soc * float(self.rng.normal()), 0.0, 1.0))
        stored = max(0.0, float(chosen["stored_bits"]) * (1.0 + self.model.sigma_stored * float(self.rng.normal())))
        reading = {
            "kind": "measurement",
            "missing": False,
            "stale": False,
            "time_sec": float(chosen["time_sec"]),
            "soc": soc,
            "stored_bits": stored,
            "in_contact": bool(chosen.get("in_contact", False)),
            "station": str(chosen.get("station", "")),
        }
        self._last = reading
        return reading


def separate_scores(*, on_time_fraction: float, safety_events: int) -> dict[str, float | int]:
    """Сдача и нарушения не складываются в одну награду."""
    return {"delivery": float(on_time_fraction), "safety": int(safety_events)}
