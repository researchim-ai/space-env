"""
Цели миссии: метрики (покрытие, сессии связи, данные, манёвры).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class MissionGoal(BaseModel):
    """Целевые метрики миссии (нормы для успеха)."""
    # Покрытие / съёмка
    coverage_percent: float | None = Field(default=None, ge=0, le=100)
    imaging_count: int | None = Field(default=None, ge=0)
    # Связь
    comms_sessions_min: int | None = Field(default=None, ge=0)
    data_downlink_Mb: float | None = Field(default=None, ge=0)
    # Ресурсы (не превышать)
    delta_v_budget_m_s: float | None = Field(default=None, ge=0)
    # Время миссии (сек) — опционально
    mission_duration_sec: float | None = Field(default=None, ge=0)

    def check(
        self,
        coverage_percent: float | None = None,
        imaging_count: int | None = None,
        comms_sessions: int | None = None,
        data_downlink_Mb: float | None = None,
        delta_v_used_m_s: float | None = None,
    ) -> dict[str, bool]:
        """Проверка достижения целей. Возвращает словарь {метрика: достигнута}."""
        out: dict[str, bool] = {}
        if self.coverage_percent is not None and coverage_percent is not None:
            out["coverage"] = coverage_percent >= self.coverage_percent
        if self.imaging_count is not None and imaging_count is not None:
            out["imaging"] = imaging_count >= self.imaging_count
        if self.comms_sessions_min is not None and comms_sessions is not None:
            out["comms_sessions"] = comms_sessions >= self.comms_sessions_min
        if self.data_downlink_Mb is not None and data_downlink_Mb is not None:
            out["data_downlink"] = data_downlink_Mb >= self.data_downlink_Mb
        if self.delta_v_budget_m_s is not None and delta_v_used_m_s is not None:
            out["delta_v_within_budget"] = delta_v_used_m_s <= self.delta_v_budget_m_s
        return out
