"""
Ограничения миссии: окна видимости, лимиты Δv, питание, температура.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    """Ограничения, которые нельзя нарушать."""
    delta_v_max_m_s: float | None = Field(default=None, ge=0)
    battery_soc_min: float = Field(default=0.2, ge=0, le=1)
    temp_min_C: float | None = Field(default=None)
    temp_max_C: float | None = Field(default=None)
    # Окна видимости станций задаются отдельно (временные интервалы) — упрощённо не моделим
    max_pass_duration_min: float | None = Field(default=None, ge=0)
