"""
Сценарии для обучения: номинальный, аномалии, конъюнкции, нехватка ресурсов.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from space_env.mission.events import EventKind


class ScenarioType(str, Enum):
    NOMINAL = "nominal"
    ANOMALY = "anomaly"
    CONJUNCTION = "conjunction"
    RESOURCE_STRESS = "resource_stress"


class AnomalyInject(BaseModel):
    """Инжект аномалии в сценарий."""
    subsystem: str = Field(description="power | thermal | propulsion | comms | payload")
    at_sec: float = Field(ge=0)
    severity: str = Field(default="degraded")  # degraded | failed
    duration_sec: float | None = Field(default=None, ge=0)  # None = до конца эпизода


class ConjunctionInject(BaseModel):
    """Инжект предупреждения о сближении."""
    at_sec: float = Field(ge=0)
    miss_distance_km: float = Field(gt=0)
    probability: float = Field(default=1e-5, ge=0, le=1)


class ScenarioConfig(BaseModel):
    """Конфиг сценария: тип + инжекты (для воспроизводимости по seed)."""
    scenario_type: ScenarioType = ScenarioType.NOMINAL
    name: str = ""
    seed: int | None = Field(default=None)
    anomaly_injects: list[AnomalyInject] = Field(default_factory=list)
    conjunction_injects: list[ConjunctionInject] = Field(default_factory=list)
    # Ресурсы: начальное состояние (например, пониженный SoC)
    initial_battery_soc: float | None = Field(default=None, ge=0, le=1)
    initial_delta_v_fraction: float | None = Field(default=None, ge=0, le=1)
