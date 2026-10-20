"""
Сценарии для обучения: номинальный, аномалии, конъюнкции, нехватка ресурсов.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from space_env.core.ground import GroundStation
from space_env.mission.events import PlannedEvent


class GroundTarget(BaseModel):
    """Точка съёмки на поверхности Земли."""

    name: str = "target"
    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=180)


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
    """
    Встреча в плоскости радиус/нормаль. Промах задаётся компонентами;
    если их нет, весь miss_distance_km кладётся на радиальную ось.
    hard_body_radius_m и combined_sigma_m — поля уровня CDM, Pc из них считается, а не подставляется.
    Поле probability сохранено для старых файлов и в расчёт не входит.
    """

    at_sec: float = Field(ge=0)
    miss_distance_km: float = Field(gt=0)
    probability: float = Field(default=1e-5, ge=0, le=1)
    tca_in_sec: float = Field(default=600.0, gt=0, description="Секунды от инжекта до номинального TCA")
    miss_radial_m: float | None = None
    miss_cross_m: float | None = None
    approach_m_s: float = Field(default=150.0, gt=0, description="Скорость сближения вдоль трека, м/с")
    hard_body_radius_m: float = Field(default=15.0, gt=0)
    combined_sigma_m: float = Field(default=200.0, gt=0)


class ScenarioConfig(BaseModel):
    """Конфиг сценария: тип + инжекты (для воспроизводимости по seed)."""
    scenario_type: ScenarioType = ScenarioType.NOMINAL
    name: str = ""
    seed: int | None = Field(default=None)
    anomaly_injects: list[AnomalyInject] = Field(default_factory=list)
    conjunction_injects: list[ConjunctionInject] = Field(default_factory=list)
    planned_events: list[PlannedEvent] = Field(default_factory=list)
    ground_targets: list[GroundTarget] = Field(default_factory=list)
    # None — станция по умолчанию (Москва). Пустой список — без наземного сегмента.
    ground_stations: list[GroundStation] | None = None
    # Ресурсы: начальное состояние (например, пониженный SoC)
    initial_battery_soc: float | None = Field(default=None, ge=0, le=1)
    initial_delta_v_fraction: float | None = Field(default=None, ge=0, le=1)
