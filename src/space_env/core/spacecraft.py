"""
Модель космического аппарата и подсистем: Power, Thermal, Propulsion, Comms, Payload.
Состояние и конфигурация через Pydantic.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SubsystemHealth(str, Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    FAILED = "failed"


# --- Конфигурация подсистем (YAML/JSON) ---


class PowerConfig(BaseModel):
    """Конфиг подсистемы электропитания."""
    solar_panel_area_m2: float = Field(ge=0, description="Площадь СБ, м²")
    battery_capacity_Wh: float = Field(gt=0, description="Ёмкость батареи, Вт·ч")
    consumption_nominal_W: float = Field(ge=0, description="Номинальное потребление, Вт")
    consumption_peak_W: float = Field(ge=0, description="Пиковое потребление, Вт")


class ThermalConfig(BaseModel):
    """Конфиг терморегулирования (упрощённо)."""
    temp_min_C: float = Field(default=-20.0, description="Минимальная рабочая температура, °C")
    temp_max_C: float = Field(default=50.0, description="Максимальная рабочая температура, °C")
    heater_power_W: float = Field(ge=0, default=0, description="Мощность нагревателей, Вт")


class PropulsionConfig(BaseModel):
    """Конфиг ДУ."""
    delta_v_total_m_s: float = Field(ge=0, description="Суммарный доступный Δv, м/с")
    thrust_N: float = Field(ge=0, default=0, description="Тяга, Н")
    mass_kg: float = Field(gt=0, description="Масса КА, кг")


class CommsConfig(BaseModel):
    """Конфиг связи (упрощённо)."""
    data_rate_max_bps: float = Field(default=1e6, ge=0, description="Макс. скорость передачи, бит/с")
    pass_duration_min: float = Field(ge=0, default=10.0, description="Типовая длительность сессии, мин")


class PayloadConfig(BaseModel):
    """Конфиг полезной нагрузки (режимы)."""
    modes: list[str] = Field(default_factory=lambda: ["idle", "imaging"], description="Режимы работы")
    default_mode: str = Field(default="idle", description="Режим по умолчанию")


class SpacecraftConfig(BaseModel):
    """Полная конфигурация КА."""
    name: str = "SC"
    power: PowerConfig = Field(default_factory=PowerConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    propulsion: PropulsionConfig = Field(default_factory=PropulsionConfig)
    comms: CommsConfig = Field(default_factory=CommsConfig)
    payload: PayloadConfig = Field(default_factory=PayloadConfig)


# --- Состояние подсистем (в процессе симуляции) ---


class PowerState(BaseModel):
    """Текущее состояние питания."""
    battery_soc: float = Field(default=1.0, ge=0, le=1, description="Уровень заряда 0..1")
    in_sun: bool = True
    consumption_W: float = 0.0


class ThermalState(BaseModel):
    """Текущее тепловое состояние."""
    temp_C: float = 20.0
    heater_on: bool = False


class PropulsionState(BaseModel):
    """Состояние ДУ."""
    delta_v_remaining_m_s: float = 0.0


class CommsState(BaseModel):
    """Состояние связи."""
    in_contact: bool = False
    data_sent_this_pass_Mb: float = 0.0


class SpacecraftState(BaseModel):
    """Состояние КА в текущий момент."""
    power: PowerState = Field(default_factory=PowerState)
    thermal: ThermalState = Field(default_factory=ThermalState)
    propulsion: PropulsionState = Field(default_factory=PropulsionState)
    comms: CommsState = Field(default_factory=CommsState)
    payload_mode: str = "idle"
    health: dict[str, SubsystemHealth] = Field(
        default_factory=lambda: {
            "power": SubsystemHealth.NOMINAL,
            "thermal": SubsystemHealth.NOMINAL,
            "propulsion": SubsystemHealth.NOMINAL,
            "comms": SubsystemHealth.NOMINAL,
            "payload": SubsystemHealth.NOMINAL,
        }
    )


class Spacecraft(BaseModel):
    """
    Космический аппарат: конфигурация + текущее состояние.
    Можно загружать из YAML/JSON и обновлять состояние в симуляции.
    """
    config: SpacecraftConfig = Field(default_factory=SpacecraftConfig)
    state: SpacecraftState = Field(default_factory=SpacecraftState)

    model_config = {"use_enum_values": False}

    def reset_state(self, delta_v_initial_m_s: Optional[float] = None) -> None:
        """Сброс состояния к начальному (например, для нового эпизода)."""
        self.state = SpacecraftState(
            power=PowerState(battery_soc=1.0, in_sun=True, consumption_W=self.config.power.consumption_nominal_W),
            thermal=ThermalState(temp_C=20.0, heater_on=False),
            propulsion=PropulsionState(
                delta_v_remaining_m_s=delta_v_initial_m_s or self.config.propulsion.delta_v_total_m_s
            ),
            comms=CommsState(in_contact=False, data_sent_this_pass_Mb=0.0),
            payload_mode=self.config.payload.default_mode,
        )
