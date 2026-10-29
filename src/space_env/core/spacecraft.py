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
    """
    Опорный малый аппарат ~100 кг: около 1.2 м² тройного перехода, шина ~35 Вт, батарея ~280 Вт·ч.
    Это порядок коммерческого EO-спутника, не паспорт конкретного борта.
    """
    solar_panel_area_m2: float = Field(default=1.2, ge=0, description="Площадь СБ, м²")
    battery_capacity_Wh: float = Field(default=280.0, gt=0, description="Ёмкость батареи, Вт·ч")
    consumption_nominal_W: float = Field(default=35.0, ge=0, description="Номинальное потребление, Вт")
    consumption_peak_W: float = Field(default=90.0, ge=0, description="Пиковое потребление, Вт")
    panel_efficiency: float = Field(default=0.30, ge=0, le=1, description="КПД фотопреобразователей")
    safe_soc_enter: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Автоматический безопасный режим при заряде не выше этого. 0 вместе с выходом — гистерезис выключен.",
    )
    safe_soc_exit: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Снятие автоматического безопасного режима при заряде не ниже этого. Должен быть выше входа.",
    )


class ThermalConfig(BaseModel):
    """Одноузловая термика: теплоёмкость, излучение, нагреватели."""
    temp_min_C: float = Field(default=-20.0, description="Минимальная рабочая температура, °C")
    temp_max_C: float = Field(default=50.0, description="Максимальная рабочая температура, °C")
    heater_power_W: float = Field(ge=0, default=40.0, description="Мощность нагревателей, Вт")
    heat_capacity_J_K: float = Field(default=90_000.0, gt=0, description="Теплоёмкость узла, Дж/К")
    radiating_area_m2: float = Field(default=1.0, ge=0, description="Площадь излучения, м²")
    projected_area_m2: float = Field(default=0.40, ge=0, description="Проекция, принимающая Солнце, м²")
    earth_facing_area_m2: float = Field(default=0.35, ge=0, description="Площадь, смотрящая на Землю, м²")
    absorptivity: float = Field(default=0.40, ge=0, le=1)
    emissivity: float = Field(default=0.80, ge=0, le=1)


class PropulsionConfig(BaseModel):
    """
    Два профиля. Пока тяга, удельный импульс и сухая масса не заданы вместе,
    манёвр мгновенный и масса не меняется. Конечное включение — отдельный профиль:
    тяга, Isp и сухая масса ниже текущей. Бюджет Δv его не заменяет.
    """
    delta_v_total_m_s: float = Field(default=80.0, ge=0, description="Предел импульса миссии, м/с. Не модель двигателя.")
    thrust_N: float = Field(ge=0, default=0, description="Тяга конечного включения, Н. Ноль — импульсный профиль.")
    isp_s: float = Field(ge=0, default=0, description="Удельный импульс, с. Ноль — импульсный профиль.")
    dry_mass_kg: float = Field(ge=0, default=0, description="Сухая масса, кг. Топливо — разница с mass_kg.")
    mass_kg: float = Field(default=100.0, gt=0, description="Текущая масса КА, кг")
    drag_area_m2: float = Field(default=0.9, ge=0, description="Мидель для сопротивления, м²")
    drag_coefficient: float = Field(default=2.2, ge=0, description="Коэффициент сопротивления")
    srp_area_m2: float = Field(default=1.2, ge=0, description="Площадь для солнечного давления, м²")
    srp_coefficient: float = Field(default=1.2, ge=0, le=2, description="Cr пушечного ядра, 1 поглощение … 2 зеркало")


class CommsConfig(BaseModel):
    """
    S-диапазон малого аппарата на станцию с зеркалом около 3 м.
    2 Вт, усиление борта 6 дБи, станции 32 дБи, требуемый уровень −125 дБВт:
    в зените запас положительный, у горизонта линия не закрывается.
    """
    data_rate_max_bps: float = Field(default=1e6, ge=0, description="Макс. скорость передачи, бит/с")
    pass_duration_min: float = Field(ge=0, default=10.0, description="Типовая длительность сессии, мин")
    frequency_Hz: float = Field(default=2.2e9, gt=0)
    tx_power_W: float = Field(default=2.0, gt=0)
    tx_gain_dBi: float = Field(default=6.0)
    rx_gain_dBi: float = Field(default=32.0)
    system_loss_dB: float = Field(default=3.0, ge=0)
    required_power_dBW: float = Field(default=-125.0)
    min_margin_dB: float = Field(default=0.0)
    setup_sec: float = Field(
        default=0.0,
        ge=0,
        description="Подготовка антенны перед первым битом сеанса, с. Ноль — сдача с начала контакта.",
    )
    ack_sec: float = Field(
        default=0.0,
        ge=0,
        description="Задержка подтверждения custody после конца передачи, с. Ноль — квитанция в момент прихода бит.",
    )
    ttl_sec: float = Field(
        default=0.0,
        ge=0,
        description="Срок жизни продукта на борту, с. Ноль — биты не сгорают по времени.",
    )
    forward_hold_sec: float = Field(
        default=0.0,
        ge=0,
        description="После приёма по линии биты нельзя отдать дальше раньше этого срока, с. Ноль — сразу.",
    )
    full_duplex: bool = Field(
        default=False,
        description="Можно ретранслировать в том же интервале, где идёт сдача на станцию. По умолчанию радио полудуплексное.",
    )
    weather_loss_db: float = Field(
        default=0.0,
        ge=0,
        description="Дополнительная потеря модема, дБ. Ноль — прежний бюджет. Контакт по углу места от этой потери не зависит.",
    )


class PayloadConfig(BaseModel):
    """Конфиг полезной нагрузки (режимы)."""
    modes: list[str] = Field(default_factory=lambda: ["idle", "imaging"], description="Режимы работы")
    default_mode: str = Field(default="idle", description="Режим по умолчанию")
    half_fov_deg: float = Field(default=30.0, gt=0, le=70, description="Половина поля зрения надирного прибора, °")
    imaging_power_w: float = Field(default=0.0, ge=0, description="Дополнительная мощность в режиме съёмки, Вт")
    images_in_eclipse: bool = Field(default=False, description="Съёмка возможна в тени. У радара да, у оптики нет")
    quality_off_nadir_deg: float | None = Field(
        default=None,
        description="Уже этого угла от надира кадр только partial. Пусто — совпадает с полем зрения.",
    )
    smear_rate_deg_s: float = Field(
        default=0.0,
        ge=0,
        description="Выше этой скорости разворота оптический кадр partial. Ноль — смаз не проверяется.",
    )
    prep_sec: float = Field(
        default=0.0,
        ge=0,
        description="Подготовка экспозиции, с. Ноль — кадр с начала команды. Не паспорт прибора.",
    )
    exclusive_with_downlink: bool = Field(
        default=False,
        description="Съёмка и включённый передатчик не совмещаются. По умолчанию совмещаются.",
    )
    memory_bits: float = Field(default=0.0, ge=0, description="Ёмкость регистратора, бит. 0 — предел не задан")
    memory_policy: str = Field(
        default="reject",
        pattern="^(reject|drop_oldest)$",
        description="reject отбрасывает новый объём. drop_oldest снимает целые старые продукты.",
    )
    store_rate_bps: float = Field(default=0.0, ge=0, description="Поток в память, бит/с. 0 — один кадр image_bits")
    process_rate_bps: float = Field(
        default=0.0,
        ge=0,
        description="Очередь обработки, бит/с. 0 — продукт доступен сразу после съёмки.",
    )
    process_power_w: float = Field(default=0.0, ge=0, description="Мощность обработки, Вт. Тратится, только пока очередь не пуста.")
    compression_ratio: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Доля сенсорного потока, которая остаётся в продукте. 1 — без сжатия. Не модель кодека.",
    )
    sar_duty_fraction: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Доля витка для радара. Ноль — скважность не проверяется. Оптика этот порог не читает.",
    )
    image_bits: float = Field(
        default=8.0e6,
        ge=0,
        description="Биты одного кадра, если поток в память не задан. Допущение, не паспорт",
    )


class SpacecraftConfig(BaseModel):
    """Полная конфигурация КА."""
    name: str = "SC"
    slew_rate_deg_s: float = Field(default=2.0, gt=0, description="Максимальная угловая скорость разворота, °/с")
    slew_accel_deg_s2: float = Field(
        default=0.0,
        ge=0,
        description="Угловое ускорение, °/с². Ноль набирает предельную скорость сразу.",
    )
    slew_settle_deg: float = Field(
        default=1.0,
        ge=0,
        description="Съёмка не идёт, пока остаток разворота больше этого угла.",
    )
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
    pointing: str = Field(default="nadir", description="nadir | sun")
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
                delta_v_remaining_m_s=(
                    self.config.propulsion.delta_v_total_m_s
                    if delta_v_initial_m_s is None
                    else delta_v_initial_m_s
                )
            ),
            comms=CommsState(in_contact=False, data_sent_this_pass_Mb=0.0),
            payload_mode=self.config.payload.default_mode,
            pointing="nadir",
        )
