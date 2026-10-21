"""
Именованные аппараты по открытым карточкам миссий.

Опубликованные числа лежат в полях. Всё, чего в карточке нет
(ЭИИМ X-диапазона, импульс гидразина, теплоёмкость узла), перечислено в assumptions
и посчитано в geometry.py. Это опорные борта для задач обучения, не паспорт изделия.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from space_env.core.constants import SOLAR_CONSTANT
from space_env.core.ground import DEFAULT_GROUND_STATIONS, KIRUNA, MATERA, SIOUX_FALLS, GroundStation
from space_env.core.spacecraft import (
    CommsConfig,
    PayloadConfig,
    PowerConfig,
    PropulsionConfig,
    SpacecraftConfig,
    ThermalConfig,
)

from space_env.catalog.geometry import (
    ASSUMED_HYDRAZINE_ISP_S,
    closure_required_dbw,
    dish_gain_dbi,
    half_swath_deg,
    radiating_area_m2,
    rocket_delta_v_m_s,
)

# Середина распределения EESS (Земля-космос) 8025–8400 МГц. Несущая в карточках не названа.
X_BAND_CENTER_HZ: float = 8.2125e9
# Середина опубликованного диапазона кросс-линков Iridium 22.55–23.55 ГГц.
IRIDIUM_CROSSLINK_HZ: float = 23.05e9


@dataclass(frozen=True)
class ReferenceVehicle:
    key: str
    title: str
    norad_id: int | None
    launch_mass_kg: float
    propellant_kg: float | None
    altitude_m: float
    inclination_deg: float
    ltdn_hours: float | None
    swath_m: float | None
    array_area_m2: float
    array_area_published: bool
    array_power_w: float
    battery_wh: float
    battery_published: bool
    consumption_nominal_w: float
    consumption_peak_w: float
    consumption_published: bool
    downlink_bps: float
    downlink_hz: float
    tx_power_w: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    required_power_dbw: float
    crosslink_hz: float | None
    # Сжатый поток в память, объём памяти и предел съёмки за виток. None — в карточке этого нет.
    store_rate_bps: float | None
    memory_bits: float | None
    max_imaging_s: float | None
    sources: tuple[str, ...]
    assumptions: tuple[str, ...]

    def spacecraft_config(self, name: str | None = None) -> SpacecraftConfig:
        efficiency = self.array_power_w / (SOLAR_CONSTANT * self.array_area_m2)
        delta_v = 0.0
        if self.propellant_kg is not None:
            delta_v = rocket_delta_v_m_s(self.launch_mass_kg, self.propellant_kg)
        drag_area = 1.0
        srp_area = self.array_area_m2
        if self.key.startswith("sentinel-2"):
            # Земная грань по габариту 3.4 × 1.8 м из карточки ESA. Крыло в мидель не включено.
            drag_area = 3.4 * 1.8
        half_fov = 30.0
        if self.swath_m is not None:
            half_fov = half_swath_deg(self.altitude_m, self.swath_m)
        return SpacecraftConfig(
            name=name or self.title,
            slew_rate_deg_s=2.0,
            power=PowerConfig(
                solar_panel_area_m2=self.array_area_m2,
                battery_capacity_Wh=self.battery_wh,
                consumption_nominal_W=self.consumption_nominal_w,
                consumption_peak_W=self.consumption_peak_w,
                panel_efficiency=efficiency,
            ),
            thermal=ThermalConfig(
                heater_power_W=max(40.0, 0.02 * self.consumption_nominal_w),
                heat_capacity_J_K=self.launch_mass_kg * 900.0,
                radiating_area_m2=radiating_area_m2(self.consumption_nominal_w),
                projected_area_m2=self.array_area_m2,
                earth_facing_area_m2=drag_area,
            ),
            propulsion=PropulsionConfig(
                delta_v_total_m_s=delta_v,
                mass_kg=self.launch_mass_kg,
                drag_area_m2=drag_area,
                drag_coefficient=2.2,
                srp_area_m2=srp_area,
                srp_coefficient=1.2,
            ),
            comms=CommsConfig(
                data_rate_max_bps=self.downlink_bps,
                frequency_Hz=self.downlink_hz,
                tx_power_W=self.tx_power_w,
                tx_gain_dBi=self.tx_gain_dbi,
                rx_gain_dBi=self.rx_gain_dbi,
                system_loss_dB=3.0,
                required_power_dBW=self.required_power_dbw,
            ),
            payload=PayloadConfig(half_fov_deg=half_fov),
        )


def _xband_closure(altitude_m: float, dish_m: float, tx_power_w: float, tx_gain_dbi: float) -> tuple[float, float]:
    gain = dish_gain_dbi(dish_m, X_BAND_CENTER_HZ)
    required = closure_required_dbw(
        altitude_m=altitude_m,
        frequency_hz=X_BAND_CENTER_HZ,
        tx_power_w=tx_power_w,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=gain,
        system_loss_db=3.0,
    )
    return gain, required


_S2_GAIN, _S2_REQUIRED = _xband_closure(786_000.0, 15.0, 10.0, 0.0)
_L9_GAIN, _L9_REQUIRED = _xband_closure(705_000.0, 11.0, 10.0, 0.0)

# Площадь, которая при КПД 0.30 даёт опубликованные ватты решётки. Сама площадь в источнике не дана.
_L9_AREA = 4300.0 / (SOLAR_CONSTANT * 0.30)
_IRIDIUM_AREA = 2000.0 / (SOLAR_CONSTANT * 0.30)


REFERENCES: dict[str, ReferenceVehicle] = {
    "sentinel-2a": ReferenceVehicle(
        key="sentinel-2a",
        title="Sentinel-2A",
        norad_id=40697,
        launch_mass_kg=1140.0,
        propellant_kg=123.0,
        altitude_m=786_000.0,
        inclination_deg=98.62,
        ltdn_hours=10.5,
        swath_m=290_000.0,
        array_area_m2=7.2,
        array_area_published=True,
        array_power_w=2300.0,
        battery_wh=102.0 * 28.0,
        battery_published=True,
        consumption_nominal_w=1400.0,
        consumption_peak_w=1700.0,
        consumption_published=True,
        downlink_bps=560e6,
        downlink_hz=X_BAND_CENTER_HZ,
        tx_power_w=10.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=_S2_GAIN,
        required_power_dbw=_S2_REQUIRED,
        crosslink_hz=None,
        store_rate_bps=450e6,
        memory_bits=2.4e12,
        max_imaging_s=40.0 * 60.0,
        sources=(
            "https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures",
            "https://esamultimedia.esa.int/docs/S2-Data_Sheet.pdf",
            "https://www.eoportal.org/satellite-missions/copernicus-sentinel-2",
            "https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf",
            "https://www.esa.int/Enabling_Support/Operations/ESA_Ground_Stations/Kiruna_station",
        ),
        assumptions=(
            "Масса 1140 кг и 123 кг топлива — страница ESA Facts and figures. Лист 2013 года писал 1200 кг и 117 кг.",
            "Полоса 290 км и наклонение 98.62°, нисходящий узел 10:30 — Sentinel-2 User Handbook.",
            "СБ 7.2 м² и 1.4 кВт номинала — лист ESA 2013. eoPortal даёт 7.1 м², 2300 Вт BOL и 1700 Вт максимума платформы.",
            "Батарея 102 А·ч EOL и шина 28 В ± 4 В — eoPortal. Ёмкость в Вт·ч = 102 × 28, без учёта допуска шины.",
            "X-band 560 Мбит/с — SP-1322 и eoPortal. Несущая принята серединой полосы 8025–8400 МГц.",
            "ЭИИМ борта в источниках нет. Принято 10 Вт и 0 дБи изофлюкса. Порог станции выставлен в ноль запаса на маске 5°.",
            "Усиление Кируны посчитано по опубликованному зеркалу KIR-1 15 м и КПД 0.55, это не измеренный G/T.",
            "Δv — уравнение Циолковского при 220 с. Импульс гидразина в карточке не дан.",
            "Теплоёмкость 900 Дж/(кг·К) и площадь излучения под номинальную нагрузку при 20 °C — одноузловая замыкалка, не тепловой расчёт Airbus.",
            "В память пишется 450 Мбит/с после сжатия 1.3 Гбит/с — eoPortal. Память 2.4 Тбит и 40 минут съёмки на виток — лист ESA 2013.",
        ),
    ),
    "landsat-9": ReferenceVehicle(
        key="landsat-9",
        title="Landsat 9",
        norad_id=49260,
        launch_mass_kg=2623.0,
        propellant_kg=2623.0 - 1512.0,
        altitude_m=705_000.0,
        inclination_deg=98.2,
        ltdn_hours=10.0,
        swath_m=185_000.0,
        array_area_m2=_L9_AREA,
        array_area_published=False,
        array_power_w=4300.0,
        battery_wh=1800.0,
        battery_published=False,
        consumption_nominal_w=1800.0,
        consumption_peak_w=2500.0,
        consumption_published=False,
        downlink_bps=384e6,
        downlink_hz=X_BAND_CENTER_HZ,
        tx_power_w=10.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=_L9_GAIN,
        required_power_dbw=_L9_REQUIRED,
        crosslink_hz=None,
        store_rate_bps=384e6,
        memory_bits=3.14e12,
        max_imaging_s=None,
        sources=(
            "https://science.nasa.gov/mission/landsat-9/",
            "https://space.oscar.wmo.int/satellites/view/landsat_9",
            "https://landsat.usgs.gov/LGS",
            "https://repository.arizona.edu/handle/10150/581626",
            "https://www.usgs.gov/media/files/landsat-9-data-users-handbook",
        ),
        assumptions=(
            "Масса 2623 кг, сухая 1512 кг и мощность 4300 Вт — WMO OSCAR. Орбита 705 км, 98.2°, полоса 185 км, узел около 10:00 — NASA.",
            "384 Мбит/с — проект X-band ряда LDCM (Landsat 8). Для Landsat 9 USGS подтверждает X-band науки, отдельную скорость на странице миссии не выделяет.",
            "Площадь СБ выведена из 4300 Вт при КПД 0.30. В OSCAR площади нет.",
            "Потребление 1800/2500 Вт и батарея 1800 Вт·ч в карточках нет: запас на одно затмение при этой нагрузке.",
            "Зеркало 11 м для LGS не опубликовано. Усиление посчитано так же, как для Кируны, с этим диаметром и КПД 0.55.",
            "Порог линии снова замкнут на маску 5° при ЭИИМ 10 дБВт. Это не измеренный бюджет USGS.",
            "Память 3.14 Тбит — Landsat 9 Data Users Handbook, твердотельный регистратор. Сцена 185×180 км оттуда же.",
            "Скорость записи в регистратор в справочнике не дана. Пока цель в полосе, в память пишется не быстрее 384 Мбит/с — это скорость сброса, не скорость OLI.",
        ),
    ),
    "iridium-next": ReferenceVehicle(
        key="iridium-next",
        title="Iridium NEXT",
        norad_id=None,
        launch_mass_kg=860.0,
        propellant_kg=None,
        altitude_m=780_000.0,
        inclination_deg=86.4,
        ltdn_hours=None,
        swath_m=4_700_000.0,
        array_area_m2=_IRIDIUM_AREA,
        array_area_published=False,
        array_power_w=2000.0,
        battery_wh=600.0,
        battery_published=False,
        consumption_nominal_w=600.0,
        consumption_peak_w=800.0,
        consumption_published=False,
        downlink_bps=1.0e6,
        downlink_hz=1.616e9,
        tx_power_w=1.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=0.0,
        required_power_dbw=-140.0,
        crosslink_hz=IRIDIUM_CROSSLINK_HZ,
        store_rate_bps=None,
        memory_bits=None,
        max_imaging_s=None,
        sources=(
            "https://www.eoportal.org/satellite-missions/iridium-next",
            "https://www.microwavejournal.com/articles/37587-the-iridium-leo-satellite-system-for-global-mobile-communications",
            "https://en.wikiversity.org/wiki/Satellite_constellation/Design",
        ),
        assumptions=(
            "66 аппаратов, 6 плоскостей, 780 км, 86.4°, Ka-кросс-линки 22.55–23.55 ГГц и четыре соседа — eoPortal и Microwave Journal.",
            "Walker-star 86.4°:66/6/2 — сводка Wikiversity. Узлы в коде разнесены на 180°.",
            "Масса около 860 кг и 2 кВт — таблица eoPortal. Microwave Journal пишет около 800 кг. Взята цифра eoPortal.",
            "ЭИИМ кросс-линка не опубликован, запас линии для него не считается. Считаются дистанция и закрытие Землёй.",
            "Площадь СБ выведена из 2 кВт при КПД 0.30. Размах 9.4 м опубликован, хорда панелей — нет.",
            "Потребление и батарея не опубликованы. 600 Вт и 600 Вт·ч держат одно затмение и не выдаются за борт Thales.",
            "Пятно L-band 4700 км в диаметре — Microwave Journal. Угол от надира посчитан по этой дуге, это не камера.",
            "L-band абонента (около 1.616 ГГц) не замкнут на зеркало: это не линия на станцию Кируны.",
        ),
    ),
}


SENTINEL_STATIONS: tuple[GroundStation, ...] = (
    KIRUNA,
    next(station for station in DEFAULT_GROUND_STATIONS if station.name == "Svalbard"),
    MATERA,
)
LANDSAT_STATIONS: tuple[GroundStation, ...] = (
    SIOUX_FALLS,
    next(station for station in DEFAULT_GROUND_STATIONS if station.name == "Svalbard"),
)


REFERENCES["sentinel-2b"] = replace(
    REFERENCES["sentinel-2a"],
    key="sentinel-2b",
    title="Sentinel-2B",
    norad_id=42063,
)


def reference(key: str) -> ReferenceVehicle:
    try:
        return REFERENCES[key]
    except KeyError as exc:
        known = ", ".join(sorted(REFERENCES))
        raise KeyError(f"нет аппарата {key}. известны: {known}") from exc
