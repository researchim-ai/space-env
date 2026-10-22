"""
Геометрия, которая выводится из опубликованных высоты, полосы и размеров антенны.
Здесь нет подгонки под ответ: формулы названы, константы — из core.constants.
"""
from __future__ import annotations

import math

from space_env.core.constants import C_LIGHT, J2_EARTH, MU_EARTH, R_EARTH_EQUATOR, R_EARTH_MEAN, SIGMA_SB
from space_env.core.link import free_space_path_loss_db
from space_env.core.sun import sun_position_m
from space_env.core.time import Time

# Монопропеллентный гидразин. Удельный импульс в карточках Sentinel и Landsat не опубликован.
ASSUMED_HYDRAZINE_ISP_S: float = 220.0
G0_M_S2: float = 9.80665
# Среднее движение Солнца по эклиптике, рад/с. Узел солнечно-синхронной орбиты держит эту скорость.
MEAN_SOLAR_RATE_RAD_S: float = 2.0 * math.pi / (365.2421897 * 86400.0)


def j2_nodal_rate_rad_s(semimajor_m: float, inclination_rad: float, eccentricity: float = 0.0) -> float:
    """
    dΩ/dt от J2, рад/с. Форма Кёртиса: −(3/2) n J2 (Re/a)² cos i / (1−e²)².
    Для наклонения больше 90° скорость положительная: узел идёт на восток вместе с Солнцем.
    """
    a_m = float(semimajor_m)
    motion = math.sqrt(MU_EARTH / a_m**3)
    denom = (1.0 - float(eccentricity) ** 2) ** 2
    factor = 1.5 * motion * J2_EARTH * (R_EARTH_EQUATOR / a_m) ** 2 / denom
    return -factor * math.cos(float(inclination_rad))


def sun_sync_inclination_rad(semimajor_m: float, eccentricity: float = 0.0) -> float:
    """Наклонение, при котором узловая скорость J2 равна среднему движению Солнца."""
    a_m = float(semimajor_m)
    motion = math.sqrt(MU_EARTH / a_m**3)
    denom = (1.0 - float(eccentricity) ** 2) ** 2
    factor = 1.5 * motion * J2_EARTH * (R_EARTH_EQUATOR / a_m) ** 2 / denom
    cosine = max(-1.0, min(1.0, -MEAN_SOLAR_RATE_RAD_S / factor))
    return math.acos(cosine)


def geosynchronous_semimajor_axis_m() -> float:
    """Большая полуось круговой орбиты с периодом звёздных суток, м."""
    sidereal_sec = 86164.0905
    return (MU_EARTH * sidereal_sec**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)


def geosynchronous_altitude_m() -> float:
    """Высота над сферой равного объёма для геосинхронной круговой орбиты, м."""
    return geosynchronous_semimajor_axis_m() - R_EARTH_MEAN


def mean_semimajor_axis_m(altitude_m: float) -> float:
    """Средняя высота над сферой равного объёма, не над экватором."""
    return R_EARTH_MEAN + float(altitude_m)


def half_swath_deg(altitude_m: float, swath_m: float) -> float:
    """
    Половина угла от надира до края полосы на сфере.
    swath_m — длина дуги на поверхности, как в карточке миссии.
    """
    theta = (float(swath_m) / 2.0) / R_EARTH_MEAN
    opposite = R_EARTH_MEAN * math.sin(theta)
    adjacent = R_EARTH_MEAN + float(altitude_m) - R_EARTH_MEAN * math.cos(theta)
    return math.degrees(math.atan2(opposite, adjacent))


def slant_range_m(altitude_m: float, elevation_deg: float) -> float:
    """Наклонная дальность круговой орбиты при заданном угле места."""
    elevation = math.radians(float(elevation_deg))
    radius = mean_semimajor_axis_m(altitude_m)
    ratio = radius / R_EARTH_MEAN
    return R_EARTH_MEAN * (math.sqrt(ratio * ratio - math.cos(elevation) ** 2) - math.sin(elevation))


def dish_gain_dbi(diameter_m: float, frequency_hz: float, efficiency: float = 0.55) -> float:
    """Усиление круглой антенны: 10 log10(η (π D / λ)²)."""
    wavelength = C_LIGHT / float(frequency_hz)
    gain = float(efficiency) * (math.pi * float(diameter_m) / wavelength) ** 2
    return 10.0 * math.log10(gain)


def rocket_delta_v_m_s(launch_mass_kg: float, propellant_kg: float, isp_s: float = ASSUMED_HYDRAZINE_ISP_S) -> float:
    """Уравнение Циолковского. Импульс — допущение, массы — из источника."""
    dry = float(launch_mass_kg) - float(propellant_kg)
    if dry <= 0.0 or propellant_kg < 0.0:
        raise ValueError("сухая масса должна быть положительной")
    return float(isp_s) * G0_M_S2 * math.log(float(launch_mass_kg) / dry)


def radiating_area_m2(heat_w: float, temp_c: float = 20.0, emissivity: float = 0.8) -> float:
    """
    Площадь, которая при этой температуре излучает heat_w в космос 0 К.
    Замыкает одноузловую модель на опубликованную электрическую нагрузку.
    """
    temp_k = float(temp_c) + 273.15
    return float(heat_w) / (SIGMA_SB * float(emissivity) * temp_k**4)


def raan_rad_for_ltdn(moment: Time, ltdn_hours: float) -> float:
    """
    Долгота восходящего узла, при которой нисходящий узел имеет заданное местное солнечное время.
    Прямое восхождение Солнца берётся из core.sun. Уравнение времени отдельно не вводится.
    """
    sun = sun_position_m(moment)
    sun_ra = math.atan2(float(sun[1]), float(sun[0]))
    descending_ra = sun_ra + (float(ltdn_hours) - 12.0) * math.pi / 12.0
    return descending_ra - math.pi


def closure_required_dbw(
    *,
    altitude_m: float,
    frequency_hz: float,
    tx_power_w: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    system_loss_db: float,
    mask_deg: float = 5.0,
) -> float:
    """Порог, при котором запас линии на маске угла места равен нулю."""
    slant = slant_range_m(altitude_m, mask_deg)
    eirp = 10.0 * math.log10(tx_power_w) + tx_gain_dbi
    return eirp + rx_gain_dbi - free_space_path_loss_db(slant, frequency_hz) - system_loss_db
