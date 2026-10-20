"""
Пространственная среда: Земля, коническая тень (umbra/penumbra), поток.
Положение Солнца передаётся снаружи (аналитика или SPICE).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from space_env.core.constants import AU, R_EARTH_EQUATOR, R_SUN, SOLAR_CONSTANT


class ShadowState(NamedTuple):
    """Тень Земли: sun — полный свет, penumbra — частичная, umbra — полная."""

    kind: str
    flux_fraction: float
    r_perp_m: float
    umbra_radius_m: float
    penumbra_radius_m: float


def _shadow_radii(x_behind_m: float, sun_distance_m: float) -> tuple[float, float]:
    """Радиусы umbra и penumbra на расстоянии x_behind_m за центром Земли (антисолнечная сторона)."""
    d = max(float(sun_distance_m), R_SUN + R_EARTH_EQUATOR)
    # Длина конуса полной тени от центра Земли до вершины.
    l_umbra = R_EARTH_EQUATOR * d / (R_SUN - R_EARTH_EQUATOR)
    # Характерная длина полутени.
    l_pen = R_EARTH_EQUATOR * d / (R_SUN + R_EARTH_EQUATOR)
    if x_behind_m <= 0.0 or x_behind_m >= l_umbra:
        r_umbra = 0.0
    else:
        r_umbra = R_EARTH_EQUATOR * (l_umbra - x_behind_m) / l_umbra
    r_pen = R_EARTH_EQUATOR * (l_pen + max(x_behind_m, 0.0)) / l_pen
    return float(r_umbra), float(r_pen)


def shadow_state(r_sat_m: np.ndarray, r_sun_m: np.ndarray) -> ShadowState:
    """
    Коническая тень Земли.
    r_sat_m, r_sun_m — положения КА и Солнца в одной инерциальной СК, м.
    КА на солнечной стороне (между Землёй и Солнцем, вне Земли) освещён.
    """
    r_sat = np.asarray(r_sat_m, dtype=float)
    r_sun = np.asarray(r_sun_m, dtype=float)
    sun_norm = float(np.linalg.norm(r_sun))
    sat_norm = float(np.linalg.norm(r_sat))
    if sat_norm < 1e3 or sun_norm < 1e3:
        return ShadowState("sun", 1.0, 0.0, 0.0, 0.0)
    s_hat = r_sun / sun_norm
    proj = float(np.dot(r_sat, s_hat))
    r_perp = float(np.linalg.norm(r_sat - proj * s_hat))
    x_behind = -proj
    r_umbra, r_pen = _shadow_radii(x_behind, sun_norm)
    if x_behind <= 0.0:
        return ShadowState("sun", 1.0, r_perp, r_umbra, r_pen)
    if r_perp <= r_umbra and r_umbra > 0.0:
        return ShadowState("umbra", 0.0, r_perp, r_umbra, r_pen)
    if r_perp <= r_pen:
        span = max(r_pen - r_umbra, 1.0)
        fraction = float(np.clip((r_perp - r_umbra) / span, 0.0, 1.0))
        return ShadowState("penumbra", fraction, r_perp, r_umbra, r_pen)
    return ShadowState("sun", 1.0, r_perp, r_umbra, r_pen)


def in_eclipse(r_sat_m: np.ndarray, r_sun_m: np.ndarray) -> bool:
    """True только в полной тени (umbra). Полутень сюда не входит."""
    return shadow_state(r_sat_m, r_sun_m).kind == "umbra"


def solar_flux_at_satellite_W_m2(r_sat_m: np.ndarray, r_sun_m: np.ndarray, in_umbra: bool | None = None) -> float:
    """
    Солнечный поток на КА, Вт/м².
    Учитывает 1/r² и долю диска в полутени. Аргумент in_umbra сохранён для совместимости:
    если передан и тень ещё не посчитана снаружи, поток берётся из shadow_state, а не из флага.
    """
    del in_umbra  # поток всегда считается по геометрии, флаг больше не режет полутень
    r_sun = np.asarray(r_sun_m, dtype=float)
    sun_norm = float(np.linalg.norm(r_sun))
    if sun_norm < 1e3:
        return 0.0
    fraction = shadow_state(r_sat_m, r_sun).flux_fraction
    return float(SOLAR_CONSTANT * (AU / sun_norm) ** 2 * fraction)


def altitude_from_position_m(r_m: np.ndarray) -> float:
    """Высота над сферой R_EARTH_EQUATOR, м."""
    return float(np.linalg.norm(r_m) - R_EARTH_EQUATOR)
