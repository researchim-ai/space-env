"""Одноузловой тепловой баланс: Солнце, альбедо, ИК Земли, внутренний теплоприток, излучение."""
from __future__ import annotations

import numpy as np

from space_env.core.constants import EARTH_ALBEDO, EARTH_IR_W_M2, R_EARTH_EQUATOR, SIGMA_SB

_T_SPACE_K = 3.0


def environmental_heat_w(
    *,
    r_sat_m: np.ndarray,
    r_sun_m: np.ndarray,
    solar_flux_w_m2: float,
    absorptivity: float,
    emissivity: float,
    solar_area_m2: float,
    earth_area_m2: float,
) -> tuple[float, float, float]:
    """
    (Солнце, альбедо, ИК Земли), Вт.
    Вид-фактор диска Земли для пластины в надир — (Re/r)².
    Альбедо пропорционально косинусу зенитного угла Солнца в подспутниковой точке
    и обнуляется на ночной стороне.
    """
    r = np.asarray(r_sat_m, dtype=float)
    sun = np.asarray(r_sun_m, dtype=float)
    radius = float(np.linalg.norm(r))
    sun_norm = float(np.linalg.norm(sun))
    q_solar = absorptivity * solar_area_m2 * solar_flux_w_m2
    if radius < 1.0 or sun_norm < 1.0:
        return q_solar, 0.0, 0.0
    view = (R_EARTH_EQUATOR / radius) ** 2
    cos_zenith = float(np.clip(np.dot(r / radius, sun / sun_norm), -1.0, 1.0))
    # Поток Солнца у Земли для альбедо берём тот же, что уже пришёл на КА вне тени;
    # в тени solar_flux_w_m2 = 0, и альбедо тоже гаснет вместе с cos_zenith < 0.
    q_albedo = absorptivity * earth_area_m2 * solar_flux_w_m2 * EARTH_ALBEDO * view * max(cos_zenith, 0.0)
    q_ir = emissivity * earth_area_m2 * EARTH_IR_W_M2 * view
    return float(q_solar), float(q_albedo), float(q_ir)


def step_temperature_c(
    temp_c: float,
    dt_sec: float,
    *,
    q_sun_w: float,
    q_diss_w: float,
    q_heater_w: float,
    radiating_area_m2: float,
    emissivity: float,
    heat_capacity_j_k: float,
) -> float:
    """
    Явный шаг Эйлера. Излучение — закон Стефана–Больцмана в небо 3 К.
    q_sun_w здесь — уже сумма Солнца, альбедо и ИК Земли.
    """
    if heat_capacity_j_k <= 0.0:
        return temp_c
    t_k = temp_c + 273.15
    q_rad = emissivity * SIGMA_SB * radiating_area_m2 * (t_k**4 - _T_SPACE_K**4)
    q_net = q_sun_w + q_diss_w + q_heater_w - q_rad
    return float(temp_c + q_net / heat_capacity_j_k * dt_sec)
