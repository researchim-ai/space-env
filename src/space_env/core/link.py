"""
Линк-бюджет S-диапазона: ЭИИМ, потери в свободном пространстве, запас над чувствительностью.
FSPL = 20 log10(d_км) + 20 log10(f_МГц) + 32.44.
Сеанс возможен, когда запас неотрицательный. Скорость ступенчатая, как у простого модема.
Межспутниковая линия дополнительно проверяет, не пересекает ли луч Землю.
"""
from __future__ import annotations

import math

import numpy as np

from space_env.core.constants import R_EARTH_EQUATOR


def free_space_path_loss_db(range_m: float, frequency_hz: float) -> float:
    d_km = max(float(range_m), 1.0) / 1000.0
    f_mhz = max(float(frequency_hz), 1.0) / 1e6
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.44


def link_margin_db(
    range_m: float,
    *,
    frequency_hz: float,
    tx_power_w: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    system_loss_db: float,
    required_power_dbw: float,
) -> float:
    """Запас принятой мощности над требуемой, дБ. Отрицательный запас — линия не закрывается."""
    if tx_power_w <= 0.0:
        return -99.0
    eirp_dbw = 10.0 * math.log10(tx_power_w) + tx_gain_dbi
    received_dbw = eirp_dbw + rx_gain_dbi - free_space_path_loss_db(range_m, frequency_hz) - system_loss_db
    return received_dbw - required_power_dbw


def earth_occults(r1_m: np.ndarray, r2_m: np.ndarray, limb_radius_m: float | None = None) -> bool:
    """Луч между двумя точками проходит сквозь лимб. Радиус по умолчанию — экватор плюс 80 км."""
    start = np.asarray(r1_m, dtype=float)
    chord = np.asarray(r2_m, dtype=float) - start
    length2 = float(np.dot(chord, chord))
    if length2 < 1.0:
        return False
    closest = float(np.clip(np.dot(-start, chord) / length2, 0.0, 1.0))
    approach = start + closest * chord
    limit = R_EARTH_EQUATOR + 80_000.0 if limb_radius_m is None else limb_radius_m
    return float(np.linalg.norm(approach)) < limit


ISL_RATE_MAX_BPS: float = 1.0e6


def isl_rate_bps(range_m: float) -> float:
    """Ступени того же модема на межспутниковой линии. Закрытие Землёй проверяют снаружи."""
    return stepped_rate_bps(isl_margin_db(range_m), ISL_RATE_MAX_BPS)


def isl_margin_db(range_m: float) -> float:
    """
    Межспутниковая линия S-диапазона: 1 Вт и 6 дБи с каждой стороны, порог −140 дБВт.
    Закрытие Землёй сюда не входит — его проверяют отдельно.
    """
    return link_margin_db(
        range_m,
        frequency_hz=2.2e9,
        tx_power_w=1.0,
        tx_gain_dbi=6.0,
        rx_gain_dbi=6.0,
        system_loss_db=2.0,
        required_power_dbw=-140.0,
    )


def stepped_rate_bps(margin_db: float, rate_max_bps: float) -> float:
    """Четыре ступени модема. Ниже нуля запаса передачи нет."""
    if margin_db < 0.0 or rate_max_bps <= 0.0:
        return 0.0
    if margin_db < 3.0:
        return 0.25 * rate_max_bps
    if margin_db < 8.0:
        return 0.5 * rate_max_bps
    return rate_max_bps
