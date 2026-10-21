"""
Линк-бюджет S-диапазона: ЭИИМ, потери в свободном пространстве, запас над чувствительностью.
FSPL = 20 log10(d_км) + 20 log10(f_МГц) + 32.44.
Сеанс возможен, когда запас неотрицательный. Скорость ступенчатая, как у простого модема.
"""
from __future__ import annotations

import math


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


def stepped_rate_bps(margin_db: float, rate_max_bps: float) -> float:
    """Четыре ступени модема. Ниже нуля запаса передачи нет."""
    if margin_db < 0.0 or rate_max_bps <= 0.0:
        return 0.0
    if margin_db < 3.0:
        return 0.25 * rate_max_bps
    if margin_db < 8.0:
        return 0.5 * rate_max_bps
    return rate_max_bps
