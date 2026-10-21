"""
Окна связи: шаг орбиты вперёд, угол места и запас линка.
AOS — первый отсчёт, где станция выше маски и линия закрывается.
Точность границы — шаг поиска (по умолчанию 60 с).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.core.frames import eci_to_ecef, elevation_deg, geodetic_to_ecef
from space_env.core.ground import GroundStation
from space_env.core.link import link_margin_db
from space_env.core.propagate import propagate_state
from space_env.core.time import Time


@dataclass(frozen=True)
class PassWindow:
    station: str
    aos_sec: float
    los_sec: float
    max_elevation_deg: float


def find_passes(
    r_m: np.ndarray,
    v_m_s: np.ndarray,
    epoch: Time,
    stations: list[GroundStation],
    *,
    horizon_sec: float,
    mass_kg: float,
    drag_area_m2: float,
    drag_coefficient: float,
    frequency_hz: float,
    tx_power_w: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    system_loss_db: float,
    required_power_dbw: float,
    min_margin_db: float = 0.0,
    sample_sec: float = 60.0,
    atmosphere: str = "table",
    f107: float = 150.0,
    density_scale: float = 1.0,
) -> list[PassWindow]:
    """Список сеансов на горизонте. aos_sec и los_sec — секунды от эпохи `epoch`."""
    if horizon_sec <= 0.0 or not stations:
        return []
    r = np.asarray(r_m, dtype=float).copy()
    v = np.asarray(v_m_s, dtype=float).copy()
    open_pass: dict[str, tuple[float, float]] = {}
    finished: list[PassWindow] = []
    elapsed = 0.0
    while elapsed <= horizon_sec + 1e-6:
        moment = epoch + elapsed
        for station in stations:
            elevation = elevation_deg(r, station.lat_deg, station.lon_deg, station.alt_m, moment)
            slant = float(np.linalg.norm(eci_to_ecef(r, moment) - geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)))
            margin = link_margin_db(
                slant,
                frequency_hz=frequency_hz,
                tx_power_w=tx_power_w,
                tx_gain_dbi=tx_gain_dbi,
                rx_gain_dbi=rx_gain_dbi,
                system_loss_db=system_loss_db,
                required_power_dbw=required_power_dbw,
            )
            visible = elevation >= station.min_elevation_deg and margin >= min_margin_db
            current = open_pass.get(station.name)
            if visible and current is None:
                open_pass[station.name] = (elapsed, elevation)
            elif visible and current is not None:
                open_pass[station.name] = (current[0], max(current[1], elevation))
            elif not visible and current is not None:
                finished.append(PassWindow(station.name, current[0], elapsed, current[1]))
                del open_pass[station.name]
        if elapsed >= horizon_sec:
            break
        step = min(sample_sec, horizon_sec - elapsed)
        if step <= 1e-6:
            break
        r, v, _samples = propagate_state(
            r,
            v,
            step,
            mass_kg=mass_kg,
            drag_area_m2=drag_area_m2,
            drag_coefficient=drag_coefficient,
            atmosphere=atmosphere,
            f107=f107,
            density_scale=density_scale,
            when=moment.datetime.replace(tzinfo=None),
        )
        elapsed += step
    for name, (aos, max_el) in open_pass.items():
        finished.append(PassWindow(name, aos, horizon_sec, max_el))
    finished.sort(key=lambda item: item.aos_sec)
    return finished
