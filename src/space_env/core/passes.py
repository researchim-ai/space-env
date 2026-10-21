"""
Окна связи: шаг орбиты вперёд, угол места и запас линка.
AOS — первый момент, где станция выше маски и линия закрывается.
Грубая сетка 60 с, затем половинное деление до refine_sec.
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
    include_third_body: bool = False,
    refine_sec: float = 1.0,
) -> list[PassWindow]:
    """Список сеансов на горизонте. aos_sec и los_sec — секунды от эпохи `epoch`."""
    if horizon_sec <= 0.0 or not stations:
        return []

    def _visible(position: np.ndarray, moment: Time, station: GroundStation) -> tuple[bool, float]:
        elevation = elevation_deg(position, station.lat_deg, station.lon_deg, station.alt_m, moment)
        slant = float(
            np.linalg.norm(
                eci_to_ecef(position, moment) - geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
            )
        )
        margin = link_margin_db(
            slant,
            frequency_hz=frequency_hz,
            tx_power_w=tx_power_w,
            tx_gain_dbi=tx_gain_dbi,
            rx_gain_dbi=rx_gain_dbi,
            system_loss_db=system_loss_db,
            required_power_dbw=required_power_dbw,
        )
        return elevation >= station.min_elevation_deg and margin >= min_margin_db, elevation

    def _propagate(position: np.ndarray, velocity: np.ndarray, dt: float, moment: Time):
        return propagate_state(
            position,
            velocity,
            dt,
            mass_kg=mass_kg,
            drag_area_m2=drag_area_m2,
            drag_coefficient=drag_coefficient,
            atmosphere=atmosphere,
            f107=f107,
            density_scale=density_scale,
            when=moment.datetime.replace(tzinfo=None),
            include_third_body=include_third_body,
        )

    def _refine(
        start: tuple[np.ndarray, np.ndarray, float],
        end_sec: float,
        station: GroundStation,
        *,
        want_visible: bool,
    ) -> float:
        lo_r, lo_v, lo = start
        hi = end_sec
        if refine_sec <= 0.0 or hi - lo <= refine_sec:
            return hi if want_visible else lo
        while hi - lo > refine_sec:
            mid = 0.5 * (lo + hi)
            step = mid - lo
            if step <= 1e-6:
                break
            mid_r, mid_v, _samples = _propagate(lo_r, lo_v, step, epoch + lo)
            seen, _elevation = _visible(mid_r, epoch + mid, station)
            if seen == want_visible:
                hi = mid
            else:
                lo, lo_r, lo_v = mid, mid_r, mid_v
        return hi if want_visible else lo

    r = np.asarray(r_m, dtype=float).copy()
    v = np.asarray(v_m_s, dtype=float).copy()
    open_pass: dict[str, tuple[float, float]] = {}
    finished: list[PassWindow] = []
    elapsed = 0.0
    previous: tuple[np.ndarray, np.ndarray, float] | None = None
    while elapsed <= horizon_sec + 1e-6:
        moment = epoch + elapsed
        for station in stations:
            visible, elevation = _visible(r, moment, station)
            current = open_pass.get(station.name)
            if visible and current is None:
                aos = elapsed if previous is None else _refine(previous, elapsed, station, want_visible=True)
                open_pass[station.name] = (aos, elevation)
            elif visible and current is not None:
                open_pass[station.name] = (current[0], max(current[1], elevation))
            elif not visible and current is not None:
                los = elapsed if previous is None else _refine(previous, elapsed, station, want_visible=False)
                finished.append(PassWindow(station.name, current[0], los, current[1]))
                del open_pass[station.name]
        previous = (r.copy(), v.copy(), elapsed)
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
            include_third_body=include_third_body,
        )
        elapsed += step
    for name, (aos, max_el) in open_pass.items():
        finished.append(PassWindow(name, aos, horizon_sec, max_el))
    finished.sort(key=lambda item: item.aos_sec)
    return finished
