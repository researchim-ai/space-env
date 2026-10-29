"""
Спрос вне удобного прохода.
Точки выбираются на сфере, затем честно проверяются доступом прибора.
Длинная полоса не становится достижимой от того, что её центр лежит на трассе.
"""
from __future__ import annotations

import numpy as np

from space_env.catalog.build import make_engine
from space_env.core.access import footprint_diameter_m, subsatellite_latlon, target_access
from space_env.core.constants import R_EARTH_EQUATOR

_EPOCH = "2026-10-20T10:00:00Z"


def _samples(horizon_sec: float, step_sec: float):
    engine = make_engine("sentinel-2a", epoch=_EPOCH, nu_deg=80.0, with_stations=False)
    engine.spacecraft.config.name = "optic"
    taken = []
    elapsed = 0.0
    while elapsed <= horizon_sec + 1e-9:
        taken.append((elapsed, engine.orbit.r.copy(), engine.now()))
        if elapsed >= horizon_sec - 1e-9:
            break
        step = min(step_sec, horizon_sec - elapsed)
        engine.step(step, "wait")
        elapsed += step
    return engine, taken


def _visible(engine, position, lat: float, lon: float, moment) -> bool:
    radar = bool(engine.spacecraft.config.payload.images_in_eclipse)
    access = target_access(
        position,
        lat,
        lon,
        moment,
        half_fov_deg=engine.spacecraft.config.payload.half_fov_deg,
        frame=engine.orbit.frame,
        images_in_eclipse=radar,
    )
    return access.visible


def independent_demand(
    seed: int = 0,
    *,
    count: int = 8,
    horizon_sec: float = 3600.0,
    step_sec: float = 60.0,
) -> dict:
    """
    count точек на сфере плюс одна полоса 5000 км на подспутниковой точке старта.
    Достижимость — проход прибора, а не открытая станция.
    """
    if count < 1:
        raise ValueError("в спросе нужна хотя бы одна точка")
    engine, samples = _samples(horizon_sec, step_sec)
    altitude = float(np.linalg.norm(samples[0][1]) - R_EARTH_EQUATOR)
    half_fov = engine.spacecraft.config.payload.half_fov_deg
    footprint_m = footprint_diameter_m(altitude, half_fov)
    rng = np.random.default_rng(seed)
    issued: list[dict] = []
    for number in range(count):
        lat = float(rng.uniform(-70.0, 70.0))
        lon = float(rng.uniform(-180.0, 180.0))
        hits = [mark for mark, position, moment in samples if _visible(engine, position, lat, lon, moment)]
        reachable = bool(hits)
        issued.append(
            {
                "id": f"aoi-{number:02d}",
                "lat_deg": lat,
                "lon_deg": lon,
                "geometry": "point",
                "length_km": 0.0,
                "priority": int(1 + number % 3),
                "reachable": reachable,
                "reason": "" if reachable else "вне трассы",
                "first_sec": None if not hits else float(hits[0]),
            }
        )
    feasible = None
    for mark, position, moment in samples:
        lat, lon = subsatellite_latlon(position, moment)
        if _visible(engine, position, float(lat), float(lon), moment):
            feasible = (float(mark), float(lat), float(lon))
            break
    if feasible is not None:
        issued.append(
            {
                "id": "track-point",
                "lat_deg": feasible[1],
                "lon_deg": feasible[2],
                "geometry": "point",
                "length_km": 0.0,
                "priority": 1,
                "reachable": True,
                "reason": "",
                "first_sec": feasible[0],
            }
        )
    start_lat, start_lon = subsatellite_latlon(samples[0][1], samples[0][2])
    strip_m = 5_000_000.0
    issued.append(
        {
            "id": "strip-long",
            "lat_deg": float(start_lat),
            "lon_deg": float(start_lon),
            "geometry": "strip",
            "length_km": strip_m / 1000.0,
            "priority": 1,
            "reachable": strip_m <= footprint_m,
            "reason": "" if strip_m <= footprint_m else "полоса длиннее кадра",
            "first_sec": 0.0,
            "footprint_diameter_m": footprint_m,
        }
    )
    return {
        "epoch": _EPOCH,
        "horizon_sec": horizon_sec,
        "footprint_diameter_m": footprint_m,
        "issued": issued,
        "reachable": sum(1 for item in issued if item["reachable"]),
        "unreachable": sum(1 for item in issued if not item["reachable"]),
    }
