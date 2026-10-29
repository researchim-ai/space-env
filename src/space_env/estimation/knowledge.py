"""
Что можно утверждать по дуге измерений.
Одна станция на коротком проходе почти не видит поперёк трека.
Даже две станции на короткой дуге не становятся эксплуатационным определением орбиты.
"""
from __future__ import annotations

import numpy as np

from space_env.core.relative import rtn_basis
from space_env.estimation.range_arc import geometric_range_m
from space_env.core.time import Time


def knowledge_claim(stations: int, duration_sec: float, *, measurement: str = "range") -> dict[str, bool | str]:
    one_station = int(stations) < 2
    short = float(duration_sec) < 600.0
    angles = measurement == "angles"
    if measurement not in {"range", "angles"}:
        raise ValueError("измерение должно быть range или angles")
    if one_station and not angles:
        reason = "одна станция по дальности не видит поперёк трека и не является эксплуатационной оценкой"
    elif one_station:
        reason = "углы одной станции видят поперёк трека, но это не фильтр каталога"
    elif short:
        reason = "две станции на короткой дуге всё ещё не фильтр каталога"
    else:
        reason = "длинная дуга двух станций здесь не доводится до эксплуатационной оценки"
    return {
        "cross_track": angles or (not one_station and not short),
        "operational_od": False,
        "reason": reason,
    }


def range_shift_m(position: np.ndarray, velocity: np.ndarray, moment: Time, lat_deg: float, lon_deg: float, step_m: float = 1_000.0) -> tuple[float, float]:
    """Насколько меняется дальность от сдвига 1 км вдоль трека и поперёк него."""
    basis = rtn_basis(position, velocity)
    base = geometric_range_m(position, moment, lat_deg, lon_deg)
    along = geometric_range_m(position + basis[:, 1] * step_m, moment, lat_deg, lon_deg)
    cross = geometric_range_m(position + basis[:, 2] * step_m, moment, lat_deg, lon_deg)
    return abs(along - base), abs(cross - base)


def look_shift_rad(
    position: np.ndarray,
    velocity: np.ndarray,
    moment: Time,
    lat_deg: float,
    lon_deg: float,
    step_m: float = 1_000.0,
) -> tuple[float, float]:
    """Насколько поворачивается направление станция–аппарат от сдвига 1 км вдоль трека и поперёк."""
    from space_env.core.frames import eci_to_ecef, geodetic_to_ecef

    basis = rtn_basis(position, velocity)
    station = geodetic_to_ecef(lat_deg, lon_deg, 0.0)

    def direction(shift: np.ndarray) -> np.ndarray:
        delta = eci_to_ecef(position + shift, moment) - station
        return delta / np.linalg.norm(delta)

    base = direction(np.zeros(3))

    def angle(shift: np.ndarray) -> float:
        cosine = float(np.clip(np.dot(base, direction(shift)), -1.0, 1.0))
        return float(np.arccos(cosine))

    return angle(basis[:, 1] * step_m), angle(basis[:, 2] * step_m)
