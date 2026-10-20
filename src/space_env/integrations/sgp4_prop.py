"""Пропагация TLE через sgp4. Пакет опциональный: pip install 'space-env[sgp4]'."""
from __future__ import annotations

import numpy as np

from space_env.core.orbit import Orbit
from space_env.core.time import Time


def orbit_from_tle(line1: str, line2: str, epoch: Time | None = None) -> Orbit:
    """
    Состояние SGP4 в метрах и м/с, обёрнутое в Orbit.
    Без epoch берётся эпоха TLE.
    """
    try:
        from sgp4.api import Satrec
    except ImportError as exc:
        raise ImportError("Для TLE установите extra sgp4: pip install 'space-env[sgp4]'") from exc
    satellite = Satrec.twoline2rv(line1, line2)
    if epoch is None:
        epoch = Time(float(satellite.jdsatepoch) + float(satellite.jdsatepochF))
    minutes = (epoch.jd - (float(satellite.jdsatepoch) + float(satellite.jdsatepochF))) * 1440.0
    error, r_km, v_km_s = satellite.sgp4_tsince(minutes)
    if error != 0:
        raise RuntimeError(f"sgp4 вернул код ошибки {error}")
    return Orbit.from_vectors(np.asarray(r_km, dtype=float) * 1000.0, np.asarray(v_km_s, dtype=float) * 1000.0, epoch)
