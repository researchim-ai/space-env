"""Положение Солнца через spiceypy. Нужны загруженные ядра SPICE."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_spice_kernels(paths: list[str | Path]) -> None:
    """furnsh для каждого ядра. Без spiceypy — ImportError."""
    try:
        import spiceypy
    except ImportError as exc:
        raise ImportError("Для SPICE установите extra spice: pip install 'space-env[spice]'") from exc
    for path in paths:
        spiceypy.furnsh(str(path))


def sun_position_spice_m(ephemeris_time: float) -> np.ndarray:
    """
    Вектор Земля → Солнце в J2000, м.
    ephemeris_time — ET (секунды TDB от J2000), как его ждёт spiceypy.spkpos.
    """
    try:
        import spiceypy
    except ImportError as exc:
        raise ImportError("Для SPICE установите extra spice: pip install 'space-env[spice]'") from exc
    position_km, _light_time = spiceypy.spkpos("SUN", ephemeris_time, "J2000", "NONE", "EARTH")
    return np.asarray(position_km, dtype=float) * 1000.0
