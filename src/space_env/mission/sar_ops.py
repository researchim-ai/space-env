"""
Операционный SAR: угол падения, полоса режима, скважность и энергия.
Сырой радиолокационный кадр здесь не строится.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from space_env.core.access import footprint_diameter_m


def incidence_deg(elevation_deg: float) -> float | None:
    """Угол падения у цели, градусы. Ниже горизонта луча нет."""
    if elevation_deg < 0.0:
        return None
    return 90.0 - float(elevation_deg)


def mode_swath_m(altitude_m: float, beam_half_deg: float) -> float:
    return footprint_diameter_m(altitude_m, beam_half_deg)


@dataclass
class SarDuty:
    """Доля витка, которую можно светить. Сверх доли следующий импульс не начинается."""

    orbit_sec: float
    duty_fraction: float
    used_sec: float = 0.0

    def allow(self, seconds: float) -> bool:
        return self.used_sec + float(seconds) <= self.orbit_sec * self.duty_fraction + 1e-9

    def consume(self, seconds: float) -> bool:
        if not self.allow(seconds):
            return False
        self.used_sec += float(seconds)
        return True


def energy_wh(power_w: float, seconds: float) -> float:
    return float(power_w) * float(seconds) / 3600.0


def energy_allows(soc: float, capacity_wh: float, power_w: float, seconds: float) -> bool:
    return float(soc) * float(capacity_wh) + 1e-9 >= energy_wh(power_w, seconds)


def sar_look(
    *,
    elevation_deg: float,
    altitude_m: float,
    beam_half_deg: float,
    power_w: float,
    seconds: float,
    soc: float,
    capacity_wh: float,
    duty: SarDuty,
) -> dict:
    """Один операционный взгляд. Отказ по углу, скважности или энергии не тратит ресурс."""
    incidence = incidence_deg(elevation_deg)
    if incidence is None:
        return {"ok": False, "reason": "ниже горизонта", "incidence_deg": None, "swath_m": 0.0}
    swath = mode_swath_m(altitude_m, beam_half_deg)
    if not duty.allow(seconds):
        return {"ok": False, "reason": "скважность", "incidence_deg": incidence, "swath_m": swath}
    if not energy_allows(soc, capacity_wh, power_w, seconds):
        return {"ok": False, "reason": "энергия", "incidence_deg": incidence, "swath_m": swath}
    duty.consume(seconds)
    return {"ok": True, "reason": "", "incidence_deg": incidence, "swath_m": swath, "energy_wh": energy_wh(power_w, seconds)}
