"""
Сверка геометрического Cd·A/m с B* из снимка TLE.
B* — подгоночный член SGP4 в 1/радиус Земли. Он впитывает ошибки модели и бывает
отрицательным, поэтому геометрический мидель карточки им не заменяется.
"""
from __future__ import annotations

from dataclasses import dataclass

from space_env.catalog.build import read_tle, spacecraft_config

# ρ0 = 2.461e-5 кг/м²/R_E · XKMPER, XKMPER = 6378.135 км. Так B* переводится в Cd·A/m.
SGP4_RHO0_KG_M2_PER_ER: float = 2.461e-5 * 6378.135


@dataclass(frozen=True)
class DragComparison:
    key: str
    bstar_per_er: float
    tle_cd_area_per_kg: float
    geometric_cd_area_per_kg: float

    @property
    def geometric_over_tle(self) -> float:
        return self.geometric_cd_area_per_kg / self.tle_cd_area_per_kg


def bstar_per_er(line1: str) -> float:
    """Поле B* строки 1 TLE, колонки 54–61, в 1/радиус Земли."""
    raw = line1[53:61].strip()
    sign = -1.0 if raw.startswith("-") else 1.0
    body = raw[1:] if raw[:1] in "+-" else raw
    mantissa = float(body[:-2])
    exponent = int(body[-2:])
    return sign * mantissa * 10.0 ** (exponent - 5)


def cd_area_per_kg_from_bstar(bstar: float) -> float:
    """Cd·A/m, м²/кг. B* = ρ0 · Cd · A / (2 m), ρ0 в кг/(м²·R_E)."""
    if bstar == 0.0:
        return 0.0
    return 2.0 * float(bstar) / SGP4_RHO0_KG_M2_PER_ER


def drag_comparison(key: str) -> DragComparison:
    line1, _line2 = read_tle(key)
    bstar = bstar_per_er(line1)
    prop = spacecraft_config(key).propulsion
    geometric = prop.drag_coefficient * prop.drag_area_m2 / prop.mass_kg
    return DragComparison(key, bstar, cd_area_per_kg_from_bstar(bstar), geometric)
