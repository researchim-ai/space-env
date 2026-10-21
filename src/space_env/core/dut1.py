"""
Параметры вращения Земли по суточному ряду IERS finals2000A.
Вне интервала снимка DUT1 = 0 и полюс в нуле: UT1 совпадает с UTC, полярного движения нет.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from space_env.core.time import Time

_TABLE = Path(__file__).resolve().parent / "eop" / "finals2000A.daily.csv"


@dataclass(frozen=True)
class EarthOrientation:
    jd: float
    dut1_s: float
    xp_arcsec: float
    yp_arcsec: float


@lru_cache(maxsize=1)
def _series() -> tuple[EarthOrientation, ...]:
    """Узлы на 0h UTC, по возрастанию JD."""
    rows: list[EarthOrientation] = []
    with _TABLE.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for record in reader:
            moment = datetime(int(record["Year"]), int(record["Month"]), int(record["Day"]), tzinfo=timezone.utc)
            rows.append(
                EarthOrientation(
                    Time(moment).jd,
                    float(record["UT1-UTC"]),
                    float(record["x_pole"]),
                    float(record["y_pole"]),
                )
            )
    rows.sort(key=lambda item: item.jd)
    return tuple(rows)


def bulletin_bounds() -> tuple[Time, Time]:
    series = _series()
    return Time(series[0].jd), Time(series[-1].jd)


def _interpolate(time: Time) -> EarthOrientation | None:
    series = _series()
    jd = time.jd
    if jd < series[0].jd or jd > series[-1].jd:
        return None
    for index in range(len(series) - 1):
        left = series[index]
        right = series[index + 1]
        if left.jd <= jd <= right.jd:
            span = right.jd - left.jd
            weight = 0.0 if span == 0.0 else (jd - left.jd) / span
            return EarthOrientation(
                jd,
                left.dut1_s + (right.dut1_s - left.dut1_s) * weight,
                left.xp_arcsec + (right.xp_arcsec - left.xp_arcsec) * weight,
                left.yp_arcsec + (right.yp_arcsec - left.yp_arcsec) * weight,
            )
    last = series[-1]
    return EarthOrientation(jd, last.dut1_s, last.xp_arcsec, last.yp_arcsec)


def dut1_seconds(time: Time) -> float:
    """Линейная интерполяция UT1−UTC, секунды. Вне снимка — 0."""
    sample = _interpolate(time)
    return 0.0 if sample is None else sample.dut1_s


def pole_arcsec(time: Time) -> tuple[float, float]:
    """Координаты полюса x, y в угловых секундах. Вне снимка — (0, 0)."""
    sample = _interpolate(time)
    if sample is None:
        return 0.0, 0.0
    return sample.xp_arcsec, sample.yp_arcsec
