"""
Суточные F10.7 и планетарный Ap по снимку NOAA SWPC.
f107a — среднее скорректированного потока Пенктитона за 81 сутки, кончающиеся этим днём
(компиляция IUP Bremen). Центрированное среднее потребовало бы дни после 20 октября 2026,
а их в выпуске от 21 октября нет.
Вне окна снимка SWPC индекса нет: сценарий остаётся при своих значениях по умолчанию.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from space_env.core.time import Time

_TABLE = Path(__file__).resolve().parent / "eop" / "swpc_daily.csv"
_FLUX = Path(__file__).resolve().parent / "eop" / "f107_trailing.csv"
_TRAILING_DAYS = 81


@dataclass(frozen=True)
class SolarIndices:
    f107: float
    f107a: float
    ap: float


@lru_cache(maxsize=1)
def _rows() -> tuple[tuple[float, float, float], ...]:
    found: list[tuple[float, float, float]] = []
    with _TABLE.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            moment = datetime.fromisoformat(record["date"]).replace(tzinfo=timezone.utc)
            found.append((Time(moment).jd, float(record["f107"]), float(record["ap"])))
    found.sort()
    return tuple(found)


@lru_cache(maxsize=1)
def _flux() -> tuple[tuple[float, float], ...]:
    found: list[tuple[float, float]] = []
    with _FLUX.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            moment = datetime.fromisoformat(record["date"]).replace(tzinfo=timezone.utc)
            found.append((Time(moment).jd, float(record["f107_sfu"])))
    found.sort()
    return tuple(found)


def trailing_f107a(time: Time) -> tuple[float, int] | None:
    """Среднее и число суток. None, если до этой даты нет 81 измерения."""
    series = _flux()
    ended = [flux for jd, flux in series if jd <= time.jd + 1e-6]
    if len(ended) < _TRAILING_DAYS:
        return None
    window = ended[-_TRAILING_DAYS:]
    return sum(window) / _TRAILING_DAYS, _TRAILING_DAYS


def indices_for(time: Time) -> SolarIndices | None:
    """Индексы суток UTC, в которые попадает момент. None, если дата не попала в снимок SWPC."""
    rows = _rows()
    if time.jd < rows[0][0] or time.jd >= rows[-1][0] + 1.0:
        return None
    chosen = rows[0]
    for item in rows:
        if item[0] <= time.jd + 1e-9:
            chosen = item
        else:
            break
    averaged = trailing_f107a(time)
    if averaged is None:
        return None
    return SolarIndices(chosen[1], averaged[0], chosen[2])
