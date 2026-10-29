"""
Суточные F10.7 и планетарный Ap по снимку NOAA SWPC.
f107a — среднее скорректированного потока Пенктитона за 81 сутки, кончающиеся этим днём
(компиляция IUP Bremen). Центрированное среднее — ``centered_f107a``: оно есть только там,
где в снимке есть 40 суток до даты и 40 после. На 20 октября 2026 хвоста нет.
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
    f107_centered: float | None = None


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


def centered_f107a(time: Time) -> tuple[float, int] | None:
    """
    Среднее F10.7 за 81 сутки, симметрично вокруг даты: 40 до, эти сутки и 40 после.
    None, если снимок не покрывает оба края. На 20 октября 2026 хвоста уже нет.
    """
    series = _flux()
    index = None
    for place, (jd, _flux_value) in enumerate(series):
        if jd <= time.jd + 0.5:
            index = place
        else:
            break
    if index is None:
        return None
    lo = index - 40
    hi = index + 40
    if lo < 0 or hi >= len(series):
        return None
    window = series[lo : hi + 1]
    span_days = window[-1][0] - window[0][0]
    if len(window) != 81 or not 79.0 <= span_days <= 82.0:
        return None
    return sum(item[1] for item in window) / 81.0, 81


def indices_for(time: Time) -> SolarIndices | None:
    """Индексы суток UTC. f107a — центрированные 81 сутки, если оба края есть в снимке, иначе скользящие 81."""
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
    centered = centered_f107a(time)
    f107a = averaged[0] if centered is None else centered[0]
    return SolarIndices(chosen[1], f107a, chosen[2], None if centered is None else centered[0])
