"""
Время и эпоха миссии (UTC).
Поддержка JD и ISO 8601.

Это календарный UTC datetime: високосные секунды не вставляются, шкала не равна TAI.
JD хранится как float64. Около 2026 года младший бит — около 50 мкс, поэтому
сравнение моментов в этом модуле обещает 1 мс, не наносекунды.
Проверенные фикстуры лежат в 1970–2100, григорианский календарь.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Union

import numpy as np

# Константы для JD
# JD(2000-01-01 12:00 UTC) = 2451545.0
JD_EPOCH_2000: float = 2451545.0
SEC_PER_DAY: float = 86400.0
# Допуск календарного roundtrip: float64 JD не держит наносекунды.
TIME_TOLERANCE_SEC: float = 1.0e-3


def datetime_to_jd(dt: datetime) -> float:
    """Преобразование datetime (UTC) в юлианскую дату."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # Формула из Meeus, упрощённо
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn - 0.5 + (
        dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3600.0 / 1e6
    ) / 24.0
    return jd


def jd_to_datetime(jd: float) -> datetime:
    """Юлианская дата → datetime UTC.

    Целая часть JD исторически начинается в полдень. Сдвиг +0.5 переносит её
    на календарные сутки. Прежняя формула брала int(jd) без сдвига и для
    полуночи 2026-10-20 возвращала предыдущий полдень, минус 12 часов.
    """
    if not math.isfinite(jd):
        raise ValueError("юлианская дата должна быть конечной")
    shifted = float(jd) + 0.5
    z = math.floor(shifted)
    frac_day = shifted - z
    if z >= 2299161:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - math.floor(alpha / 4.0)
    else:
        a = z
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day_float = b - d - math.floor(30.6001 * e) + frac_day
    day = int(math.floor(day_float))
    month = int(e - 1 if e < 14 else e - 13)
    year = int(c - 4716 if month > 2 else c - 4715)
    microseconds = int(round((day_float - day) * SEC_PER_DAY * 1.0e6))
    return datetime(year, month, day, tzinfo=timezone.utc) + timedelta(microseconds=microseconds)


class Time:
    """
    Момент времени в UTC.
    Хранит юлианскую дату и позволяет получать datetime, секунды от эпохи и т.д.
    """

    __slots__ = ("_jd",)

    def __init__(self, value: Union[datetime, float, "Time", str]) -> None:
        if isinstance(value, Time):
            self._jd = value._jd
        elif isinstance(value, datetime):
            self._jd = datetime_to_jd(value)
        elif isinstance(value, (int, float)):
            # Считаем юлианской датой
            self._jd = float(value)
        elif isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self._jd = datetime_to_jd(dt)
        else:
            raise TypeError(f"Time не поддерживает тип {type(value)}")

    @property
    def jd(self) -> float:
        """Юлианская дата."""
        return self._jd

    @property
    def datetime(self) -> datetime:
        """datetime в UTC."""
        return jd_to_datetime(self._jd)

    def __add__(self, delta_seconds: float) -> "Time":
        """Сдвиг на delta_seconds секунд."""
        return Time(self._jd + delta_seconds / SEC_PER_DAY)

    def __sub__(self, other: Union["Time", float]) -> Union["Time", float]:
        if isinstance(other, Time):
            return (self._jd - other._jd) * SEC_PER_DAY
        return self + (-other)

    def seconds_since(self, epoch: "Time") -> float:
        """Секунды от заданной эпохи."""
        return (self._jd - epoch._jd) * SEC_PER_DAY

    def __repr__(self) -> str:
        return f"Time({self.datetime.isoformat()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return abs(self._jd - other._jd) < 1e-9

    def to_numpy(self) -> np.floating:
        """JD в виде numpy для совместимости с boinor."""
        return np.float64(self._jd)
