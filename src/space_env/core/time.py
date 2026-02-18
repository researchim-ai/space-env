"""
Время и эпоха миссии (UTC).
Поддержка JD и ISO 8601; при установленном astropy — расширенная шкала (TAI, UT1).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

import numpy as np

# Константы для JD
# JD(2000-01-01 12:00 UTC) = 2451545.0
JD_EPOCH_2000: float = 2451545.0
SEC_PER_DAY: float = 86400.0


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
    """Юлианская дата → datetime UTC."""
    jd = float(jd)
    z = int(jd)
    f = jd - z
    if z >= 2299161:
        a = (z - 1867216.25) // 36524.25
        z += 1 + a - a // 4
    b = z + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = int(b - d - int(30.6001 * e))
    month = int(e - 1 if e < 14 else e - 13)
    year = int(c - 4716 if month > 2 else c - 4715)
    frac = f * 24.0
    hour = int(frac)
    frac = (frac - hour) * 60.0
    minute = int(frac)
    frac = (frac - minute) * 60.0
    second = int(frac)
    micro = int((frac - second) * 1e6)
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=timezone.utc)


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
