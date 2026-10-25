"""
Орбита: Keplerian / Cartesian, пропагация через boinor.
Все величины в СИ (м, м/с, рад) на границе; внутри — конвертация в boinor (km, astropy).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from space_env.core.constants import MU_EARTH, R_EARTH_EQUATOR
from space_env.core.frames import NUMERICAL_FRAME, require_orbit_frame
from space_env.core.time import Time

if TYPE_CHECKING:
    pass

# boinor + astropy (boinor тянет astropy)
from astropy import units as u
from astropy.time import Time as AstropyTime, TimeDelta
from boinor.bodies import Earth
from boinor.twobody import Orbit as BoinorOrbit


class KeplerianElements(NamedTuple):
    """Кеплеровы элементы (СИ: м, рад)."""
    a: float       # большая полуось, м
    ecc: float     # эксцентриситет
    inc: float     # наклонение, рад
    raan: float    # долгота восходящего узла, рад
    argp: float    # аргумент перицентра, рад
    nu: float      # истинная аномалия, рад


def _to_astropy_time(t: Time) -> AstropyTime:
    return AstropyTime(t.jd, format="jd", scale="utc")


def _from_astropy_time(t: AstropyTime) -> Time:
    return Time(t.jd)


class Orbit:
    """
    Орбита вокруг Земли. Эпоха и состояние (r, v) в СИ.
    Пропагация — двухтелая модель через boinor.
    """

    __slots__ = ("_orb", "_epoch", "_frame")

    def __init__(self, boinor_orbit: BoinorOrbit, epoch: Time, frame: str = NUMERICAL_FRAME) -> None:
        self._orb = boinor_orbit
        self._epoch = epoch
        self._frame = require_orbit_frame(frame)

    @classmethod
    def from_keplerian(
        cls,
        a_m: float,
        ecc: float,
        inc_rad: float,
        raan_rad: float,
        argp_rad: float,
        nu_rad: float,
        epoch: Time,
    ) -> "Orbit":
        """Орбита из кеплеровых элементов. a_m — в метрах, углы в радианах."""
        epoch_ap = _to_astropy_time(epoch)
        a_km = (a_m / 1000.0) * u.km
        orb = BoinorOrbit.from_classical(
            Earth,
            a_km,
            ecc * u.dimensionless_unscaled,
            inc_rad * u.rad,
            raan_rad * u.rad,
            argp_rad * u.rad,
            nu_rad * u.rad,
            epoch=epoch_ap,
        )
        return cls(orb, epoch, NUMERICAL_FRAME)

    @classmethod
    def from_vectors(
        cls,
        r_m: np.ndarray,
        v_m_s: np.ndarray,
        epoch: Time,
        frame: str = NUMERICAL_FRAME,
    ) -> "Orbit":
        """Орбита из векторов (м, м/с). Подпись сохраняется и не заменяется преобразованием."""
        r_km = (np.asarray(r_m, dtype=float) / 1000.0) * u.km
        v_km_s = (np.asarray(v_m_s, dtype=float) / 1000.0) * u.km / u.s
        epoch_ap = _to_astropy_time(epoch)
        orb = BoinorOrbit.from_vectors(Earth, r_km, v_km_s, epoch=epoch_ap)
        return cls(orb, epoch, frame)

    @property
    def epoch(self) -> Time:
        return self._epoch

    @property
    def frame(self) -> str:
        """Подпись системы координат векторов. SGP4 — TEME, численная дуга — GCRF."""
        return self._frame

    @property
    def r(self) -> np.ndarray:
        """Вектор положения в м (инерциальная СК)."""
        r_km = self._orb.r.to_value(u.km)
        return np.array(r_km, dtype=float) * 1000.0

    @property
    def v(self) -> np.ndarray:
        """Вектор скорости в м/с."""
        v_km_s = self._orb.v.to_value(u.km / u.s)
        return np.array(v_km_s, dtype=float) * 1000.0

    def keplerian(self) -> KeplerianElements:
        """Кеплеровы элементы (a в м, углы в рад)."""
        el = self._orb.classical()
        # boinor .classical() возвращает tuple (a, ecc, inc, raan, argp, nu) с astropy Quantity
        a_km = el[0].to_value(u.km)
        ecc = float(el[1].value) if hasattr(el[1], "value") else float(el[1])
        return KeplerianElements(
            a=a_km * 1000.0,
            ecc=ecc,
            inc=float(el[2].to_value(u.rad)),
            raan=float(el[3].to_value(u.rad)),
            argp=float(el[4].to_value(u.rad)),
            nu=float(el[5].to_value(u.rad)),
        )

    def propagate(self, delta_seconds: float) -> "Orbit":
        """Двухтелый шаг на delta_seconds. Подпись системы сохраняется; это не шаг SGP4 и не переход кадров."""
        delta = TimeDelta(delta_seconds, format="sec")
        new_orb = self._orb.propagate(delta)
        new_epoch = self._epoch + delta_seconds
        return Orbit(new_orb, new_epoch, self._frame)

    def altitude_km(self) -> float:
        """Высота над сферой R_EARTH_EQUATOR, км."""
        r_norm = np.linalg.norm(self.r)
        return (r_norm - R_EARTH_EQUATOR) / 1000.0

    def with_state(self, r_m: np.ndarray, v_m_s: np.ndarray, epoch: Time | None = None) -> "Orbit":
        """Новые векторы в той же системе. Подпись не меняется и векторы не переводятся."""
        return Orbit.from_vectors(r_m, v_m_s, self.epoch if epoch is None else epoch, frame=self._frame)

    def with_delta_v(self, dv_m_s: np.ndarray) -> "Orbit":
        """Импульс в той же системе: новая скорость, та же эпоха и та же подпись."""
        dv = np.asarray(dv_m_s, dtype=float)
        return self.with_state(self.r, self.v + dv)
