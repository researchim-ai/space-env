"""
Двухтелая пропагация через Orekit (orekit-jpype).
Нужны JDK 11+ и данные Orekit (setup_orekit_curdir / orekit-data), иначе Java-вызов завершится ошибкой.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

_VM_READY = False


def _start_vm() -> None:
    global _VM_READY
    if _VM_READY:
        return
    try:
        import orekit_jpype as orekit
    except ImportError as exc:
        raise ImportError(
            "Для Orekit установите extra orekit и JDK 11+: pip install 'space-env[orekit]'"
        ) from exc
    orekit.initVM()
    _VM_READY = True


def propagate_keplerian_m(
    a_m: float,
    ecc: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    nu_rad: float,
    epoch: datetime,
    dt_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Пропагация кеплеровой орбиты на dt_sec. Возвращает (r_m, v_m_s) в EME2000.
    epoch — timezone-aware UTC.
    """
    _start_vm()
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import Constants

    utc = TimeScalesFactory.getUTC()
    if epoch.tzinfo is None:
        raise ValueError("epoch должен быть timezone-aware UTC")
    epoch = epoch.astimezone(timezone.utc)
    initial = AbsoluteDate(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        float(epoch.second) + epoch.microsecond / 1e6,
        utc,
    )
    orbit = KeplerianOrbit(
        float(a_m),
        float(ecc),
        float(inc_rad),
        float(argp_rad),
        float(raan_rad),
        float(nu_rad),
        PositionAngleType.TRUE,
        FramesFactory.getEME2000(),
        initial,
        Constants.WGS84_EARTH_MU,
    )
    state = KeplerianPropagator(orbit).propagate(initial.shiftedBy(float(dt_sec)))
    pv = state.getPVCoordinates()
    position = pv.getPosition()
    velocity = pv.getVelocity()
    r_m = np.array([position.getX(), position.getY(), position.getZ()], dtype=float)
    v_m_s = np.array([velocity.getX(), velocity.getY(), velocity.getZ()], dtype=float)
    return r_m, v_m_s
