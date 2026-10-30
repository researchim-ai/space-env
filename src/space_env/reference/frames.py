"""Explicit GCRS/ITRS position, velocity and covariance transformations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.reference.time import EOPSeries, Epoch, erfa_backend

_ARCSEC = np.pi / (180. * 3600.)


def c2t_matrix(tt1: float, tt2: float, ut11: float, ut12: float, xp_rad: float, yp_rad: float) -> np.ndarray:
    """IAU 2006/2000A; arguments are explicit TT/UT1 JD parts and pole radians."""
    if not np.isfinite([tt1, tt2, ut11, ut12, xp_rad, yp_rad]).all():
        raise ValueError("frame arguments must be finite")
    return np.asarray(erfa_backend().c2t06a(tt1, tt2, ut11, ut12, xp_rad, yp_rad), dtype=float)


def _vector(value, length: int) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"expected a finite {length}-vector")
    return vector


def _covariance(value) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (6, 6) or not np.isfinite(covariance).all():
        raise ValueError("expected finite 6x6 covariance in SI units")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    scale = max(1., float(np.linalg.norm(covariance, 2)))
    if np.linalg.eigvalsh(covariance).min() < -1e-12 * scale:
        raise ValueError("covariance must be positive semidefinite")
    return covariance


@dataclass(frozen=True)
class FrameContext:
    eop: EOPSeries

    def matrix(self, epoch: Epoch) -> np.ndarray:
        orientation = self.eop.at(epoch)
        u1, u2 = erfa_backend().utcut1(*epoch.utc_jd, orientation.dut1_s)
        return c2t_matrix(*epoch.tt_jd, u1, u2, orientation.xp_arcsec * _ARCSEC, orientation.yp_arcsec * _ARCSEC)

    def jacobian(self, epoch: Epoch, *, inverse: bool = False) -> np.ndarray:
        rotation = self.matrix(epoch)
        # SI-second derivative in TAI; no float-JD quantization or fixed omega.
        h = .5
        derivative = (self.matrix(epoch.shifted(-2*h)) - 8*self.matrix(epoch.shifted(-h))
                      + 8*self.matrix(epoch.shifted(h)) - self.matrix(epoch.shifted(2*h))) / (12*h)
        jacobian = np.zeros((6, 6))
        if inverse:
            jacobian[:3, :3] = rotation.T
            jacobian[3:, 3:] = rotation.T
            jacobian[3:, :3] = -rotation.T @ derivative @ rotation.T
        else:
            jacobian[:3, :3] = rotation
            jacobian[3:, 3:] = rotation
            jacobian[3:, :3] = derivative
        return jacobian

    def transform(self, state, epoch: Epoch, *, source: str, destination: str, covariance=None) -> dict:
        if (source, destination) not in {("GCRS", "ITRS"), ("ITRS", "GCRS")}:
            raise ValueError("reference transform requires explicit GCRS/ITRS; legacy GCRF is not GCRS")
        jacobian = self.jacobian(epoch, inverse=source == "ITRS")
        result = {"frame": destination, "state_si": jacobian @ _vector(state, 6)}
        if covariance is not None:
            transformed = jacobian @ _covariance(covariance) @ jacobian.T
            result["covariance_si"] = .5 * (transformed + transformed.T)
        return result

    def manifest(self) -> dict:
        erfa = erfa_backend()
        table = erfa.leap_seconds.get()
        import hashlib

        return {"profile": "gcrs-reference/1", "rotation": "ERFA c2t06a; IAU2006/2000A",
                "time": "two-part UTC/TAI/TT/UT1; TAI SI-second arithmetic",
                "leap_table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
                "leap_applicability": "1972-2026 UTC input", "eop": self.eop.manifest(),
                "celestial_pole_offsets": "dX/dY not applied",
                "velocity_derivative": "five-point central derivative, h=0.5 SI seconds"}
