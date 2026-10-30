"""Adaptive GCRS orbit propagation with explicit terrestrial/ephemeris conventions.

Separate from the legacy mission executor. This is a model profile, not a
flight-qualified reference for all spacecraft regimes or arbitrary force settings.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from space_env.core.constants import (
    AU,
    J2_EARTH,
    MU_EARTH,
    R_EARTH_EQUATOR,
    SOLAR_PRESSURE_1AU_N_M2,
)
from space_env.reference.frames import FrameContext, _vector
from space_env.reference.time import Epoch, erfa_backend


@dataclass(frozen=True)
class ForceModel:
    j2: bool = False
    drag: bool = False
    srp: bool = False
    third_body: bool = False
    mass_kg: float = 100.
    drag_area_m2: float = 1.
    cd: float = 2.2
    srp_area_m2: float = 1.
    cr: float = 1.2
    atmosphere: str = "table"
    f107: float = 150.
    f107a: float = 150.
    ap: float = 4.
    density_scale: float = 1.

    def __post_init__(self):
        for name in ("j2", "drag", "srp", "third_body"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("force selectors must be booleans")
        for name in ("mass_kg", "drag_area_m2", "cd", "srp_area_m2", "cr", "f107", "f107a", "ap", "density_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mass_kg <= 0 or self.density_scale <= 0 or self.f107 <= 0 or self.f107a <= 0:
            raise ValueError("mass, density scale and fluxes must be positive")
        if self.atmosphere not in {"table", "nrlmsise"}:
            raise ValueError("atmosphere must be explicitly table or nrlmsise")
        if self.drag and self.atmosphere == "nrlmsise":
            try:
                import nrlmsise00  # noqa: F401
            except ImportError as exc:
                raise ImportError("explicit reference NRLMSISE requires nrlmsise00") from exc


@dataclass(frozen=True)
class ReferenceState:
    epoch: Epoch
    r_m: np.ndarray
    v_m_s: np.ndarray
    frame: str = "GCRS"

    def __post_init__(self):
        if self.frame != "GCRS":
            raise ValueError("reference orbit accepts GCRS only; legacy GCRF is not relabeled")
        for name in ("r_m", "v_m_s"):
            value = _vector(getattr(self, name), 3).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if np.linalg.norm(self.r_m) <= R_EARTH_EQUATOR:
            raise ValueError("reference orbit must be outside Earth")


def _ephemerides(epoch: Epoch) -> tuple[np.ndarray, np.ndarray]:
    erfa = erfa_backend()
    tt1, tt2 = epoch.tt_jd
    # Geocentric observer: dtdb's topocentric u/v terms are zero.
    tdb1, tdb2 = erfa.tttdb(tt1, tt2, erfa.dtdb(tt1, tt2, 0., 0., 0., 0.))
    earth, _barycentric = erfa.epv00(tdb1, tdb2)
    sun = -np.asarray(earth["p"]) * AU
    moon = np.asarray(erfa.moon98(tt1, tt2)["p"]) * AU
    return sun, moon


def acceleration(state_si, epoch: Epoch, context: FrameContext, model: ForceModel) -> np.ndarray:
    """Force vectors in GCRS; J2/drag evaluated in the declared terrestrial frame."""
    state = _vector(state_si, 6)
    r = state[:3]
    radius = float(np.linalg.norm(r))
    if radius <= R_EARTH_EQUATOR:
        raise ValueError("Earth intersection is outside this orbit profile")
    a = -MU_EARTH * r / radius**3
    rotation = context.matrix(epoch) if model.j2 or model.drag else None
    if model.j2:
        fixed = rotation @ r
        z2, r2 = fixed[2]**2, radius**2
        factor = 1.5 * J2_EARTH * MU_EARTH * R_EARTH_EQUATOR**2 / radius**5
        correction = factor * fixed * np.array([5*z2/r2-1, 5*z2/r2-1, 5*z2/r2-3])
        a += rotation.T @ correction
    if model.drag:
        from space_env.core.frames import ecef_to_geodetic
        from space_env.core.gravity import ambient_density_kg_m3

        fixed_state = context.jacobian(epoch) @ state
        lat, lon, height = ecef_to_geodetic(fixed_state[:3])
        if not 150e3 <= height <= 800e3:
            raise ValueError("reference drag applicability is 150-800 km; no table extrapolation")
        rho = ambient_density_kg_m3(height, atmosphere=model.atmosphere, f107=model.f107,
                                   f107a=model.f107a, ap=model.ap, density_scale=model.density_scale,
                                   lat_deg=lat, lon_deg=lon,
                                   when=epoch.datetime() if model.atmosphere == "nrlmsise" else None)
        relative_velocity = fixed_state[3:]  # Rdot*r + R*v relative to rotating atmosphere.
        a -= rotation.T @ (.5 * model.cd * model.drag_area_m2 / model.mass_kg * rho
                           * np.linalg.norm(relative_velocity) * relative_velocity)
    if model.srp or model.third_body:
        sun, moon = _ephemerides(epoch)
        if model.srp:
            from space_env.core.environment import shadow_state

            away = r - sun
            distance = float(np.linalg.norm(away))
            flux = shadow_state(r, sun).flux_fraction
            a += model.cr * model.srp_area_m2 / model.mass_kg * SOLAR_PRESSURE_1AU_N_M2 * (AU/distance)**2 * flux * away/distance
        if model.third_body:
            for body, mu in ((sun, 1.32712440018e20), (moon, 4.9048695e12)):
                relative = body - r
                a += mu * (relative / np.linalg.norm(relative)**3 - body / np.linalg.norm(body)**3)
    return a


def propagate(
    initial: ReferenceState, duration_sec: float, context: FrameContext, *,
    model: ForceModel | None = None, sample_sec: float = 60., rtol: float = 1e-11,
    position_atol_m: float = 1e-4, velocity_atol_m_s: float = 1e-7, max_step_sec: float = 60.,
) -> dict:
    """DOP853; numerical tolerances are not physical accuracy guarantees."""
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise ImportError("adaptive reference propagation requires scipy") from exc
    model = ForceModel() if model is None else model
    for value in (duration_sec, sample_sec, rtol, position_atol_m, velocity_atol_m_s, max_step_sec):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("duration, sampling and integration controls must be finite and positive")
    if duration_sec > 72*3600 or rtol < 1e-13 or rtol > 1e-5:
        raise ValueError("reference profile supports arcs <=72h and 1e-13 <= rtol <= 1e-5")
    if math.ceil(duration_sec/sample_sec) > 100_000:
        raise ValueError("reference output exceeds 100000 samples")
    context.matrix(initial.epoch)
    context.matrix(initial.epoch.shifted(duration_sec))
    if model.drag:
        context.jacobian(initial.epoch)
        context.jacobian(initial.epoch.shifted(duration_sec))
    times = np.append(np.arange(0, duration_sec, sample_sec), duration_sec)

    def derivative(clock, state):
        # Central-only propagation needs no celestial-to-terrestrial calculation.
        if not any((model.j2, model.drag, model.srp, model.third_body)):
            radius = float(np.linalg.norm(state[:3]))
            if radius <= R_EARTH_EQUATOR:
                raise ValueError("Earth intersection is outside this orbit profile")
            a = -MU_EARTH * state[:3] / radius**3
        else:
            a = acceleration(state, initial.epoch.shifted(float(clock)), context, model)
        return np.concatenate((state[3:], a))

    solution = solve_ivp(derivative, (0., duration_sec), np.concatenate((initial.r_m, initial.v_m_s)),
                         method="DOP853", t_eval=times, rtol=rtol,
                         atol=[position_atol_m]*3 + [velocity_atol_m_s]*3, max_step=max_step_sec)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(f"reference integration failed: {solution.message}")
    import scipy

    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "reference").glob("*.py")),
               *[root / "core" / name for name in ("constants.py", "frames.py", "gravity.py", "environment.py")]]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    return {"schema": "space-env.reference-orbit/1", "frame": "GCRS", "epoch_utc": initial.epoch.iso(),
            "epoch_tai_jd": [initial.epoch.tai1, initial.epoch.tai2],
            "times_sec": solution.t.tolist(), "states_si": solution.y.T.tolist(),
            "manifest": {**context.manifest(), "forces": asdict(model),
                         "integrator": "scipy DOP853", "scipy_version": scipy.__version__,
                         "erfa_version": erfa_backend().__version__, "numpy_version": np.__version__,
                         "model_source_sha256": hashes, "rtol": rtol,
                         "position_atol_m": position_atol_m, "velocity_atol_m_s": velocity_atol_m_s,
                         "sample_sec": sample_sec, "max_step_sec": max_step_sec, "evaluations": solution.nfev,
                         "scope": "Newtonian Earth satellite orbit; no burn/ADCS/mission executor qualification"}}
