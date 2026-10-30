"""External force-formula comparison with explicitly shared frame/data inputs.

Boinor implements the comparison formulas. This is NOT independent validation of
EOP, atmospheric density, ephemerides, illumination or an operational orbit truth.
"""
from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path

import numpy as np
from boinor.core import perturbations
from scipy.integrate import solve_ivp

from space_env.core.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATOR
from space_env.core.frames import ecef_to_geodetic
from space_env.core.gravity import ambient_density_kg_m3
from space_env.reference import (
    EOPSeries,
    Epoch,
    ForceModel,
    FrameContext,
    ReferenceState,
    acceleration,
    propagate,
)
from space_env.reference.orbit import _ephemerides


def _function(name):
    function = getattr(perturbations, name)
    return getattr(function, "py_func", function)


def external_j2(state, epoch: Epoch, context: FrameContext) -> np.ndarray:
    rotation = context.matrix(epoch)
    fixed = np.concatenate((rotation @ state[:3], rotation @ state[3:])) / 1000.
    return rotation.T @ (_function("J2_perturbation")(0., fixed, MU_EARTH/1e9, J2_EARTH, R_EARTH_EQUATOR/1000.)*1000.)


def force_crosscheck(*, long_arcs: bool = False) -> dict:
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    rows = []
    cases = []
    for index, state in enumerate((np.array([6.8e6, 0, 0, 0, 7600, 0]),
                                    np.array([6e6, -3e6, 2e6, 1200, 6500, -2000]),
                                    np.array([0, -1e6, 6.9e6, 7400, 0, 900]))):
        central = acceleration(state, epoch, context, ForceModel())
        expected_j2 = external_j2(state, epoch, context)
        actual_j2 = acceleration(state, epoch, context, ForceModel(j2=True))-central
        fixed = context.jacobian(epoch) @ state
        _lat, _lon, height = ecef_to_geodetic(fixed[:3])
        rho = ambient_density_kg_m3(height, atmosphere="table", f107=150, density_scale=1)
        # Convert ALL units: kg/m3 -> kg/km3; m2/kg -> km2/kg; km/s2 -> m/s2.
        expected_drag = context.matrix(epoch).T @ (_function("atmospheric_drag")(
            0., fixed/1000, MU_EARTH/1e9, 2.2, 1e-6/100., rho*1e9)*1000)
        actual_drag = acceleration(state, epoch, context, ForceModel(drag=True))-central
        cases.append({"id": index, "state_si": state.tolist(), "density_kg_m3": rho})
        expected_third = np.zeros(3)
        for body, mu in zip(_ephemerides(epoch), (1.32712440018e20, 4.9048695e12)):
            expected_third += _function("third_body")(0., state/1000, MU_EARTH/1e9, mu/1e9,
                                                      lambda _t, body=body: body/1000)*1000
        actual_third = acceleration(state, epoch, context, ForceModel(third_body=True))-central
        for name, actual, expected in (("j2", actual_j2, expected_j2), ("drag", actual_drag, expected_drag),
                                        ("third_body", actual_third, expected_third)):
            error = float(np.linalg.norm(actual-expected))
            rows.append({"name": f"{name}_{index}", "error_m_s2": error, "tolerance_m_s2": 1e-12,
                         "passed": error <= 1e-12, "expected_m_s2": expected.tolist(), "actual_m_s2": actual.tolist()})
    arcs = []
    for duration in ([86400., 259200.] if long_arcs else [600.]):
        initial = ReferenceState(epoch, np.array([7e6, 0, 0]), np.array([0, 5000, 5700]))
        ours = propagate(initial, duration, context, model=ForceModel(j2=True), sample_sec=600)
        y0 = np.concatenate((initial.r_m, initial.v_m_s))

        def comparison_rhs(clock, state):
            r = state[:3]
            central = -MU_EARTH*r/np.linalg.norm(r)**3
            return np.concatenate((state[3:], central+external_j2(state, epoch.shifted(float(clock)), context)))

        # Different integration family (ODEPACK LSODA vs production DOP853).
        other = solve_ivp(comparison_rhs, (0., duration), y0, method="LSODA", t_eval=ours["times_sec"],
                          rtol=1e-12, atol=[1e-5]*3+[1e-8]*3, max_step=30.)
        if not other.success:
            raise RuntimeError(f"external comparison integration failed: {other.message}")
        difference = np.array(ours["states_si"])-other.y.T
        position = float(np.max(np.linalg.norm(difference[:, :3], axis=1)))
        velocity = float(np.max(np.linalg.norm(difference[:, 3:], axis=1)))
        arcs.append({"duration_sec": duration, "position_error_m": position, "velocity_error_m_s": velocity,
                     "position_tolerance_m": 10., "velocity_tolerance_m_s": .01,
                     "passed": position <= 10. and velocity <= .01})
    source = Path(inspect.getfile(perturbations))
    return {"schema": "space-env.force-crosscheck/1", "passed": all(x["passed"] for x in [*rows, *arcs]),
            "epoch_utc": epoch.iso(), "cases": cases, "context": context.manifest(),
            "harness_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "production_model_sources": ours["manifest"]["model_source_sha256"],
            "scipy_version": ours["manifest"]["scipy_version"], "forces": rows, "j2_arcs": arcs, "oracle": {"library": "boinor", "version": version("boinor"),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "functions": ["J2_perturbation", "atmospheric_drag", "third_body"], "integrator": "SciPy LSODA"},
            "shared_inputs": ["ERFA/EOP frames", "density", "Sun/Moon positions", "physical constants"],
            "not_qualified": ["SRP/eclipse comparison", "independent ephemerides and atmosphere",
                              "independent EOP/frame truth", "full perturbed mission operational accuracy"]}
