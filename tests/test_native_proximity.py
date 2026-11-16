import json

import numpy as np
import pytest

from space_env.core.constants import MU_EARTH
from space_env.core.relative import rtn_basis, rtn_state_to_eci
from space_env.industrial import NativeProximitySession
from space_env.industrial.native_proximity import (
    native_proximity_policy,
    native_proximity_recipe,
)
from space_env.industrial.safety import central_pericenter_m, check_central_interval


def step(s, a):
    return s.step(a, expected_revision=s.revision, idempotency_key=str(s.revision))


def test_nonlinear_native_inspection_and_delayed_estimates_replay():
    s = NativeProximitySession(native_proximity_recipe("latency"))
    while not s._done:
        step(s, native_proximity_policy(s.observe()))
    assert s.metrics()["success"] and s.metrics()["safe_intervals"]
    assert (
        NativeProximitySession.from_checkpoint(
            json.loads(json.dumps(s.checkpoint()))
        ).checkpoint()
        == s.checkpoint()
    )


def test_public_observation_never_contains_truth_or_future_packets():
    s = NativeProximitySession(native_proximity_recipe("blind"))
    obs = s.observe()
    assert "initial_truth_si" not in json.dumps(obs)
    assert "true_bias" not in json.dumps(obs)
    assert obs["relative_estimate_rtn_si"] != s._relative_truth(s._runtime).tolist()
    assert all(
        "value" not in p
        for n in obs["navigation"].values()
        for p in n["received_samples"]
    )


def test_impulse_axes_use_estimated_chief_and_native_fuel_is_paid():
    s = NativeProximitySession(native_proximity_recipe())
    obs = s.observe()
    chief = np.array(obs["navigation"][s.spec.chief]["estimate_si"])
    expected = rtn_basis(chief[:3], chief[3:]) @ np.array([0.1, 0.0, 0.0])
    step(s, {"kind": "impulse", "delta_v_rtn_m_s": [0.1, 0.0, 0.0]})
    control = s._runtime.command_status()[0]["command"]
    np.testing.assert_allclose(control["impulse"]["delta_v_m_s"], expected, atol=1e-12)
    assert s.metrics()["propellant_kg"] < 10.0
    assert s._runtime._backend.mission.consumed_wh["a"] > 0


def test_between_step_crossing_detected_for_real_gcrs_orbits():
    cfg = native_proximity_recipe("crossing")
    cfg["decision_sec"] = 10.0
    chief = cfg["scenario"]["definition"]["orbits"]["b"]["initial"]
    own = rtn_state_to_eci(
        [100.0, 0.0, 0.0, -20.0, 0.0, 0.0], chief["r_m"], chief["v_m_s"]
    )
    cfg["scenario"]["definition"]["orbits"]["a"]["initial"].update(
        r_m=own[:3].tolist(), v_m_s=own[3:].tolist()
    )
    cfg["sensors"]["a"]["initial_estimate_si"] = own.tolist()
    s = NativeProximitySession(cfg)
    response = step(s, {"kind": "coast"})
    assert response["terminated"] and not s.metrics()["safe_intervals"]
    providers = s._runtime._backend.mission.ephemerides
    assert (
        np.linalg.norm(providers["a"].state(10.0)[:3] - providers["b"].state(10.0)[:3])
        > 20
    )
    assert s._intervals[-1]["status"] == "unsafe"


def test_indeterminate_is_failure_not_safe():
    cfg = native_proximity_recipe()
    cfg.update(keepout_m=99.0, waypoint_m=[200.0, 0.0, 0.0], max_interval_evaluations=2)
    s = NativeProximitySession(cfg)
    step(s, {"kind": "coast"})
    assert (
        s._intervals[-1]["status"] == "indeterminate"
        and not s.metrics()["safe_intervals"]
    )


@pytest.mark.parametrize("change", ["j2", "coarse", "beacon"])
def test_unsupported_fidelity_profiles_rejected(change):
    cfg = native_proximity_recipe()
    if change == "j2":
        cfg["scenario"]["definition"]["orbits"]["a"]["forces"] = {"j2": True}
        cfg["sensors"]["a"]["force_profile"] = "matched_native"
    elif change == "coarse":
        cfg["scenario"]["definition"]["orbits"]["a"]["integration"] = {
            "position_atol_m": 1.0
        }
    else:
        cfg["sensors"]["a"].update(
            kind="range_range_rate", reference_actor="b", standard_deviation=[1.0, 0.1]
        )
    with pytest.raises(ValueError):
        NativeProximitySession(cfg)


def test_oversized_impulse_is_atomic_and_public_lists_are_detached():
    s = NativeProximitySession(native_proximity_recipe())
    before = s.checkpoint()
    with pytest.raises(ValueError):
        step(s, {"kind": "impulse", "delta_v_rtn_m_s": [50.0, 0.0, 0.0]})
    obs = s.observe()
    obs["task"]["waypoint_m"][0] = 9999
    assert s.checkpoint() == before


def test_central_bounds_against_separate_nonlinear_ode():
    from scipy.integrate import solve_ivp

    r = 7000000.0
    speed = (MU_EARTH / r) ** 0.5
    left = np.array([r, 0, 0, 0, speed, 0])
    right = np.array([r + 100, 0, 0, 0, speed + 0.1, 0])

    def solve(x):
        def rhs(t, y):
            return np.r_[y[3:], -MU_EARTH * y[:3] / np.linalg.norm(y[:3]) ** 3]

        return solve_ivp(rhs, (0, 10), x, rtol=1e-12, atol=1e-8, dense_output=True).sol

    a, b = solve(left), solve(right)
    result = check_central_interval(
        left, right, lambda t: a(t)[:3], lambda t: b(t)[:3], 10.0, 20.0
    )
    minimum = min(np.linalg.norm(a(t)[:3] - b(t)[:3]) for t in np.linspace(0, 10, 1001))
    assert result["status"] == "safe" and result["lower_bound_m"] <= minimum
    assert abs(central_pericenter_m(left) - r) < 1e-6


def test_caller_owned_sensor_mapping_cannot_change_live_navigation():
    from space_env.industrial.native_proximity import NativeProximitySpec

    cfg = NativeProximitySpec.model_validate(native_proximity_recipe())
    session = NativeProximitySession(cfg)
    cfg.sensors.clear()
    response = step(session, {"kind": "inspect"})
    assert set(response["observation"]["navigation"]) == {"a", "b"}
    assert (
        NativeProximitySession.from_checkpoint(session.checkpoint()).observe()
        == session.observe()
    )
