"""Native causal measurements, EKF consistency kernels and public-view boundaries."""

import copy

import numpy as np
import pytest

from space_env.navigation import NavigationSession, SensorSpec, measurement_model
from space_env.navigation.models import predict, update
from space_env.simulation.snapshot import digest
from space_env.tasks.modeling_suite import compile_suite


@pytest.fixture(scope="module")
def scenario():
    return next(
        c["scenario"] for c in compile_suite()["cases"] if c["id"] == "orbit-nominal"
    )


def sensor(scenario, **changes):
    o = scenario["definition"]["orbits"]["a"]["initial"]
    state = np.array([*o["r_m"], *o["v_m_s"]])
    state[:3] += [20.0, -10.0, 5.0]
    return {
        "kind": "cartesian_position",
        "initial_estimate_si": state.tolist(),
        "initial_covariance_si": np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]).tolist(),
        "standard_deviation": [3.0, 3.0, 3.0],
        "seed": 42,
        **changes,
    }


def test_cartesian_update_matches_independent_information_form():
    x = np.array([7e6, 10.0, 20.0, 0.0, 7500.0, 0.0])
    p = np.diag([100.0, 50.0, 25.0, 1.0, 1.0, 1.0])
    r = np.diag([9.0, 4.0, 1.0])
    z = x[:3] + [5.0, -2.0, 1.0]
    actual, cov, info = update(x, p, z, "cartesian_position", r)
    h = np.eye(6)[:3]
    expected_cov = np.linalg.inv(np.linalg.inv(p) + h.T @ np.linalg.inv(r) @ h)
    expected = expected_cov @ (np.linalg.solve(p, x) + h.T @ np.linalg.solve(r, z))
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(cov, expected_cov, atol=1e-12)
    assert info["accepted"] and info["nis"] > 0


def test_range_rate_jacobian_matches_symmetric_finite_differences():
    x = np.array([7e6, 1000.0, 500.0, -10.0, 7400.0, 300.0])
    ref = np.array([7e6 + 1000.0, -100.0, 200.0, 1.0, 7410.0, 310.0])
    value, h = measurement_model("range_range_rate", x, ref)
    actual = np.zeros((2, 6))
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = 0.01 if i < 3 else 0.001
        actual[:, i] = (
            measurement_model("range_range_rate", x + delta, ref)[0]
            - measurement_model("range_range_rate", x - delta, ref)[0]
        ) / (2 * delta[i])
    np.testing.assert_allclose(h, actual, atol=3e-8, rtol=1e-6)
    assert value[0] > 0


def test_covariance_prediction_psd_and_process_noise(scenario):
    spec = sensor(scenario)
    x = np.array(spec["initial_estimate_si"])
    p = np.array(spec["initial_covariance_si"])
    a, base = predict(x, p, 2.0, 0.0)
    b, noisy = predict(x, p, 2.0, 0.01)
    np.testing.assert_allclose(a, b, atol=1e-8)
    assert np.linalg.eigvalsh(noisy - base).min() > 0
    assert np.linalg.eigvalsh(noisy).min() > 0


def test_outlier_gate_preserves_prior():
    x = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    p = np.eye(6)
    actual, cov, info = update(
        x, p, x[:3] + [100.0, 0.0, 0.0], "cartesian_position", np.eye(3), nis_gate=10.0
    )
    assert not info["accepted"]
    np.testing.assert_array_equal(actual, x)
    np.testing.assert_array_equal(cov, p)


def test_singular_innovation_rejected():
    with pytest.raises(ValueError, match="nonsingular"):
        update(
            [7e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
            np.zeros((6, 6)),
            [7e6, 0.0, 0.0],
            "cartesian_position",
            np.zeros((3, 3)),
        )


def test_latency_and_observations_do_not_consume_rng_or_future(scenario):
    s = NavigationSession(scenario, {"a": sensor(scenario, latency_sec=0.25)})
    initial = s.checkpoint()
    assert not s.observe("a")["received_samples"]
    assert s.checkpoint() == initial
    assert not s.advance(0.2, expected_revision=0, idempotency_key="early")[
        "observations"
    ]["a"]["received_samples"]
    row = s.advance(0.05, expected_revision=1, idempotency_key="receive")[
        "observations"
    ]["a"]["received_samples"][0]
    assert row["acquired_sec"] == 0 and row["received_sec"] == 0.25
    assert row["received_sec"] <= s.observe("a")["time_sec"]


def test_step_partition_invariance_and_cold_replay(scenario):
    spec = sensor(scenario, latency_sec=0.25, dropout_probability=0.3)
    a = NavigationSession(scenario, {"a": spec})
    b = NavigationSession(scenario, {"a": spec})
    for i in range(5):
        a.advance(1.0, expected_revision=i, idempotency_key=str(i))
    last = b.advance(5.0, expected_revision=0, idempotency_key="all")
    assert a.observe("a") == b.observe("a")
    assert a.measurement_history("a") == b.measurement_history("a")
    restored = NavigationSession.from_checkpoint(b.checkpoint())
    assert restored.observe("a") == b.observe("a")
    assert restored.advance(5.0, expected_revision=0, idempotency_key="all") == last
    with pytest.raises(ValueError, match="conflict"):
        restored.advance(1.0, expected_revision=0, idempotency_key="all")


def test_all_dropout_is_retained_and_covariance_not_artificially_reduced(scenario):
    s = NavigationSession(scenario, {"a": sensor(scenario, dropout_probability=1.0)})
    s.advance(5.0, expected_revision=0, idempotency_key="drop")
    rows = s.measurement_history("a")
    assert len(rows) == 6 and all(r["dropped"] for r in rows)
    assert all("value" not in row for row in rows)
    assert s.observe("a")["filter_epoch_sec"] == 0


def test_range_known_beacon_is_explicit_and_truth_errors_not_public(scenario):
    s = NavigationSession(
        scenario,
        {
            "a": sensor(
                scenario,
                kind="range_range_rate",
                reference_actor="b",
                standard_deviation=[2.0, 0.02],
            )
        },
    )
    s.advance(2.0, expected_revision=0, idempotency_key="range")
    view = s.observe("a")
    row = view["received_samples"][0]
    assert row["reference_actor"] == "b" and len(row["known_reference_state_si"]) == 6
    assert len(row["value"]) == 2
    assert not {"true_bias", "error_si", "nees", "scenario", "truth"} & set(view)
    assert "true_bias" not in str(row)
    assert s.diagnostics("a")["nees"] is not None
    with pytest.raises(ValueError):
        s.observe("b")


def test_advance_failure_is_atomic(scenario):
    s = NavigationSession(scenario, {"a": sensor(scenario)})
    before = s.checkpoint()
    with pytest.raises(ValueError):
        s.advance(
            s._runtime.horizon_sec + 1.0,
            expected_revision=0,
            idempotency_key="past-horizon",
        )
    assert s.checkpoint() == before
    with pytest.raises(ValueError):
        s.advance(1.0, expected_revision=1, idempotency_key="stale")
    assert s.checkpoint() == before


def test_resealed_checkpoint_cannot_invent_observations(scenario):
    s = NavigationSession(scenario, {"a": sensor(scenario)})
    payload = s.checkpoint()
    payload["observations_sha256"] = "0" * 64
    payload["sha256"] = digest({k: v for k, v in payload.items() if k != "sha256"})
    with pytest.raises(ValueError, match="replay mismatch"):
        NavigationSession.from_checkpoint(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"standard_deviation": [0.0, 1.0, 1.0]},
        {"standard_deviation": [1.0, 1.0]},
        {"true_bias": [1.0]},
        {"initial_covariance_si": [[1.0]]},
        {"reference_actor": "b"},
        {"frame": "ITRS"},
        {"initial_estimate_si": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"dropout_probability": 1.1},
        {"seed": True},
        {"cadence_sec": 0.0},
        {"standard_deviation": [float("nan"), 1.0, 1.0]},
    ],
)
def test_invalid_sensor_contract(scenario, change):
    with pytest.raises(ValueError):
        SensorSpec.model_validate(sensor(scenario, **change))


@pytest.mark.parametrize(
    "mode",
    ["force", "budget", "self_reference", "unknown_actor", "unknown_beacon", "horizon"],
)
def test_unsupported_profile_rejected(scenario, mode):
    c = copy.deepcopy(scenario)
    spec = sensor(c)
    actor = "a"
    if mode == "force":
        c["definition"]["orbits"]["a"]["forces"]["j2"] = True
    if mode == "budget":
        spec["max_samples"] = 1
    if mode == "self_reference":
        spec.update(
            kind="range_range_rate", reference_actor="a", standard_deviation=[2.0, 0.1]
        )
    if mode == "unknown_actor":
        actor = "unknown"
    if mode == "unknown_beacon":
        spec.update(
            kind="range_range_rate",
            reference_actor="unknown",
            standard_deviation=[2.0, 0.1],
        )
    if mode == "horizon":
        c["definition"]["horizon_sec"] = 3601.0
    with pytest.raises(ValueError):
        NavigationSession(c, {actor: spec})


@pytest.mark.parametrize("psd", [-1.0, float("nan"), True])
def test_invalid_process_noise_rejected(scenario, psd):
    spec = sensor(scenario)
    with pytest.raises(ValueError):
        predict(spec["initial_estimate_si"], spec["initial_covariance_si"], 1.0, psd)
