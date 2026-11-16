import copy

import numpy as np
import pytest

from space_env.industrial import ProximitySession
from space_env.industrial.proximity import ProximitySpec
from space_env.simulation.snapshot import digest


def step(session, action):
    return session.step(
        action,
        expected_revision=session.revision,
        idempotency_key=str(session.revision),
    )


def test_inspection_success_and_replay():
    session = ProximitySession()
    for _ in range(3):
        step(session, {"kind": "inspect"})
    assert session.metrics()["success"]
    assert (
        ProximitySession.from_checkpoint(session.checkpoint()).checkpoint()
        == session.checkpoint()
    )
    assert "truth" not in session.observe() and "intervals" not in session.observe()


def test_abort_is_latched_and_not_success():
    session = ProximitySession(
        {
            "initial_truth_si": [160.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            "initial_estimate_si": [160.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        }
    )
    step(session, {"kind": "abort"})
    assert session.metrics()["safe_retreat"] and not session.metrics()["success"]
    with pytest.raises(ValueError, match="done"):
        step(session, {"kind": "inspect"})


def test_midstep_violation_terminates_even_when_endpoints_safe():
    session = ProximitySession(
        {
            "mean_motion_rad_s": 0.0,
            "step_sec": 10.0,
            "initial_truth_si": [100.0, 0.0, 0.0, -20.0, 0.0, 0.0],
        }
    )
    response = step(session, {"kind": "coast"})
    assert response["terminated"] and not session.metrics()["safe_intervals"]


def test_budget_failure_is_atomic_and_idempotency_pays_once():
    session = ProximitySession()
    before = session.checkpoint()
    with pytest.raises(ValueError, match="budget"):
        step(session, {"kind": "impulse", "delta_v_rtn_m_s": [10.0, 0.0, 0.0]})
    assert session.checkpoint() == before
    response = step(session, {"kind": "coast"})
    assert (
        session.step({"kind": "coast"}, expected_revision=0, idempotency_key="0")
        == response
    )
    assert session.revision == 1
    with pytest.raises(ValueError, match="conflict"):
        session.step({"kind": "inspect"}, expected_revision=0, idempotency_key="0")


def test_delayed_filter_replays_acquisition_epoch():
    cfg = {"measurement_latency_steps": 2, "required_observations": 128}
    a, b = (
        ProximitySession(cfg),
        ProximitySession({**cfg, "measurement_latency_steps": 0}),
    )
    for _ in range(3):
        step(a, {"kind": "coast"})
        step(b, {"kind": "coast"})
    assert a.observe()["measurement_age_steps"] == 2
    assert a.observe()["estimate_si"] != b.observe()["estimate_si"]
    # Compare delayed posterior against a separate linear batch calculation:
    # at t=3 only the first frozen position sample has arrived.
    from space_env.core.relative import cw_stm

    f = cw_stm(a.spec.mean_motion_rad_s, a.spec.step_sec)
    p = np.array(a.spec.initial_covariance_si)
    x = np.array(a.spec.initial_estimate_si)
    t, q = a.spec.step_sec, a.spec.acceleration_psd_m2_s3
    noise = q * np.kron([[t**3 / 3, t**2 / 2], [t**2 / 2, t]], np.eye(3))
    x, p = f @ x, f @ p @ f.T + noise
    k = np.linalg.solve(p[:3, :3] + np.eye(3) * a.spec.position_noise_m**2, p[:3, :]).T
    x += k @ (a._measurements[1] - x[:3])
    for _ in range(2):
        x = f @ x
    np.testing.assert_allclose(a.observe()["estimate_si"], x, atol=1e-10)
    assert ProximitySession.from_checkpoint(a.checkpoint()).observe() == a.observe()


def test_no_measurements_does_not_substitute_truth():
    session = ProximitySession({"dropout_probability": 1.0})
    step(session, {"kind": "coast"})
    assert session.observe()["measurement_age_steps"] is None
    assert session.observe()["estimate_si"] != session._truth.tolist()


def test_resealed_forged_state_rejected():
    session = ProximitySession()
    record = copy.deepcopy(session.checkpoint())
    record["state_sha256"] = "0" * 64
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError, match="state"):
        ProximitySession.from_checkpoint(record)


def test_declared_retreat_goal_can_succeed_without_becoming_inspection():
    session = ProximitySession(
        {
            "mode": "retreat",
            "initial_truth_si": [160.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            "initial_estimate_si": [160.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        }
    )
    step(session, {"kind": "coast"})
    assert session.metrics()["success"] and session.metrics()["safe_retreat"]
    assert session.metrics()["inspection_acknowledgements"] == 0


def test_abort_impulse_changes_trajectory_in_same_interval():
    session = ProximitySession(
        {
            "mean_motion_rad_s": 0.0,
            "step_sec": 10.0,
            "initial_truth_si": [100.0, 0.0, 0.0, -10.0, 0.0, 0.0],
            "max_impulse_m_s": 3.0,
        }
    )
    step(session, {"kind": "abort", "delta_v_rtn_m_s": [0.0, 3.0, 0.0]})
    assert session.metrics()["safe_intervals"] and session.observe()["aborting"]


def test_dependency_mismatch_and_detached_checkpoint():
    session = ProximitySession()
    checkpoint = session.checkpoint()
    checkpoint["environment"]["python"] = "different"
    checkpoint["sources"]["invented.py"] = "0" * 64
    assert "invented.py" not in session.checkpoint()["sources"]
    checkpoint["sha256"] = digest(
        {k: v for k, v in checkpoint.items() if k != "sha256"}
    )
    with pytest.raises(ValueError, match="source or integrity"):
        ProximitySession.from_checkpoint(checkpoint)


def test_public_task_description_cannot_mutate_checkpoint_configuration():
    session = ProximitySession()
    before = session.checkpoint()
    view = session.observe()
    view["task"]["waypoint_m"][0] = 99999.0
    view["covariance_si"][0][0] = 99999.0
    assert session.checkpoint() == before


@pytest.mark.parametrize(
    "cfg",
    [
        {"keepout_m": 200.0},
        {"initial_covariance_si": [[0.0]]},
        {"initial_truth_si": [100000.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"seed": True},
    ],
)
def test_invalid_envelopes(cfg):
    with pytest.raises(ValueError):
        ProximitySpec.model_validate(cfg)
