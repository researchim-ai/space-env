"""Causal delayed fixes across native impulses and real battery accounting."""

import copy

import numpy as np
import pytest

from space_env.navigation import NavigationSession, SensorSpec
from space_env.navigation.task_suite import compile_suite
from space_env.tasks import TaskSession


@pytest.fixture(scope="module")
def base():
    return compile_suite()[0]


def recipe(base, *, power=0.0, latency=0.5, dv=(0.0, 0.0, 1.0)):
    c = copy.deepcopy(base)
    c["task"].pop("goal")
    c["task"]["navigation_sensors"]["a"].update(
        continuous_power_w=power, latency_sec=latency
    )
    c["task"]["actions"].insert(
        0,
        {
            "id": "kick",
            "role": "a",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "kick",
                    "actor_id": "a",
                    "kind": "impulse",
                    "duration_sec": 0.0,
                    "impulse": {"frame": "GCRS", "delta_v_m_s": list(dv)},
                }
            ],
        },
    )
    return c


def step(s, choice, dt=1.0):
    return s.step(
        {"a": [choice]},
        expected_revision=s.revision,
        idempotency_key="step-" + str(s.revision),
        advance_sec=dt,
    )


def test_known_impulse_changes_estimated_velocity_without_truth_prediction(
    base, monkeypatch
):
    c = recipe(base, latency=3.0)
    s = TaskSession(c["scenario"], c["task"])
    prior = s.observe("a")["navigation"]["a"]["estimate_si"]
    row = step(s, "kick")
    assert row["feedback"]["accepted"]
    v = s.observe("a")["navigation"]["a"]
    assert v["received_sample_count"] == 0

    def forbidden(*args):
        raise AssertionError("truth read in predictor")

    monkeypatch.setattr(s._navigation, "_state", forbidden)
    assert s._navigation._view(s._runtime, s._navigation._filters, "a") == v
    plain = recipe(base, latency=3.0)
    plain["task"]["actions"] = []
    other = TaskSession(plain["scenario"], plain["task"])
    step(other, "wait")
    delta = np.array(v["estimate_si"]) - np.array(
        other.observe("a")["navigation"]["a"]["estimate_si"]
    )
    assert delta[5] == pytest.approx(1.0, abs=1e-5) and delta[2] == pytest.approx(
        1.0, abs=1e-5
    )
    assert prior == c["task"]["navigation_sensors"]["a"]["initial_estimate_si"]


def test_pending_range_rate_packet_preserves_pre_impulse_measurement(base):
    c = recipe(base)
    c["task"]["navigation_sensors"]["a"].update(
        kind="range_range_rate", reference_actor="b", standard_deviation=[2.0, 0.02]
    )
    s = TaskSession(c["scenario"], c["task"])
    reference = TaskSession(c["scenario"], c["task"])
    step(s, "kick")
    step(reference, "wait")
    packet = s._navigation.measurement_history("a")[0]
    assert packet == reference._navigation.measurement_history("a")[0]
    assert packet["acquired_sec"] == 0.0 and packet["received_sec"] == 0.5
    assert s._navigation._filters["a"]["controls"] == []
    assert len(s._navigation._filters["a"]["pending"]) == 1
    step(s, "wait", 0.5)
    assert len(s._navigation._filters["a"]["controls"]) == 1


def test_second_impulse_on_sample_boundary_applied_once_and_replays(base):
    c = recipe(base, power=3.0)
    s = TaskSession(c["scenario"], c["task"])
    step(s, "kick")
    last = step(s, "kick")
    step(s, "wait")
    assert len(s._navigation._filters["a"]["controls"]) == 2
    restored = TaskSession.from_checkpoint(s.checkpoint(compact=True))
    assert restored.observe() == s.observe()
    before = restored._navigation.measurement_history("a")
    assert (
        restored.step(
            {"a": ["kick"]},
            expected_revision=1,
            idempotency_key="step-1",
            advance_sec=1.0,
        )
        == last
    )
    assert restored._navigation.measurement_history("a") == before
    assert restored._runtime._backend.mission.vehicles["a"].base_power_w == 13.0


def test_failed_native_impulse_never_enters_estimator_control_history(base):
    c = recipe(base, dv=(0.0, 0.0, 100000.0))
    s = TaskSession(c["scenario"], c["task"])
    row = step(s, "kick")
    assert not row["feedback"]["accepted"]
    assert s._navigation._filters["a"]["controls"] == []
    assert not [
        r for r in s._runtime.command_status() if r["command"]["kind"] == "impulse"
    ]


def test_impulse_covariance_jump_is_declared_not_truth_measurement(base):
    c = recipe(base, latency=3.0)
    c["task"]["navigation_sensors"]["a"]["impulse_velocity_cov_m2_s2"] = (
        np.eye(3) * 0.25
    ).tolist()
    s = TaskSession(c["scenario"], c["task"])
    other = copy.deepcopy(c)
    other["task"]["navigation_sensors"]["a"]["impulse_velocity_cov_m2_s2"] = np.zeros(
        (3, 3)
    ).tolist()
    plain = TaskSession(other["scenario"], other["task"])
    step(s, "kick")
    step(plain, "kick")
    p = np.array(s.observe("a")["navigation"]["a"]["covariance_si"])
    q = np.array(plain.observe("a")["navigation"]["a"]["covariance_si"])
    assert np.linalg.eigvalsh(p - q).min() > -1e-10
    assert np.diag(p - q)[3:].min() > 0.249


def test_native_power_is_real_additive_load_and_does_not_mutate_recipe(base):
    c = recipe(base, power=3.0)
    original = copy.deepcopy(c)
    s = TaskSession(c["scenario"], c["task"])
    step(s, "wait", 2.0)
    report = s._runtime._backend.report()
    assert report["energy_ledger"]["a"]["consumed_wh"] * 3600 == pytest.approx(26.0)
    assert abs(report["energy_ledger"]["a"]["balance_error_wh"]) < 1e-12
    assert c == original
    free = recipe(base)
    other = TaskSession(free["scenario"], free["task"])
    step(other, "wait", 2.0)
    delta = (
        report["energy_ledger"]["a"]["consumed_wh"]
        - other._runtime._backend.report()["energy_ledger"]["a"]["consumed_wh"]
    )
    assert delta * 3600 == pytest.approx(6.0)


def test_battery_exhaustion_drops_fixes_at_acquisition(base):
    c = recipe(base, power=36.0, latency=0.5)
    c["scenario"]["definition"]["operations"]["vehicles"]["a"].update(
        base_power_w=0.0, battery_wh=0.01, initial_soc=1.0
    )
    s = TaskSession(c["scenario"], c["task"])
    step(s, "wait", 2.0)
    history = s._navigation.measurement_history("a")
    assert not history[0]["dropped"]
    assert history[1]["acquired_sec"] == 1.0 and history[1]["dropped"]
    assert history[1]["unavailable"] == "power" and "value" not in history[1]
    assert s._runtime._backend.mission.energy_wh["a"] == 0.0
    assert s._runtime._backend.report()["energy_ledger"]["a"][
        "consumed_wh"
    ] * 3600 == pytest.approx(36.0)


def test_standalone_power_replay_not_double_charged(base):
    c = recipe(base, power=2.0)
    nav = NavigationSession(c["scenario"], c["task"]["navigation_sensors"])
    nav.advance(2.0, expected_revision=0, idempotency_key="two")
    restored = NavigationSession.from_checkpoint(nav.checkpoint())
    assert restored._runtime._backend.mission.vehicles["a"].base_power_w == 12.0
    assert restored.observe("a") == nav.observe("a")


def test_estimated_control_path_independent_of_hidden_initial_truth(base):
    c = recipe(base, latency=3.0)
    other = copy.deepcopy(c)
    other["scenario"]["definition"]["orbits"]["a"]["initial"]["r_m"][0] -= 1000.0
    first = TaskSession(c["scenario"], c["task"])
    second = TaskSession(other["scenario"], other["task"])
    step(first, "kick")
    step(second, "kick")
    assert first.observe("a") == second.observe("a")


@pytest.mark.parametrize(
    "field,value",
    [
        ("continuous_power_w", -1.0),
        ("continuous_power_w", True),
        ("minimum_soc", 1.0),
        (
            "impulse_velocity_cov_m2_s2",
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ),
    ],
)
def test_invalid_power_or_control_uncertainty_contract(base, field, value):
    c = recipe(base)
    spec = c["task"]["navigation_sensors"]["a"]
    spec[field] = value
    with pytest.raises(ValueError):
        SensorSpec.model_validate(spec)


def test_step_partition_does_not_change_controlled_estimate_or_noise(base):
    c = recipe(base, power=3.0, latency=0.5)
    first = TaskSession(c["scenario"], c["task"])
    second = TaskSession(c["scenario"], c["task"])
    step(first, "kick", 1.0)
    step(first, "wait", 4.0)
    step(second, "kick", 3.0)
    step(second, "wait", 2.0)
    assert first.observe("a")["navigation"] == second.observe("a")["navigation"]
    assert first._navigation.measurement_history(
        "a"
    ) == second._navigation.measurement_history("a")
    assert first._runtime._backend.report()["energy_ledger"]["a"][
        "consumed_wh"
    ] == pytest.approx(
        second._runtime._backend.report()["energy_ledger"]["a"]["consumed_wh"]
    )


@pytest.mark.parametrize(
    "name,goal,sft",
    [
        ("powered-impulse", True, True),
        ("rejected-control", True, False),
        ("no-contact", False, False),
        ("power-exhaustion", False, False),
        ("uncertain-control", True, True),
        ("boundary-delay", False, False),
    ],
)
def test_controlled_mission_retains_resource_and_timing_failures(name, goal, sft):
    from space_env.navigation.controlled_suite import compile_suite, execute

    c = next(c for c in compile_suite() if c["id"] == name)
    record, witness = execute(c, "reference")
    assert record["metrics"]["success"] is goal
    assert record["metrics"]["sft_eligible"] is sft
    assert all(
        abs(row["balance_error_wh"]) < 1e-12
        for row in witness["energy_ledger"].values()
    )
    if name == "boundary-delay":
        assert witness["accepted_impulses"] == 2
        assert any(
            e["response"].get("feedback", {}).get("code") == "no_current_contact"
            for e in record["events"]
        )
    if name == "power-exhaustion":
        assert witness["sensors"]["a"]["unavailable_power"] > 0
    if name == "rejected-control":
        assert witness["accepted_impulses"] == 0
