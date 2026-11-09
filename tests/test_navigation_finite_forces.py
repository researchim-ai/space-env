"""Known finite-thrust and declared-force estimation without truth substitution."""

import copy

import numpy as np
import pytest

from space_env.core.propulsion import G0_M_S2
from space_env.navigation.controls import own_impulses
from space_env.navigation.task_suite import compile_suite
from space_env.tasks import TaskSession, TaskSpec


@pytest.fixture(scope="module")
def original():
    return compile_suite()[0]


def setup(original, *, forces=None, latency=0.5, dropout=1.0):
    c = copy.deepcopy(original)
    c["task"].pop("goal")
    spec = c["task"]["navigation_sensors"]["a"]
    spec.update(
        control_profile="inertial_finite_burn",
        force_profile="matched_native",
        latency_sec=latency,
        dropout_probability=dropout,
        continuous_power_w=3.0,
    )
    initial = c["scenario"]["definition"]["orbits"]["a"]["initial"]
    spec["initial_estimate_si"] = [*initial["r_m"], *initial["v_m_s"]]
    c["scenario"]["definition"]["orbits"]["a"]["forces"] = forces or {}
    if forces and forces.get("drag"):
        c["scenario"]["definition"]["weather"] = {
            "source": "synthetic fixed weather assumptions",
            "status": "synthetic",
            "valid_from_utc": "2026-07-18T11:00:00Z",
            "valid_until_utc": "2026-07-18T13:00:00Z",
        }
    c["task"]["actions"] = [
        {
            "id": "burn",
            "role": "a",
            "kind": "submit",
            "commands": [
                {
                    "command_id": "burn",
                    "actor_id": "a",
                    "kind": "finite_burn",
                    "duration_sec": 2.0,
                    "burn": {
                        "frame": "GCRS",
                        "direction_gcrs": [0.0, 0.0, 1.0],
                        "thrust_n": 0.5,
                    },
                }
            ],
        },
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
                    "impulse": {"frame": "GCRS", "delta_v_m_s": [0.0, 0.0, 0.5]},
                }
            ],
        },
        {"id": "stop", "role": "a", "kind": "cancel_actor", "actor": "a"},
    ]
    return c, TaskSession(c["scenario"], c["task"])


def step(s, action, duration=1.0):
    return s.step(
        {"a": [action]},
        expected_revision=s.revision,
        idempotency_key="s" + str(s.revision),
        advance_sec=duration,
    )


def mass(s):
    impulses = own_impulses(s._runtime, "a")
    seen = {r["command"]["command_id"] for r in impulses}
    f = s._navigation._filters["a"]
    return s._navigation._environments["a"].nominal_mass(
        s._runtime.time_sec, impulses, seen, f["burns"]
    )


@pytest.mark.parametrize(
    "forces",
    [
        {},
        {"j2": True},
        {"third_body": True},
        {"srp": True},
        {"drag": True},
        {"j2": True, "drag": True, "srp": True, "third_body": True},
    ],
)
def test_matched_mean_and_nominal_mass_against_native_finite_arc(original, forces):
    _, s = setup(original, forces=forces)
    assert step(s, "burn", 3.0)["feedback"]["accepted"]
    estimate = np.array(s.observe("a")["navigation"]["a"]["estimate_si"])
    truth = s._runtime._backend.mission.ephemerides["a"].state(3.0)
    np.testing.assert_allclose(estimate[:3], truth[:3], rtol=0, atol=1e-3)
    np.testing.assert_allclose(estimate[3:], truth[3:], rtol=0, atol=1e-6)
    assert mass(s) == pytest.approx(
        s._runtime._backend.mission.ephemerides["a"].mass(3.0), abs=1e-9
    )
    assert s.modeling_profile()["actors"]["a"]["matched_force_estimation"] == "enabled"
    assert s.modeling_profile()["actors"]["a"]["finite_burn_estimation"] == "enabled"
    assert (
        s.modeling_profile()["actors"]["b"]["finite_burn_estimation"]
        == "not_configured"
    )


def test_mid_burn_cold_restore_cancel_and_resume(original):
    _, s = setup(original, forces={"j2": True}, dropout=0.0)
    step(s, "burn", 1.0)
    resumed = TaskSession.from_checkpoint(s.checkpoint(compact=True))
    assert resumed.observe() == s.observe()
    assert step(resumed, "stop", 1.0)["feedback"]["accepted"]
    step(resumed, "wait", 1.0)
    burns = resumed._navigation._filters["a"]["burns"]
    assert next(iter(burns.values()))["end"] == 1.0
    assert mass(resumed) == pytest.approx(
        resumed._runtime._backend.mission.ephemerides["a"].mass(3.0), abs=1e-9
    )
    assert (
        TaskSession.from_checkpoint(resumed.checkpoint(compact=True)).observe()
        == resumed.observe()
    )


@pytest.mark.parametrize("order", [["burn", "kick"], ["kick", "burn"]])
def test_mass_history_is_chronological_not_sum_of_independent_losses(original, order):
    _, s = setup(original)
    for action in order:
        step(s, action, 2.0)
    backend = s._runtime._backend
    prop = backend.config.orbits["a"].propulsion
    m0 = prop.dry_mass_kg + prop.propellant_kg
    flow = 0.5 / (prop.isp_s * G0_M_S2)
    factor = np.exp(-0.5 / (prop.isp_s * G0_M_S2))
    expected = (
        (m0 - 2 * flow) * factor if order[0] == "burn" else m0 * factor - 2 * flow
    )
    assert mass(s) == pytest.approx(expected, abs=1e-9)
    assert mass(s) == pytest.approx(
        backend.mission.ephemerides["a"].mass(4.0), abs=1e-9
    )


def test_prediction_does_not_query_native_truth_state_or_mass(original, monkeypatch):
    _, s = setup(original, forces={"j2": True})
    step(s, "burn", 1.0)

    def forbidden(*args):
        raise AssertionError("truth substitution")

    provider = s._runtime._backend.mission.ephemerides["a"]
    monkeypatch.setattr(provider, "state", forbidden)
    monkeypatch.setattr(provider, "mass", forbidden)
    view = s._navigation._view(s._runtime, s._navigation._filters, "a")
    assert len(view["estimate_si"]) == 6


def test_cancelled_burn_does_not_rewrite_prior_delayed_packets(original):
    _, s = setup(original, latency=1.5, dropout=0.0)
    step(s, "burn", 1.0)
    before = copy.deepcopy(s._navigation._filters["a"]["pending"])
    step(s, "stop", 1.0)
    first = s._navigation.measurement_history("a")[0]
    assert first["value"] == before[0]["value"]
    assert first["acquired_sec"] == 0.0 and first["received_sec"] == 1.5
    step(s, "wait", 1.0)
    assert len(s._navigation.measurement_history("a")) == 2


def test_finite_cutoff_is_partition_invariant(original):
    _, first = setup(original, forces={"j2": True}, dropout=0.0)
    _, second = setup(original, forces={"j2": True}, dropout=0.0)
    step(first, "burn", 1.0)
    step(first, "wait", 3.0)
    step(second, "burn", 3.0)
    step(second, "wait", 1.0)
    assert first.observe("a")["navigation"] == second.observe("a")["navigation"]
    assert mass(first) == mass(second)


def test_overthrust_rejection_creates_no_estimated_burn(original):
    c, _ = setup(original)
    c["task"]["actions"][0]["commands"][0]["burn"]["thrust_n"] = 1e6
    s = TaskSession(c["scenario"], c["task"])
    r = step(s, "burn")
    assert not r["feedback"]["accepted"]
    assert s._navigation._filters["a"]["burns"] == {}


@pytest.mark.parametrize(
    "mode", ["force_optin", "control_optin", "callback", "force_field"]
)
def test_unsupported_or_implicit_profiles_fail_explicitly(original, mode):
    c, _ = setup(original, forces={"j2": True})
    if mode == "force_optin":
        c["task"]["navigation_sensors"]["a"]["force_profile"] = "central"
    if mode == "control_optin":
        c["task"]["navigation_sensors"]["a"]["control_profile"] = "impulse"
    if mode == "callback":
        c["task"]["navigation_sensors"]["a"]["dynamics"] = "arbitrary callback"
    if mode == "force_field":
        c["scenario"]["definition"]["orbits"]["a"]["forces"]["unsupported_field"] = True
    with pytest.raises((ValueError, TypeError)):
        TaskSession(c["scenario"], c["task"])


def test_recipe_for_matched_features_has_no_implicit_actor_promotion(original):
    c, _ = setup(original)
    c["task"]["model_requirements"].extend(
        [
            {"feature": f, "actors": ["a"]}
            for f in ["matched_force_estimation", "finite_burn_estimation"]
        ]
    )
    assert (
        TaskSession(c["scenario"], c["task"]).modeling_profile()["actors"]["a"][
            "finite_burn_estimation"
        ]
        == "enabled"
    )
    c["task"]["model_requirements"][-1]["actors"] = ["b"]
    with pytest.raises(ValueError, match="not_configured"):
        TaskSession(c["scenario"], c["task"])
    assert TaskSpec.model_validate(c["task"])


def test_inconsistent_actor_epochs_not_silently_treated_as_simultaneous(original):
    c, _ = setup(original)
    c["scenario"]["definition"]["orbits"]["b"]["initial"]["epoch_utc"] = (
        "2026-07-18T12:00:01Z"
    )
    with pytest.raises(ValueError, match="common .*epoch"):
        TaskSession(c["scenario"], c["task"])


def test_cancelled_burn_mean_matches_native_without_received_measurements(original):
    _, s = setup(original, forces={"j2": True}, dropout=1.0)
    step(s, "burn", 0.5)
    step(s, "stop", 0.5)
    step(s, "wait", 3.0)
    estimate = np.array(s.observe("a")["navigation"]["a"]["estimate_si"])
    truth = s._runtime._backend.mission.ephemerides["a"].state(4.0)
    np.testing.assert_allclose(estimate[:3], truth[:3], atol=1e-3, rtol=0)
    np.testing.assert_allclose(estimate[3:], truth[3:], atol=1e-6, rtol=0)


def test_central_finite_arc_matches_separate_fixed_step_oracle(original):
    from space_env.tasks.modeling_suite import _orbit_target

    c, s = setup(original, dropout=1.0)
    step(s, "burn", 3.0)
    oracle = _orbit_target(
        c["scenario"]["definition"]["orbits"]["a"],
        [0.0, 0.0, 1.0],
        thrust=0.5,
        firing=2.0,
        end=3.0,
    )
    estimate = np.array(s.observe("a")["navigation"]["a"]["estimate_si"])
    np.testing.assert_allclose(estimate[:3], oracle[:3], atol=1e-3, rtol=0)
    np.testing.assert_allclose(estimate[3:], oracle[3:], atol=1e-6, rtol=0)


def test_j2_finite_covariance_matches_mean_finite_difference_map(original):
    _, s = setup(original, forces={"j2": True}, dropout=1.0)
    step(s, "burn", 3.0)
    filters = s._navigation._filters
    actual = np.array(s.observe("a")["navigation"]["a"]["covariance_si"])
    jac = np.zeros((6, 6))
    for i in range(6):
        h = 1.0 if i < 3 else 0.01
        plus = copy.deepcopy(filters)
        minus = copy.deepcopy(filters)
        plus["a"]["x"][i] += h
        minus["a"]["x"][i] -= h
        plus["a"]["p"] = np.zeros((6, 6))
        minus["a"]["p"] = np.zeros((6, 6))
        xp = np.array(s._navigation._view(s._runtime, plus, "a")["estimate_si"])
        xm = np.array(s._navigation._view(s._runtime, minus, "a")["estimate_si"])
        jac[:, i] = (xp - xm) / (2 * h)
    expected = jac @ filters["a"]["p"] @ jac.T
    assert (
        np.linalg.norm(actual[:3, :3] - expected[:3, :3])
        / np.linalg.norm(expected[:3, :3])
        < 5e-6
    )
    assert (
        np.linalg.norm(actual[3:, 3:] - expected[3:, 3:])
        / np.linalg.norm(expected[3:, 3:])
        < 5e-6
    )


def test_finite_suite_preflight_failures_are_explicit():
    from space_env.navigation.finite_suite import rejections

    rows = rejections()
    assert {r["id"] for r in rows} == {
        "force-optin",
        "control-optin",
        "body-thrust",
        "mixed-epochs",
    }
    assert all(r["rejected"] for r in rows)
    assert "sensor task supports" in next(
        r["error"] for r in rows if r["id"] == "body-thrust"
    )
