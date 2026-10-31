"""Command-driven reference trajectory, mass conservation and replay contracts."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from space_env.core.propulsion import G0_M_S2, mass_after_thrust
from space_env.reference.providers import StateProvider
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    dynamic_reference_scenario,
)
from tests import test_reference_operations as fixtures


@pytest.fixture(scope="module")
def dynamic_spec():
    fixed = fixtures.spec.__wrapped__()
    horizon = fixed.get("horizon_sec", 30.0)
    record = next(iter(fixed["ephemerides"].values()))
    return {
        "schema": "space-env.reference-dynamic-spec/1",
        "horizon_sec": float(horizon),
        "eop": record["manifest"]["eop"],
        "orbits": {
            actor: {
                "initial": {
                    "frame": "GCRS",
                    "epoch_utc": ephem["epoch_utc"],
                    "r_m": ephem["states_si"][0][:3],
                    "v_m_s": ephem["states_si"][0][3:],
                },
                "propulsion": {
                    "dry_mass_kg": 90.0,
                    "propellant_kg": 10.0,
                    "isp_s": 300.0,
                    "max_thrust_n": 2.0,
                },
                "integration": {"sample_sec": 1.0, "max_step_sec": 5.0},
            }
            for actor, ephem in fixed["ephemerides"].items()
        },
        "operations": {
            k: v for k, v in fixed.items() if k not in {"ephemerides", "horizon_sec"}
        },
    }


def runtime(spec):
    return MissionRuntime(
        dynamic_reference_scenario(spec, id="dynamic", lineage="B02"),
        access=AccessPolicy(truth=True, forecast=True),
    )


def impulse(id="kick", actor="a", dv=(0.0, 10.0, 0.0)):
    return {
        "command_id": id,
        "actor_id": actor,
        "kind": "impulse",
        "duration_sec": 0.0,
        "impulse": {"frame": "GCRS", "delta_v_m_s": dv},
    }


def burn(duration=4.0):
    return {
        "command_id": "burn",
        "actor_id": "a",
        "kind": "finite_burn",
        "duration_sec": duration,
        "burn": {"frame": "GCRS", "direction_gcrs": [0.0, 1.0, 0.0], "thrust_n": 2.0},
    }


def test_impulse_changes_velocity_mass_geometry_and_replays(dynamic_spec, tmp_path):
    model = runtime(dynamic_spec)
    provider = model._backend.mission.ephemerides["a"]
    assert isinstance(provider, StateProvider)
    before = provider.state(0).copy()
    old_future = provider.state(20).copy()
    old_forecast = model.forecast("a", "Ground")
    answer = model.submit([impulse()], expected_revision=0, idempotency_key="kick")
    assert answer["effect"]["commands"][0]["status"] == "completed"
    np.testing.assert_allclose(provider.state(0)[:3], before[:3], atol=0)
    np.testing.assert_allclose(
        provider.state(0)[3:] - before[3:], [0, 10, 0], atol=1e-12
    )
    assert provider.mass(0) == pytest.approx(100 * math.exp(-10 / (300 * G0_M_S2)))
    assert np.linalg.norm(provider.state(20) - old_future) > 100
    assert (
        model.forecast("a", "Ground")["geometry_revision"]
        == old_forecast["geometry_revision"] + 1
    )
    model.advance(5, expected_revision=1, idempotency_key="coast")
    path = tmp_path / "dynamic.json"
    model.save(path)
    restored = MissionRuntime.load(path, access=AccessPolicy(truth=True))
    assert restored.checkpoint() == model.checkpoint()
    assert (
        restored.submit([impulse()], expected_revision=0, idempotency_key="kick")
        == answer
    )


def test_finite_burn_partial_cancel_preserves_executed_mass_and_history(dynamic_spec):
    model = runtime(dynamic_spec)
    model.submit([burn()], expected_revision=0, idempotency_key="burn")
    provider = model._backend.mission.ephemerides["a"]
    assert (
        provider.mass(0) == 100
    )  # Forecasting a burn does not spend fuel at admission.
    model.advance(2, expected_revision=1, idempotency_key="half")
    state, mass = provider.state(2).copy(), provider.mass(2)
    assert mass == pytest.approx(mass_after_thrust(100, 2, 2, 300, 90)[0])
    past = provider.state(1).copy()
    model.cancel("burn", expected_revision=2, idempotency_key="cancel")
    np.testing.assert_allclose(provider.state(2), state, atol=0)
    np.testing.assert_allclose(provider.state(1), past, atol=0)
    model.advance(3, expected_revision=3, idempotency_key="after")
    assert provider.mass(5) == mass
    assert model.report()["commands"][0]["status"] == "cancelled"
    restored = MissionRuntime.from_checkpoint(
        model.checkpoint(), access=AccessPolicy(truth=True)
    )
    assert restored.checkpoint() == model.checkpoint()


def test_mid_burn_checkpoint_and_coarse_fine_evolution(dynamic_spec):
    coarse, fine = runtime(dynamic_spec), runtime(dynamic_spec)
    for item in (coarse, fine):
        item.submit([burn()], expected_revision=0, idempotency_key="burn")
    coarse.advance(8, expected_revision=1, idempotency_key="whole")
    fine.advance(1, expected_revision=1, idempotency_key="prefix")
    restored = MissionRuntime.from_checkpoint(
        fine.checkpoint(), access=AccessPolicy(truth=True)
    )
    for item in (fine, restored):
        for index in range(7):
            item.advance(1, expected_revision=index + 2, idempotency_key=f"tick{index}")
    assert restored.checkpoint() == fine.checkpoint()
    for actor in ("a", "b"):
        np.testing.assert_allclose(
            coarse._backend.mission.ephemerides[actor].state(8),
            fine._backend.mission.ephemerides[actor].state(8),
            atol=1e-8,
            rtol=0,
        )
    assert coarse._backend.report()["mass_kg"] == fine._backend.report()["mass_kg"]


@pytest.mark.parametrize("change", ["frame", "fuel", "thrust", "duration", "direction"])
def test_invalid_burn_or_impulse_is_atomic(dynamic_spec, change):
    model = runtime(dynamic_spec)
    command = burn()
    if change == "frame":
        command["burn"]["frame"] = "ITRS"
    elif change == "fuel":
        command = impulse(dv=(0.0, 1000.0, 0.0))
    elif change == "thrust":
        command["burn"]["thrust_n"] = 3.0
    elif change == "duration":
        command["duration_sec"] = 0
    else:
        command["burn"]["direction_gcrs"] = [0.0, 2.0, 0.0]
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError):
        model.submit([command], expected_revision=0, idempotency_key="bad")
    assert model.checkpoint() == before


def test_unpowered_actor_and_batch_failure_rollback_after_first_maneuver(
    dynamic_spec, monkeypatch
):
    model = runtime(dynamic_spec)
    original = model._backend.admit

    def fail_second(command):
        if command.actor_id == "b":
            raise RuntimeError("actuator failure")
        return original(command)

    monkeypatch.setattr(model._backend, "admit", fail_second)
    before = model.checkpoint()
    with pytest.raises(RuntimeError, match="actuator"):
        model.submit(
            [impulse(), impulse(id="b-kick", actor="b")],
            expected_revision=0,
            idempotency_key="batch",
        )
    assert model.checkpoint() == before


def test_weather_eop_and_ephemeris_coverage_are_explicit(dynamic_spec):
    data = copy.deepcopy(dynamic_spec)
    for orbit in data["orbits"].values():
        orbit["forces"] = {"drag": True}
    with pytest.raises(RuntimeContractError, match="weather"):
        runtime(data)
    data["weather"] = {
        "source": "synthetic test",
        "status": "prediction",
        "allow_predictions": False,
        "valid_from_utc": "2026-07-18T00:00:00Z",
        "valid_until_utc": "2026-07-19T00:00:00Z",
    }
    with pytest.raises(RuntimeContractError, match="opt-in"):
        runtime(data)
    data["weather"].update(status="synthetic", valid_until_utc="2026-07-18T12:00:01Z")
    with pytest.raises(RuntimeContractError, match="validity"):
        runtime(data)
    model = runtime(dynamic_spec)
    with pytest.raises(ValueError, match="coverage"):
        model._backend.mission.ephemerides["a"].state(model.report()["horizon_sec"] + 1)


def test_maneuver_recomputes_los_and_preserves_other_actor_and_executed_past(
    dynamic_spec,
):
    data = copy.deepcopy(dynamic_spec)
    data["horizon_sec"] = 1200.0
    for orbit in data["orbits"].values():
        orbit["integration"] = {"sample_sec": 10.0, "max_step_sec": 30.0}
    model = runtime(data)
    before = copy.deepcopy(model._backend.mission.windows)
    model.advance(10, expected_revision=0, idempotency_key="coast")
    provider = model._backend.mission.ephemerides["a"]
    past, other = (
        provider.state(5).copy(),
        model._backend.mission.ephemerides["b"].state(1000).copy(),
    )
    model.submit(
        [impulse(dv=(0.0, 100.0, 0.0))], expected_revision=1, idempotency_key="kick"
    )
    after = model._backend.mission.windows
    assert abs(after["a", "Ground"][0][1] - before["a", "Ground"][0][1]) > 0.1
    assert after["b", "Ground"] == before["b", "Ground"]
    np.testing.assert_array_equal(provider.state(5), past)
    np.testing.assert_array_equal(
        model._backend.mission.ephemerides["b"].state(1000), other
    )


def test_all_force_switches_work_with_mass_changing_burn(dynamic_spec):
    data = copy.deepcopy(dynamic_spec)
    for orbit in data["orbits"].values():
        orbit["forces"] = {"j2": True, "drag": True, "srp": True, "third_body": True}
    data["weather"] = {
        "source": "declared synthetic constants, not operational weather",
        "status": "synthetic",
        "valid_from_utc": "2026-07-18T00:00:00Z",
        "valid_until_utc": "2026-07-19T00:00:00Z",
    }
    model = runtime(data)
    model.submit([burn()], expected_revision=0, idempotency_key="burn")
    model.advance(4, expected_revision=1, idempotency_key="step")
    assert model._backend.report()["mass_kg"]["a"] < 100
    assert np.isfinite(model.observe("a")["actors"]["a"]["orbit"]["state_si"]).all()


def test_fuel_and_maneuver_budgets_are_enforced_before_mutation(dynamic_spec):
    data = copy.deepcopy(dynamic_spec)
    data["max_maneuvers"] = 1
    data["orbits"]["a"]["propulsion"]["propellant_kg"] = 1e-6
    model = runtime(data)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="propellant"):
        model.submit([burn()], expected_revision=0, idempotency_key="fuel")
    assert model.checkpoint() == before
    model.submit([impulse(actor="b")], expected_revision=0, idempotency_key="one")
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="budget"):
        model.submit(
            [impulse(actor="b", id="second")],
            expected_revision=1,
            idempotency_key="two",
        )
    assert model.checkpoint() == before


def test_finite_burn_independent_rhs_and_integration_family():
    from space_env.validation.maneuvers import finite_burn_report

    report = finite_burn_report()
    assert report["passed"], report


def test_frozen_external_j2_fixture_suite_and_corruption():
    import json
    from pathlib import Path

    from space_env.simulation.snapshot import digest
    from space_env.validation.orbit_fixtures import (
        verify_orbit_fixture,
        verify_packaged_orbit_fixtures,
    )

    report = verify_packaged_orbit_fixtures()
    assert report["passed"], report
    assert {r["duration_sec"] for r in report["cases"]} == {86400, 259200}
    assert report["qualification_status"] == "needs_external_evidence"
    path = (
        Path(__file__).resolve().parents[1]
        / "src/space_env/reference/fixtures/boinor_j2_24h.json"
    )
    data = json.loads(path.read_text())
    data["expected"]["states_si"][1][0] += 100
    with pytest.raises(ValueError, match="corrupt"):
        verify_orbit_fixture(data)
    data["sha256"] = digest({k: v for k, v in data.items() if k != "sha256"})
    assert not verify_orbit_fixture(data)["passed"]


def test_batch_maneuver_budget_and_cancel_before_first_advance(dynamic_spec):
    data = copy.deepcopy(dynamic_spec)
    data["max_maneuvers"] = 1
    model = runtime(data)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="budget"):
        model.submit(
            [impulse(), impulse(id="b", actor="b")],
            expected_revision=0,
            idempotency_key="two",
        )
    assert model.checkpoint() == before
    model.submit([burn()], expected_revision=0, idempotency_key="burn")
    model.cancel("burn", expected_revision=1, idempotency_key="cancel")
    model.advance(4, expected_revision=2, idempotency_key="coast")
    assert model.observe("a")["actors"]["a"]["orbit"]["mass_kg"] == 100


def test_interpolation_error_is_separate_from_integrator_tolerance():
    from space_env.core.constants import MU_EARTH
    from space_env.reference import (
        EOPSeries,
        Epoch,
        FrameContext,
        ReferenceState,
        propagate,
    )
    from space_env.reference.access import ReferenceEphemeris

    radius = 7e6
    motion = math.sqrt(MU_EARTH / radius**3)
    speed = radius * motion
    initial = ReferenceState(
        Epoch.utc("2026-07-18T12:00:00Z"), [radius, 0, 0], [0, speed, 0]
    )
    record = propagate(
        initial, 600, FrameContext(EOPSeries.packaged()), sample_sec=60.0
    )
    ephemeris = ReferenceEphemeris.from_result(record)
    errors = []
    for t in np.arange(1.0, 600.0, 3.0):
        exact = np.array(
            [
                radius * math.cos(motion * t),
                radius * math.sin(motion * t),
                0,
                -speed * math.sin(motion * t),
                speed * math.cos(motion * t),
                0,
            ]
        )
        difference = ephemeris.state(t) - exact
        errors.append((np.linalg.norm(difference[:3]), np.linalg.norm(difference[3:])))
    assert max(p for p, _ in errors) < 1.0
    assert max(v for _, v in errors) < 0.03
    assert max(p for p, _ in errors) > record["manifest"]["position_atol_m"]


def test_missing_required_covariance_capability_and_legacy_maneuver_do_not_fallback(
    dynamic_spec,
):
    scenario = dynamic_reference_scenario(dynamic_spec, id="covariance", lineage="B02")
    scenario["required_capabilities"] = ["state.covariance_propagation"]
    with pytest.raises(RuntimeContractError, match="capabilities"):
        MissionRuntime(scenario)
    from space_env.runtime import legacy_scenario
    from tests.test_execution_integrity import world

    model = MissionRuntime(
        legacy_scenario(world(), id="legacy", lineage="B02"),
        access=AccessPolicy(truth=True),
    )
    before = model.checkpoint()
    command = impulse(actor=model.describe_capabilities()["actors"][0])
    with pytest.raises(RuntimeContractError) as error:
        model.submit([command], expected_revision=0, idempotency_key="impulse")
    assert error.value.code == "unsupported_capability"
    assert model.checkpoint() == before
