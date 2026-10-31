"""Body-frame engine execution, finite mechanisms, paid resources and rollback."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from space_env.core.attitude import align_z
from space_env.core.propulsion import G0_M_S2
from space_env.reference.spacecraft_bus import angle_deg
from space_env.reference.thrust import FiniteThrust
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    spacecraft_scenario,
)
from tests import test_reference_lifecycle as life
from tests import test_spacecraft_bus as bus_fixtures


def engine(kind="chemical"):
    result = {
        "kind": kind,
        "model_id": f"synthetic-{kind}/1",
        "source_status": "synthetic",
        "source": "declared engineering test assumptions",
        "max_thrust_n": 1.0,
        "max_gimbal_deg": 30.0,
        "gimbal_rate_deg_s": 5.0,
        "gimbal_settling_sec": 0.5,
        "ignition_sec": 0.5,
        "ignition_power_w": 3.0,
        "firing_aux_power_w": 2.0,
        "chemical_wall_heat_w_at_max": 4.0 if kind == "chemical" else 0.0,
    }
    if kind == "electric":
        result["electric_efficiency"] = 0.5
    return result


@pytest.fixture(scope="module")
def scenario():
    definition = copy.deepcopy(bus_fixtures.scenario.__wrapped__()["definition"])
    definition["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    definition["actuation"] = {"engines": {"a": engine()}}
    return spacecraft_scenario(definition, id="coupled-thrust", lineage="B04-D1")


def runtime(data):
    return MissionRuntime(data, access=AccessPolicy(truth=True, forecast=True))


def burn(
    model, *, id="burn", duration=4.0, firing=2.0, thrust=1.0, direction=(0.0, 0.0, 1.0)
):
    return model.submit(
        [
            {
                "command_id": id,
                "actor_id": "a",
                "kind": "body_burn",
                "duration_sec": duration,
                "body_burn": {
                    "frame": "BODY",
                    "direction_body": list(direction),
                    "thrust_n": thrust,
                    "firing_sec": firing,
                },
            }
        ],
        expected_revision=model.revision,
        idempotency_key=id,
    )


def report(model):
    return life.lifecycle(model)["actuation"]


def op(model, id="burn"):
    return report(model)["operations"][id]


def test_admission_and_ignition_never_preapply_future_fuel_or_delta_v(scenario):
    model = runtime(scenario)
    provider = model._backend.mission.ephemerides["a"]
    mass = provider.mass(0)
    burn(model)
    assert provider.mass(20) == mass and not provider.maneuvers
    life.advance(model, 0.25, "half-ignition")
    assert op(model)["ignition_sec"] == 0.25 and op(model)["firing_sec"] == 0
    assert provider.mass(model.time_sec) == mass
    assert op(model)["dc_energy_j"] == 0.75
    life.advance(model, 2.25, "fire")
    expected = 2 / (300 * G0_M_S2)
    assert mass - provider.mass(model.time_sec) == pytest.approx(expected, abs=1e-12)
    assert op(model)["status"] == "completed"
    assert op(model)["ideal_scalar_delta_v_m_s"] == pytest.approx(
        300 * G0_M_S2 * math.log(mass / (mass - expected)), abs=1e-11
    )
    assert report(model)["fuel_balance_error_kg"]["a"] == pytest.approx(0, abs=1e-12)


def test_body_quaternion_rotates_actual_thrust_direction(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        align_z(np.array([1.0, 0.0, 0.0])).tolist()
    )
    model = runtime(data)
    burn(model)
    life.advance(model, 2.5, "finish")
    np.testing.assert_allclose(
        op(model)["impulse_vector_gcrs_ns"], [2.0, 0.0, 0.0], atol=1e-12
    )
    assert all(
        np.allclose(row["direction_gcrs"], [1.0, 0.0, 0.0], atol=1e-12)
        for row in report(model)["intervals"]
    )


def test_finite_gimbal_rotation_settling_and_ignition_are_paid_before_thrust(scenario):
    model = runtime(scenario)
    angle = math.radians(10.0)
    goal = [math.sin(angle), 0.0, math.cos(angle)]
    burn(model, direction=goal, duration=6.0)
    life.advance(model, 1.0, "moving")
    assert angle_deg(
        [0.0, 0.0, 1.0], report(model)["states"]["a"]["gimbal_body"]
    ) == pytest.approx(5.0, abs=1e-10)
    assert op(model)["firing_sec"] == 0
    life.advance(model, 2.0, "ready")
    assert op(model)["gimbal_sec"] == pytest.approx(2.0)
    assert op(model)["settling_sec"] == pytest.approx(0.5)
    assert op(model)["ignition_sec"] == pytest.approx(0.5)
    life.advance(model, 2.0, "fired")
    np.testing.assert_allclose(
        op(model)["impulse_vector_gcrs_ns"], np.asarray(goal) * 2, atol=1e-10
    )
    assert op(model)["dc_energy_j"] == pytest.approx(2 * 2 + 0.5 * 2 + 0.5 * 3 + 2 * 2)


def test_cancel_preserves_spent_fuel_and_coast_forecast_has_no_residual_burn(scenario):
    model = runtime(scenario)
    burn(model, firing=5.0, duration=7.0)
    life.advance(model, 1.0, "part")
    before_mass = model._backend.mission.ephemerides["a"].mass(model.time_sec)
    spent = op(model)["dc_energy_j"]
    model.cancel("burn", expected_revision=model.revision, idempotency_key="cancel")
    life.advance(model, 2.0, "coast")
    assert op(model)["firing_sec"] == 0.5 and op(model)["dc_energy_j"] == spent
    assert model._backend.mission.ephemerides["a"].mass(model.time_sec) == before_mass
    assert op(model)["status"] == "cancelled"


def test_short_window_does_not_fabricate_requested_firing(scenario):
    model = runtime(scenario)
    burn(model, duration=0.25)
    life.advance(model, 0.25, "window")
    assert (
        op(model)["status"] == "failed" and op(model)["reason"] == "burn_window_ended"
    )
    assert op(model)["firing_sec"] == 0 and op(model)["consumed_propellant_kg"] == 0


@pytest.mark.parametrize("subsystem", ["propulsion", "adcs", "bus"])
def test_fault_stops_engine_without_replenishing_resources(scenario, subsystem):
    model = runtime(scenario)
    burn(model, firing=5.0, duration=7.0)
    life.advance(model, 1.0, "part")
    before = op(model).copy()
    model.inject(
        [
            life.event(
                model,
                "spacecraft_fault",
                {"actor": "a", "subsystem": subsystem},
                "fault",
            )
        ],
        expected_revision=model.revision,
        idempotency_key="fault",
    )
    life.advance(model, 2.0, "abort")
    assert op(model)["status"] == "failed"
    for key in ("dc_energy_j", "consumed_propellant_kg", "firing_sec"):
        assert op(model)[key] == before[key]


def test_electric_dc_heat_and_jet_equations_match_independent_energy_balance(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["actuation"]["engines"]["a"] = engine("electric")
    model = runtime(data)
    initial = model._backend.mission.energy_wh["a"]
    burn(model, thrust=0.01)
    life.advance(model, 2.5, "complete")
    jet = 0.01 * 300 * G0_M_S2 / 2 * 2
    assert op(model)["jet_energy_j"] == pytest.approx(jet)
    assert op(model)["dc_energy_j"] == pytest.approx(1.5 + 4 + jet / 0.5)
    assert op(model)["engine_heat_j"] == pytest.approx(1.5 + 4 + jet)
    assert initial - model._backend.mission.energy_wh["a"] == pytest.approx(
        op(model)["dc_energy_j"] / 3600 / 0.9, abs=1e-12
    )
    assert abs(report(model)["electrical_heat_balance_error_j"]["burn"]) < 1e-10
    bus = model.observe(view="truth")["backend"]["report"]["spacecraft_bus"]
    assert abs(bus["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-8


def test_electric_power_limit_blocks_force_even_if_requested_thrust_is_small(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["actuation"]["engines"]["a"] = engine("electric")
    model = runtime(data)
    burn(model, thrust=1.0)
    life.advance(model, 3.0, "too-much-power")
    assert (
        op(model)["status"] == "failed" and op(model)["reason"] == "bus_power_capacity"
    )
    assert op(model)["firing_sec"] == 0 and op(model)["consumed_propellant_kg"] == 0
    assert op(model)["dc_energy_j"] == 1.5


def test_fuel_exhaustion_consumes_only_available_mass_and_does_not_cross_dry_mass(
    scenario,
):
    data = copy.deepcopy(scenario)
    data["definition"]["orbits"]["a"]["propulsion"]["propellant_kg"] = 0.0001
    model = runtime(data)
    burn(model)
    life.advance(model, 3.0, "exhausted")
    assert op(model)["reason"] == "propellant_exhausted"
    assert op(model)["firing_sec"] == pytest.approx(0.0001 * 300 * G0_M_S2, abs=1e-9)
    assert model._backend.mission.ephemerides["a"].mass(
        model.time_sec
    ) == pytest.approx(90.0, abs=1e-12)
    assert op(model)["consumed_propellant_kg"] == pytest.approx(0.0001, abs=1e-12)


def test_thermal_trip_from_chemical_wall_heat_stops_remaining_thrust(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["actuation"]["engines"]["a"]["chemical_wall_heat_w_at_max"] = (
        10000.0
    )
    model = runtime(data)
    burn(model, firing=5.0, duration=7.0)
    life.advance(model, 4.0, "hot-engine")
    assert op(model)["reason"] == "safe_mode"
    assert 0 < op(model)["firing_sec"] < 5
    assert abs(report(model)["fuel_balance_error_kg"]["a"]) < 1e-12


def test_only_applied_thrust_changes_geometry_binding(scenario):
    model = runtime(scenario)
    forecast = model.forecast_acquisition("a", "site")
    before = forecast["binding"]["trajectory_sha256"]
    burn(model)
    life.advance(model, 0.25, "warm")
    assert model._backend.geometry_revision == 0
    assert model._backend.geometry_token("a") == before
    life.advance(model, 1.0, "thrust")
    assert model._backend.geometry_revision > 0
    assert model._backend.geometry_token("a") != before
    assert "applied thrust only" in model.forecast("a", "Ground")["scope"]
    with pytest.raises(RuntimeContractError) as error:
        model.submit(
            [
                {
                    "command_id": "stale",
                    "actor_id": "a",
                    "kind": "wait",
                    "duration_sec": 1.0,
                    "forecast": forecast["binding"],
                }
            ],
            expected_revision=model.revision,
            idempotency_key="stale",
        )
    assert error.value.code == "stale_forecast"


def test_segment_budget_fails_closed_after_paid_partial_burn(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["actuation"]["max_segments"] = 1
    model = runtime(data)
    burn(model)
    life.advance(model, 3.0, "limited")
    assert (
        op(model)["reason"] == "burn_segment_budget" and op(model)["firing_sec"] == 0.5
    )


def test_mid_gimbal_and_mid_burn_checkpoint_replay(scenario, tmp_path):
    model = runtime(scenario)
    angle = math.radians(5.0)
    burn(model, direction=[math.sin(angle), 0.0, math.cos(angle)], duration=5.0)
    life.advance(model, 0.5, "gimbal-prefix")
    model.save(tmp_path / "gimbal.json")
    restored = MissionRuntime.load(
        tmp_path / "gimbal.json", access=AccessPolicy(truth=True)
    )
    for item in (model, restored):
        life.advance(item, 2.0, "partial-fire")
    assert model.checkpoint() == restored.checkpoint()
    model.save(tmp_path / "firing.json")
    restored = MissionRuntime.load(
        tmp_path / "firing.json", access=AccessPolicy(truth=True)
    )
    for item in (model, restored):
        life.advance(item, 2.5, "finish")
    assert model.checkpoint() == restored.checkpoint()


@pytest.mark.parametrize(
    "invalid",
    ["mount", "efficiency", "cone", "hardware", "actor", "profile", "fidelity"],
)
def test_invalid_actuation_scenario_has_no_fallback(scenario, invalid):
    data = copy.deepcopy(scenario)
    setting = data["definition"]["actuation"]["engines"]["a"]
    if invalid == "mount":
        setting["mount_body"] = [0.0, 0.0, 2.0]
    elif invalid == "efficiency":
        setting["electric_efficiency"] = 0.5
    elif invalid == "cone":
        setting["max_gimbal_deg"] = 80.0
    elif invalid == "hardware":
        setting["max_thrust_n"] = 3.0
    elif invalid == "actor":
        data["definition"]["actuation"]["engines"]["unknown"] = setting
    elif invalid == "fidelity":
        data["fidelity"]["trajectory"] = "propagated-gcrs-coast"
    else:
        data.update(
            backend="reference-lifecycle",
            physics_profile="reference-lifecycle/1",
            fidelity={
                "trajectory": "controlled-gcrs",
                "resources": "explicit-surrogate",
            },
        )
        del data["definition"]["spacecraft_bus"]
        for v in data["definition"]["operations"]["vehicles"].values():
            v["solar_mode"] = "ideal_sun_tracking"
    with pytest.raises(ValueError):
        runtime(data)


def test_partitioned_applied_burn_matches_single_inertial_reference_arc(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["actuation"]["engines"]["a"]["ignition_sec"] = 0.0
    model = runtime(data)
    reference = copy.deepcopy(model._backend.mission.ephemerides["a"])
    reference.burn(0.0, 2.0, FiniteThrust(1.0, 300.0, 90.0, (0.0, 0.0, 1.0)))
    burn(model, duration=2.0)
    life.advance(model, 2.0, "applied")
    actual = model._backend.mission.ephemerides["a"]
    np.testing.assert_allclose(
        actual.state(2.0)[:3], reference.state(2.0)[:3], rtol=0, atol=1e-3
    )
    np.testing.assert_allclose(
        actual.state(2.0)[3:], reference.state(2.0)[3:], rtol=0, atol=1e-6
    )


def test_low_soc_interrupts_paid_ignition_before_any_thrust(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["vehicles"]["a"]["initial_soc"] = 0.1002
    model = runtime(data)
    burn(model)
    life.advance(model, 2.0, "floor")
    assert op(model)["status"] == "failed" and op(model)["reason"] == "propulsion_power"
    assert 0 < op(model)["ignition_sec"] < 0.5
    assert op(model)["dc_energy_j"] == pytest.approx(0.0002 * 3600 * 0.9, abs=1e-9)
    assert op(model)["consumed_propellant_kg"] == 0


def test_body_settling_delays_engine_and_short_wait_window_fails(scenario):
    model = runtime(scenario)
    model.submit(
        [
            {
                "command_id": "slew",
                "actor_id": "a",
                "kind": "slew",
                "duration_sec": 0.25,
                "slew": {"frame": "GCRS", "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0]},
            }
        ],
        expected_revision=0,
        idempotency_key="slew",
    )
    life.advance(model, 0.25, "slew-window")
    burn(model, duration=0.25)
    life.advance(model, 0.25, "body-settling")
    assert op(model)["active_phase"] == "attitude_wait"
    assert (
        op(model)["status"] == "failed" and op(model)["reason"] == "burn_window_ended"
    )
    assert op(model)["dc_energy_j"] == 0 and op(model)["consumed_propellant_kg"] == 0


def test_cancelled_gimbal_keeps_real_position_and_no_free_restart(scenario):
    model = runtime(scenario)
    angle = math.radians(10.0)
    direction = [math.sin(angle), 0.0, math.cos(angle)]
    burn(model, direction=direction, duration=5.0)
    life.advance(model, 0.5, "gimbal-part")
    before = report(model)["states"]["a"]["gimbal_body"]
    model.cancel(
        "burn", expected_revision=model.revision, idempotency_key="cancel-gimbal"
    )
    assert report(model)["states"]["a"]["gimbal_body"] == before
    assert op(model)["gimbal_sec"] == 0.5 and op(model)["firing_sec"] == 0
    burn(model, id="again", direction=direction, duration=5.0)
    life.advance(model, 1.5, "finish-gimbal")
    assert op(model, "again")["firing_sec"] == 0
    assert op(model, "again")["gimbal_sec"] == pytest.approx(1.5)


def test_parallel_cpu_radio_and_body_thrust_share_bus_once():
    from tests import test_resource_lanes as lanes

    data = copy.deepcopy(lanes.scenario.__wrapped__())
    definition = data["definition"]
    definition["actuation"] = {"engines": {"a": engine()}}
    data = spacecraft_scenario(definition, id="parallel-thrust", lineage="B04-D1")
    model = runtime(data)
    life.acquire(model)
    initial = model._backend.mission.energy_wh["a"]
    rows = [
        {
            "command_id": "cpu",
            "actor_id": "a",
            "kind": "transform",
            "duration_sec": 2.5,
            "job_id": "compress",
        },
        {
            "command_id": "radio",
            "actor_id": "a",
            "kind": "downlink",
            "duration_sec": 2.5,
            "product_id": "archive",
            "destination_id": "Ground",
        },
        {
            "command_id": "burn",
            "actor_id": "a",
            "kind": "body_burn",
            "duration_sec": 2.5,
            "body_burn": {
                "frame": "BODY",
                "direction_body": [0.0, 0.0, 1.0],
                "thrust_n": 1.0,
                "firing_sec": 1.0,
            },
        },
    ]
    model.submit(rows, expected_revision=model.revision, idempotency_key="three")
    life.advance(model, 2.5, "together")
    assert op(model)["status"] == "completed"
    assert life.lifecycle(model)["jobs"]["compress"]["status"] == "completed"
    assert model._backend.mission.remaining["archive"] == 0
    assert initial - model._backend.mission.energy_wh["a"] == pytest.approx(
        (2 * 2.5 + 5 * 2 + 8 * 1 + 3.5) / 3600 / 0.9, abs=1e-12
    )


@pytest.mark.parametrize("other", ["slew", "imaging", "directed-radio"])
def test_engine_exclusively_owns_attitude_in_parallel_mode(scenario, other):
    data = copy.deepcopy(scenario)
    data["definition"]["execution"] = {
        "actors": {a: {"memory_bandwidth_bps": 10000000.0} for a in data["actor_order"]}
    }
    rows = [
        {
            "command_id": "burn",
            "actor_id": "a",
            "kind": "body_burn",
            "duration_sec": 3.0,
            "body_burn": {
                "frame": "BODY",
                "direction_body": [0.0, 0.0, 1.0],
                "thrust_n": 1.0,
                "firing_sec": 1.0,
            },
        }
    ]
    if other == "slew":
        rows.append(
            {
                "command_id": "slew",
                "actor_id": "a",
                "kind": "slew",
                "duration_sec": 1.0,
                "slew": {"frame": "GCRS", "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            }
        )
    elif other == "imaging":
        rows.append(
            {
                "command_id": "image",
                "actor_id": "a",
                "kind": "imaging",
                "duration_sec": 2.0,
                "request_id": "request",
            }
        )
    else:
        data["definition"]["spacecraft_bus"]["a"]["attitude"]["radio_tolerance_deg"] = (
            5.0
        )
        data["definition"]["operations"]["products"] = [
            {
                "id": "archive",
                "actor": "a",
                "size_bits": 1000000,
                "deadline_sec": 20.0,
                "destination": "Ground",
            }
        ]
        rows.append(
            {
                "command_id": "radio",
                "actor_id": "a",
                "kind": "downlink",
                "duration_sec": 1.0,
                "product_id": "archive",
                "destination_id": "Ground",
            }
        )
    model = runtime(data)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError) as error:
        model.submit(rows, expected_revision=0, idempotency_key="conflict")
    assert error.value.code == "resource_busy" and model.checkpoint() == before


def test_gcrs_maneuver_shortcuts_remain_unsupported_in_body_profile(scenario):
    model = runtime(scenario)
    assert "commands.body_burn" in model.describe_capabilities()["supported"]
    assert "commands.impulse" not in model.describe_capabilities()["supported"]
    assert "commands.finite_burn" not in model.describe_capabilities()["supported"]
    with pytest.raises(RuntimeContractError, match="gimbal cone"):
        burn(model, direction=[1.0, 0.0, 0.0])
    with pytest.raises(RuntimeContractError, match="engine limit"):
        burn(model, thrust=1.5)


def test_applied_thrust_native_failure_rolls_back_trajectory_fuel_and_resources(
    scenario, monkeypatch
):
    model = runtime(scenario)
    burn(model)
    life.advance(model, 0.5, "ignition")
    before = model.checkpoint()
    cls = type(model._backend.mission)
    original = cls.step

    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected native failure after thrust")

    with monkeypatch.context() as patch:
        patch.setattr(cls, "step", fail)
        with pytest.raises(RuntimeError, match="injected"):
            life.advance(model, 0.5, "failed")
    assert model.checkpoint() == before
    assert model._backend.mission.bus is model._backend.bus
    life.advance(model, 2.0, "complete")
    assert op(model)["status"] == "completed"


def test_coupled_burn_preserves_navigation_covariance_prediction(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["orbits"]["a"]["navigation"] = {
        "initial_covariance_si": np.eye(6).tolist()
    }
    model = runtime(data)
    burn(model, firing=0.5, duration=1.0)
    life.advance(model, 1.0, "with-covariance")
    provider = model._backend.mission.ephemerides["a"]
    covariance = provider.covariance(model.time_sec)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert op(model)["status"] == "completed"


def test_global_segment_budget_cannot_be_double_spent_by_two_engines(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["actuation"]["engines"]["b"] = engine()
    data["definition"]["actuation"]["max_segments"] = 1
    model = runtime(data)
    rows = [
        {
            "command_id": "burn-" + actor,
            "actor_id": actor,
            "kind": "body_burn",
            "duration_sec": 3.0,
            "body_burn": {
                "frame": "BODY",
                "direction_body": [0.0, 0.0, 1.0],
                "thrust_n": 1.0,
                "firing_sec": 2.0,
            },
        }
        for actor in ("a", "b")
    ]
    model.submit(rows, expected_revision=0, idempotency_key="fleet")
    life.advance(model, 3.0, "bounded-fleet")
    data = report(model)
    assert len(data["intervals"]) == 1
    assert sum(row["firing_sec"] for row in data["operations"].values()) == 0.5
    assert all(abs(value) < 1e-12 for value in data["fuel_balance_error_kg"].values())


def test_actor_observation_contains_no_peer_engine_commands(scenario):
    model = runtime(scenario)
    burn(model)
    life.advance(model, 0.25, "warm")
    own = model.observe("a")["actors"]["a"]["lifecycle"]["actuation"]
    peer = model.observe("b")["actors"]["b"]["lifecycle"]["actuation"]
    assert "burn" in own["operations"] and peer is None
    own["state"]["gimbal_body"][0] = 42.0
    assert report(model)["states"]["a"]["gimbal_body"][0] == 0


@pytest.mark.parametrize("invalid", ["frame", "unit", "zero-window"])
def test_body_command_rejects_frame_shortcuts_and_invalid_geometry(scenario, invalid):
    model = runtime(scenario)
    row = {
        "command_id": "bad",
        "actor_id": "a",
        "kind": "body_burn",
        "duration_sec": 3.0,
        "body_burn": {
            "frame": "BODY",
            "direction_body": [0.0, 0.0, 1.0],
            "thrust_n": 1.0,
            "firing_sec": 1.0,
        },
    }
    if invalid == "frame":
        row["body_burn"]["frame"] = "GCRS"
    elif invalid == "unit":
        row["body_burn"]["direction_body"] = [0.0, 0.0, 2.0]
    else:
        row["duration_sec"] = 0.0
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError) as error:
        model.submit([row], expected_revision=0, idempotency_key="bad")
    assert error.value.code == "invalid_command" and model.checkpoint() == before


def test_gimbal_boundary_does_not_strand_simultaneous_cpu_at_roundoff():
    from tests import test_resource_lanes as lanes

    data = copy.deepcopy(lanes.scenario.__wrapped__())
    data["definition"]["actuation"] = {"engines": {"a": engine()}}
    model = runtime(
        spacecraft_scenario(data["definition"], id="gimbal-cpu", lineage="B04-D1")
    )
    life.acquire(model)
    angle = math.radians(10.0)
    model.submit(
        [
            {
                "command_id": "burn",
                "actor_id": "a",
                "kind": "body_burn",
                "duration_sec": 5.0,
                "body_burn": {
                    "frame": "BODY",
                    "direction_body": [math.sin(angle), 0.0, math.cos(angle)],
                    "thrust_n": 1.0,
                    "firing_sec": 1.0,
                },
            },
            {
                "command_id": "cpu",
                "actor_id": "a",
                "kind": "transform",
                "duration_sec": 2.0,
                "job_id": "compress",
            },
        ],
        expected_revision=model.revision,
        idempotency_key="gimbal-and-cpu",
    )
    life.advance(model, 2.0, "shared-endpoint")
    assert life.lifecycle(model)["jobs"]["compress"]["status"] == "completed"
    assert "final" in life.lifecycle(model)["artifacts"]
