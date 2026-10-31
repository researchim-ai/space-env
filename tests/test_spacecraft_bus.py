"""B04-A coupled attitude/power/thermal/health engineering contracts."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from space_env.core.attitude import align_z, boresight, slew_step
from space_env.reference.orbit import _ephemerides
from space_env.reference.spacecraft_bus import (
    bus_preset,
    pointing_quaternion,
    quaternion_angle,
    rotate_body,
)
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    spacecraft_scenario,
)
from tests import test_reference_lifecycle as life


@pytest.fixture(scope="module")
def scenario():
    previous = life.scenario.__wrapped__()
    template = life.runtime(previous)
    definition = copy.deepcopy(previous["definition"])
    definition["spacecraft_bus"] = {
        a: bus_preset("synthetic-agile/1") for a in definition["orbits"]
    }
    for a, model in definition["spacecraft_bus"].items():
        q = pointing_quaternion(
            template._backend.mission.ephemerides[a],
            template._backend.acquisition_targets["site"],
            0.0,
        )
        model["attitude"]["initial_quaternion_wxyz"] = q.tolist()
        definition["operations"]["vehicles"][a]["solar_mode"] = "body_fixed_bus"
    return spacecraft_scenario(definition, id="coupled", lineage="B04-A")


def runtime(scenario):
    return MissionRuntime(scenario, access=AccessPolicy(truth=True, forecast=True))


def bus(model):
    return model.observe(view="truth")["backend"]["report"]["spacecraft_bus"]


def command(model, goal, *, duration=10.0, id="slew"):
    return model.submit(
        [
            {
                "command_id": id,
                "actor_id": "a",
                "kind": "slew",
                "duration_sec": duration,
                "slew": {"frame": "GCRS", "quaternion_wxyz": np.asarray(goal).tolist()},
            }
        ],
        expected_revision=model.revision,
        idempotency_key=id,
    )


def test_zero_time_or_authority_does_not_teleport_core_attitude():
    start = np.array([1.0, 0.0, 0.0, 0.0])
    goal = align_z([1.0, 0.0, 0.0])
    for rate, dt in ((2.0, 0.0), (0.0, 1.0), (-1.0, 1.0)):
        q, remaining = slew_step(start, goal, rate, dt)
        np.testing.assert_array_equal(q, start)
        assert remaining == pytest.approx(90.0)
    q, remaining = slew_step(start, goal, 2.0, 1.0)
    assert quaternion_angle(q, start) == pytest.approx(2.0)
    assert remaining == pytest.approx(88.0)


def test_prepointed_acquisition_uses_energy_thermal_and_true_attitude(scenario):
    model = runtime(scenario)
    life.acquire(model)
    report = life.lifecycle(model)
    assert "raw" in report["artifacts"]
    assert report["artifacts"]["raw"]["spacecraft_bus"]["quaternion_wxyz"]
    state = bus(model)["states"]["a"]
    assert state["electrical_load_wh"] > 0 and state["discharge_loss_wh"] > 0
    assert state["heat_j"] > 0
    for key in (
        "thermal_balance_error_j",
        "pv_balance_error_wh",
        "discharge_balance_error_wh",
    ):
        assert abs(bus(model)["ledgers"]["a"][key]) < 1e-8
    assert model.report()["profile"] == "reference-spacecraft-bus/1"


def test_wrong_pointing_prevents_raw_without_hidden_slew(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    model = runtime(data)
    life.acquire(model)
    assert not life.lifecycle(model)["artifacts"]
    assert life.lifecycle(model)["requests"]["request"]["reason"] == "pointing"
    np.testing.assert_allclose(
        bus(model)["states"]["a"]["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0]
    )


def test_slew_takes_time_and_energy_and_settling_before_completion(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["attitude"].update(
        initial_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0], max_rate_deg_s=10.0
    )
    model = runtime(data)
    goal = align_z([1.0, 0.0, 0.0])
    command(model, goal, duration=12.0)
    life.advance(model, 4.0, "partial")
    state = bus(model)["states"]["a"]
    assert quaternion_angle(
        [1.0, 0.0, 0.0, 0.0], state["quaternion_wxyz"]
    ) == pytest.approx(40.0, abs=1e-6)
    assert model.report()["commands"][0]["status"] == "running"
    assert state["slew_energy_wh"] == pytest.approx(3 * 4 / 3600)
    life.advance(model, 5.0, "rotation-end")
    assert bus(model)["states"]["a"]["settle_remaining_sec"] == pytest.approx(1.0)
    life.advance(model, 1.0, "settle")
    assert model.report()["commands"][0]["status"] == "completed"
    assert quaternion_angle(bus(model)["states"]["a"]["quaternion_wxyz"], goal) < 1e-6


def test_partial_slew_checkpoint_cancel_preserves_quaternion_energy_and_settling(
    scenario, tmp_path
):
    model = runtime(scenario)
    start = np.array(bus(model)["states"]["a"]["quaternion_wxyz"])
    command(model, [1.0, 0.0, 0.0, 0.0], duration=10.0)
    life.advance(model, 1.0, "partial")
    path = tmp_path / "bus.json"
    model.save(path)
    restored = MissionRuntime.load(path, access=AccessPolicy(truth=True))
    q = bus(model)["states"]["a"]["quaternion_wxyz"]
    for item in (model, restored):
        item.cancel("slew", expected_revision=item.revision, idempotency_key="cancel")
        np.testing.assert_allclose(bus(item)["states"]["a"]["quaternion_wxyz"], q)
        assert bus(item)["states"]["a"]["settle_remaining_sec"] == 1.0
        life.advance(item, 1.0, "idle")
    assert model.checkpoint() == restored.checkpoint()
    assert quaternion_angle(start, q) == pytest.approx(2.0, abs=1e-6)


def test_short_slew_window_does_not_fabricate_reached_orientation(scenario):
    model = runtime(scenario)
    initial = bus(model)["states"]["a"]["quaternion_wxyz"]
    command(model, [1.0, 0.0, 0.0, 0.0], duration=0.5)
    life.advance(model, 0.5, "short")
    assert model.report()["commands"][0]["status"] == "failed"
    assert bus(model)["operations"]["slew"]["reason"] == "slew_window_ended"
    assert quaternion_angle(
        initial, bus(model)["states"]["a"]["quaternion_wxyz"]
    ) == pytest.approx(1.0, abs=1e-6)


def test_body_fixed_panel_orientation_and_conversion_losses_change_charge(scenario):
    data = copy.deepcopy(scenario)
    for orbit in data["definition"]["orbits"].values():
        orbit["initial"]["r_m"] = [-x for x in orbit["initial"]["r_m"]]
        orbit["initial"]["v_m_s"] = [-x for x in orbit["initial"]["v_m_s"]]
    template = runtime(data)
    p = template._backend.mission.ephemerides["a"]
    sun, _ = _ephemerides(p.epoch)
    direction = sun - p.state(0)[:3]
    for vehicle in data["definition"]["operations"]["vehicles"].values():
        vehicle["solar_area_m2"] = 0.1
        vehicle["initial_soc"] = 0.5
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        align_z(direction).tolist()
    )
    toward = runtime(data)
    life.advance(toward, 1.0, "charge")
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        align_z(-direction).tolist()
    )
    away = runtime(data)
    life.advance(away, 1.0, "dark-panel")
    assert (
        bus(toward)["states"]["a"]["raw_pv_wh"] > bus(away)["states"]["a"]["raw_pv_wh"]
    )
    assert bus(toward)["states"]["a"]["charge_loss_wh"] > 0
    assert toward._backend.mission.energy_wh["a"] > away._backend.mission.energy_wh["a"]
    assert abs(bus(toward)["ledgers"]["a"]["pv_balance_error_wh"]) < 1e-12


def test_faults_pause_cpu_and_recovery_does_not_heal_state_or_degradation(scenario):
    model = runtime(scenario)
    life.acquire(model)
    life.submit(model, "transform", 2.0, "cpu", job_id="compress")
    life.advance(model, 0.5, "work")
    progress = life.lifecycle(model)["jobs"]["compress"]["remaining_sec"]
    model.inject(
        [life.event(model, "spacecraft_fault", {"actor": "a", "subsystem": "cpu"})],
        expected_revision=model.revision,
        idempotency_key="fault",
    )
    life.advance(model, 0.5, "pause")
    assert life.lifecycle(model)["jobs"]["compress"]["remaining_sec"] == progress
    model.inject(
        [
            life.event(
                model,
                "spacecraft_fault_cleared",
                {"actor": "a", "subsystem": "cpu"},
                "clear",
            )
        ],
        expected_revision=model.revision,
        idempotency_key="clear",
    )
    model.inject(
        [
            life.event(
                model,
                "spacecraft_degraded",
                {"actor": "a", "solar_scale": 0.5, "rate_scale": 0.5},
                "degrade",
            )
        ],
        expected_revision=model.revision,
        idempotency_key="degrade",
    )
    model.inject(
        [life.event(model, "spacecraft_safe_mode", {"actor": "a"}, "safe")],
        expected_revision=model.revision,
        idempotency_key="safe",
    )
    before = bus(model)["states"]["a"].copy()
    model.inject(
        [life.event(model, "spacecraft_recover", {"actor": "a"}, "recover")],
        expected_revision=model.revision,
        idempotency_key="recover",
    )
    after = bus(model)["states"]["a"]
    for key in ("temperature_k", "quaternion_wxyz", "solar_scale", "rate_scale"):
        assert after[key] == before[key]
    assert not after["safe"]


def test_safe_mode_aborts_continuous_acquisition_and_low_soc_cannot_recover(scenario):
    model = runtime(scenario)
    life.submit(model, "imaging", 2.0, "image", request_id="request")
    life.advance(model, 0.5, "partial")
    model.inject(
        [life.event(model, "spacecraft_safe_mode", {"actor": "a"})],
        expected_revision=model.revision,
        idempotency_key="safe",
    )
    life.advance(model, 1.5, "abort")
    assert (
        not life.lifecycle(model)["artifacts"]
        and life.lifecycle(model)["requests"]["request"]["reason"] == "safe_mode"
    )
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["vehicles"]["a"]["initial_soc"] = 0.04
    low = runtime(data)
    snapshot = low.checkpoint()
    assert bus(low)["states"]["a"]["safe"]
    with pytest.raises(RuntimeContractError, match="recovery"):
        low.inject(
            [life.event(low, "spacecraft_recover", {"actor": "a"})],
            expected_revision=0,
            idempotency_key="recover",
        )
    assert low.checkpoint() == snapshot


def test_thermal_trip_changes_pipeline_and_conserves_node_energy(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["thermal"].update(
        initial_k=280.0, capacity_j_per_k=1.0, trip_k=281.0, resume_k=280.5
    )
    model = runtime(data)
    life.acquire(model)
    assert bus(model)["states"]["a"]["safe"]
    assert not life.lifecycle(model)["artifacts"]
    assert abs(bus(model)["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-8


def test_presets_have_distinct_executed_bus_constraints(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["lifecycle"]["payloads"]["a"]["power_w"] = 20.0
    agile = runtime(data)
    life.acquire(agile)
    constrained = bus_preset("synthetic-constrained/1")
    constrained["attitude"]["initial_quaternion_wxyz"] = data["definition"][
        "spacecraft_bus"
    ]["a"]["attitude"]["initial_quaternion_wxyz"]
    data["definition"]["spacecraft_bus"]["a"] = constrained
    slow = runtime(data)
    life.acquire(slow)
    assert "raw" in life.lifecycle(agile)["artifacts"]
    assert life.lifecycle(slow)["requests"]["request"]["reason"] == "bus_power_capacity"
    assert all(
        m["parameter_status"] == "synthetic" for m in bus(agile)["models"].values()
    )


@pytest.mark.parametrize(
    "change", ["quaternion", "frame", "solar_mode", "actor", "thermal", "profile"]
)
def test_invalid_bus_profiles_reject_without_fallback(scenario, change):
    data = copy.deepcopy(scenario)
    if change == "quaternion":
        data["definition"]["spacecraft_bus"]["a"]["attitude"][
            "initial_quaternion_wxyz"
        ] = [0.0, 0.0, 0.0, 0.0]
    elif change == "frame":
        model = runtime(data)
        with pytest.raises(RuntimeContractError):
            model.submit(
                [
                    {
                        "command_id": "bad",
                        "actor_id": "a",
                        "kind": "slew",
                        "duration_sec": 1.0,
                        "slew": {
                            "frame": "ITRS",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    }
                ],
                expected_revision=0,
                idempotency_key="bad",
            )
        return
    elif change == "solar_mode":
        data["definition"]["operations"]["vehicles"]["a"]["solar_mode"] = (
            "ideal_sun_tracking"
        )
    elif change == "actor":
        data["definition"]["spacecraft_bus"].pop("b")
    elif change == "thermal":
        data["definition"]["operations"]["vehicles"]["a"]["thermal"] = {
            "capacity_j_per_k": 10.0,
            "conductance_w_per_k": 1.0,
            "heat_tx_w": 1.0,
            "trip_k": 310.0,
            "resume_k": 300.0,
        }
    else:
        from space_env.runtime import lifecycle_scenario

        with pytest.raises(RuntimeContractError):
            MissionRuntime(
                lifecycle_scenario(data["definition"], id="bad", lineage="B04")
            )
        return
    with pytest.raises(RuntimeContractError):
        runtime(data)


def test_bus_fault_batch_and_runtime_rollback_preserve_aliases(scenario, monkeypatch):
    model = runtime(scenario)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError):
        model.inject(
            [
                life.event(model, "spacecraft_safe_mode", {"actor": "a"}, "safe"),
                life.event(
                    model,
                    "spacecraft_degraded",
                    {"actor": "b", "solar_scale": 2.0, "rate_scale": 1.0},
                    "bad",
                ),
            ],
            expected_revision=0,
            idempotency_key="batch",
        )
    assert model.checkpoint() == before
    assert model._backend.mission.bus is model._backend.bus
    command(model, [1.0, 0.0, 0.0, 0.0])
    before = model.checkpoint()

    def fail(*args):
        raise RuntimeError("bus integration failure")

    monkeypatch.setattr(model._backend.bus, "integrate", fail)
    with pytest.raises(RuntimeError):
        life.advance(model, 1.0, "advance")
    assert (
        model.checkpoint() == before
        and model._backend.mission.bus is model._backend.bus
    )


def test_sun_keepout_and_radio_faults_gate_actual_work(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["sun_keep_out_deg"] = 90.0
    guarded = runtime(data)
    life.acquire(guarded)
    assert not life.lifecycle(guarded)["artifacts"]
    assert life.lifecycle(guarded)["requests"]["request"]["reason"] == "sun_keep_out"
    model = runtime(scenario)
    life.acquire(model)
    life.transform(model)
    model.inject(
        [life.event(model, "spacecraft_fault", {"actor": "a", "subsystem": "radio"})],
        expected_revision=model.revision,
        idempotency_key="fault",
    )
    life.submit(
        model,
        "downlink",
        1.0,
        "blocked-radio",
        destination_id="Ground",
        product_id="final",
    )
    life.advance(model, 1.0, "blocked")
    assert not life.lifecycle(model)["acknowledgements"]
    assert bus(model)["denials"][-1]["reason"] == "radio_fault"
    model.inject(
        [
            life.event(
                model,
                "spacecraft_fault_cleared",
                {"actor": "a", "subsystem": "radio"},
                "clear",
            )
        ],
        expected_revision=model.revision,
        idempotency_key="clear",
    )
    life.submit(
        model, "downlink", 1.0, "radio", destination_id="Ground", product_id="final"
    )
    life.advance(model, 1.0, "delivery")
    assert life.lifecycle(model)["metrics"]["fulfilled_requests"] == 1


def test_slew_settling_energy_and_preset_rate_difference(scenario):
    completed = []
    for name in ("synthetic-agile/1", "synthetic-constrained/1"):
        data = copy.deepcopy(scenario)
        data["definition"]["spacecraft_bus"]["a"] = bus_preset(name)
        model = runtime(data)
        goal = align_z([math.sin(math.radians(10)), 0.0, math.cos(math.radians(10))])
        command(model, goal, duration=10.0)
        life.advance(model, 10.0, "slew-time")
        completed.append(model.report()["commands"][0]["status"])
        if name == "synthetic-agile/1":
            assert bus(model)["states"]["a"]["slew_energy_wh"] == pytest.approx(
                3 * 6 / 3600, abs=1e-10
            )
    assert completed == ["completed", "failed"]


def test_failed_or_cancelled_slew_has_no_hidden_attitude_reset_and_recovery_needs_hysteresis(
    scenario,
):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["thermal"].update(initial_k=311.0)
    model = runtime(data)
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="recovery"):
        model.inject(
            [life.event(model, "spacecraft_recover", {"actor": "a"})],
            expected_revision=0,
            idempotency_key="recover",
        )
    assert model.checkpoint() == before
    command(model, [1.0, 0.0, 0.0, 0.0])
    life.advance(model, 1.0, "blocked")
    assert model.report()["commands"][0]["status"] == "failed"
    np.testing.assert_allclose(
        bus(model)["states"]["a"]["quaternion_wxyz"],
        data["definition"]["spacecraft_bus"]["a"]["attitude"][
            "initial_quaternion_wxyz"
        ],
    )


def test_degradation_is_monotonic_and_cross_actor_bus_observation_is_private(scenario):
    model = runtime(scenario)
    model.inject(
        [
            life.event(
                model,
                "spacecraft_degraded",
                {"actor": "a", "solar_scale": 0.5, "rate_scale": 0.5},
            )
        ],
        expected_revision=0,
        idempotency_key="degrade",
    )
    before = model.checkpoint()
    with pytest.raises(RuntimeContractError, match="monotonic"):
        model.inject(
            [
                life.event(
                    model,
                    "spacecraft_degraded",
                    {"actor": "a", "solar_scale": 1.0, "rate_scale": 1.0},
                    "repair",
                )
            ],
            expected_revision=1,
            idempotency_key="repair",
        )
    assert model.checkpoint() == before
    local = model.agent("a").observe()
    assert set(local["actors"]) == {"a"} and "spacecraft_bus" in local["actors"]["a"]


def test_body_to_gcrs_rotation_and_quaternion_sign_equivalence():
    q = align_z([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        rotate_body(q, [0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(boresight(q), boresight(-q), atol=1e-12)
    assert quaternion_angle(q, -q) == 0


def test_constant_load_battery_and_rc_temperature_match_independent_equations(scenario):
    data = copy.deepcopy(scenario)
    vehicle = data["definition"]["operations"]["vehicles"]["a"]
    vehicle["base_power_w"] = 10.0
    model = runtime(data)
    initial_energy = model._backend.mission.energy_wh["a"]
    life.advance(model, 10.0, "constant-load")
    state = bus(model)["states"]["a"]
    expected_useful = 100 / 3600
    expected_withdrawal = expected_useful / 0.9
    assert state["electrical_load_wh"] == pytest.approx(expected_useful, abs=1e-12)
    assert model._backend.mission.energy_wh["a"] == pytest.approx(
        initial_energy - expected_withdrawal, abs=1e-12
    )
    heat_w = 10 * 0.8 + 10 / 0.9 - 10
    expected_temperature = 280 + heat_w + (285 - (280 + heat_w)) * math.exp(-10 / 200)
    assert state["temperature_k"] == pytest.approx(expected_temperature, abs=1e-9)
    assert abs(bus(model)["ledgers"]["a"]["thermal_balance_error_j"]) < 1e-8


def test_bus_limited_native_cpu_and_radio_pointing_do_not_bypass_arbitration(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["products"] = [
        {
            "id": "preloaded",
            "actor": "a",
            "size_bits": 100000,
            "deadline_sec": 20.0,
            "processing_sec": 1.0,
        }
    ]
    data["definition"]["operations"]["vehicles"]["a"]["processing_power_w"] = 30.0
    data["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 20.0
    model = runtime(data)
    life.submit(model, "processing", 1.0, "cpu", product_id="preloaded")
    life.advance(model, 1.0, "cpu-time")
    assert model._backend.mission.processing_remaining["preloaded"] == 1.0
    assert bus(model)["denials"][-1]["reason"] == "bus_power_capacity"
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["attitude"]["radio_tolerance_deg"] = 1.0
    model = runtime(data)
    life.acquire(model)
    life.transform(model)
    command(model, [1.0, 0.0, 0.0, 0.0], duration=0.5)
    life.advance(model, 0.5, "wrong-radio-attitude")
    life.submit(
        model, "downlink", 1.0, "radio", destination_id="Ground", product_id="final"
    )
    life.advance(model, 1.0, "radio-time")
    assert not life.lifecycle(model)["acknowledgements"]
    assert bus(model)["denials"][-1]["reason"] == "radio_pointing"


def test_full_pipeline_credits_only_delivered_product_and_checkpoint_replays(
    scenario, tmp_path
):
    model = runtime(scenario)
    life.acquire(model)
    life.transform(model)
    life.submit(
        model, "downlink", 1.0, "radio", destination_id="Ground", product_id="final"
    )
    life.advance(model, 0.25, "partial")
    assert life.lifecycle(model)["metrics"]["fulfilled_requests"] == 0
    path = tmp_path / "pipeline.json"
    model.save(path)
    restored = MissionRuntime.load(path, access=AccessPolicy(truth=True))
    assert restored._backend.mission.bus is restored._backend.bus
    for item in (model, restored):
        life.advance(item, 0.75, "delivery")
    assert model.checkpoint() == restored.checkpoint()
    assert life.lifecycle(model)["metrics"]["fulfilled_requests"] == 1


def test_slew_power_floor_prevents_free_rotation(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["operations"]["vehicles"]["a"]["initial_soc"] = 0.01
    model = runtime(data)
    before = bus(model)["states"]["a"]["quaternion_wxyz"]
    command(model, [1.0, 0.0, 0.0, 0.0])
    life.advance(model, 1.0, "unpowered")
    assert model.report()["commands"][0]["status"] == "failed"
    np.testing.assert_array_equal(bus(model)["states"]["a"]["quaternion_wxyz"], before)


def test_safe_base_demand_cannot_exceed_bus_capacity(scenario):
    data = copy.deepcopy(scenario)
    data["definition"]["spacecraft_bus"]["a"]["power"]["safe_base_power_w"] = 101.0
    with pytest.raises(ValueError, match="safe base power exceeds"):
        runtime(data)


def test_coupled_profile_does_not_advertise_uncoupled_propulsion(scenario):
    model = runtime(scenario)
    assert "commands.impulse" not in model.describe_capabilities()["supported"]
    assert "commands.finite_burn" not in model.describe_capabilities()["supported"]
    assert scenario["fidelity"]["trajectory"] == "propagated-gcrs-coast"

    before = model.checkpoint()
    with pytest.raises(RuntimeContractError) as error:
        model.submit([{"actor_id": "a", "command_id": "burn", "kind": "impulse",
                       "duration_sec": 0.0, "impulse": {"frame": "GCRS", "delta_v_m_s": [0.0, 0.0, 1.0]}}],
                     expected_revision=0, idempotency_key="unsupported-burn")
    assert error.value.code == "unsupported_capability"
    assert model.checkpoint() == before
