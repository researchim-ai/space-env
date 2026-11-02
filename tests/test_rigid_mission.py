"""Opt-in rigid mission attitude: no ideal hold, paid control, replay and faults."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from space_env.reference.spacecraft_bus import quaternion_angle
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    spacecraft_scenario,
)
from tests import test_reference_lifecycle as life
from tests import test_spacecraft_bus as fixtures


def config(**updates):
    value = {
        "model": {
            "model_id": "synthetic-wheel-bus/1",
            "source": "declared analytical assumptions",
            "inertia_kg_m2": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            "wheel_inertia_kg_m2": [0.01, 0.01, 0.01],
            "max_torque_nm": [0.1, 0.1, 0.1],
            "max_momentum_nms": [0.2, 0.2, 0.2],
        },
        "supply_w": 12.0,
        "control_step_sec": 0.1,
    }
    return value | updates


@pytest.fixture(scope="module")
def scenario():
    d = copy.deepcopy(fixtures.scenario.__wrapped__()["definition"])
    a = d["spacecraft_bus"]["a"]["attitude"]
    a["initial_quaternion_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    a["rigid"] = config()
    return spacecraft_scenario(d, id="rigid-mission", lineage="B04-D2b1")


def runtime(s):
    return MissionRuntime(s, access=AccessPolicy(truth=True, forecast=True))


def view(m):
    return m.observe("a")["actors"]["a"]["spacecraft_bus"]["rigid_attitude"]


def torque(m, value=(0.05, 0.0, 0.0), duration=1.0, key="wheel"):
    return life.submit(
        m,
        "wheel_torque",
        duration,
        key,
        wheel_torque={"frame": "BODY", "motor_torque_nm": list(value)},
    )


def inject(m, type, **fields):
    return m.inject(
        [life.event(m, type, {"actor": "a", **fields})],
        expected_revision=m.revision,
        idempotency_key=type + str(m.revision),
    )


def test_admission_no_free_motion(scenario):
    m = runtime(scenario)
    before = view(m)
    torque(m)
    assert view(m) == before
    assert "commands.wheel_torque" in m.describe_capabilities()["supported"]


def test_paid_motor_reaction_and_common_bus_ledger(scenario):
    m = runtime(scenario)
    torque(m)
    before = m._backend.mission.energy_wh["a"]
    life.advance(m, 1.0, "run")
    s = view(m)
    assert s["omega_body_rad_s"][0] == pytest.approx(-0.025)
    assert s["wheel_momentum_nms"][0] == pytest.approx(0.05)
    assert s["dc_energy_j"] == pytest.approx(12.0)
    assert s["dc_heat_work_balance_error_j"] == pytest.approx(0.0, abs=1e-10)
    assert s["kinetic_work_balance_error_j"] == pytest.approx(0.0, abs=1e-10)
    assert before - m._backend.mission.energy_wh["a"] == pytest.approx(
        12.0 / (3600 * 0.9), abs=1e-10
    )
    for value in fixtures.bus(m)["ledgers"]["a"].values():
        assert math.isfinite(value)
    assert fixtures.bus(m)["ledgers"]["a"]["thermal_balance_error_j"] == pytest.approx(
        0.0, abs=1e-9
    )


def test_after_window_free_rotation_without_free_braking(scenario):
    m = runtime(scenario)
    torque(m)
    life.advance(m, 1.0, "motor")
    before = view(m)
    life.advance(m, 1.0, "coast")
    after = view(m)
    assert after["omega_body_rad_s"] == pytest.approx(before["omega_body_rad_s"])
    assert after["wheel_momentum_nms"] == pytest.approx(before["wheel_momentum_nms"])
    assert quaternion_angle(after["quaternion_wxyz"], before["quaternion_wxyz"]) > 1.0
    assert after["dc_energy_j"] == before["dc_energy_j"]


def test_cancel_preserves_spin(scenario):
    m = runtime(scenario)
    torque(m, duration=2.0)
    life.advance(m, 0.5, "partial")
    before = view(m)
    m.cancel("wheel", expected_revision=m.revision, idempotency_key="cancel")
    life.advance(m, 0.5, "coast")
    after = view(m)
    assert after["omega_body_rad_s"] == pytest.approx(before["omega_body_rad_s"])
    assert after["wheel_momentum_nms"] == pytest.approx(before["wheel_momentum_nms"])
    assert after["dc_energy_j"] == before["dc_energy_j"]


@pytest.mark.parametrize("subsystem", ["adcs", "bus"])
def test_fault_stops_motor_not_momentum(scenario, subsystem):
    m = runtime(scenario)
    torque(m, duration=2.0)
    life.advance(m, 0.5, "partial")
    before = view(m)
    inject(m, "spacecraft_fault", subsystem=subsystem)
    life.advance(m, 0.5, "faulted")
    after = view(m)
    assert after["omega_body_rad_s"] == pytest.approx(before["omega_body_rad_s"])
    assert after["wheel_momentum_nms"] == pytest.approx(before["wheel_momentum_nms"])
    assert after["dc_energy_j"] == before["dc_energy_j"]


def test_safe_mode_does_not_despin(scenario):
    m = runtime(scenario)
    torque(m, duration=2.0)
    life.advance(m, 0.5, "partial")
    before = view(m)
    inject(m, "spacecraft_safe_mode")
    life.advance(m, 0.5, "safe")
    assert view(m)["omega_body_rad_s"] == pytest.approx(before["omega_body_rad_s"])


def test_finite_torque_and_saturation(scenario):
    m = runtime(scenario)
    torque(m, (100.0, 0.0, 0.0), duration=4.0)
    life.advance(m, 4.0, "saturate")
    s = view(m)
    assert s["wheel_momentum_nms"][0] == pytest.approx(0.2)
    assert s["capacity_limited_sec"][0] > 1.9
    assert s["omega_body_rad_s"][0] == pytest.approx(-0.1)
    np.testing.assert_allclose(s["momentum_inertial_nms"], [0, 0, 0], atol=1e-10)


def test_checkpoint_mid_motor_and_continue(scenario, tmp_path):
    m = runtime(scenario)
    torque(m, duration=2.0)
    life.advance(m, 0.55, "partial")
    p = tmp_path / "rigid.json"
    m.save(p)
    restored = MissionRuntime.load(p, access=AccessPolicy(truth=True, forecast=True))
    assert restored.checkpoint() == m.checkpoint()
    life.advance(m, 0.45, "continue")
    life.advance(restored, 0.45, "continue")
    assert restored.checkpoint() == m.checkpoint()
    assert restored._backend.mission.bus is restored._backend.bus


def test_other_actor_not_given_wheels(scenario):
    m = runtime(scenario)
    with pytest.raises(RuntimeContractError):
        m.submit(
            [
                {
                    "command_id": "bad",
                    "actor_id": "b",
                    "kind": "wheel_torque",
                    "duration_sec": 1.0,
                    "wheel_torque": {
                        "frame": "BODY",
                        "motor_torque_nm": [0.01, 0.0, 0.0],
                    },
                }
            ],
            expected_revision=m.revision,
            idempotency_key="bad",
        )
    assert "rigid_attitude" not in m.observe("b")["actors"]["b"]["spacecraft_bus"]
    assert "b" not in m.observe("a")["actors"]


def test_attitude_conflict_with_slew(scenario):
    m = runtime(scenario)
    torque(m, duration=2.0)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        fixtures.command(m, [1.0, 0.0, 0.0, 0.0])
    assert m.checkpoint() == before


def test_short_slew_does_not_snap_to_target_or_stop(scenario):
    m = runtime(scenario)
    fixtures.command(m, [math.cos(0.1), math.sin(0.1), 0.0, 0.0], duration=0.5)
    life.advance(m, 0.5, "slew-run")
    assert fixtures.bus(m)["operations"]["slew"]["status"] == "failed"
    assert abs(view(m)["omega_body_rad_s"][0]) > 0
    assert (
        quaternion_angle(
            view(m)["quaternion_wxyz"], [math.cos(0.1), math.sin(0.1), 0, 0]
        )
        > 1.0
    )


def test_slew_completion_needs_real_angle_rate_and_settling(scenario):
    d = copy.deepcopy(scenario)
    a = d["definition"]["spacecraft_bus"]["a"]["attitude"]
    a["settling_sec"] = 0.2
    a["max_rate_deg_s"] = 10.0
    a["rigid"]["damping_nm_per_rad_s"] = 3.0
    m = runtime(d)
    goal = [math.cos(0.01), math.sin(0.01), 0.0, 0.0]
    fixtures.command(m, goal, duration=15.0)
    life.advance(m, 15.0, "slew-run")
    assert fixtures.bus(m)["operations"]["slew"]["status"] == "completed"
    operation = fixtures.bus(m)["operations"]["slew"]
    assert operation["final_pointing_error_deg"] <= 0.1
    assert operation["final_rate_rad_s"] <= 1e-3
    assert operation["completed_sec"] < 15.0
    # Residual spin after completion is physical; there is no automatic hold.


@pytest.mark.parametrize(
    "updates",
    [
        {"supply_w": 1.0},
        {"initial_wheel_momentum_nms": [1.0, 0.0, 0.0]},
        {"initial_omega_body_rad_s": [11.0, 0.0, 0.0]},
        {"rate_tolerance_rad_s": 11.0},
    ],
)
def test_bad_controller_rejected(scenario, updates):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"].update(updates)
    with pytest.raises(RuntimeContractError):
        runtime(d)


def test_power_rejection_has_no_motor_work(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 6.0
    m = runtime(d)
    torque(m)
    life.advance(m, 1.0, "power")
    assert view(m)["dc_energy_j"] == 0.0
    assert view(m)["wheel_momentum_nms"] == [0.0, 0.0, 0.0]
    assert fixtures.bus(m)["operations"]["wheel"]["status"] == "failed"


def test_budget_failure_atomic(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"][
        "max_total_substeps"
    ] = 2
    m = runtime(d)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        life.advance(m, 1.0, "budget")
    assert m.checkpoint() == before


def test_low_soc_controller_stops_at_reserve(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["slew_min_soc"] = 0.1
    m = runtime(d)
    v = m._backend.mission.vehicles["a"]
    floor = m._backend.bus.models["a"].attitude.slew_min_soc * v.battery_wh
    m._backend.mission.energy_wh["a"] = floor + 6 / (3600 * 0.9)
    torque(m)
    life.advance(m, 1.0, "low")
    assert view(m)["dc_energy_j"] == pytest.approx(6.0, abs=1e-8)
    assert view(m)["wheel_momentum_nms"][0] == pytest.approx(0.025, abs=1e-8)


def test_imaging_uses_actual_rigid_pointing(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"][
        "initial_omega_body_rad_s"
    ] = [0.1, 0.0, 0.0]
    m = runtime(d)
    life.acquire(m)
    assert not life.lifecycle(m)["artifacts"]
    assert quaternion_angle(view(m)["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0]) > 0


def test_prepointed_rigid_acquisition_produces_real_product(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        fixtures.scenario.__wrapped__()["definition"]["spacecraft_bus"]["a"][
            "attitude"
        ]["initial_quaternion_wxyz"]
    )
    m = runtime(d)
    life.acquire(m)
    assert "raw" in life.lifecycle(m)["artifacts"]


def test_rotating_body_burn_fail_closed(scenario):
    from tests.test_actuation import burn, engine

    d = copy.deepcopy(scenario)
    d["definition"]["actuation"] = {"engines": {"a": engine()}}
    d = spacecraft_scenario(d["definition"], id="rigid-burn-rejected", lineage="test")
    m = runtime(d)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        burn(m)
    assert m.checkpoint() == before


def test_post_native_failure_rolls_back_rigid_and_bus(scenario, monkeypatch):
    m = runtime(scenario)
    torque(m)
    before = m.checkpoint()
    from space_env.reference.spacecraft_bus import SpacecraftBus

    original = SpacecraftBus.integrate

    def broken(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ValueError("injected after rigid integration")

    monkeypatch.setattr(SpacecraftBus, "integrate", broken)
    with pytest.raises(RuntimeContractError):
        life.advance(m, 0.5, "failure")
    assert m.checkpoint() == before
    assert m._backend.mission.bus is m._backend.bus


@pytest.mark.parametrize(
    "wheel",
    [
        {"frame": "GCRS", "motor_torque_nm": [0.1, 0.0, 0.0]},
        {"frame": "BODY", "motor_torque_nm": [float("nan"), 0.0, 0.0]},
        {"frame": "BODY", "motor_torque_nm": [0.1, 0.0]},
    ],
)
def test_bad_torque_command_atomic(scenario, wheel):
    m = runtime(scenario)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        life.submit(m, "wheel_torque", 1.0, "bad", wheel_torque=wheel)
    assert m.checkpoint() == before


def test_dynamic_rate_applicability_failure_rolls_back(scenario):
    d = copy.deepcopy(scenario)
    c = d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]
    c["max_applicable_rate_rad_s"] = 0.001
    c["rate_tolerance_rad_s"] = 0.0001
    m = runtime(d)
    torque(m)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        life.advance(m, 1.0, "over-rate")
    assert m.checkpoint() == before


def test_degradation_limits_wheel_authority_without_reset(scenario):
    m = runtime(scenario)
    inject(m, "spacecraft_degraded", solar_scale=1.0, rate_scale=0.5)
    torque(m, (1.0, 0.0, 0.0))
    life.advance(m, 1.0, "degraded")
    assert view(m)["wheel_momentum_nms"][0] == pytest.approx(0.05)


def test_payload_rate_gate_rejects_fast_prepointed_body(scenario):
    d = copy.deepcopy(scenario)
    a = d["definition"]["spacecraft_bus"]["a"]["attitude"]
    a["initial_quaternion_wxyz"] = fixtures.scenario.__wrapped__()["definition"][
        "spacecraft_bus"
    ]["a"]["attitude"]["initial_quaternion_wxyz"]
    a["rigid"]["initial_omega_body_rad_s"] = [0.0, 0.0, 0.1]
    m = runtime(d)
    life.acquire(m)
    assert not life.lifecycle(m)["artifacts"]
    assert any(r["reason"] == "rigid_payload_rate" for r in fixtures.bus(m)["denials"])


def test_directed_radio_checks_rotating_attitude_through_interval(scenario):
    from space_env.reference.spacecraft_bus import pointing_quaternion

    d = copy.deepcopy(scenario)
    template = runtime(d)
    station = template._backend.mission.stations["Ground"]
    q = pointing_quaternion(template._backend.mission.ephemerides["a"], station, 0.0)
    a = d["definition"]["spacecraft_bus"]["a"]["attitude"]
    a["initial_quaternion_wxyz"] = q.tolist()
    a["rigid"]["initial_omega_body_rad_s"] = [0.5, 0.0, 0.0]
    a["radio_tolerance_deg"] = 0.1
    m = runtime(d)
    bus = m._backend.bus
    bus.prepare(m._backend, [])
    row = {
        "command": {
            "actor_id": "a",
            "kind": "downlink",
            "destination_id": "Ground",
            "command_id": "radio",
        }
    }
    assert bus.ordinary_allowed(row, m._backend, span=0.0)
    assert not bus.ordinary_allowed(row, m._backend, span=0.1)


def test_idle_asymmetric_rigid_mission_conserves_inertial_momentum(scenario):
    d = copy.deepcopy(scenario)
    c = d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]
    c["initial_omega_body_rad_s"] = [0.1, 0.2, 0.3]
    c["initial_wheel_momentum_nms"] = [0.02, -0.01, 0.03]
    m = runtime(d)
    initial = view(m)
    life.advance(m, 2.0, "coast")
    final = view(m)
    np.testing.assert_allclose(
        final["momentum_inertial_nms"], initial["momentum_inertial_nms"], atol=1e-10
    )
    assert final["kinetic_work_balance_error_j"] == pytest.approx(0.0, abs=1e-10)
    assert final["dc_energy_j"] == 0.0


def test_rigid_control_grid_does_not_strand_complete_acquisition(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = (
        fixtures.scenario.__wrapped__()["definition"]["spacecraft_bus"]["a"][
            "attitude"
        ]["initial_quaternion_wxyz"]
    )
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]["payload_max_body_rate_rad_s"] = .1
    m = runtime(d)
    life.advance(m, 4.5, "coast-to-half-second")
    assert m.time_sec == 4.5
    life.acquire(m)
    assert m.time_sec == 6.5
    assert "raw" in life.lifecycle(m)["artifacts"]
