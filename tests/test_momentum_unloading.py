"""Resource-accounted balanced RCS/wheel unloading; no free momentum reset."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from space_env.core.propulsion import G0_M_S2
from space_env.reference.thrust import BalancedCoupleThrust
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    spacecraft_scenario,
)
from tests import test_reference_lifecycle as life
from tests import test_rigid_mission as rigid


def rcs(**updates):
    return {
        "model_id": "synthetic-rcs/1",
        "parameter_status": "synthetic",
        "source": "declared engineering assumptions",
        "propellant_pool": "shared-lumped",
        "max_total_nozzle_thrust_n": 1.0,
        "max_torque_nm": [0.1, 0.1, 0.1],
        "lever_arm_m": [0.5, 0.5, 0.5],
        "isp_s": 60.0,
        "startup_sec": 0.2,
        **updates,
    }


@pytest.fixture(scope="module")
def scenario():
    d = copy.deepcopy(rigid.scenario.__wrapped__()["definition"])
    d["spacecraft_bus"]["a"]["attitude"]["rigid"]["initial_wheel_momentum_nms"] = [
        0.08,
        0.0,
        0.0,
    ]
    d["spacecraft_bus"]["a"]["attitude"]["settling_sec"] = 0.2
    d["unloading"] = {"actors": {"a": rcs()}, "max_segments": 100}
    return spacecraft_scenario(d, id="unloading", lineage="B04-D2b3")


def runtime(s):
    return MissionRuntime(s, access=AccessPolicy(truth=True, forecast=True))


def report(m):
    return m._backend.report()["lifecycle"]["unloading"]


def unload(m, *, duration=3.0, key="unload", target=(0.0, 0.0, 0.0), tolerance=1e-5):
    return life.submit(
        m,
        "momentum_unload",
        duration,
        key,
        momentum_unload={
            "frame": "BODY",
            "target_momentum_nms": list(target),
            "tolerance_nms": tolerance,
        },
    )


def op(m):
    return report(m)["operations"]["unload"]


def test_admission_no_free_fuel_energy_or_momentum(scenario):
    m = runtime(scenario)
    before = rigid.view(m)
    unload(m)
    assert rigid.view(m) == before
    assert report(m)["fuel_used_kg"]["a"] == 0
    assert op(m)["status"] == "accepted"


def test_startup_paid_without_actuation(scenario):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.1, "boot")
    assert rigid.view(m)["wheel_momentum_nms"] == [0.08, 0.0, 0.0]
    assert op(m)["dc_energy_j"] == pytest.approx(0.3)
    assert op(m)["consumed_propellant_kg"] == 0
    assert rigid.view(m)["dc_energy_j"] == 0


def test_analytic_balanced_unload_closes_fuel_energy_and_angular_ledgers(scenario):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 3.0, "run")
    s = rigid.view(m)
    r = report(m)
    assert op(m)["status"] == "completed"
    np.testing.assert_allclose(s["wheel_momentum_nms"], [0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(s["omega_body_rad_s"], [0, 0, 0], atol=1e-12)
    assert op(m)["consumed_propellant_kg"] == pytest.approx(
        0.08 / (0.5 * 60 * G0_M_S2), abs=1e-11
    )
    np.testing.assert_allclose(
        s["external_angular_impulse_gcrs_nms"], [-0.08, 0, 0], atol=1e-10
    )
    np.testing.assert_allclose(
        s["angular_momentum_balance_error_gcrs_nms"], [0, 0, 0], atol=1e-10
    )
    assert abs(s["kinetic_work_balance_error_j"]) < 1e-10
    assert abs(r["energy_balance_error_j"]["unload"]) < 1e-9
    assert abs(r["fuel_balance_error_kg"]["a"]) < 1e-10
    assert s["heat_j"] > s["dc_energy_j"]  # Braking releases stored wheel energy.


def test_pair_mass_law_has_zero_force_and_correct_total_nozzle_flow():
    law = BalancedCoupleThrust(
        0.2, 60.0, 90.0, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.5, 0.5, 0.5)
    )
    np.testing.assert_array_equal(law.direction_at(0.5), [0, 0, 0])
    assert law.mass(100.0, 1.0) == pytest.approx(100 - 0.2 / (60 * G0_M_S2))
    with pytest.raises(ValueError):
        BalancedCoupleThrust(
            0.1, 60.0, 90.0, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.5, 0.5, 0.5)
        )


def test_zero_net_force_matches_unpowered_central_orbit(scenario):
    m = runtime(scenario)
    control = runtime(scenario)
    unload(m)
    life.advance(m, 2.0, "rcs")
    life.advance(control, 2.0, "coast")
    np.testing.assert_allclose(
        m._backend.mission.ephemerides["a"].state(2.0),
        control._backend.mission.ephemerides["a"].state(2.0),
        atol=1e-6,
        rtol=0,
    )
    assert all(row["net_force_gcrs_n"] == [0, 0, 0] for row in report(m)["intervals"])


def test_short_window_retains_remaining_momentum(scenario):
    m = runtime(scenario)
    unload(m, duration=0.5)
    life.advance(m, 0.5, "short")
    assert op(m)["status"] == "failed" and op(m)["reason"] == "unload_window_ended"
    assert rigid.view(m)["wheel_momentum_nms"][0] == pytest.approx(0.05)
    assert op(m)["consumed_propellant_kg"] > 0


def test_cancel_preserves_residual_and_paid_work(scenario):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.5, "partial")
    s = rigid.view(m)
    used = op(m)["consumed_propellant_kg"]
    m.cancel("unload", expected_revision=m.revision, idempotency_key="cancel")
    life.advance(m, 0.5, "coast")
    assert rigid.view(m)["wheel_momentum_nms"] == pytest.approx(s["wheel_momentum_nms"])
    assert op(m)["consumed_propellant_kg"] == used
    assert op(m)["status"] == "cancelled"


@pytest.mark.parametrize("subsystem", ["adcs", "propulsion", "bus"])
def test_fault_stops_both_actuators(scenario, subsystem):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.5, "partial")
    s = rigid.view(m)
    used = op(m)["consumed_propellant_kg"]
    rigid.inject(m, "spacecraft_fault", subsystem=subsystem)
    life.advance(m, 0.5, "faulted")
    assert rigid.view(m)["wheel_momentum_nms"] == pytest.approx(s["wheel_momentum_nms"])
    assert op(m)["consumed_propellant_kg"] == used
    assert op(m)["status"] == "failed"


def test_safe_mode_retains_momentum(scenario):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.5, "partial")
    before = rigid.view(m)
    rigid.inject(m, "spacecraft_safe_mode")
    life.advance(m, 0.5, "safe")
    assert rigid.view(m)["wheel_momentum_nms"] == pytest.approx(
        before["wheel_momentum_nms"]
    )


def test_rcs_authority_limits_progress(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["unloading"]["actors"]["a"]["max_torque_nm"] = [0.02, 0.02, 0.02]
    m = runtime(d)
    unload(m, duration=1.0)
    life.advance(m, 1.0, "limited")
    assert rigid.view(m)["wheel_momentum_nms"][0] == pytest.approx(0.064)
    assert op(m)["status"] == "failed"


def test_total_nozzle_limit_scales_actual_authority(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["unloading"]["actors"]["a"]["max_total_nozzle_thrust_n"] = 0.04
    m = runtime(d)
    unload(m, duration=1.0)
    life.advance(m, 1.0, "limited")
    assert rigid.view(m)["wheel_momentum_nms"][0] == pytest.approx(0.064)
    assert all(
        i["scalar_nozzle_thrust_n"] <= 0.04 + 1e-12 for i in report(m)["intervals"]
    )


def test_peak_power_rejection_never_moves_wheels(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["power"]["max_load_w"] = 14.0
    m = runtime(d)
    unload(m)
    life.advance(m, 1.0, "peak")
    assert rigid.view(m)["wheel_momentum_nms"] == [0.08, 0, 0]
    assert op(m)["status"] == "failed"
    assert op(m)["consumed_propellant_kg"] == 0


def test_low_soc_cuts_at_shared_reserve(scenario):
    m = runtime(scenario)
    v = m._backend.mission.vehicles["a"]
    # startup .6 J plus .2 s of wheel+RCS at 16 W.
    m._backend.mission.energy_wh["a"] = 0.1 * v.battery_wh + 3.8 / (3600 * 0.9)
    unload(m)
    life.advance(m, 1.0, "soc")
    assert rigid.view(m)["wheel_momentum_nms"][0] == pytest.approx(0.06, abs=1e-8)
    assert op(m)["status"] == "failed"


def test_propellant_exhaustion_does_not_continue_wheel_unload(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["orbits"]["a"]["propulsion"]["propellant_kg"] = 1e-5
    m = runtime(d)
    unload(m)
    life.advance(m, 2.0, "dry")
    assert op(m)["status"] == "failed" and op(m)["reason"] == "propellant_exhausted"
    assert op(m)["consumed_propellant_kg"] == pytest.approx(1e-5, abs=1e-11)
    assert rigid.view(m)["wheel_momentum_nms"][0] > 0.07


def test_constant_inertia_scope_and_segment_budget_fail_paid_partial(scenario):
    for field, value, reason in [
        ("max_mass_loss_fraction", 1e-7, "constant_inertia_mass_limit"),
        ("segments", 1, "unloading_segment_budget"),
    ]:
        d = copy.deepcopy(scenario)
        if field == "segments":
            d["definition"]["unloading"]["max_segments"] = value
        else:
            d["definition"]["unloading"]["actors"]["a"][field] = value
        m = runtime(d)
        unload(m)
        life.advance(m, 2.0, "limit")
        assert op(m)["status"] == "failed" and op(m)["reason"] == reason
        assert op(m)["consumed_propellant_kg"] > 0
        assert rigid.view(m)["wheel_momentum_nms"][0] > 0


def test_mid_unload_checkpoint_replays_both_actuators(scenario, tmp_path):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.45, "partial")
    p = tmp_path / "unloading.json"
    m.save(p)
    restored = MissionRuntime.load(p, access=AccessPolicy(truth=True, forecast=True))
    life.advance(m, 1.55, "finish")
    life.advance(restored, 1.55, "finish")
    assert restored.checkpoint() == m.checkpoint()
    assert restored._backend.mission.bus is restored._backend.bus


def test_attitude_and_propulsion_exclusive_ownership(scenario):
    m = runtime(scenario)
    unload(m)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        rigid.torque(m)
    assert m.checkpoint() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_torque_nm", [0.0, 0.1, 0.1]),
        ("lever_arm_m", [0.0, 0.5, 0.5]),
        ("isp_s", 0.0),
        ("propellant_pool", "unlimited"),
        ("parameter_status", "flight-qualified"),
        ("max_total_nozzle_thrust_n", float("inf")),
        ("source", " "),
    ],
)
def test_bad_rcs_configuration_rejected(scenario, field, value):
    d = copy.deepcopy(scenario)
    d["definition"]["unloading"]["actors"]["a"][field] = value
    with pytest.raises(RuntimeContractError):
        runtime(d)


def test_unloader_on_kinematic_actor_rejected(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"].pop("rigid")
    with pytest.raises(RuntimeContractError):
        runtime(d)


def test_bad_target_and_frame_rejected_atomically(scenario):
    m = runtime(scenario)
    before = m.checkpoint()
    for target in [(1.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)]:
        with pytest.raises(RuntimeContractError):
            unload(m, target=target)
    with pytest.raises(RuntimeContractError):
        life.submit(m, "momentum_unload", 1.0, "bad", momentum_unload={"frame": "GCRS"})
    assert m.checkpoint() == before


def test_actor_isolation_and_no_undeclared_rcs(scenario):
    m = runtime(scenario)
    unload(m)
    assert "b" not in m.observe("a")["actors"]
    with pytest.raises(RuntimeContractError):
        m.submit(
            [
                {
                    "command_id": "bad",
                    "actor_id": "b",
                    "kind": "momentum_unload",
                    "duration_sec": 1.0,
                    "momentum_unload": {"frame": "BODY"},
                }
            ],
            expected_revision=m.revision,
            idempotency_key="bad",
        )


def test_post_integration_failure_rolls_back_everything(scenario, monkeypatch):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.2, "boot")
    before = m.checkpoint()
    from space_env.reference.spacecraft_bus import SpacecraftBus

    original = SpacecraftBus.integrate

    def broken(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ValueError("injected after RCS integration")

    monkeypatch.setattr(SpacecraftBus, "integrate", broken)
    with pytest.raises(RuntimeContractError):
        life.advance(m, 0.2, "failure")
    assert m.checkpoint() == before
    assert m._backend.mission.bus is m._backend.bus


@pytest.mark.parametrize("initial", [[-0.08, 0.0, 0.0], [0.04, -0.05, 0.03]])
def test_signed_multi_axis_unloading_conserves_angular_momentum(scenario, initial):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"][
        "initial_wheel_momentum_nms"
    ] = initial
    m = runtime(d)
    unload(m)
    life.advance(m, 3.0, "run-unload")
    assert op(m)["status"] == "completed"
    assert op(m)["consumed_propellant_kg"] == pytest.approx(
        sum(abs(v) for v in initial) / (0.5 * 60 * G0_M_S2), abs=1e-11
    )
    np.testing.assert_allclose(
        rigid.view(m)["angular_momentum_balance_error_gcrs_nms"], [0, 0, 0], atol=1e-10
    )


def test_nonzero_target_is_physical_not_a_reset(scenario):
    m = runtime(scenario)
    unload(m, target=(0.02, 0.0, 0.0))
    life.advance(m, 3.0, "target")
    assert op(m)["status"] == "completed"
    assert rigid.view(m)["wheel_momentum_nms"][0] == pytest.approx(0.02, abs=1e-5)
    assert op(m)["consumed_propellant_kg"] == pytest.approx(
        0.06 / (0.5 * 60 * G0_M_S2), abs=1e-11
    )


def test_degradation_reduces_authority(scenario):
    m = runtime(scenario)
    rigid.inject(m, "spacecraft_degraded", solar_scale=1.0, rate_scale=0.5)
    unload(m, duration=1.0)
    life.advance(m, 1.0, "degraded")
    assert rigid.view(m)["wheel_momentum_nms"][0] == pytest.approx(0.04)
    assert op(m)["status"] == "failed"


def test_cancelled_startup_cannot_be_reused_for_free(scenario):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.1, "boot")
    m.cancel("unload", expected_revision=m.revision, idempotency_key="cancel")
    unload(m, key="second")
    life.advance(m, 0.1, "second-boot")
    assert rigid.view(m)["wheel_momentum_nms"] == [0.08, 0, 0]
    assert report(m)["operations"]["second"]["dc_energy_j"] == pytest.approx(0.3)
    assert report(m)["operations"]["second"]["startup_remaining_sec"] == pytest.approx(
        0.1
    )


def test_clearing_fault_does_not_restore_momentum_or_fuel(scenario):
    m = runtime(scenario)
    unload(m)
    life.advance(m, 0.5, "partial")
    rigid.inject(m, "spacecraft_fault", subsystem="propulsion")
    life.advance(m, 0.1, "faulted")
    before = rigid.view(m)
    used = op(m)["consumed_propellant_kg"]
    m.inject(
        [
            life.event(
                m,
                "spacecraft_fault_cleared",
                {"actor": "a", "subsystem": "propulsion"},
                id="clear",
            )
        ],
        expected_revision=m.revision,
        idempotency_key="clear",
    )
    assert rigid.view(m) == before
    assert op(m)["consumed_propellant_kg"] == used
    assert op(m)["status"] == "failed"


def test_primary_engine_and_rcs_share_global_segment_budget(scenario):
    from tests.test_actuation import engine
    from tests.test_rotating_thrust import engines

    d = copy.deepcopy(scenario)
    e = engine()
    e.update(ignition_sec=0.0, gimbal_settling_sec=0.0, max_gimbal_deg=0.0)
    d["definition"]["actuation"] = {"engines": {"b": e}}
    d["definition"]["unloading"]["actors"]["a"]["startup_sec"] = 0.0
    d["definition"]["max_maneuvers"] = 1
    m = runtime(d)
    unload(m)
    m.submit(
        [
            {
                "command_id": "main",
                "actor_id": "b",
                "kind": "body_burn",
                "duration_sec": 1.0,
                "body_burn": {
                    "frame": "BODY",
                    "direction_body": [0.0, 0.0, 1.0],
                    "thrust_n": 1.0,
                    "firing_sec": 1.0,
                },
            }
        ],
        expected_revision=m.revision,
        idempotency_key="main",
    )
    life.advance(m, 1.0, "shared-budget")
    assert sum(len(p.maneuvers) for p in m._backend.mission.ephemerides.values()) == 1
    assert op(m)["reason"] == "unloading_segment_budget"
    assert report(m)["fuel_used_kg"]["a"] == 0
    assert len(engines.report(m)["intervals"]) == 1


def test_primary_and_rcs_shared_tank_ledger(scenario):
    from tests.test_actuation import burn, engine

    d = copy.deepcopy(scenario)
    e = engine()
    e.update(
        ignition_sec=0.0,
        gimbal_settling_sec=0.0,
        max_gimbal_deg=0.0,
        rigid_coupling={"lever_arm_body_m": [0.0, 0.0, 0.0]},
    )
    d["definition"]["actuation"] = {"engines": {"a": e}}
    m = runtime(d)
    burn(m, firing=0.2, duration=0.2)
    life.advance(m, 0.2, "burn-first")
    unload(m)
    life.advance(m, 3.0, "unload-second")
    assert op(m)["status"] == "completed"
    assert abs(report(m)["fuel_balance_error_kg"]["a"]) < 1e-10
    assert (
        abs(
            m._backend.lifecycle.actuation.report(m._backend)["fuel_balance_error_kg"][
                "a"
            ]
        )
        < 1e-10
    )


def test_rcs_nominal_covariance_uses_zero_force_mass_flow(scenario):
    from tests.test_reference_covariance import settings

    d = copy.deepcopy(scenario)
    d["definition"]["orbits"]["a"]["navigation"] = settings().model_dump(mode="json")
    m = runtime(d)
    unload(m)
    life.advance(m, 0.5, "rcs-covariance")
    p = np.asarray(m.observe("a")["actors"]["a"]["orbit"]["covariance_si"])
    assert np.linalg.eigvalsh(p).min() >= -1e-9


def test_cpu_and_omni_radio_can_run_during_unload_with_common_limits(scenario):
    from tests import test_resource_lanes as lanes

    d = copy.deepcopy(lanes.scenario.__wrapped__()["definition"])
    d["spacecraft_bus"]["a"]["attitude"]["rigid"] = copy.deepcopy(
        scenario["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]
    )
    d["unloading"] = copy.deepcopy(scenario["definition"]["unloading"])
    m = runtime(spacecraft_scenario(d, id="parallel-unload", lineage="test"))
    life.acquire(m)
    unload(m, duration=3.0)
    lanes.submit(
        m,
        [
            lanes.command("transform", "cpu", job_id="compress"),
            lanes.command(
                "downlink", "radio", product_id="archive", destination_id="Ground"
            ),
        ],
        key="cpu-radio",
    )
    life.advance(m, 3.0, "concurrent")
    assert max(abs(v) for v in rigid.view(m)["wheel_momentum_nms"]) < 1e-5
    assert op(m)["reason"] == "unload_window_ended"  # Body rate has not settled yet.
    assert life.lifecycle(m)["jobs"]["compress"]["status"] == "completed"
    assert m._backend.mission.remaining["archive"] == 0
    assert abs(report(m)["energy_balance_error_j"]["unload"]) < 1e-8


def test_angular_undersampling_rejected(scenario):
    d = copy.deepcopy(scenario)
    r = d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]
    r.update(max_applicable_rate_rad_s=100.0, supply_w=100.0)
    with pytest.raises(RuntimeContractError, match="angular sampling"):
        runtime(d)


def test_initial_near_unit_quaternion_is_normalized_consistently(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["initial_quaternion_wxyz"] = [
        1 + 1e-11,
        0.0,
        0.0,
        0.0,
    ]
    m = runtime(d)
    assert rigid.view(m)["quaternion_wxyz"] == [1, 0, 0, 0]
    unload(m)
    life.advance(m, 3.0, "normalized")
    assert op(m)["status"] == "completed"


def test_rcs_wall_heat_trips_bus_without_resource_healing(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["unloading"]["actors"]["a"]["wall_heat_w_per_n"] = 1e6
    m = runtime(d)
    unload(m)
    life.advance(m, 0.5, "hot")
    assert m._backend.bus.states["a"]["safe"]
    used = op(m)["consumed_propellant_kg"]
    momentum = rigid.view(m)["wheel_momentum_nms"]
    life.advance(m, 0.5, "cooling")
    assert op(m)["status"] == "failed"
    assert op(m)["consumed_propellant_kg"] == used
    assert rigid.view(m)["wheel_momentum_nms"] == pytest.approx(momentum)
    assert (
        m._backend.bus.states["a"]["temperature_k"]
        > m._backend.bus.models["a"].thermal.initial_k
    )
