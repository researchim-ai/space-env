"""Rotating force, off-center moment, conservation and causal runtime execution."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from space_env.core.propulsion import G0_M_S2
from space_env.reference.rigid_attitude import (
    RigidAttitudeState,
    advance_rigid_attitude,
)
from space_env.reference.thrust import RotatingThrust
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    spacecraft_scenario,
)
from tests import test_actuation as engines
from tests import test_reference_lifecycle as life
from tests import test_rigid_mission as rigid


@pytest.fixture(scope="module")
def scenario():
    d = copy.deepcopy(rigid.scenario.__wrapped__()["definition"])
    e = engines.engine()
    e.update(
        ignition_sec=0.0,
        max_gimbal_deg=0.0,
        gimbal_settling_sec=0.0,
        rigid_coupling={
            "lever_arm_body_m": [0.0, 0.0, 0.0],
            "max_mass_loss_fraction": 0.001,
        },
    )
    d["actuation"] = {"engines": {"a": e}, "max_segments": 100}
    return spacecraft_scenario(d, id="rotating-burn", lineage="B04-D2b2")


def runtime(s):
    return MissionRuntime(s, access=AccessPolicy(truth=True, forecast=True))


def advance(m, dt=1.0, key="run"):
    return life.advance(m, dt, key)


def engine_report(m):
    return m._backend.report()["lifecycle"]["actuation"]


def path_thrust(omega=1.0, duration=1.0, step=0.01):
    from tests.test_rigid_attitude import model

    cfg = model(
        inertia_kg_m2=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
        max_step_sec=step,
    )
    state = RigidAttitudeState(omega_body_rad_s=(0.0, omega, 0.0))
    _, p = advance_rigid_attitude(cfg, state, duration, return_path=True)
    return RotatingThrust(
        1.0,
        300.0,
        90.0,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        tuple(p["attitude_times_sec"]),
        tuple(p["attitude_quaternions_wxyz"]),
    )


def test_rotating_law_integrated_impulse_matches_analytic_rotation():
    law = path_thrust(omega=2.0, duration=0.7)
    expected = np.array([(1 - math.cos(1.4)) / 2, 0, math.sin(1.4) / 2])
    np.testing.assert_allclose(law.impulse_vector(), expected, atol=1e-10)
    np.testing.assert_allclose(
        law.direction_at(0.35), [math.sin(0.7), 0, math.cos(0.7)], atol=1e-10
    )
    assert law.manifest()["direction_model"] == "space-env.rotating-body-thrust/1"


def test_complete_rotation_does_not_invent_net_linear_impulse():
    law = path_thrust(omega=2.0, duration=math.pi)
    np.testing.assert_allclose(law.impulse_vector(), [0, 0, 0], atol=1e-9)


def test_centerline_rotating_burn_uses_current_body_orientation(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"][
        "initial_omega_body_rad_s"
    ] = [0.0, 1.0, 0.0]
    m = runtime(d)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m)
    r = engine_report(m)
    op = r["operations"]["burn"]
    np.testing.assert_allclose(
        op["impulse_vector_gcrs_ns"], [1 - math.cos(1), 0, math.sin(1)], atol=1e-10
    )
    assert op["status"] == "completed"
    assert rigid.view(m)["omega_body_rad_s"] == pytest.approx([0, 1, 0])
    assert r["states"]["a"]["consumed_propellant_kg"] == pytest.approx(
        1 / (300 * G0_M_S2), abs=1e-11
    )
    assert all(
        i["direction_model"] == "space-env.rotating-body-thrust/1"
        for i in r["intervals"]
    )
    assert m._backend.geometry_revision == len(r["intervals"])


def test_offcenter_thrust_drives_attitude_and_preserves_momentum_balance(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [1.0, 0.0, 0.0]
    m = runtime(d)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m)
    s = rigid.view(m)
    assert s["omega_body_rad_s"][1] == pytest.approx(-1 / 3, abs=1e-11)
    assert s["external_work_j"] == pytest.approx(1 / 6, abs=1e-11)
    assert s["wheel_momentum_nms"] == [0, 0, 0]
    assert s["dc_energy_j"] == 0.0  # No hidden wheel hold/power during burn.
    np.testing.assert_allclose(
        s["angular_momentum_balance_error_gcrs_nms"], [0, 0, 0], atol=1e-10
    )
    assert s["kinetic_work_balance_error_j"] == pytest.approx(0.0, abs=1e-10)
    assert engine_report(m)["operations"]["burn"]["impulse_vector_gcrs_ns"][0] < 0


def test_ignition_spins_without_preapplying_force_or_external_torque(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"].update(
        ignition_sec=1.0, rigid_coupling={"lever_arm_body_m": [1.0, 0.0, 0.0]}
    )
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"][
        "initial_omega_body_rad_s"
    ] = [0.0, 0.2, 0.0]
    m = runtime(d)
    before = m._backend.mission.ephemerides["a"].mass(0)
    engines.burn(m, firing=1.0, duration=2.0)
    advance(m, 0.5)
    assert m._backend.mission.ephemerides["a"].mass(0.5) == before
    assert m._backend.geometry_revision == 0
    assert rigid.view(m)["external_work_j"] == 0.0
    assert rigid.view(m)["omega_body_rad_s"] == pytest.approx([0, 0.2, 0])


def test_cancel_preserves_spent_fuel_and_spin_with_future_coast(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [1.0, 0.0, 0.0]
    m = runtime(d)
    engines.burn(m, firing=2.0, duration=2.0)
    advance(m, 0.5)
    before = rigid.view(m)
    mass = m._backend.mission.ephemerides["a"].mass(0.5)
    m.cancel("burn", expected_revision=m.revision, idempotency_key="cancel")
    advance(m, 1.0, "coast")
    after = rigid.view(m)
    assert m._backend.mission.ephemerides["a"].mass(1.5) == mass
    assert after["omega_body_rad_s"] == pytest.approx(before["omega_body_rad_s"])
    assert after["external_work_j"] == before["external_work_j"]
    assert engine_report(m)["operations"]["burn"]["status"] == "cancelled"


@pytest.mark.parametrize("subsystem", ["adcs", "propulsion", "bus"])
def test_fault_removes_force_and_torque_without_healing(scenario, subsystem):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [1.0, 0.0, 0.0]
    m = runtime(d)
    engines.burn(m, firing=2.0, duration=2.0)
    advance(m, 0.25)
    before = rigid.view(m)
    mass = m._backend.mission.ephemerides["a"].mass(0.25)
    rigid.inject(m, "spacecraft_fault", subsystem=subsystem)
    advance(m, 0.5, "faulted")
    assert rigid.view(m)["omega_body_rad_s"] == pytest.approx(
        before["omega_body_rad_s"]
    )
    assert m._backend.mission.ephemerides["a"].mass(0.75) == mass
    assert engine_report(m)["operations"]["burn"]["status"] == "failed"


def test_mid_burn_checkpoint_restores_path_state_and_future_geometry(
    scenario, tmp_path
):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [0.1, 0.2, 0.0]
    m = runtime(d)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m, 0.35)
    p = tmp_path / "rotating.json"
    m.save(p)
    restored = MissionRuntime.load(p, access=AccessPolicy(truth=True, forecast=True))
    assert restored.checkpoint() == m.checkpoint()
    advance(m, 0.65, "finish")
    advance(restored, 0.65, "finish")
    assert restored.checkpoint() == m.checkpoint()
    assert restored._backend.mission.bus is restored._backend.bus


def test_constant_inertia_applicability_stops_paid_partial_burn(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "max_mass_loss_fraction"
    ] = 1e-6
    m = runtime(d)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m)
    r = engine_report(m)
    assert r["operations"]["burn"]["status"] == "failed"
    assert r["operations"]["burn"]["reason"] == "constant_inertia_mass_limit"
    assert r["states"]["a"]["consumed_propellant_kg"] == pytest.approx(
        0.0001, abs=1e-11
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("lever_arm_body_m", [101.0, 0.0, 0.0]),
        ("lever_arm_body_m", [float("nan"), 0.0, 0.0]),
        ("max_mass_loss_fraction", 0.1),
        ("max_mass_loss_fraction", 0.0),
    ],
)
def test_bad_rigid_coupling_rejected(scenario, field, value):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][field] = value
    with pytest.raises(RuntimeContractError):
        runtime(d)


def test_rotating_config_on_kinematic_actor_rejected(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"].pop("rigid")
    with pytest.raises(RuntimeContractError):
        runtime(d)


def test_fixed_inertial_impulse_and_burn_commands_still_not_available(scenario):
    m = runtime(scenario)
    assert not {"commands.impulse", "commands.finite_burn"} & set(
        m.describe_capabilities()["supported"]
    )


def test_attitude_lane_ownership_prevents_unpaid_simultaneous_wheel_hold(scenario):
    m = runtime(scenario)
    engines.burn(m, firing=1.0, duration=1.0)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        rigid.torque(m)
    assert m.checkpoint() == before


def test_post_orbit_failure_restores_mass_geometry_and_angular_state(
    scenario, monkeypatch
):
    m = runtime(scenario)
    engines.burn(m, firing=1.0, duration=1.0)
    before = m.checkpoint()
    from space_env.reference.spacecraft_bus import SpacecraftBus

    original = SpacecraftBus.integrate

    def broken(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ValueError("injected after orbit and rigid integration")

    monkeypatch.setattr(SpacecraftBus, "integrate", broken)
    with pytest.raises(RuntimeContractError):
        advance(m, 0.2)
    assert m.checkpoint() == before
    assert m._backend.mission.bus is m._backend.bus


def test_old_state_history_preserved_after_new_rotating_arc(scenario):
    m = runtime(scenario)
    advance(m, 0.5, "coast")
    before = m._backend.mission.ephemerides["a"].state(0.25).copy()
    engines.burn(m, firing=0.5, duration=0.5)
    advance(m, 0.5, "burn-run")
    np.testing.assert_array_equal(
        m._backend.mission.ephemerides["a"].state(0.25), before
    )


@pytest.mark.parametrize("time", [float("nan"), -0.1, 2.0])
def test_rotating_path_rejects_times_outside_sealed_interval(time):
    law = path_thrust()
    with pytest.raises(ValueError):
        law.direction_at(time)


def test_rotating_path_integrity_validation():
    law = path_thrust()
    with pytest.raises(ValueError):
        replace(law, attitude_times_sec=(0.0, 0.0))
    with pytest.raises(ValueError):
        replace(law, direction_gcrs=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        replace(
            law,
            attitude_quaternions_wxyz=((2.0, 0.0, 0.0, 0.0),)
            * len(law.attitude_times_sec),
        )


def test_rotating_orbit_matches_independent_analytic_angle_rhs_and_refines(scenario):
    from scipy.integrate import solve_ivp

    from space_env.core.constants import MU_EARTH

    errors = []
    for step in (0.1, 0.01):
        d = copy.deepcopy(scenario)
        d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"].update(
            initial_omega_body_rad_s=[0.0, 1.0, 0.0], max_applicable_rate_rad_s=1.1
        )
        d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]["model"][
            "max_step_sec"
        ] = step
        d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
            "lever_arm_body_m"
        ] = [0.5, 0.0, 0.0]
        m = runtime(d)
        provider = m._backend.mission.ephemerides["a"]
        initial = provider.state(0.0).copy()
        mass = provider.mass(0.0)

        def rhs(t, y, mass=mass):
            theta = t - t * t / 12  # omega0=1, torque_y=-.5, inertia_y=3.
            r = y[:3]
            current_mass = mass - t / (300 * G0_M_S2)
            return np.r_[
                y[3:],
                -MU_EARTH * r / np.linalg.norm(r) ** 3
                + np.array([math.sin(theta), 0, math.cos(theta)]) / current_mass,
            ]

        expected = solve_ivp(
            rhs, (0, 1), initial, method="DOP853", rtol=1e-12, atol=1e-10, max_step=0.01
        ).y[:, -1]
        engines.burn(m, firing=1.0, duration=1.0)
        advance(m)
        actual = provider.state(1.0)
        error = np.linalg.norm(actual[3:] - expected[3:])
        errors.append(error)
        assert error < 2e-6
    assert errors[1] < errors[0] / 20


def test_rotating_burn_electric_power_and_heat_balance(scenario):
    d = copy.deepcopy(scenario)
    e = engines.engine("electric")
    e.update(
        ignition_sec=0.0,
        max_gimbal_deg=0.0,
        rigid_coupling={"lever_arm_body_m": [0.2, 0.0, 0.0]},
    )
    d["definition"]["actuation"]["engines"]["a"] = e
    m = runtime(d)
    engines.burn(m, firing=0.5, duration=0.5, thrust=0.01)
    advance(m, 0.5)
    r = engine_report(m)
    assert abs(r["electrical_heat_balance_error_j"]["burn"]) < 1e-9
    assert rigid.view(m)["dc_energy_j"] == 0.0
    assert r["operations"]["burn"]["dc_energy_j"] == pytest.approx(
        (0.01 * 300 * G0_M_S2 + 2) * 0.5
    )
    assert rigid.view(m)["external_work_j"] > 0


def test_rotating_fuel_exhaustion_clears_force_and_torque(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["orbits"]["a"]["propulsion"]["propellant_kg"] = 1e-5
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [1.0, 0.0, 0.0]
    m = runtime(d)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m)
    r = engine_report(m)
    assert r["operations"]["burn"]["reason"] == "propellant_exhausted"
    assert r["states"]["a"]["consumed_propellant_kg"] == pytest.approx(1e-5, abs=1e-11)
    before = rigid.view(m)
    advance(m, 0.2, "coast-after-dry")
    assert rigid.view(m)["external_work_j"] == before["external_work_j"]


def test_rotating_segment_budget_does_not_leave_ghost_torque(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["max_segments"] = 1
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [1.0, 0.0, 0.0]
    m = runtime(d)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m)
    assert engine_report(m)["operations"]["burn"]["reason"] == "burn_segment_budget"
    assert rigid.view(m)["omega_body_rad_s"][1] == pytest.approx(-0.1 / 3, abs=1e-10)
    assert len(engine_report(m)["intervals"]) == 1


def test_rotating_navigation_covariance_is_nominal_psd(scenario):
    from tests.test_reference_covariance import settings

    d = copy.deepcopy(scenario)
    d["definition"]["orbits"]["a"]["navigation"] = settings().model_dump(mode="json")
    m = runtime(d)
    engines.burn(m, firing=0.3, duration=0.3)
    advance(m, 0.3)
    p = np.asarray(m.observe("a")["actors"]["a"]["orbit"]["covariance_si"])
    assert np.linalg.eigvalsh(p).min() >= -1e-9
    assert np.isfinite(p).all()


def test_safe_mode_removes_rotating_thrust_without_resetting_spin(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [1.0, 0.0, 0.0]
    m = runtime(d)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m, 0.2)
    before = rigid.view(m)
    rigid.inject(m, "spacecraft_safe_mode")
    advance(m, 0.3, "safe")
    assert rigid.view(m)["omega_body_rad_s"] == pytest.approx(
        before["omega_body_rad_s"]
    )
    assert engine_report(m)["operations"]["burn"]["status"] == "failed"


def test_centerline_and_offcenter_rotating_impulses_are_different(scenario):
    impulses = []
    for arm in (0.0, 1.0):
        d = copy.deepcopy(scenario)
        d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
            "lever_arm_body_m"
        ] = [arm, 0.0, 0.0]
        m = runtime(d)
        engines.burn(m, firing=1.0, duration=1.0)
        advance(m)
        impulses.append(
            engine_report(m)["operations"]["burn"]["impulse_vector_gcrs_ns"]
        )
    assert abs(impulses[0][0]) < 1e-12
    assert impulses[1][0] < -0.05


def test_malformed_large_path_gap_rejected_before_orbit_integration():
    law = path_thrust()
    with pytest.raises(ValueError):
        replace(
            law,
            attitude_times_sec=(0.0, 1e-12, 1.0),
            attitude_quaternions_wxyz=(law.attitude_quaternions_wxyz[0],) * 3,
        )


def test_covariance_rejects_changed_attitude_path_even_with_same_initial_force(
    scenario,
):
    from space_env.reference.covariance import CovarianceArc
    from tests.test_reference_covariance import settings

    m = runtime(scenario)
    engines.burn(m, firing=0.2, duration=0.2)
    advance(m, 0.2)
    _, arc, _, law = next(
        seg
        for seg in m._backend.mission.ephemerides["a"].segments
        if isinstance(seg[3], RotatingThrust)
    )
    bad_path = law.attitude_quaternions_wxyz[:-1] + (
        (math.cos(0.01), math.sin(0.01), 0.0, 0.0),
    )
    bad = replace(law, attitude_quaternions_wxyz=bad_path)
    with pytest.raises(ValueError, match="covariance thrust model differs"):
        CovarianceArc(arc, settings(), thrust=bad)


def test_rotating_soc_limit_interrupts_engine_without_ghost_moment(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [1.0, 0.0, 0.0]
    m = runtime(d)
    v = m._backend.mission.vehicles["a"]
    m._backend.mission.energy_wh["a"] = 0.1 * v.battery_wh + 1 / (3600 * 0.9)
    engines.burn(m, firing=1.0, duration=1.0)
    advance(m)
    assert engine_report(m)["operations"]["burn"]["firing_sec"] == pytest.approx(
        0.5, abs=1e-8
    )
    assert rigid.view(m)["omega_body_rad_s"][1] == pytest.approx(-0.5 / 3, abs=1e-8)
    assert engine_report(m)["operations"]["burn"]["reason"] == "propulsion_power"


def test_combined_noncommuting_spin_and_torque_closes_angular_momentum(scenario):
    d = copy.deepcopy(scenario)
    c = d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]
    c.update(initial_omega_body_rad_s=[0.1, 0.2, 0.3])
    d["definition"]["actuation"]["engines"]["a"]["rigid_coupling"][
        "lever_arm_body_m"
    ] = [0.2, 0.3, 0.0]
    m = runtime(d)
    engines.burn(m, firing=0.5, duration=0.5)
    advance(m, 0.5)
    np.testing.assert_allclose(
        rigid.view(m)["angular_momentum_balance_error_gcrs_nms"], [0, 0, 0], atol=1e-10
    )
    assert abs(rigid.view(m)["kinetic_work_balance_error_j"]) < 1e-10


def test_angular_sampling_aliasing_rejected_before_any_execution(scenario):
    d = copy.deepcopy(scenario)
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"][
        "max_applicable_rate_rad_s"
    ] = 100.0
    d["definition"]["spacecraft_bus"]["a"]["attitude"]["rigid"]["supply_w"] = 100.0
    with pytest.raises(RuntimeContractError, match="angular step"):
        runtime(d)
