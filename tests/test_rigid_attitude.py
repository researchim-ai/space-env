import math

import numpy as np
import pytest

from space_env.reference.rigid_attitude import (
    RigidAttitudeModel,
    RigidAttitudeState,
    advance_rigid_attitude,
    invariants,
    thrust_torque_nm,
)


def model(**updates):
    data = {
        "model_id": "synthetic-rigid/1",
        "source": "analytic engineering assumptions",
        "inertia_kg_m2": ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        "wheel_inertia_kg_m2": (0.01, 0.01, 0.01),
        "max_torque_nm": (0.1, 0.1, 0.1),
        "max_momentum_nms": (0.2, 0.2, 0.2),
    }
    return RigidAttitudeModel(**(data | updates))


def test_free_spherical_rotation():
    m = model(inertia_kg_m2=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)))
    s = RigidAttitudeState(omega_body_rad_s=(0.0, 0.0, 0.4))
    end, _ = advance_rigid_attitude(m, s, 2.0)
    np.testing.assert_allclose(
        end.quaternion_wxyz, (math.cos(0.4), 0, 0, math.sin(0.4)), atol=1e-12
    )


def test_asymmetric_free_body_conservation():
    m, s = (
        model(),
        RigidAttitudeState(
            omega_body_rad_s=(0.1, 0.2, 0.3), wheel_momentum_nms=(0.02, -0.01, 0.03)
        ),
    )
    end, _ = advance_rigid_attitude(m, s, 4.0)
    np.testing.assert_allclose(
        invariants(m, end)["momentum_inertial_nms"],
        invariants(m, s)["momentum_inertial_nms"],
        atol=1e-11,
    )
    assert invariants(m, end)["kinetic_energy_j"] == pytest.approx(
        invariants(m, s)["kinetic_energy_j"], abs=1e-12
    )


def test_motor_reaction_and_energy_balance():
    m, s = model(), RigidAttitudeState()
    end, _ = advance_rigid_attitude(m, s, 1.0, motor_torque_nm=(0.05, 0, 0))
    assert end.omega_body_rad_s[0] == pytest.approx(-0.025)
    assert end.wheel_momentum_nms[0] == pytest.approx(0.05)
    np.testing.assert_allclose(
        invariants(m, end)["momentum_inertial_nms"], [0, 0, 0], atol=1e-12
    )
    assert end.dc_energy_j - end.heat_j == pytest.approx(end.motor_work_j, abs=1e-12)
    assert end.motor_work_j == pytest.approx(
        invariants(m, end)["kinetic_energy_j"], abs=1e-12
    )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_capacity_and_reversal(sign):
    m, s = model(), RigidAttitudeState()
    end, report = advance_rigid_attitude(m, s, 4.0, motor_torque_nm=(sign * 2, 0, 0))
    assert end.wheel_momentum_nms[0] == pytest.approx(sign * 0.2)
    assert report["capacity_limited_sec"][0] > 1.9
    reverse, _ = advance_rigid_attitude(
        m, end, 1.0, motor_torque_nm=(-sign * 0.1, 0, 0)
    )
    assert reverse.wheel_momentum_nms[0] == pytest.approx(sign * 0.1)


def test_off_center_external_torque():
    torque = thrust_torque_nm((1, 0, 0), (0, 2, 0))
    np.testing.assert_equal(torque, [0, 0, 2])
    m, s = model(), RigidAttitudeState()
    end, _ = advance_rigid_attitude(m, s, 1.0, external_torque_body_nm=torque)
    assert end.omega_body_rad_s[2] == pytest.approx(0.5)
    assert end.external_work_j == pytest.approx(invariants(m, end)["kinetic_energy_j"])


def test_no_free_momentum_dump_and_no_regeneration():
    m, s = model(), RigidAttitudeState(wheel_momentum_nms=(0.1, 0, 0))
    idle, _ = advance_rigid_attitude(m, s, 1.0)
    assert idle.wheel_momentum_nms == s.wheel_momentum_nms
    end, _ = advance_rigid_attitude(m, s, 0.5, motor_torque_nm=(-0.1, 0, 0))
    assert end.motor_work_j < 0
    assert end.dc_energy_j > 0 and end.heat_j > end.dc_energy_j


def test_serialized_continuation_and_pure_inputs():
    m, s = model(), RigidAttitudeState(omega_body_rad_s=(0.1, 0.2, 0.3))
    original = s.model_dump()
    whole, _ = advance_rigid_attitude(m, s, 1.0, motor_torque_nm=(0.02, 0, 0))
    half, _ = advance_rigid_attitude(m, s, 0.5, motor_torque_nm=(0.02, 0, 0))
    restored = RigidAttitudeState.model_validate_json(half.model_dump_json())
    end, _ = advance_rigid_attitude(m, restored, 0.5, motor_torque_nm=(0.02, 0, 0))
    np.testing.assert_allclose(end.quaternion_wxyz, whole.quaternion_wxyz, atol=1e-12)
    assert s.model_dump() == original


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf"), 10001.0])
def test_bad_duration(dt):
    with pytest.raises(ValueError):
        advance_rigid_attitude(model(), RigidAttitudeState(), dt)


@pytest.mark.parametrize(
    "updates",
    [
        {"inertia_kg_m2": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 4.0))},
        {"inertia_kg_m2": ((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
        {"inertia_kg_m2": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
        {"max_torque_nm": (0.0, 0.1, 0.1)},
        {"wheel_inertia_kg_m2": (-1.0, 0.1, 0.1)},
        {"source": " "},
        {"max_momentum_nms": (float("nan"), 1.0, 1.0)},
    ],
)
def test_bad_models(updates):
    with pytest.raises(ValueError):
        model(**updates)


def test_invalid_state_and_vectors():
    with pytest.raises(ValueError):
        RigidAttitudeState(quaternion_wxyz=(2.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        advance_rigid_attitude(
            model(), RigidAttitudeState(wheel_momentum_nms=(1.0, 0.0, 0.0)), 1.0
        )
    with pytest.raises(ValueError):
        thrust_torque_nm((1, 2), (1, 2, 3))
    with pytest.raises(ValueError):
        advance_rigid_attitude(
            model(), RigidAttitudeState(), 1.0, motor_torque_nm=(float("nan"), 0, 0)
        )
