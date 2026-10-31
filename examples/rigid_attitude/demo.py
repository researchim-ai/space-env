"""Standalone rigid-body experiment; intentionally not a mission bus option."""

from __future__ import annotations

import json

import numpy as np

from space_env.reference.rigid_attitude import (
    RigidAttitudeModel,
    RigidAttitudeState,
    advance_rigid_attitude,
    invariants,
)


def main():
    model = RigidAttitudeModel(
        model_id="synthetic-rigid/1",
        source="declared analytic demonstration",
        inertia_kg_m2=((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        wheel_inertia_kg_m2=(0.01, 0.01, 0.01),
        max_torque_nm=(0.1, 0.1, 0.1),
        max_momentum_nms=(0.2, 0.2, 0.2),
    )
    initial = RigidAttitudeState()
    spinning, report = advance_rigid_attitude(
        model, initial, 4.0, motor_torque_nm=(0.1, 0, 0)
    )
    assert abs(spinning.wheel_momentum_nms[0] - 0.2) < 1e-12
    np.testing.assert_allclose(
        invariants(model, spinning)["momentum_inertial_nms"], [0, 0, 0], atol=1e-12
    )
    restored = RigidAttitudeState.model_validate_json(spinning.model_dump_json())
    # Explicit externally supplied torque, not a free wheel momentum reset.
    unloaded, _ = advance_rigid_attitude(
        model,
        restored,
        2.0,
        motor_torque_nm=(-0.1, 0, 0),
        external_torque_body_nm=(-0.1, 0, 0),
    )
    assert abs(unloaded.wheel_momentum_nms[0]) < 1e-12
    balance = unloaded.dc_energy_j - unloaded.heat_j - unloaded.motor_work_j
    assert abs(balance) < 1e-10
    print(
        json.dumps(
            {
                "ok": True,
                "scope": "standalone kernel, not integrated mission ADCS",
                "saturation": report,
                "saturated_state": spinning.model_dump(mode="json"),
                "unloaded_state": unloaded.model_dump(mode="json"),
                "unloaded_invariants": invariants(model, unloaded),
                "electrical_balance_error_j": balance,
            }
        )
    )


if __name__ == "__main__":
    main()
