"""Estimated-state prediction from known inertial controls and declared dynamics.

Continuous sensor electronics are an explicit derived native base load.
"""

import copy

import numpy as np

from space_env.navigation.models import predict


def own_impulses(runtime, actor):
    return sorted(
        [
            r
            for r in runtime.command_status()
            if r["command"]["actor_id"] == actor
            and r["command"]["kind"] == "impulse"
            and r["status"] == "completed"
        ],
        key=lambda r: (r["start_sec"], r["command"]["command_id"]),
    )


def predict_controlled(
    runtime,
    actor,
    spec,
    state,
    covariance,
    start,
    end,
    applied,
    visible=None,
    *,
    environment=None,
    burns=None,
):
    """Return new estimate/P and applied IDs; no native truth state queries."""
    x = np.array(state, copy=True)
    p = np.array(covariance, copy=True)
    seen = set(applied)
    clock = start
    impulses = own_impulses(runtime, actor)
    burns = {} if burns is None else burns

    def coast(x, p, begin, finish):
        if environment is None:
            return predict(x, p, max(0.0, finish - begin), spec.acceleration_psd_m2_s3)
        return environment.propagate(
            x, p, begin, finish, spec.acceleration_psd_m2_s3, impulses, seen, burns
        )

    for row in impulses:
        cid = row["command"]["command_id"]
        epoch = row["start_sec"]
        if (
            cid in seen
            or epoch > end + 1e-10
            or (visible is not None and cid not in visible)
        ):
            continue
        if epoch < start - 1e-10:
            raise ValueError("estimator missed a past control boundary")
        x, p = coast(x, p, clock, epoch)
        x[3:] += np.array(row["command"]["impulse"]["delta_v_m_s"])
        p[3:, 3:] += np.array(spec.impulse_velocity_cov_m2_s2)
        clock = epoch
        seen.add(cid)
    x, p = coast(x, p, clock, end)
    return x, p, sorted(seen)


def sensor_power_scenario(scenario, sensors):
    """Derive an explicit continuous native base load; never mutate caller recipe."""
    from space_env.navigation.models import SensorSpec
    from space_env.reference.operations import VehicleResources

    result = copy.deepcopy(scenario)
    if not isinstance(sensors, dict) or not 1 <= len(sensors) <= 8:
        raise ValueError("one to eight actor sensors required")
    for actor, value in sensors.items():
        spec = SensorSpec.model_validate(value)
        if spec.continuous_power_w == 0:
            continue
        definition = result["definition"]
        operations = definition.get("operations", definition)
        if actor not in operations.get("vehicles", {}):
            raise ValueError("sensor power actor has no native resource model")
        row = operations["vehicles"][actor]
        baseline = VehicleResources.model_validate(row).base_power_w
        row["base_power_w"] = baseline + spec.continuous_power_w
    return result
