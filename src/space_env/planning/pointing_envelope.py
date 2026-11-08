"""Conservative finite-goal attitude cone for the initial kinematic network profile.

Positive quaternion cones are closed under shortest-arc interpolation. A strictly
positive pairwise dot lower bound therefore bounds angles between all states in
the cone, including interrupted slews. Non-acute cones retain the 180-degree cap.
No rigid-body dynamics, tracking controller or native future rollout is inferred.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from space_env.planning.joint_network import JointNetworkView, schedule_network
from space_env.simulation.snapshot import canonical_json, digest


def sealed(body):
    return body | {"sha256": digest(body)}


def cone_angle(quaternions):
    q = np.asarray(quaternions, dtype=float)
    if (
        q.ndim != 2
        or q.shape[1] != 4
        or not 1 <= len(q) <= 4096
        or not np.isfinite(q).all()
        or not np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9, rtol=0)
    ):
        raise ValueError("Invalid bounded unit-quaternion goal set")
    q = q / np.linalg.norm(q, axis=1)[:, None]
    q = q * np.where(q @ q[0] < 0, -1.0, 1.0)[:, None]
    minimum = 1.0
    for i in range(0, len(q), 128):
        minimum = min(minimum, float(np.min(q[i : i + 128] @ q.T)))
    # Subtract a numerical guard before acos. Do not infer a convex cone at its
    # boundary, where sign ambiguity could put the short arc outside the cone.
    minimum -= 1e-10
    if minimum <= 0:
        return {"angle_deg": 180.0, "fallback": True, "goals": len(q)}
    return {
        "angle_deg": min(180.0, math.degrees(2 * math.acos(minimum)) + 1e-6),
        "fallback": False,
        "goals": len(q),
    }


def collect_pointing_envelope(configuration, forecast):
    from space_env.api.network_workspace import (
        NetworkConfiguration,
        _geometry_configuration,
        collect_network_forecast,
    )
    from space_env.api.workspace import compile_configuration
    from space_env.runtime import AccessPolicy, MissionRuntime

    spec = NetworkConfiguration.model_validate(configuration)
    _, scenario, _ = compile_configuration(_geometry_configuration(spec))
    runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
    raw = collect_network_forecast(spec, runtime=runtime, scenario=scenario)
    if canonical_json(raw) != canonical_json(forecast):
        raise ValueError("Pointing envelope requires matching initial forecast")
    buses = scenario["definition"]["spacecraft_bus"]
    targets = {
        (t["lat_deg"], t["lon_deg"]): t["id"]
        for t in scenario["definition"]["acquisition_targets"]
    }
    requests = {r.id: r for r in spec.mission.requests}
    q = spec.mission.decision_sec
    actors = {}
    for actor, bus in buses.items():
        poses = [bus["attitude"]["initial_quaternion_wxyz"]]
        for capture in raw["captures"]:
            if capture["actor"] != actor:
                continue
            request = requests[capture["request"]]
            for i, allowed in enumerate(capture["cells"]):
                if allowed:
                    poses.append(
                        runtime.forecast_pointing(
                            actor,
                            targets[request.lat_deg, request.lon_deg],
                            i * q + request.acquisition_sec / 2,
                        )["goal_quaternion_wxyz"]
                    )
        bound = cone_angle(poses)
        # Imaging itself tracks the target. Its final pose need not equal the
        # midpoint goal. A successful preceding slew resets the pose error;
        # tracking can then move at most rate * reserved capture time.
        tracking_cells = max(
            (
                r["capture_cells"]
                for r in raw["demands"]
                if actor in r["owners"]
                and any(
                    c["request"] == r["id"] and c["actor"] == actor and any(c["cells"])
                    for c in raw["captures"]
                )
            ),
            default=0,
        )
        bound["cone_angle_deg"] = bound["angle_deg"]
        bound["tracking_guard_deg"] = min(
            180.0, bus["attitude"]["max_rate_deg_s"] * tracking_cells * q
        )
        bound["angle_deg"] = min(
            180.0, bound["angle_deg"] + bound["tracking_guard_deg"]
        )

        bound["slew_cells"] = max(
            1,
            math.ceil(
                (
                    bound["angle_deg"] / bus["attitude"]["max_rate_deg_s"]
                    + bus["attitude"]["settling_sec"]
                    + 1e-6
                )
                / q
            ),
        )
        actors[actor] = bound
    adapted = copy.deepcopy(raw)
    for demand in adapted["demands"]:
        demand["slew_cells"] = min(
            demand["slew_cells"], max(actors[a]["slew_cells"] for a in demand["owners"])
        )
        # Keep the old energy reserve: a shorter admitted time is NOT free energy.
    adapted = sealed({k: v for k, v in adapted.items() if k != "sha256"})
    JointNetworkView.model_validate(adapted)
    proof = sealed(
        {
            "schema": "space-env.pointing-envelope/1",
            "base_forecast_sha256": raw["sha256"],
            "forecast_sha256": adapted["sha256"],
            "actors": actors,
            "conditional_on_successful_preceding_slews": True,
            "scope": "finite declared GCRS slew goals at allowed capture-grid midpoints plus initial pose; positive quaternion cone plus capture tracking rate-time guard; conditional on successful planned slews; original energy reserve retained; no native optimality or rigid/flight qualification",
        }
    )
    return adapted, proof


def diagnose_network(forecast, schedule):
    """Explain catalogue coverage without claiming physical infeasibility.

    Isolated schedules are witnesses only. Failure of a bounded serial search is
    unknown; it is not an infeasibility proof. Prefix capacities are necessary
    bounds only for this conservative discrete model.
    """
    v = JointNetworkView.model_validate(forecast)
    if (
        forecast["sha256"]
        != digest({k: x for k, x in forecast.items() if k != "sha256"})
        or schedule.get("forecast_sha256") != forecast["sha256"]
        or schedule.get("sha256")
        != digest({k: x for k, x in schedule.items() if k != "sha256"})
    ):
        raise ValueError("Diagnostic forecast/schedule binding mismatch")
    if v.at_sec or v.recovery:
        raise ValueError("Diagnostics require a fresh initial forecast")
    planned = {a["request"] for a in schedule["allocations"]}
    known = {r.id for r in v.demands}
    if (
        not planned <= known
        or len(planned) != len(schedule["allocations"])
        or schedule.get("total") != len(known)
        or schedule.get("predicted_fulfilled") != len(planned)
    ):
        raise ValueError("Invalid diagnostic catalogue coverage")
    rows = []
    eligible = {}
    for r in v.demands:
        owners = [
            a
            for a in r.owners
            if any(
                c.request == r.id and c.actor == a and any(c.cells) for c in v.captures
            )
        ]
        eligible[r.id] = owners
        singleton = copy.deepcopy(forecast)
        singleton["demands"] = [d for d in singleton["demands"] if d["id"] == r.id]
        singleton["captures"] = [
            c for c in singleton["captures"] if c["request"] == r.id
        ]
        singleton = sealed({k: x for k, x in singleton.items() if k != "sha256"})
        alone = schedule_network(
            singleton,
            max_nodes=schedule["max_search_nodes"],
            max_hops=schedule["max_hops"],
        )
        peak = r.raw_bits + r.output_bits + 100000
        memory_impossible = all(peak > v.actors[a].memory_bits for a in r.owners)
        rows.append(
            {
                "request": r.id,
                "catalogue_planned": r.id in planned,
                "capture_owners": owners,
                "slew_sec": r.slew_cells * v.decision_sec,
                "prefix_sec": (r.slew_cells + r.capture_cells + r.cpu_cells)
                * v.decision_sec,
                "isolated_witness": bool(alone["allocations"]),
                "isolated_search_exhausted": alone["search_budget_exhausted"],
                "memory_prefix_exceeds_every_owner": memory_impossible,
                "classification": "planned"
                if r.id in planned
                else "no_sampled_capture"
                if not owners
                else "memory_prefix_bound"
                if memory_impossible
                else "shared_resource_or_greedy_choice"
                if alone["allocations"]
                else "unresolved_in_forecast_model",
            }
        )
    bounds = []
    for actor in v.actors:
        mandatory = [r for r in v.demands if eligible[r.id] == [actor]]
        cells = sum(r.slew_cells + r.capture_cells + r.cpu_cells for r in mandatory)
        horizon_cells = round(v.horizon_sec / v.decision_sec)
        if cells > horizon_cells:
            bounds.append(
                {
                    "actor": actor,
                    "requests": [r.id for r in mandatory],
                    "required_cells": cells,
                    "available_cells": horizon_cells,
                    "all_requests_fit": False,
                    "scope": "necessary contiguous EO/CPU occupancy in declared forecast, not physical mission impossibility",
                }
            )
    return sealed(
        {
            "schema": "space-env.network-completeness/1",
            "forecast_sha256": forecast["sha256"],
            "schedule_sha256": schedule["sha256"],
            "requests": rows,
            "prefix_bounds": bounds,
            "catalogue_requests": len(planned),
            "total_requests": len(rows),
            "all_requests_catalogued": len(planned) == len(rows),
            "physical_infeasibility_established": False,
        }
    )
