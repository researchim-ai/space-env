"""Sequence-aware, fixed-start compaction of an admitted exact-small plan.

Only single-terminal stations are compacted. Multiple terminals keep their
conservative widths: the native executor assigns physical terminals itself.
This pass never adds requests, changes starts, or proves a new global optimum.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from space_env.planning.network_optimal import _resources, verify_exact_schedule
from space_env.simulation.snapshot import digest


def sealed(body):
    return body | {"sha256": digest(body)}


def collect_terminal_geometry(configuration, forecast):
    """Read known ephemerides only; no command submission or future rollout."""
    from space_env.api.network_workspace import (
        NetworkConfiguration,
        _geometry_configuration,
        _hardware,
        collect_network_forecast,
    )
    from space_env.api.workspace import compile_configuration
    from space_env.reference.radio import RFReceiver
    from space_env.runtime import AccessPolicy, MissionRuntime

    spec = NetworkConfiguration.model_validate(configuration)
    _, scenario, _ = compile_configuration(_geometry_configuration(spec))
    runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
    # Recollect to bind geometry to the complete inputs and validity envelope,
    # rather than accepting an arbitrary caller-supplied geometry hash.
    if collect_network_forecast(spec, runtime=runtime, scenario=scenario) != forecast:
        raise ValueError("Terminal geometry/configuration forecast mismatch")
    mission = runtime._backend.mission
    radio, _ = _hardware(spec, scenario["actor_order"])
    step = 0.5
    times = [i * step for i in range(round(spec.mission.horizon_sec / step) + 1)]
    stations = {}
    for name, receiver in radio["receivers"].items():
        rx = RFReceiver.model_validate(receiver)
        directions = {}
        for actor, provider in mission.ephemerides.items():
            rays = []
            for t in times:
                look = provider.look(mission.stations[name], t)
                az, el = np.radians([look["azimuth_deg"], look["elevation_deg"]])
                rays.append(
                    [
                        math.cos(el) * math.cos(az),
                        math.cos(el) * math.sin(az),
                        math.sin(el),
                    ]
                )
            directions[actor] = rays
        stations[name] = {
            "acquisition_sec": rx.acquisition_sec,
            "slew_rate_deg_s": rx.slew_rate_deg_s,
            "directions": directions,
        }
    return sealed(
        {
            "schema": "space-env.terminal-geometry/1",
            "forecast_sha256": forecast["sha256"],
            "sample_sec": step,
            "angular_guard_deg": math.degrees(12000.0 / 300000.0 * step) * 2,
            "stations": stations,
            "scope": "known gravity-only LEO geometry; same range/speed applicability as conservative envelope; two-sample angular guard; no native future",
        }
    )


def _admit_geometry(view, geometry):
    if (
        geometry.get("sha256")
        != digest({k: v for k, v in geometry.items() if k != "sha256"})
        or geometry.get("schema") != "space-env.terminal-geometry/1"
        or geometry.get("forecast_sha256") != view["sha256"]
    ):
        raise ValueError("Invalid terminal geometry binding")
    if (
        geometry.get("sample_sec") != 0.5
        or geometry.get("angular_guard_deg")
        != math.degrees(12000.0 / 300000.0 * 0.5) * 2
        or set(geometry["stations"]) != set(view["stations"])
    ):
        raise ValueError("Invalid terminal geometry envelope")
    if abs(
        view["decision_sec"] / 0.5 - round(view["decision_sec"] / 0.5)
    ) > 1e-8 or any(
        not edge.get("setup_sec")
        for edge in view["edges"]
        if edge["destination"] in view["stations"]
    ):
        raise ValueError(
            "Terminal compaction requires aligned samples and physical setup envelopes"
        )
    count = round(view["horizon_sec"] / 0.5) + 1
    for row in geometry["stations"].values():
        if not all(
            type(row[k]) in (int, float)
            and math.isfinite(row[k])
            and (row[k] >= 0 if k == "acquisition_sec" else row[k] > 0)
            for k in ("acquisition_sec", "slew_rate_deg_s")
        ) or set(row["directions"]) != set(view["actors"]):
            raise ValueError("Invalid receiver parameters")
        for rays in row["directions"].values():
            rays = np.asarray(rays, dtype=float)
            if (
                rays.shape != (count, 3)
                or not np.isfinite(rays).all()
                or not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-10, rtol=0)
            ):
                raise ValueError("Invalid terminal direction samples")


def compact_terminal_schedule(view, base, geometry):
    verify_exact_schedule(view, base)
    _admit_geometry(view, geometry)
    if base.get("planner") != "exact_small":
        raise ValueError("Compaction requires an exact-small base plan")
    effective = copy.deepcopy(view)
    plan = copy.deepcopy(base)
    q, step = view["decision_sec"], geometry["sample_sec"]
    demands = {r["id"]: r for r in effective["demands"]}
    edges = {(e["source"], e["destination"]): e for e in effective["edges"]}
    witnesses = []
    for station, capacity in view["stations"].items():
        previous = None
        for allocation in sorted(
            (a for a in plan["allocations"] if a["station"] == station),
            key=lambda a: (a["stages"][-1]["cell"], a["request"]),
        ):
            ground = allocation["stages"][-1]
            old_width = ground["cells"]
            start = ground["cell"] * q
            row = geometry["stations"][station]
            actor = allocation["path"][-1]
            edge = edges[actor, station]
            predecessor = None
            setup = edge.get("setup_sec", [0.0] * round(view["horizon_sec"] / q))[
                ground["cell"]
            ]
            if capacity == 1:
                setup = row["acquisition_sec"]
                if previous is not None:
                    prior_actor, lo, hi, predecessor = previous
                    prior = np.asarray(
                        row["directions"][prior_actor][
                            round(lo / step) : round(hi / step) + 1
                        ]
                    )
                    current = np.asarray(
                        row["directions"][actor][
                            round(start / step) : round(start / step) + 2
                        ]
                    )
                    angle = min(
                        180.0,
                        math.degrees(
                            math.acos(float(np.clip((prior @ current.T).min(), -1, 1)))
                        )
                        + geometry["angular_guard_deg"],
                    )
                    setup += angle / row["slew_rate_deg_s"]
                request = demands[allocation["request"]]
                seconds = request.get("ground_seconds") or request["ground_cells"] * q
                # The old physical envelope remains a safe fallback and bounds
                # the pass to shrink-only, including finite sampling margins.
                setup = min(
                    setup,
                    edge.get("setup_sec", [0.0] * round(view["horizon_sec"] / q))[
                        ground["cell"]
                    ],
                )
                edge.setdefault("setup_sec", [0.0] * round(view["horizon_sec"] / q))[
                    ground["cell"]
                ] = setup
                ground["cells"] = max(
                    request["ground_cells"], math.ceil((seconds + setup) / q)
                )
                if ground["cells"] > old_width:
                    raise ValueError("Terminal compaction would expand a reservation")
                allocation["finish_sec"] = start + ground["cells"] * q
                _, memory, energy = _resources(effective, request, allocation)
                allocation.update(memory_bits=memory, energy_wh=energy)
                previous = (
                    actor,
                    start,
                    allocation["finish_sec"],
                    allocation["request"],
                )
            witnesses.append(
                {
                    "request": allocation["request"],
                    "station": station,
                    "terminal": 0 if capacity == 1 else None,
                    "predecessor": predecessor,
                    "start_sec": start,
                    "setup_bound_sec": setup,
                    "before_cells": old_width,
                    "after_cells": ground["cells"],
                    "mode": "sequence_envelope"
                    if capacity == 1
                    else "conservative_unpinned",
                }
            )
    effective = sealed({k: v for k, v in effective.items() if k != "sha256"})
    # Bounds from the narrower base model are not valid for a relaxed model.
    # Keep only the trivial all-demand bound and never claim sequence optimality.
    upper = sum(r["priority"] * (len(demands) + 1) + 1 for r in demands.values())
    plan.update(
        planner="sequence_compact",
        forecast_sha256=effective["sha256"],
        upper_bound_score=upper,
        optimal=base["objective_score"] == upper,
        scope="fixed starts/selected chains; single-terminal sequence enclosure; conservative multi-terminal fallback; no global sequence optimum",
    )
    admitted = sealed({k: v for k, v in plan.items() if k != "sha256"})
    verify_exact_schedule(effective, admitted)
    return sealed(
        {
            **{k: v for k, v in admitted.items() if k != "sha256"},
            "forecast_sha256": view["sha256"],
            "base_schedule_sha256": base["sha256"],
            "terminal_geometry_sha256": geometry["sha256"],
            "effective_forecast": effective,
            "resource_witness": admitted,
            "terminal_sequence": {
                "rows": witnesses,
                "saved_reservation_sec": sum(
                    (w["before_cells"] - w["after_cells"]) * q for w in witnesses
                ),
                "global_sequence_optimality_established": False,
            },
        }
    )


def verify_terminal_schedule(view, base, geometry, schedule):
    if compact_terminal_schedule(view, base, geometry) != schedule:
        raise ValueError("Terminal sequence reproduction mismatch")
    return True
