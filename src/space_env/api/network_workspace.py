"""A bounded native EO/CPU/ISL/Ground profile with joint assignment and routing."""

from __future__ import annotations

import copy
import math

import numpy as np
from pydantic import Field, model_validator

from space_env.api.workspace import (
    _PLAN_LOCK,
    MissionConfiguration,
    compile_configuration,
)
from space_env.core.constants import R_EARTH_EQUATOR
from space_env.core.frames import ecef_to_geodetic, geodetic_to_ecef
from space_env.core.link import earth_occults
from space_env.planning.joint_network import schedule_network
from space_env.reference import EOPSeries, Epoch, FrameContext
from space_env.reference.radio import C_M_S, RFReceiver, RFTransmitter, rf_budget
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.schema import StrictModel
from space_env.scaling.independent import circular_state
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSpec


class NetworkLink(StrictModel):
    source: str
    destination: str
    max_range_m: float = Field(default=3000000.0, ge=1000, le=10000000, strict=True)
    outages: tuple[tuple[float, float], ...] = Field(default=(), max_length=16)


class NetworkConfiguration(StrictModel):
    mission: MissionConfiguration
    links: tuple[NetworkLink, ...] = Field(default=(), max_length=12)
    max_hops: int = Field(default=3, ge=0, le=3, strict=True)
    max_search_nodes: int = Field(default=50000, ge=1, le=500000, strict=True)
    radio_rate_bps: float = Field(default=250000.0, ge=10000, le=1000000, strict=True)

    @model_validator(mode="after")
    def limits(self):
        cfg = self.mission
        actors = {
            f"{p.id}-{i:02d}" for p in cfg.planes for i in range(len(p.phases_deg))
        }
        if (
            len(cfg.requests) > 16
            or len(cfg.stations) > 4
            or cfg.horizon_sec > 1200
            or abs(cfg.decision_sec * 2 - round(cfg.decision_sec * 2)) > 1e-8
            or cfg.rate_bps < self.radio_rate_bps
        ):
            raise ValueError(
                "Network profile: <=16 requests, <=4 stations, <=1200 s, 0.5 s aligned decisions and adequate transmitter rate required"
            )
        if actors & {s.name for s in cfg.stations}:
            raise ValueError("Actor and station IDs must be distinct")
        if len({(e.source, e.destination) for e in self.links}) != len(self.links):
            raise ValueError("Duplicate directed network link")
        for edge in self.links:
            if (
                edge.source not in actors
                or edge.destination not in actors
                or edge.source == edge.destination
                or any(not 0 <= a < b <= cfg.horizon_sec for a, b in edge.outages)
            ):
                raise ValueError("Invalid network endpoint/outage")
        return self


def default_network_configuration():
    from space_env.scaling.independent import OrbitPlane

    plane = OrbitPlane(
        id="fleet", inclination_deg=0.0, raan_deg=0.0, phases_deg=(0.0, 15.0, 30.0)
    )
    position, _ = circular_state(plane, 0.0)
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    _, longitude, _ = ecef_to_geodetic(context.matrix(epoch) @ position)
    wrap = lambda x: (x + 180) % 360 - 180
    return NetworkConfiguration.model_validate(
        {
            "mission": {
                "name": "EO → два ISL-перехода → Ground ACK",
                "horizon_sec": 150.0,
                "decision_sec": 5.0,
                "planes": [plane.model_dump(mode="json")],
                "stations": [
                    {"name": "Ground", "lat_deg": 0.0, "lon_deg": wrap(longitude + 50)}
                ],
                "requests": [
                    {
                        "id": "survey",
                        "actor_id": None,
                        "station": "Ground",
                        "lat_deg": 0.0,
                        "lon_deg": wrap(longitude - 12),
                        "deadline_sec": 150.0,
                        "priority": 5,
                    }
                ],
            },
            "links": [
                {"source": "fleet-00", "destination": "fleet-01"},
                {"source": "fleet-01", "destination": "fleet-02"},
            ],
        }
    ).model_dump(mode="json", by_alias=True)


def _hardware(spec, actors):
    tx = {
        "frequency_hz": 2.2e9,
        "rf_power_w": 1.0,
        "gain_dbi": 6.0,
        "beamwidth_deg": None,
        "modes": [
            {
                "id": "declared-net",
                "information_rate_bps": spec.radio_rate_bps,
                "bits_per_symbol": 2,
                "code_rate": 0.5,
                "required_eb_n0_db": 5.0,
                "implementation_loss_db": 1.0,
            }
        ],
    }
    rx = {
        "bandwidth_hz": 2e6,
        "gt_db_k": 0.0,
        "min_frequency_hz": 2.1e9,
        "max_frequency_hz": 2.3e9,
    }
    radio = {
        "source": "declared synthetic S-band, constant gains/no BODY pattern; no hardware calibration",
        "parameter_status": "synthetic",
        "sample_sec": 0.5,
        "transmitters": {a: copy.deepcopy(tx) for a in actors},
        "receivers": {
            s.name: {**rx, "ack_processing_sec": 0.05} for s in spec.mission.stations
        },
    }
    dtn = {
        "source": "synthetic zero-loss whole-product custody/fragment timing",
        "parameter_status": "synthetic",
        "sample_sec": 0.5,
        "endpoints": {a: {"power_w": 2.0, "memory_bandwidth_bps": 1e8} for a in actors},
        "links": [
            {**e.model_dump(mode="json"), "receiver": rx, "acquisition_sec": 0.05}
            for e in spec.links
        ],
        "fragment_bits": 250000,
        "max_retries": 0,
        "ack_processing_sec": 0.01,
    }
    return radio, dtn


def _geometry_configuration(spec, resolved=None):
    config = spec.mission.model_dump(mode="json", by_alias=True)
    actors = [
        f"{p.id}-{i:02d}" for p in spec.mission.planes for i in range(len(p.phases_deg))
    ]
    selected = {a["request"]: a for a in resolved or []}
    for i, r in enumerate(config["requests"]):
        allocation = selected.get(r["id"])
        r["actor_id"] = (
            allocation["owner"]
            if allocation
            else r["actor_id"] or actors[i % len(actors)]
        )
        r["station"] = (
            allocation["station"]
            if allocation
            else r["station"] or config["stations"][0]["name"]
        )
        # Geometry scaffold only, never the executed scenario or training data.
        r["raw_bits"] = r["output_bits"] = 1
    return config


def _ground_setup_envelopes(
    spec, runtime, providers, positions, ground, receivers, samples, demands, edges
):
    """Worst prior visible pointing, plus a declared between-sample ray bound.

    This is conservative, not a sequence-optimal ground antenna controller.
    Hardware is unchanged: the production receiver already retains pointing.
    """
    q = spec.mission.decision_sec
    n = round(spec.mission.horizon_sec / q)
    first = next(iter(providers.values()))
    rotations = np.array(
        [first.context.matrix(first.epoch.shifted(t)) for t in samples]
    )
    earliest = min(
        math.ceil(r["release_sec"] / q) * q
        + (r["slew_cells"] + r["capture_cells"] + r["cpu_cells"]) * q
        for r in demands
    )
    now = runtime.time_sec
    if now > 0:
        earliest = min(earliest, now)  # residual products can already be ready
    observed = (runtime._backend.report().get("radio_network") or {}).get(
        "terminals", {}
    )
    result = {}
    for station in spec.mission.stations:
        lat, lon = np.radians([station.lat_deg, station.lon_deg])
        basis = np.array(
            [
                [-math.sin(lon), math.cos(lon), 0.0],
                [
                    -math.sin(lat) * math.cos(lon),
                    -math.sin(lat) * math.sin(lon),
                    math.cos(lat),
                ],
                [
                    math.cos(lat) * math.cos(lon),
                    math.cos(lat) * math.sin(lon),
                    math.sin(lat),
                ],
            ]
        )
        rays, eligible = {}, {}
        for actor in positions:
            edge = next(
                e
                for e in edges
                if e["source"] == actor and e["destination"] == station.name
            )
            state, site = np.asarray(positions[actor]), np.asarray(ground[station.name])
            delta = state[:, :3] - site[:, :3]
            ranges = np.linalg.norm(delta, axis=1)
            relative_speed_bound = (
                np.linalg.norm(state[:, 3:] - site[:, 3:], axis=1) + 8e-5 * ranges
            )
            # Gravity-only low-orbit applicability: 12 m/s² acceleration guard,
            # 12 km/s relative-speed bound and >=300 km ground range between
            # 0.5 s samples. Reject unsupported envelopes rather than extrapolate.
            if (
                np.min(ranges) - 12000.0 * 0.5 < 300000.0
                or np.max(relative_speed_bound) + 12.0 * 0.5 > 12000.0
            ):
                raise ValueError(
                    "Ground setup envelope outside declared range/speed applicability"
                )
            fixed = np.einsum("tij,tj->ti", rotations, delta) / ranges[:, None]
            rays[actor] = fixed @ basis.T
            flags = []
            for t in samples:
                cell = min(n - 1, int(t // q))
                flags.append(
                    bool(
                        edge["cells"][cell]
                        or (
                            cell > 0
                            and abs(t / q - round(t / q)) < 1e-8
                            and edge["cells"][cell - 1]
                        )
                    )
                )
            eligible[actor] = np.array(flags, dtype=bool)
        history = []
        for row in observed.get(station.name, []):
            if row["pointing"] is not None:
                az, el = np.radians(row["pointing"])
                history.append(
                    [
                        math.cos(el) * math.sin(az),
                        math.cos(el) * math.cos(az),
                        math.sin(el),
                    ]
                )
        rx = receivers[station.name]
        between_sample_deg = math.degrees(12000.0 / 300000.0 * 0.5)
        for actor in positions:
            delays = []
            for cell in range(n):
                start = cell * q
                if not history and start <= earliest + 1e-8:
                    delays.append(rx.acquisition_sec)
                    continue
                lo = max(0, math.ceil(max(now, earliest) / 0.5))
                hi = round(start / 0.5) + 1
                previous = [
                    ray[lo:hi][eligible[a][lo:hi]] for a, ray in rays.items() if hi > lo
                ]
                if history:
                    previous.append(np.asarray(history))
                previous = [x for x in previous if len(x)]
                if previous:
                    dots = np.concatenate(previous) @ rays[actor][round(start / 0.5)]
                    angle = min(
                        180.0,
                        math.degrees(math.acos(float(np.clip(dots.min(), -1.0, 1.0))))
                        + between_sample_deg,
                    )
                    delays.append(rx.acquisition_sec + angle / rx.slew_rate_deg_s)
                else:
                    delays.append(rx.acquisition_sec)
            result[actor, station.name] = delays
    return result


def collect_network_forecast(spec, *, runtime=None, scenario=None):
    """Known gravity-only geometry and sampled zero-fade RF; no native rollout."""
    if scenario is None:
        _, scenario, _ = compile_configuration(_geometry_configuration(spec))
    if runtime is None:
        runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
    cfg = spec.mission
    q, h = cfg.decision_sec, cfg.horizon_sec
    n = round(h / q)
    actors = list(scenario["actor_order"])
    radio, dtn = _hardware(spec, actors)
    providers = runtime._backend.mission.ephemerides
    targets = {
        (t["lat_deg"], t["lon_deg"]): t["id"]
        for t in scenario["definition"]["acquisition_targets"]
    }
    samples = [i * 0.5 for i in range(round(h / 0.5) + 1)]
    positions = {
        a: [np.asarray(providers[a].state(t)) for t in samples] for a in actors
    }
    first = providers[actors[0]]
    ground = {}
    for station in cfg.stations:
        fixed = np.concatenate(
            (
                geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m),
                np.zeros(3),
            )
        )
        ground[station.name] = [
            np.asarray(
                first.context.transform(
                    fixed, first.epoch.shifted(t), source="ITRS", destination="GCRS"
                )["state_si"]
            )
            for t in samples
        ]
    transmitters = {
        a: RFTransmitter.model_validate(v) for a, v in radio["transmitters"].items()
    }
    receivers = {s: RFReceiver.model_validate(v) for s, v in radio["receivers"].items()}
    edges = []

    def rate(left, right, tx, rx):
        delta = right[:3] - left[:3]
        distance = float(np.linalg.norm(delta))
        if distance < 1:
            return 0.0
        return rf_budget(
            tx,
            rx,
            distance,
            rate_cap_bps=cfg.rate_bps,
            doppler_hz=-tx.frequency_hz
            * float((delta / distance) @ (right[3:] - left[3:]))
            / C_M_S,
        )["rate_bps"]

    def cells(samples_ok):
        width = round(q / 0.5)
        return [all(samples_ok[i * width : (i + 1) * width + 1]) for i in range(n)]

    for link in spec.links:
        flags = []
        rx = RFReceiver.model_validate(
            next(
                e["receiver"]
                for e in dtn["links"]
                if (e["source"], e["destination"]) == (link.source, link.destination)
            )
        )
        for j, t in enumerate(samples):
            left, right = positions[link.source][j], positions[link.destination][j]
            distance = float(np.linalg.norm(right[:3] - left[:3]))
            eligible = (
                distance <= link.max_range_m
                and not earth_occults(left[:3], right[:3], R_EARTH_EQUATOR + 80000.0)
                and not any(a <= t < b for a, b in link.outages)
            )
            flags.append(
                bool(
                    eligible
                    and rate(left, right, transmitters[link.source], rx)
                    >= spec.radio_rate_bps
                )
            )
        allowed_cells = cells(flags)
        for i in range(n):
            if any(a < (i + 1) * q and b > i * q for a, b in link.outages):
                allowed_cells[i] = False
        edges.append(
            {
                "source": link.source,
                "destination": link.destination,
                "cells": allowed_cells,
            }
        )
    for actor in actors:
        for station in cfg.stations:
            windows = runtime.forecast(actor, station.name)["windows"]
            flags = [
                bool(
                    any(
                        w["start_sec"] + 1e-3 < t < w["end_sec"] - 1e-3 for w in windows
                    )
                    and rate(
                        positions[actor][j],
                        ground[station.name][j],
                        transmitters[actor],
                        receivers[station.name],
                    )
                    >= spec.radio_rate_bps
                )
                for j, t in enumerate(samples)
            ]
            allowed_cells = cells(flags)
            for i in range(n):
                if not any(
                    w["start_sec"] + 1e-3 <= i * q
                    and (i + 1) * q <= w["end_sec"] - 1e-3
                    for w in windows
                ):
                    allowed_cells[i] = False
            edges.append(
                {"source": actor, "destination": station.name, "cells": allowed_cells}
            )
    captures, demands = [], []
    buses = scenario["definition"]["spacecraft_bus"]
    for request in cfg.requests:
        owners = [request.actor_id] if request.actor_id else actors
        for actor in owners:
            opportunities = runtime.forecast_acquisition(
                actor, targets[request.lat_deg, request.lon_deg]
            )["result"]["opportunities"]
            captures.append(
                {
                    "request": request.id,
                    "actor": actor,
                    "cells": [
                        bool(
                            any(
                                o["usable_start_sec"] + 1e-3 <= i * q
                                and i * q + request.acquisition_sec
                                <= o["usable_end_sec"] - 1e-3
                                for o in opportunities
                            )
                        )
                        for i in range(n)
                    ],
                }
            )
        bus = buses[actors[0]]
        sl = math.ceil(
            (
                180 / bus["attitude"]["max_rate_deg_s"]
                + bus["attitude"]["settling_sec"]
                + 1e-6
            )
            / q
        )
        im = math.ceil(request.acquisition_sec / q)
        cp = math.ceil(
            (
                request.processing_sec
                + (request.raw_bits + request.output_bits) / 1e8
                + 1.0
                + 1e-6
            )
            / q
        )
        fragments = math.ceil(request.output_bits / 250000)
        relay = math.ceil(
            (
                request.output_bits / spec.radio_rate_bps
                + fragments * (2 * 1e7 / C_M_S + 0.06 + 4 * 0.5)
            )
            / q
        )
        ground_seconds = (
            request.output_bits / spec.radio_rate_bps + 0.05 + 2 * 1e7 / C_M_S + 2 * 0.5
        )
        ground_cells = math.ceil(ground_seconds / q)
        efficiency = bus["power"]["discharge_efficiency"]
        demands.append(
            {
                "id": request.id,
                "owners": owners,
                "stations": [request.station]
                if request.station
                else [s.name for s in cfg.stations],
                "release_sec": request.release_sec,
                "deadline_sec": request.deadline_sec,
                "priority": request.priority,
                "raw_bits": request.raw_bits,
                "output_bits": request.output_bits,
                "slew_cells": sl,
                "capture_cells": im,
                "cpu_cells": cp,
                "relay_cells": relay,
                "ground_cells": ground_cells,
                "ground_seconds": ground_seconds,
                "owner_wh": (
                    bus["attitude"]["slew_power_w"] * sl * q + 10 * im * q + 5 * cp * q
                )
                / 3600
                / efficiency,
                "relay_wh": 20 * relay * q / 3600 / efficiency,
                "receiver_wh": 2 * relay * q / 3600 / efficiency,
                "ground_wh": 20 * ground_cells * q / 3600 / efficiency,
            }
        )
    setups = _ground_setup_envelopes(
        spec, runtime, providers, positions, ground, receivers, samples, demands, edges
    )
    for edge in edges:
        if (edge["source"], edge["destination"]) in setups:
            edge["setup_sec"] = setups[edge["source"], edge["destination"]]
    body = {
        "schema": "space-env.joint-network-view/1",
        "horizon_sec": h,
        "decision_sec": q,
        "actors": {
            a: {
                "memory_bits": cfg.memory_mbit * 1e6,
                "usable_wh": max(
                    0.0,
                    cfg.battery_wh * 0.8
                    - h / 3600 / buses[a]["power"]["discharge_efficiency"],
                ),
            }
            for a in actors
        },
        "stations": {s.name: s.terminals for s in cfg.stations},
        "demands": demands,
        "edges": edges,
        "captures": captures,
        "geometry_sha256": digest(
            {"geometry_scaffold": scenario["definition"], "radio": radio, "dtn": dtn}
        ),
    }
    return body | {"sha256": digest(body)}


def create_network(value, store, *, request_order=None, planning_method="serial"):
    from space_env.api.operator import OperatorError

    spec = NetworkConfiguration.model_validate(value)
    if not _PLAN_LOCK.acquire(blocking=False):
        raise OperatorError("planner_busy", "Планировщик занят.", 429)
    try:
        view = collect_network_forecast(spec)
        base_forecast = pointing_envelope = None
        if planning_method == "pointing_bounded":
            from space_env.planning.pointing_envelope import collect_pointing_envelope

            base_forecast = view
            view, pointing_envelope = collect_pointing_envelope(spec, view)
        if planning_method == "sequence_lanes":
            from space_env.planning.sequence_lanes import (
                collect_sequence_geometry,
                schedule_sequence,
            )

            sequence_geometry = collect_sequence_geometry(spec, view)
            schedule = schedule_sequence(
                view,
                sequence_geometry,
                max_nodes=spec.max_search_nodes,
                max_hops=spec.max_hops,
                max_orders=spec.mission.max_assignment_trials,
                request_order=request_order,
            )
        elif (
            planning_method == "pointing_bounded"
            and request_order is None
            and len(view["demands"]) <= 6
            and round(view["horizon_sec"] / view["decision_sec"]) <= 48
        ):
            from space_env.planning.network_optimal import optimize_network

            schedule = optimize_network(
                view, max_nodes=spec.max_search_nodes, max_hops=spec.max_hops
            )
        elif planning_method in ("serial", "pointing_bounded"):
            schedule = schedule_network(
                view,
                max_nodes=spec.max_search_nodes,
                max_hops=spec.max_hops,
                request_order=request_order,
            )
        elif (
            planning_method in ("exact_small", "sequence_compact")
            and request_order is None
        ):
            from space_env.planning.network_optimal import optimize_network

            schedule = optimize_network(
                view, max_nodes=spec.max_search_nodes, max_hops=spec.max_hops
            )
            if planning_method == "sequence_compact":
                from space_env.planning.terminal_sequence import (
                    collect_terminal_geometry,
                    compact_terminal_schedule,
                )

                base_schedule = schedule
                terminal_geometry = collect_terminal_geometry(spec, view)
                schedule = compact_terminal_schedule(
                    view, base_schedule, terminal_geometry
                )
        else:
            raise ValueError("Unknown/incompatible planning method or request order")
        _, scenario, task = compile_configuration(
            _geometry_configuration(spec, schedule["allocations"])
        )
        cfg = spec.mission
        radio, dtn = _hardware(spec, scenario["actor_order"])
        definition = scenario["definition"]
        definition["operations"]["radio_network"] = radio
        if spec.links:
            definition["operations"]["dtn_network"] = dtn
        requests = {r.id: r for r in cfg.requests}
        for r in definition["lifecycle"]["requests"]:
            r["raw_size_bits"] = requests[r["id"]].raw_bits
        for job in definition["lifecycle"]["jobs"]:
            rid = job["id"].removeprefix("cpu-")
            job["outputs"][0]["size_bits"] = requests[rid].output_bits
        runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
        targets = {r["id"]: r["target_id"] for r in definition["lifecycle"]["requests"]}
        actions, steps = (
            [],
            [
                {
                    "at_sec": i * cfg.decision_sec,
                    "advance_sec": cfg.decision_sec,
                    "action_ids": [],
                }
                for i in range(round(cfg.horizon_sec / cfg.decision_sec))
            ],
        )
        for allocation in schedule["allocations"]:
            rid = allocation["request"]
            for i, stage in enumerate(allocation["stages"]):
                aid = f"net/{rid}/{i}/{stage['kind']}"
                command = {
                    "command_id": aid,
                    "actor_id": stage["actor"],
                    "kind": stage["kind"],
                    "duration_sec": stage["cells"] * cfg.decision_sec,
                }
                if stage["kind"] == "slew":
                    cap = next(
                        s for s in allocation["stages"] if s["kind"] == "imaging"
                    )
                    point = runtime.forecast_pointing(
                        stage["actor"],
                        targets[rid],
                        cap["cell"] * cfg.decision_sec
                        + requests[rid].acquisition_sec / 2,
                    )
                    command["slew"] = {
                        "frame": "GCRS",
                        "quaternion_wxyz": point["goal_quaternion_wxyz"],
                    }
                elif stage["kind"] == "imaging":
                    command.update(
                        request_id=rid, duration_sec=requests[rid].acquisition_sec
                    )
                elif stage["kind"] == "transform":
                    command["job_id"] = "cpu-" + rid
                else:
                    command.update(
                        product_id="final-" + rid, destination_id=stage["destination"]
                    )
                    if stage["kind"] == "relay":
                        command["relay"] = {
                            "ttl_sec": requests[rid].deadline_sec
                            - stage["cell"] * cfg.decision_sec
                        }
                actions.append(
                    {
                        "id": aid,
                        "role": stage["actor"],
                        "kind": "submit",
                        "request_ref": rid,
                        "commands": [command],
                    }
                )
                steps[stage["cell"]]["action_ids"].append(aid)
        for step in steps:
            step["action_ids"] = sorted(step["action_ids"]) or ["wait"]
        task.update(
            actions=actions,
            max_actions=len(actions) + 1,
            policy_origin="external",
            runtime_audit="compact",
        )
        task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
        meta = {
            "schema": "space-env.network-workspace/1",
            "planning_method": planning_method,
            "configuration": spec.model_dump(mode="json", by_alias=True),
            "forecast": view,
            "schedule": schedule,
            "steps": steps,
            "definition_sha256": digest(definition),
            "task_sha256": digest(task),
            "scope": "bounded zero-loss sampled-RF/kinematic metadata profile; forecast selection is not native feasibility or hardware calibration",
        }
        if planning_method == "sequence_lanes":
            meta["sequence_geometry"] = sequence_geometry
        if planning_method == "pointing_bounded":
            meta["base_forecast"] = base_forecast
            meta["pointing_envelope"] = pointing_envelope
            from space_env.planning.pointing_envelope import diagnose_network

            meta["completeness"] = diagnose_network(view, schedule)
        if planning_method == "sequence_compact":
            meta["base_schedule"] = base_schedule
            meta["terminal_geometry"] = terminal_geometry
        meta["sha256"] = digest(meta)
        scenario["metadata"] = {
            **scenario.get("metadata", {}),
            "network_workspace": meta,
        }
        return store.create({"scenario": scenario, "task": task})
    finally:
        _PLAN_LOCK.release()


def metadata(operator):
    m = operator.session._scenario.get("metadata", {}).get("network_workspace")
    if m is not None and (
        m["sha256"] != digest({k: v for k, v in m.items() if k != "sha256"})
        or m["definition_sha256"] != digest(operator.session._scenario["definition"])
        or m["task_sha256"]
        != digest(operator.session._task.model_dump(mode="json", by_alias=True))
    ):
        raise ValueError("Network plan binding mismatch")
    return m


def suggest(operator):
    from space_env.api.operator import OperatorError

    m = metadata(operator)
    if m is None:
        return None
    if not aligned(operator, m):
        raise OperatorError(
            "plan_diverged",
            "Сетевая миссия отклонилась от плана; продолжите вручную или создайте новый план.",
            409,
        )
    step = m["steps"][operator.session.revision]
    return {
        **copy.deepcopy(step),
        "expected_revision": operator.session.revision,
        "planner": "joint_owner_path_time",
    }


def aligned(operator, m):
    if len(operator.session._transitions) > len(m["steps"]):
        return False
    for i, row in enumerate(operator.session._transitions):
        actual = sorted(a for ids in row["request"]["choices"].values() for a in ids)
        dt = row["request"]["advance_sec"] or operator.session._task.decision_sec
        if (
            actual != m["steps"][i]["action_ids"]
            or abs(dt - m["steps"][i]["advance_sec"]) > 1e-8
        ):
            return False
    return True


def describe(operator):
    m = metadata(operator)
    if m is None:
        return None
    actual = operator.session._runtime._backend.report()["lifecycle"]["requests"]
    allocations = {a["request"]: a for a in m["schedule"]["allocations"]}
    optimization = None
    if m["schedule"].get("planner") == "exact_small":
        priorities = {
            r["id"]: r["priority"] for r in m["configuration"]["mission"]["requests"]
        }
        optimization = {
            "priority_lower_bound": sum(priorities[rid] for rid in allocations),
            "priority_upper_bound": m["schedule"]["upper_bound_score"]
            // (m["schedule"]["total"] + 1),
            "optimal_in_forecast_model": m["schedule"]["optimal"],
            "scope": m["schedule"]["scope"],
        }
    return {
        "name": m["configuration"]["mission"]["name"],
        "planned": m["schedule"]["predicted_fulfilled"],
        "remaining_planned": len(allocations),
        "planning_method": m.get("planning_method", "serial"),
        "optimization": optimization,
        "terminal_sequence": copy.deepcopy(m["schedule"].get("terminal_sequence")),
        "pointing_envelope": copy.deepcopy(m.get("pointing_envelope")),
        "sequence_lanes": (
            {
                "scope": m["schedule"]["scope"],
                "conditional_on_schedule": True,
                "order_trials": len(m["schedule"]["trials"]),
                "optimal": False,
            }
            if m.get("planning_method") == "sequence_lanes"
            else None
        ),
        "completeness": copy.deepcopy(m.get("completeness")),
        "fulfilled": sum(r["status"] == "completed" for r in actual.values()),
        "total": len(actual),
        "search_nodes": m["schedule"]["search_nodes"],
        "budget_exhausted": m["schedule"]["search_budget_exhausted"],
        "dynamic": copy.deepcopy(m.get("dynamic")),
        "plan_aligned": aligned(operator, m),
        "requests": [
            {
                "id": rid,
                "status": r["status"],
                "reason": r.get("reason"),
                "plan": copy.deepcopy(allocations.get(rid)),
                "plan_reason": m["schedule"]["unserved"].get(rid),
            }
            for rid, r in actual.items()
        ],
        "scope": m["scope"],
    }
