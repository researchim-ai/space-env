"""Forecast-first serial acquisition/CPU/delivery chains with paid pointing.

The search is value-only. Native execution remains the feasibility authority.
Only declared kinematic single-owner metadata chains are supported.
"""

from __future__ import annotations

import copy
import math

from pydantic import Field

from space_env.core.ground import GroundStation
from space_env.planning.mission_bundle import strict_json
from space_env.planning.mission_plan import PlanExecution, _sealed, make_mission_plan
from space_env.reference.lifecycle import LifecycleSpec
from space_env.reference.operations import VehicleResources
from space_env.reference.spacecraft_bus import SpacecraftBusModel
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.schema import FiniteScalar, ForecastBinding, StrictModel
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSpec


class ChainSettings(StrictModel):
    edge_guard_sec: float = Field(default=0.01, ge=0, le=1, strict=True)
    max_search_nodes: int = Field(default=1000000, ge=1, le=2000000, strict=True)


class ChainActor(StrictModel):
    available_wh: float = Field(ge=0, strict=True)
    base_wh: float = Field(ge=0, strict=True)
    battery_wh: float = Field(gt=0, strict=True)
    tx_remaining_sec: float | None = Field(ge=0, strict=True)


class ChainDemand(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    actor: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)
    station: str = Field(min_length=1, max_length=128)
    release_sec: float = Field(ge=0, strict=True)
    deadline_sec: float = Field(gt=0, strict=True)
    priority: int = Field(ge=0, strict=True)
    durations_sec: tuple[FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar]
    acquisition_sec: float = Field(gt=0, strict=True)
    energy_wh: float = Field(ge=0, strict=True)
    min_soc: float = Field(ge=0, lt=1, strict=True)
    memory_bits: int = Field(gt=0, strict=True)
    actions: dict[str, str]


class ChainContact(StrictModel):
    actor: str
    station: str
    binding: ForecastBinding
    windows: tuple[tuple[FiniteScalar, FiniteScalar], ...] = Field(max_length=128)


class ChainAcquisition(StrictModel):
    request: str
    binding: ForecastBinding
    windows: tuple[tuple[FiniteScalar, FiniteScalar], ...] = Field(max_length=128)


class ChainView(StrictModel):
    schema_id: str = Field(alias="schema")
    scenario_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    at_sec: float = Field(ge=0, strict=True)
    revision: int = Field(ge=0, strict=True)
    horizon_sec: float = Field(gt=0, le=21600, strict=True)
    decision_sec: float = Field(ge=1, le=30, strict=True)
    actors: dict[str, ChainActor]
    stations: dict[str, int]
    chains: tuple[ChainDemand, ...] = Field(min_length=1, max_length=64)
    completed: tuple[str, ...] = Field(max_length=64)
    contacts: tuple[ChainContact, ...] = Field(min_length=1, max_length=64)
    acquisitions: tuple[ChainAcquisition, ...] = Field(max_length=64)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def validate_view(value):
    v = ChainView.model_validate(value)
    ids = {c.id for c in v.chains}
    if (
        len(ids) != len(v.chains)
        or len(v.completed) != len(set(v.completed))
        or set(v.completed) - ids
        or len(v.contacts) != len(v.actors) * len(v.stations)
        or {(p.actor, p.station) for p in v.contacts}
        != {(a, s) for a in v.actors for s in v.stations}
        or len(v.acquisitions) != len(ids - set(v.completed))
        or {a.request for a in v.acquisitions} != ids - set(v.completed)
        or any(
            type(x) is not int or not 1 <= x <= 4 for x in value["stations"].values()
        )
    ):
        raise ValueError("incomplete chain forecast ownership/pair grid")
    if any(a.available_wh > a.battery_wh + 1e-8 for a in v.actors.values()):
        raise ValueError("chain charge exceeds declared battery capacity")
    action_ids = []
    for c in v.chains:
        if (
            c.actor not in v.actors
            or c.station not in v.stations
            or not c.release_sec < c.deadline_sec <= v.horizon_sec
            or set(c.actions) != {"slew", "imaging", "transform", "downlink"}
            or c.acquisition_sec > c.durations_sec[1]
            or any(not a for a in c.actions.values())
        ):
            raise ValueError("invalid chain request ownership/dependency fields")
        action_ids.extend(c.actions.values())
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("duplicate chain action identity")
    scenarios = set()
    trajectories = {}
    owners = {c.id: c.actor for c in v.chains}
    for pair in (*v.contacts, *v.acquisitions):
        actor = pair.actor if isinstance(pair, ChainContact) else owners[pair.request]
        b = pair.binding
        if b.actor_id != actor or b.revision != v.revision or b.geometry_revision != 0:
            raise ValueError("stale chain forecast binding")
        scenarios.add(b.scenario_sha256)
        if (
            trajectories.setdefault(actor, b.trajectory_sha256) != b.trajectory_sha256
            or len(scenarios) != 1
        ):
            raise ValueError("inconsistent chain forecast bindings")
        last = v.at_sec
        for left, right in pair.windows:
            if not last <= left < right <= v.horizon_sec:
                raise ValueError("invalid chain forecast windows")
            last = right
    return v


def profile(scenario, task):
    """Reject unsupported mechanisms rather than silently simplify them."""
    if scenario.get("backend") != "reference-spacecraft":
        raise ValueError("chain planner requires reference-spacecraft")
    d = scenario["definition"]
    t = TaskSpec.model_validate(task)
    if set(d) - {
        "schema",
        "orbits",
        "eop",
        "horizon_sec",
        "operations",
        "acquisition_targets",
        "acquisition_sensors",
        "spacecraft_bus",
        "execution",
        "lifecycle",
    }:
        raise ValueError("unsupported chain physics fields")
    life = LifecycleSpec.model_validate(d["lifecycle"])
    vehicles = {
        a: VehicleResources.model_validate(v)
        for a, v in d["operations"]["vehicles"].items()
    }
    buses = {
        a: SpacecraftBusModel.model_validate(v) for a, v in d["spacecraft_bus"].items()
    }
    stations = {
        g["name"]: GroundStation.model_validate(g) for g in d["operations"]["stations"]
    }
    h = d["horizon_sec"]
    q = t.decision_sec
    if (
        not 1 <= len(vehicles) <= 4
        or not 1 <= len(life.requests) <= 64
        or not 1 <= len(stations) <= 16
        or not 0 < h <= 21600
        or not 1 <= q <= 30
        or abs(h / q - round(h / q)) > 1e-8
        or h / q > min(10000, t.max_decisions)
        or set(buses) != set(vehicles)
        or set(d["orbits"]) != set(vehicles)
        or t.max_joint_actions < len(vehicles)
    ):
        raise ValueError("chain actor/request/grid budget exceeded")
    if (
        set(d["operations"])
        - {"schema", "vehicles", "stations", "products", "outages", "resource_step_sec"}
        or d["operations"]["products"]
        or len(stations) != len(d["operations"]["stations"])
        or any(s.share_bandwidth or s.band_hz for s in stations.values())
    ):
        raise ValueError(
            "chain profile requires no preload and ideal unreserved per-terminal radio"
        )
    if any(
        b.attitude.rigid is not None
        or b.attitude.radio_tolerance_deg != 180
        or b.attitude.sun_keep_out_deg
        or b.thermal_network is not None
        for b in buses.values()
    ):
        raise ValueError(
            "chain profile excludes rigid ADCS, radio pointing, keepout and thermal networks"
        )
    if any(v.thermal is not None for v in vehicles.values()):
        raise ValueError("chain profile excludes standalone transmitter thermal model")
    if any(
        set(o["forces"]) - {"j2"} or o.get("navigation") is not None
        for o in d["orbits"].values()
    ):
        raise ValueError(
            "chain profile requires unforced gravity-only forecast trajectories"
        )
    if (
        life.areas_of_interest
        or life.cloud_masks
        or life.allow_final_discard
        or life.transformer_manifests
        or len(life.jobs) != len(life.requests)
    ):
        raise ValueError(
            "chain profile requires one metadata transform per point request"
        )
    commands = {}
    for action in t.actions:
        if action.kind != "submit" or len(action.commands) != 1:
            raise ValueError("chain catalogue requires single-command submissions")
        c = action.commands[0]
        if c.kind not in {"slew", "imaging", "transform", "downlink"}:
            raise ValueError("unsupported chain command")
        commands[action.id] = (action, c)
    chains = []
    used = set()
    for r in life.requests:
        if (
            len(r.eligible_actors) != 1
            or r.deadline_sec > h
            or r.coverage
            or r.quality
            or r.observations
            or r.raw_ttl_sec is not None
            or r.max_attempts != 1
        ):
            raise ValueError(
                "chain profile requires one actor and point metadata request without retries/TTL/quality"
            )
        a = r.eligible_actors[0]
        if a not in vehicles or r.destination_id not in stations:
            raise ValueError("unknown chain owner/destination")
        jobs = [
            j
            for j in life.jobs
            if j.inputs == (r.raw_product_id,)
            and len(j.outputs) == 1
            and j.outputs[0].id == r.delivery_product_id
        ]
        if (
            len(jobs) != 1
            or jobs[0].transformer != "metadata-surrogate/1"
            or not jobs[0].consume_inputs
        ):
            raise ValueError("chain requires unique consuming metadata transform")
        j = jobs[0]
        found = {}
        for key, (act, c) in commands.items():
            if act.request_ref not in (None, r.id):
                continue
            match = (
                c.kind == "slew"
                and c.slew is not None
                and c.slew.target_id == r.target_id
                or c.kind == "imaging"
                and c.request_id == r.id
                or c.kind == "transform"
                and c.job_id == j.id
                or c.kind == "downlink"
                and c.product_id == r.delivery_product_id
            )
            if match and c.actor_id == a:
                if c.kind in found:
                    raise ValueError(
                        "ambiguous chain catalogue; use per-request targets"
                    )
                if act.role not in t.roles or t.roles[act.role].actors != (a,):
                    raise ValueError("chain role owner mismatch")
                if c.kind == "downlink" and c.destination_id != r.destination_id:
                    raise ValueError("chain destination mismatch")
                found[c.kind] = key
        if set(found) != {"slew", "imaging", "transform", "downlink"}:
            raise ValueError("incomplete chain catalogue")
        if used & set(found.values()):
            raise ValueError("shared chain action templates unsupported")
        used.update(found.values())
        bus = buses[a]
        v = vehicles[a]
        bandwidth = d["execution"]["actors"][a]["memory_bandwidth_bps"]
        if (
            type(bandwidth) not in (int, float)
            or not math.isfinite(bandwidth)
            or bandwidth <= 0
        ):
            raise ValueError("chain requires finite positive memory bandwidth")
        slew = (
            math.ceil(
                (180 / bus.attitude.max_rate_deg_s + bus.attitude.settling_sec + 1e-6)
                / q
            )
            * q
        )
        capture = math.ceil(r.acquisition_sec / q) * q
        cpu = (
            math.ceil(
                (
                    j.work_sec
                    + j.switching_sec
                    + (r.raw_size_bits + j.outputs[0].size_bits) / bandwidth
                    + 2 * life.execution_step_sec
                    + 1e-6
                )
                / q
            )
            * q
        )
        tx = math.ceil(j.outputs[0].size_bits / v.rate_bps / q) * q
        payload = life.payloads[a]
        if payload.footprint is not None or payload.imaging_model is not None:
            raise ValueError("chain profile requires point metadata payload")
        floor = max(
            v.min_tx_soc,
            bus.attitude.slew_min_soc,
            payload.min_soc,
            j.min_soc,
            bus.power.recovery_soc,
        )
        watts = [bus.attitude.slew_power_w, payload.power_w, j.power_w, v.tx_power_w]
        if max(watts) + v.base_power_w > bus.power.max_load_w:
            raise ValueError("chain stage exceeds declared bus load")
        chains.append(
            {
                "id": r.id,
                "actor": a,
                "target": r.target_id,
                "station": r.destination_id,
                "release_sec": r.release_sec,
                "deadline_sec": r.deadline_sec,
                "priority": r.priority,
                "durations_sec": [slew, capture, cpu, tx],
                "acquisition_sec": r.acquisition_sec,
                "energy_wh": sum(w * x for w, x in zip(watts, [slew, capture, cpu, tx]))
                / 3600
                / bus.power.discharge_efficiency,
                "min_soc": floor,
                "memory_bits": r.raw_size_bits
                + j.outputs[0].size_bits
                + j.workspace_bits,
                "actions": found,
            }
        )
    if used != set(commands):
        raise ValueError("catalogue contains actions outside declared chains")
    if any(
        sum(c["memory_bits"] for c in chains if c["actor"] == a) > v.memory_bits
        for a, v in vehicles.items()
    ):
        raise ValueError("conservative chain memory budget exceeded")
    return d, t, vehicles, buses, stations, chains


def collect_chain_forecast(
    runtime, scenario, task, *, completed=(), _runtime_scenario=None
):
    """Explicit operator bridge; no truth/report/admission or native advancement."""
    d, t, vehicles, buses, stations, chains = profile(scenario, task)
    if not isinstance(runtime, MissionRuntime) or not runtime.access.forecast:
        raise PermissionError("operator-authorized chain forecasts required")
    if canonical_json(runtime._original) != canonical_json(
        scenario if _runtime_scenario is None else _runtime_scenario
    ):
        raise ValueError("chain forecast scenario mismatch")
    now = runtime.time_sec
    if now >= runtime.horizon_sec:
        raise ValueError("chain forecast requires future coverage")
    completed = sorted(completed)
    if len(completed) != len(set(completed)) or set(completed) - {
        c["id"] for c in chains
    }:
        raise ValueError("unknown/duplicate completed chain")
    local = runtime.observe()["actors"]
    pairs = []
    acquisitions = []
    for a in sorted(vehicles):
        for s in sorted(stations):
            row = runtime.forecast(a, s)
            pairs.append(
                {
                    "actor": a,
                    "station": s,
                    "binding": row["binding"],
                    "windows": [
                        [max(now, w["start_sec"]), w["end_sec"]]
                        for w in row["windows"]
                        if w["end_sec"] > now
                    ],
                }
            )
    for c in chains:
        if c["id"] in completed:
            continue
        row = runtime.forecast_acquisition(c["actor"], c["target"])
        acquisitions.append(
            {
                "request": c["id"],
                "binding": row["binding"],
                "windows": [
                    [o["usable_start_sec"], o["usable_end_sec"]]
                    for o in row["result"]["opportunities"]
                ],
            }
        )
    actors = {}
    for a, v in vehicles.items():
        obs = local[a]
        actors[a] = {
            "available_wh": v.battery_wh * obs["soc"],
            "base_wh": (d["horizon_sec"] - now)
            * v.base_power_w
            / 3600
            / buses[a].power.discharge_efficiency,
            "battery_wh": v.battery_wh,
            "tx_remaining_sec": None
            if v.max_tx_sec is None
            else max(
                0.0, v.max_tx_sec - runtime.inventory(a)["transmitter"]["seconds"]
            ),
        }
    body = {
        "schema": "space-env.chain-forecast/1",
        "scenario_sha256": digest(scenario),
        "task_sha256": digest(t.model_dump(mode="json", by_alias=True)),
        "at_sec": now,
        "revision": runtime.revision,
        "horizon_sec": d["horizon_sec"],
        "decision_sec": t.decision_sec,
        "actors": actors,
        "stations": {s: g.terminals for s, g in stations.items()},
        "chains": chains,
        "completed": completed,
        "contacts": pairs,
        "acquisitions": acquisitions,
    }
    return body | {"sha256": digest(body)}


def schedule_chains(value, settings=None, *, order=None, placement="earliest"):
    """Value-only bounded serial resource packing; no native feasibility claim."""
    v = strict_json(canonical_json(value))
    cfg = ChainSettings.model_validate(settings or {})
    if placement not in {"earliest", "compact"}:
        raise ValueError("unknown chain placement policy")
    if not _sealed(v) or v.get("schema") != "space-env.chain-forecast/1":
        raise ValueError("invalid chain forecast seal")
    validate_view(v)
    q = v["decision_sec"]
    h = v["horizon_sec"]
    now = v["at_sec"]
    if (
        type(q) not in (int, float)
        or not 1 <= q <= 30
        or not 0 <= now < h <= 21600
        or abs(now / q - round(now / q)) > 1e-8
        or abs(h / q - round(h / q)) > 1e-8
        or h / q > 10000
        or not 1 <= len(v["chains"]) <= 64
        or not 1 <= len(v["actors"]) <= 4
        or not 1 <= len(v["stations"]) <= 16
    ):
        raise ValueError("invalid chain forecast budgets/grid")
    n = round(h / q)
    begin = round(now / q)
    contacts = {(p["actor"], p["station"]): p["windows"] for p in v["contacts"]}
    acq = {a["request"]: a["windows"] for a in v["acquisitions"]}
    if len(contacts) != len(v["actors"]) * len(v["stations"]):
        raise ValueError("incomplete chain contact grid")
    if len({c["id"] for c in v["chains"]}) != len(v["chains"]):
        raise ValueError("duplicate chain")
    actor_cells = {a: set() for a in v["actors"]}
    station_cells = {s: [0] * n for s in v["stations"]}
    energy = dict.fromkeys(v["actors"], 0.0)
    tx_used = dict.fromkeys(v["actors"], 0.0)
    steps = [
        {"at_sec": i * q, "advance_sec": q, "action_ids": []} for i in range(begin, n)
    ]
    allocations = []
    unserved = {}
    nodes = 0

    def inside(windows, start, end):
        nonlocal nodes
        for left, right in windows:
            if nodes >= cfg.max_search_nodes:
                raise ValueError("chain search budget exceeded")
            nodes += 1
            guard = cfg.edge_guard_sec
            if left + (guard if left > now else 0) <= start and end <= right - (
                guard if right < h else 0
            ):
                return True
        return False

    ordered = sorted(
        v["chains"], key=lambda c: (c["deadline_sec"], -c["priority"], c["id"])
    )
    if order is not None:
        if (
            not isinstance(order, (list, tuple))
            or len(order) != len(ordered)
            or any(not isinstance(k, str) for k in order)
            or len(set(order)) != len(order)
            or set(order) != {c["id"] for c in ordered}
        ):
            raise ValueError(
                "chain order must be a complete distinct request permutation"
            )
        by_id = {c["id"]: c for c in ordered}
        ordered = [by_id[k] for k in order]
    for c in ordered:
        if c["id"] in v["completed"]:
            continue
        a, s = c["actor"], c["station"]
        ds = c["durations_sec"]
        if (
            a not in v["actors"]
            or s not in v["stations"]
            or len(ds) != 4
            or any(
                type(x) not in (int, float)
                or x <= 0
                or abs(x / q - round(x / q)) > 1e-8
                for x in ds
            )
        ):
            raise ValueError("invalid chain ownership/duration")
        widths = [round(x / q) for x in ds]
        sl, im, cp, tx = widths
        resource = v["actors"][a]
        cost = c["energy_wh"]
        if (
            resource["available_wh"] - resource["base_wh"] - energy[a] - cost
            < resource["battery_wh"] * c["min_soc"] + 1e-10
            or resource["tx_remaining_sec"] is not None
            and tx_used[a] + ds[3] > resource["tx_remaining_sec"]
        ):
            unserved[c["id"]] = "conservative_energy_or_tx_quota"
            continue
        selected = None
        first = max(begin, math.ceil(c["release_sec"] / q))
        for start in range(first, n - sum(widths) + 1):
            if nodes >= cfg.max_search_nodes:
                raise ValueError("chain search budget exceeded")
            nodes += 1
            capture = start + sl
            cpu = capture + im
            ready = cpu + cp
            if ready + tx > n or (ready + tx) * q > c["deadline_sec"]:
                break
            if any(i in actor_cells[a] for i in range(start, ready)):
                continue
            if not inside(
                acq.get(c["id"], []), capture * q, capture * q + c["acquisition_sec"]
            ):
                continue
            for delivery in range(ready, n - tx + 1):
                if nodes >= cfg.max_search_nodes:
                    raise ValueError("chain search budget exceeded")
                nodes += 1
                if (delivery + tx) * q > c["deadline_sec"]:
                    break
                if any(i in actor_cells[a] for i in range(start, delivery + tx)):
                    break
                if any(
                    station_cells[s][i] >= v["stations"][s]
                    for i in range(delivery, delivery + tx)
                ):
                    continue
                if inside(contacts[a, s], delivery * q, (delivery + tx) * q):
                    candidate = (start, capture, cpu, delivery)
                    if selected is None or (
                        delivery + tx - start,
                        delivery + tx,
                        start,
                    ) < (selected[3] + tx - selected[0], selected[3] + tx, selected[0]):
                        selected = candidate
                    break
            if selected is not None and (
                placement == "earliest" or selected[3] + tx - selected[0] == sum(widths)
            ):
                break
        if selected is None:
            unserved[c["id"]] = "no_serial_chain_slot"
            continue
        start, capture, cpu, delivery = selected
        for i in range(start, delivery + tx):
            actor_cells[a].add(i)
        for i in range(delivery, delivery + tx):
            station_cells[s][i] += 1
        for kind, index in zip(("slew", "imaging", "transform", "downlink"), selected):
            steps[index - begin]["action_ids"].append(c["actions"][kind])
        energy[a] += cost
        tx_used[a] += ds[3]
        allocations.append(
            {
                "request": c["id"],
                "actor": a,
                "station": s,
                "start_sec": start * q,
                "capture_sec": capture * q,
                "cpu_sec": cpu * q,
                "delivery_sec": delivery * q,
                "finish_sec": (delivery + tx) * q,
                "pointing_at_sec": capture * q + c["acquisition_sec"] / 2,
            }
        )
    for s in steps:
        s["action_ids"] = sorted(s["action_ids"]) or ["wait"]
    body = {
        **({"request_order": list(order)} if order is not None else {}),
        **({"placement": placement} if placement != "earliest" else {}),
        "schema": "space-env.chain-schedule/1",
        "forecast_sha256": v["sha256"],
        "settings": cfg.model_dump(mode="json"),
        "steps": steps,
        "allocations": allocations,
        "predicted_fulfilled": len(v["completed"]) + len(allocations),
        "total": len(v["chains"]),
        "unserved": unserved,
        "search_nodes": nodes,
        "reserved_energy_wh": energy,
        "scope": "serial greedy geometric proposal; no tracking/thermal/optimality guarantee",
    }
    return body | {"sha256": digest(body)}


def materialize_task(task, view, schedule, pointings):
    result = TaskSpec.model_validate(copy.deepcopy(task)).model_dump(
        mode="json", by_alias=True
    )
    actions = {a["id"]: a for a in result["actions"]}
    chains = {c["id"]: c for c in view["chains"]}
    if set(pointings) != {a["request"] for a in schedule["allocations"]}:
        raise ValueError("incomplete future pointing grid")
    for row in schedule["allocations"]:
        c = chains[row["request"]]
        p = pointings[c["id"]]
        binding = next(
            x["binding"] for x in view["contacts"] if x["actor"] == c["actor"]
        )
        if (
            p["actor_id"] != c["actor"]
            or p["target_id"] != c["target"]
            or p["at_sec"] != row["pointing_at_sec"]
            or canonical_json(p["binding"]) != canonical_json(binding)
        ):
            raise ValueError("stale or mismatched chain pointing forecast")
        for kind, duration in zip(
            ("slew", "imaging", "transform", "downlink"), c["durations_sec"]
        ):
            command = actions[c["actions"][kind]]["commands"][0]
            command["duration_sec"] = (
                c["acquisition_sec"] if kind == "imaging" else duration
            )
            if kind == "slew":
                command["slew"] = {
                    "frame": "GCRS",
                    "quaternion_wxyz": p["goal_quaternion_wxyz"],
                }
    return TaskSpec.model_validate(result).model_dump(mode="json", by_alias=True)


def propose_chain(runtime, scenario, task, *, goals=None, settings=None):
    if runtime.time_sec != 0 or runtime.revision != 0:
        raise ValueError("initial chain proposal requires revision zero")
    view = collect_chain_forecast(runtime, scenario, task)
    schedule = schedule_chains(view, settings)
    pointings = {
        a["request"]: runtime.forecast_pointing(
            a["actor"],
            next(c["target"] for c in view["chains"] if c["id"] == a["request"]),
            a["pointing_at_sec"],
        )
        for a in schedule["allocations"]
    }
    derived = materialize_task(task, view, schedule, pointings)
    plan = make_mission_plan(scenario, derived, schedule["steps"], goals=goals)
    body = {
        "schema": "space-env.chain-proposal/1",
        "source_task": copy.deepcopy(task),
        "forecast": view,
        "schedule": schedule,
        "pointings": pointings,
        "plan": plan,
    }
    return body | {"sha256": digest(body)}


def verify_chain_proposal(value, *, native=False):
    from space_env.planning.mission_plan import validate_plan

    p = strict_json(canonical_json(value))
    if not _sealed(p) or p.get("schema") != "space-env.chain-proposal/1":
        raise ValueError("invalid chain proposal")
    plan = validate_plan(p["plan"])
    profile(plan.scenario, p["source_task"])
    schedule = schedule_chains(p["forecast"], p["schedule"]["settings"])
    if (
        p["forecast"]["scenario_sha256"] != digest(plan.scenario)
        or p["forecast"]["task_sha256"]
        != digest(
            TaskSpec.model_validate(p["source_task"]).model_dump(
                mode="json", by_alias=True
            )
        )
        or canonical_json(schedule) != canonical_json(p["schedule"])
        or canonical_json(
            materialize_task(p["source_task"], p["forecast"], schedule, p["pointings"])
        )
        != canonical_json(plan.task)
        or canonical_json(schedule["steps"])
        != canonical_json(
            [
                {k: s[k] for k in ("at_sec", "advance_sec", "action_ids")}
                for s in p["plan"]["steps"]
            ]
        )
    ):
        raise ValueError("chain proposal recomputation mismatch")
    if native:
        runtime = MissionRuntime(plan.scenario, access=AccessPolicy(forecast=True))
        fresh = propose_chain(
            runtime,
            plan.scenario,
            p["source_task"],
            goals=p["plan"]["goals"],
            settings=schedule["settings"],
        )
        if canonical_json(fresh) != canonical_json(p):
            raise ValueError("chain authorized forecast recomputation mismatch")
    return {
        "integrity_passed": True,
        "forecasts_recomputed": native,
        "native_feasibility_validated": False,
        "predicted_fulfilled": schedule["predicted_fulfilled"],
        "total": schedule["total"],
        "expert_teacher_qualified": False,
    }


def replan_chain(proposal, record, *, settings=None):
    """Explicit operator boundary. One idle-boundary replan, audited prefix replay.

    No mid-operation interrupt, injected event, new demand or scenario mutation is
    supported here. Executed commands remain byte-identical; future templates may
    change. The prefix is freshly executed under both catalogues and physically
    compared before a new proposal is returned.
    """
    verify_chain_proposal(proposal, native=True)
    old = PlanExecution.from_checkpoint(proposal["plan"], record)
    if record["error"] is not None or old.complete or old.cursor == 0:
        raise ValueError("replan requires nonempty successful unfinished prefix")
    now = old._session._runtime.time_sec
    prefix = copy.deepcopy(proposal["plan"]["steps"][: old.cursor])
    touched = {a for step in prefix for a in step["action_ids"] if a != "wait"}
    original = proposal["source_task"]
    scenario = proposal["plan"]["scenario"]
    chains = profile(scenario, original)[-1]
    local = old._session._runtime.observe()["actors"]
    statuses = {
        r["id"]: r["status"] for a in local.values() for r in a["lifecycle"]["requests"]
    }
    completed = []
    for c in chains:
        if set(c["actions"].values()) & touched:
            if statuses.get(c["id"]) != "completed":
                raise ValueError(
                    "replan requires idle completed-chain boundary; unfinished touched chain"
                )
            completed.append(c["id"])
    runtime = old._session._runtime
    # This function is an explicit operator entry point, not an agent tool.
    runtime.access = AccessPolicy(truth=True, forecast=True)
    view = collect_chain_forecast(runtime, scenario, original, completed=completed)
    schedule = schedule_chains(view, settings or proposal["schedule"]["settings"])
    pointings = {
        a["request"]: runtime.forecast_pointing(
            a["actor"],
            next(c["target"] for c in chains if c["id"] == a["request"]),
            a["pointing_at_sec"],
        )
        for a in schedule["allocations"]
    }
    derived = materialize_task(original, view, schedule, pointings)
    # Preserve every previously executed template, including its paid durations.
    old_actions = {a["id"]: a for a in proposal["plan"]["task"]["actions"]}
    for index, action in enumerate(derived["actions"]):
        if action["id"] in touched:
            derived["actions"][index] = copy.deepcopy(old_actions[action["id"]])
    steps = [
        {k: s[k] for k in ("at_sec", "advance_sec", "action_ids")} for s in prefix
    ] + schedule["steps"]
    plan = make_mission_plan(
        scenario,
        derived,
        steps,
        goals=proposal["plan"]["goals"],
        episode_seed=proposal["plan"]["episode_seed"],
    )
    fresh = PlanExecution(plan)
    fresh.run(decisions=old.cursor)
    physical = digest(record["episode"]["runtime"]["backend_report"])
    if digest(fresh.checkpoint()["episode"]["runtime"]["backend_report"]) != physical:
        raise ValueError("replanned catalogue changes physical executed prefix")
    body = {
        "schema": "space-env.chain-replan/1",
        "source_proposal": copy.deepcopy(proposal),
        "source_record": copy.deepcopy(record),
        "forecast": view,
        "schedule": schedule,
        "pointings": pointings,
        "plan": plan,
        "prefix_steps": old.cursor,
        "prefix_time_sec": now,
        "prefix_native_sha256": physical,
        "scope": "single idle completed-chain replan; unchanged scenario; prefix replay, not RAM resume",
    }
    return body | {"sha256": digest(body)}


def verify_chain_replan(value):
    p = strict_json(canonical_json(value))
    if not _sealed(p) or p.get("schema") != "space-env.chain-replan/1":
        raise ValueError("invalid chain replan seal")
    fresh = replan_chain(
        p["source_proposal"], p["source_record"], settings=p["schedule"]["settings"]
    )
    if canonical_json(fresh) != canonical_json(p):
        raise ValueError("chain replan native recomputation mismatch")
    return {
        "integrity_passed": True,
        "prefix_native_replayed": True,
        "forecasts_recomputed": True,
        "native_feasibility_validated": False,
        "prefix_steps": p["prefix_steps"],
        "expert_teacher_qualified": False,
    }
