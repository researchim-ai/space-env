"""Bounded append-only attitude sequencing with native resource-lane claims.

Fresh kinematic network proposals only. CPU may overlap a later slew; relay stays
exclusive. One command start/role/tick preserves the scalar LLM wire protocol.
No global optimality, in-place memory reuse or native-future rollout is claimed.
"""

from __future__ import annotations

import copy
import itertools
import math

from space_env.planning.joint_network import JointNetworkView, ground_width
from space_env.reference.spacecraft_bus import quaternion_angle
from space_env.simulation.snapshot import canonical_json, digest

KINDS = ("slew", "imaging", "transform", "relay", "downlink")


def seal(value):
    return value | {"sha256": digest(value)}


def collect_sequence_geometry(configuration, forecast):
    from space_env.api.network_workspace import (
        NetworkConfiguration,
        _geometry_configuration,
        _hardware,
        collect_network_forecast,
    )
    from space_env.api.workspace import compile_configuration
    from space_env.runtime import AccessPolicy, MissionRuntime

    spec = NetworkConfiguration.model_validate(configuration)
    _, scenario, _ = compile_configuration(_geometry_configuration(spec))
    runtime = MissionRuntime(scenario, access=AccessPolicy(forecast=True))
    if canonical_json(
        collect_network_forecast(spec, runtime=runtime, scenario=scenario)
    ) != canonical_json(forecast):
        raise ValueError("Sequence geometry/initial forecast mismatch")
    backend = runtime._backend
    targets = {
        (t["lat_deg"], t["lon_deg"]): t["id"]
        for t in scenario["definition"]["acquisition_targets"]
    }
    _, dtn = _hardware(spec, scenario["actor_order"])
    actors = {}
    for actor, bus in scenario["definition"]["spacecraft_bus"].items():
        claims = {}
        for kind in KINDS:
            command = {
                "command_id": "claim-probe",
                "actor_id": actor,
                "kind": kind,
                "duration_sec": 1.0,
            }
            if kind == "transform":
                command["job_id"] = "claim-probe"
            if kind == "slew":
                command["slew"] = {
                    "frame": "GCRS",
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            if kind == "relay":
                command.update(
                    destination_id="probe", product_id="probe", relay={"ttl_sec": 1.0}
                )
            claims[kind] = runtime.command_claims(command)["resources"]
        if (
            claims["transform"] != ["cpu"]
            or set(claims["imaging"]) != {"payload", "attitude"}
            or claims["slew"] != ["attitude"]
        ):
            raise ValueError("Unsupported native lane profile")
        vehicle = backend.mission.vehicles[actor]
        actors[actor] = {
            "initial_quaternion": bus["attitude"]["initial_quaternion_wxyz"],
            "rate_deg_s": bus["attitude"]["max_rate_deg_s"],
            "settling_sec": bus["attitude"]["settling_sec"],
            "claims": claims,
            "bandwidth_bps": scenario["definition"]["execution"]["actors"][actor][
                "memory_bandwidth_bps"
            ],
            "base_w": vehicle.base_power_w,
            "max_load_w": bus["power"]["max_load_w"],
            "slew_w": bus["attitude"]["slew_power_w"],
            "imaging_w": backend.lifecycle.spec.payloads[actor].power_w,
            "radio_w": vehicle.tx_power_w,
            "receive_w": dtn["endpoints"][actor]["power_w"],
        }
    requests = {r.id: r for r in spec.mission.requests}
    goals = {r.id: {} for r in spec.mission.requests}
    n = round(spec.mission.horizon_sec / spec.mission.decision_sec)
    for capture in forecast["captures"]:
        request = requests[capture["request"]]
        goals[request.id][capture["actor"]] = [
            runtime.forecast_pointing(
                capture["actor"],
                targets[request.lat_deg, request.lon_deg],
                i * spec.mission.decision_sec + request.acquisition_sec / 2,
            )["goal_quaternion_wxyz"]
            if valid
            else None
            for i, valid in enumerate(capture["cells"])
        ]
    jobs = {j["id"]: j for j in scenario["definition"]["lifecycle"]["jobs"]}
    demand_io = {
        r.id: {
            "imaging_bps": r.raw_bits / r.acquisition_sec,
            "cpu_bps": (r.raw_bits + r.output_bits) / r.processing_sec,
            "cpu_w": jobs["cpu-" + r.id]["power_w"],
            "radio_bps": spec.mission.rate_bps,
        }
        for r in spec.mission.requests
    }
    return seal(
        {
            "schema": "space-env.sequence-lane-geometry/1",
            "forecast_sha256": forecast["sha256"],
            "cells": n,
            "actors": actors,
            "goals": goals,
            "io": demand_io,
            "scope": "fresh declared kinematic hold/slew, native lane ownership, no future rollout; previous pointing plus rate-times-capture tracking uncertainty, conditional on admitted schedule",
        }
    )


class _Ledger:
    def __init__(self, v, g):
        self.v, self.g = v, g
        self.n = round(v["horizon_sec"] / v["decision_sec"])
        self.occupied = {a: {} for a in v["actors"]}
        self.starts = {a: set() for a in v["actors"]}
        self.watts = {a: [0.0] * self.n for a in v["actors"]}
        self.bps = {a: [0.0] * self.n for a in v["actors"]}
        self.terminals = {s: [0] * self.n for s in v["stations"]}

    def fork(self):
        result = copy.copy(self)
        for key in ("occupied", "starts", "watts", "bps", "terminals"):
            setattr(result, key, copy.deepcopy(getattr(self, key)))
        return result

    def reserve(self, r, stage, station):
        a, kind, start, width = (
            stage["actor"],
            stage["kind"],
            stage["cell"],
            stage["cells"],
        )
        if (
            type(start) is not int
            or type(width) is not int
            or not 0 <= start < start + width <= self.n
            or start in self.starts[a]
        ):
            return False
        actor = self.g["actors"][a]
        io = self.g["io"][r["id"]]
        mask = ((1 << width) - 1) << start
        claims = actor["claims"][kind]
        watts = {
            "slew": actor["slew_w"],
            "imaging": actor["imaging_w"],
            "transform": io["cpu_w"],
            "relay": actor["radio_w"],
            "downlink": actor["radio_w"],
        }[kind]
        bps = {
            "imaging": io["imaging_bps"],
            "transform": io["cpu_bps"],
            "downlink": io["radio_bps"],
        }.get(kind, 0.0)
        users = [(a, claims, watts, bps)]
        if kind == "relay":
            b = stage["destination"]
            users.append(
                (
                    b,
                    ["cpu", "payload", "radio", "attitude", "propulsion"],
                    self.g["actors"][b]["receive_w"],
                    0.0,
                )
            )
        for user, lanes, power, bandwidth in users:
            limits = self.g["actors"][user]
            if any(self.occupied[user].get(lane, 0) & mask for lane in lanes):
                return False
            if any(
                self.watts[user][i] + power + limits["base_w"]
                > limits["max_load_w"] + 1e-9
                or self.bps[user][i] + bandwidth > limits["bandwidth_bps"] + 1e-6
                for i in range(start, start + width)
            ):
                return False
        if kind == "downlink" and any(
            self.terminals[station][i] >= self.v["stations"][station]
            for i in range(start, start + width)
        ):
            return False
        for user, lanes, power, bandwidth in users:
            for lane in lanes:
                self.occupied[user][lane] = self.occupied[user].get(lane, 0) | mask
            for i in range(start, start + width):
                self.watts[user][i] += power
                self.bps[user][i] += bandwidth
        self.starts[a].add(start)
        if kind == "downlink":
            for i in range(start, start + width):
                self.terminals[station][i] += 1
        return True


def _admit(v, g):
    parsed = JointNetworkView.model_validate(v)
    if (
        v.get("sha256") != digest({k: x for k, x in v.items() if k != "sha256"})
        or parsed.at_sec
        or parsed.recovery
    ):
        raise ValueError("Sequence planner requires sealed fresh forecast")
    if (
        g.get("schema") != "space-env.sequence-lane-geometry/1"
        or g.get("sha256") != digest({k: x for k, x in g.items() if k != "sha256"})
        or g["forecast_sha256"] != v["sha256"]
        or set(g["actors"]) != set(v["actors"])
        or set(g["goals"]) != {r["id"] for r in v["demands"]}
    ):
        raise ValueError("Invalid sequence geometry binding")
    n = round(v["horizon_sec"] / v["decision_sec"])
    if g["cells"] != n or set(g["io"]) != {r["id"] for r in v["demands"]}:
        raise ValueError("Incomplete sequence geometry")
    expected = {
        "slew": {"attitude"},
        "imaging": {"payload", "attitude"},
        "transform": {"cpu"},
        "relay": {"cpu", "payload", "radio", "attitude", "propulsion"},
    }

    def number(value, positive=False):
        return (
            type(value) in (int, float)
            and math.isfinite(value)
            and (value > 0 if positive else value >= 0)
        )

    def quaternion(value):
        return (
            isinstance(value, list)
            and len(value) == 4
            and all(type(x) in (int, float) and math.isfinite(x) for x in value)
            and abs(math.hypot(*value) - 1) < 1e-9
        )

    for actor in g["actors"].values():
        if (
            not quaternion(actor["initial_quaternion"])
            or any(
                not number(actor[k], k in {"rate_deg_s", "bandwidth_bps", "max_load_w"})
                for k in (
                    "rate_deg_s",
                    "bandwidth_bps",
                    "max_load_w",
                    "settling_sec",
                    "base_w",
                    "slew_w",
                    "imaging_w",
                    "radio_w",
                    "receive_w",
                )
            )
            or any(set(actor["claims"][k]) != lanes for k, lanes in expected.items())
            or set(actor["claims"]["downlink"])
            not in ({"radio"}, {"radio", "attitude"})
        ):
            raise ValueError("Invalid native lane or resource contract")
    captures = {(c["request"], c["actor"]): c["cells"] for c in v["captures"]}
    for request in v["demands"]:
        if set(g["goals"][request["id"]]) != set(request["owners"]) or any(
            not number(x, True) for x in g["io"][request["id"]].values()
        ):
            raise ValueError("Invalid sequence request contract")
        for actor, goals in g["goals"][request["id"]].items():
            if len(goals) != n or any(
                (goal is not None) != captures[request["id"], actor][i]
                or (goal is not None and not quaternion(goal))
                for i, goal in enumerate(goals)
            ):
                raise ValueError("Invalid sequence capture goals")


def _cost(v, r, path, ground_cells):
    memory = dict.fromkeys(v["actors"], 0)
    energy = dict.fromkeys(v["actors"], 0.0)
    memory[path[0]] = r["raw_bits"] + r["output_bits"] + 100000
    energy[path[0]] = r["owner_wh"]
    for a, b in itertools.pairwise(path):
        memory[b] = max(memory[b], r["output_bits"])
        energy[a] += r["relay_wh"]
        energy[b] += r["receiver_wh"]
    energy[path[-1]] += r["ground_wh"] * ground_cells / r["ground_cells"]
    return memory, energy


def _slew_cells(g, owner, previous, goal, q, uncertainty_deg=0.0):
    a = g["actors"][owner]
    return max(
        1,
        math.ceil(
            (
                min(180.0, quaternion_angle(previous, goal) + uncertainty_deg)
                / a["rate_deg_s"]
                + a["settling_sec"]
                + 1e-6
            )
            / q
        ),
    )


def schedule_sequence(
    v, g, *, max_nodes=50000, max_hops=3, max_orders=12, request_order=None
):
    _admit(v, g)
    if (
        type(max_nodes) is not int
        or not 1 <= max_nodes <= 500000
        or type(max_orders) is not int
        or not 1 <= max_orders <= 32
        or type(max_hops) is not int
        or not 0 <= max_hops <= 3
    ):
        raise ValueError("Invalid sequence search budgets")
    ids = [r["id"] for r in v["demands"]]
    by_id = {r["id"]: r for r in v["demands"]}
    if request_order is not None and (
        not isinstance(request_order, (list, tuple))
        or len(request_order) != len(ids)
        or set(request_order) != set(ids)
    ):
        raise ValueError("Invalid request order")
    priority = tuple(
        sorted(ids, key=lambda i: (-by_id[i]["priority"], by_id[i]["deadline_sec"], i))
    )
    earliest = tuple(
        sorted(ids, key=lambda i: (by_id[i]["deadline_sec"], -by_id[i]["priority"], i))
    )
    candidates = (
        [tuple(request_order)]
        if request_order is not None
        else [
            priority,
            earliest,
            tuple(reversed(priority)),
            *itertools.islice(itertools.permutations(sorted(ids)), max_orders),
        ]
    )
    orders = list(dict.fromkeys(candidates))[:max_orders]
    q = v["decision_sec"]
    n = round(v["horizon_sec"] / q)
    edges = {(e["source"], e["destination"]): e for e in v["edges"]}
    used = 0
    exhausted = False
    best = None
    trials = []
    for order in orders:
        limit = min(max_nodes, used + max(1, max_nodes // len(orders)))
        ledger = _Ledger(v, g)
        memory = dict.fromkeys(v["actors"], 0)
        energy = dict.fromkeys(v["actors"], 0.0)
        pose = {a: g["actors"][a]["initial_quaternion"] for a in v["actors"]}
        uncertainty = dict.fromkeys(v["actors"], 0.0)
        after = dict.fromkeys(v["actors"], 0)
        allocations = []
        unserved = {}

        def tick(limit=limit):
            nonlocal used
            if used >= limit:
                raise StopIteration
            used += 1

        for rid in order:
            r = by_id[rid]
            choice = None
            try:
                for owner in sorted(r["owners"]):
                    others = sorted(set(v["actors"]) - {owner})
                    for hops in range(min(max_hops, len(others)) + 1):
                        for tail in itertools.permutations(others, hops):
                            path = (owner, *tail)
                            if any(
                                (a, b) not in edges for a, b in itertools.pairwise(path)
                            ):
                                continue
                            for station in sorted(r["stations"]):
                                for cap, goal in enumerate(g["goals"][rid][owner]):
                                    tick()
                                    if goal is None:
                                        continue
                                    sl = _slew_cells(
                                        g,
                                        owner,
                                        pose[owner],
                                        goal,
                                        q,
                                        uncertainty[owner],
                                    )
                                    first = cap - sl
                                    deadline = min(
                                        n, math.floor((r["deadline_sec"] + 1e-8) / q)
                                    )
                                    if (
                                        first
                                        < max(
                                            after[owner],
                                            math.ceil(r["release_sec"] / q),
                                        )
                                        or cap + r["capture_cells"] >= deadline
                                    ):
                                        continue
                                    attempt = ledger.fork()
                                    stages = [
                                        {
                                            "kind": "slew",
                                            "actor": owner,
                                            "cell": first,
                                            "cells": sl,
                                        },
                                        {
                                            "kind": "imaging",
                                            "actor": owner,
                                            "cell": cap,
                                            "cells": r["capture_cells"],
                                        },
                                    ]
                                    if not all(
                                        attempt.reserve(r, s, station) for s in stages
                                    ):
                                        continue
                                    ready = cap + r["capture_cells"]
                                    found = None
                                    for start in range(
                                        ready, deadline - r["cpu_cells"] + 1
                                    ):
                                        tick()
                                        s = {
                                            "kind": "transform",
                                            "actor": owner,
                                            "cell": start,
                                            "cells": r["cpu_cells"],
                                        }
                                        if attempt.reserve(r, s, station):
                                            found = s
                                            break
                                    if found is None:
                                        continue
                                    stages.append(found)
                                    ready = found["cell"] + found["cells"]
                                    ok = True
                                    for index, left in enumerate(path):
                                        last = index == len(path) - 1
                                        right = station if last else path[index + 1]
                                        edge = edges.get((left, right))
                                        found = None
                                        if edge is None:
                                            ok = False
                                            break
                                        base = (
                                            r["ground_cells"]
                                            if last
                                            else r["relay_cells"]
                                        )
                                        for start in range(ready, deadline - base + 1):
                                            tick()
                                            width = (
                                                ground_width(r, edge, start, q)
                                                if last
                                                else base
                                            )
                                            if start + width > deadline or not all(
                                                edge["cells"][i]
                                                for i in range(start, start + width)
                                            ):
                                                continue
                                            s = {
                                                "kind": "downlink" if last else "relay",
                                                "actor": left,
                                                "destination": right,
                                                "cell": start,
                                                "cells": width,
                                            }
                                            if attempt.reserve(r, s, station):
                                                found = s
                                                break
                                        if found is None:
                                            ok = False
                                            break
                                        stages.append(found)
                                        ready = found["cell"] + found["cells"]
                                    if not ok:
                                        continue
                                    mem, cost = _cost(v, r, path, stages[-1]["cells"])
                                    if any(
                                        memory[a] + mem[a]
                                        > v["actors"][a]["memory_bits"]
                                        or energy[a] + cost[a]
                                        > v["actors"][a]["usable_wh"] + 1e-12
                                        for a in v["actors"]
                                    ):
                                        continue
                                    allocation = {
                                        "request": rid,
                                        "owner": owner,
                                        "station": station,
                                        "path": list(path),
                                        "stages": stages,
                                        "finish_sec": ready * q,
                                        "memory_bits": mem,
                                        "energy_wh": cost,
                                    }
                                    key = (
                                        ready,
                                        len(path),
                                        owner,
                                        station,
                                        first,
                                        path,
                                    )
                                    if choice is None or key < choice[0]:
                                        choice = (
                                            key,
                                            allocation,
                                            attempt,
                                            goal,
                                            cap + r["capture_cells"],
                                        )
            except StopIteration:
                exhausted = True
            if choice is None:
                unserved[rid] = (
                    "search_budget_exhausted"
                    if used >= limit
                    else "no_admitted_sequence_route"
                )
                continue
            _, allocation, ledger, goal, owner_after = choice
            allocations.append(allocation)
            pose[allocation["owner"]] = goal
            uncertainty[allocation["owner"]] = min(
                180.0,
                g["actors"][allocation["owner"]]["rate_deg_s"] * r["capture_cells"] * q,
            )
            after[allocation["owner"]] = owner_after
            for a in v["actors"]:
                memory[a] += allocation["memory_bits"][a]
                energy[a] += allocation["energy_wh"][a]
        score = (
            sum(by_id[a["request"]]["priority"] for a in allocations),
            len(allocations),
            -sum(a["finish_sec"] for a in allocations),
        )
        trials.append(
            {
                "order": list(order),
                "fulfilled": len(allocations),
                "priority": score[0],
                "nodes_so_far": used,
            }
        )
        if best is None or score > best[0]:
            best = (score, allocations, unserved, order)
        if used >= max_nodes:
            break
    result = seal(
        {
            "schema": "space-env.sequence-lane-schedule/1",
            "planner": "sequence_lanes",
            "forecast_sha256": v["sha256"],
            "geometry_sha256": g["sha256"],
            "allocations": best[1],
            "unserved": best[2],
            "request_order": list(best[3]),
            "predicted_fulfilled": len(best[1]),
            "completed": [],
            "total": len(ids),
            "search_nodes": used,
            "max_search_nodes": max_nodes,
            "max_hops": max_hops,
            "max_orders": max_orders,
            "trials": trials,
            "search_budget_exhausted": exhausted,
            "optimal": False,
            "scope": "bounded order portfolio and append-only attitude sequence, native lanes, peak power/bandwidth, conservative lifetime memory/energy; one start per role/tick; conditional proposal, not global/native optimum",
        }
    )
    verify_sequence_schedule(v, g, result)
    return result


def verify_sequence_schedule(v, g, schedule):
    _admit(v, g)
    if (
        schedule.get("schema") != "space-env.sequence-lane-schedule/1"
        or schedule.get("sha256")
        != digest({k: x for k, x in schedule.items() if k != "sha256"})
        or schedule["forecast_sha256"] != v["sha256"]
        or schedule["geometry_sha256"] != g["sha256"]
        or schedule.get("optimal") is not False
    ):
        raise ValueError("Invalid sequence schedule binding or optimum claim")
    if (
        type(schedule["search_nodes"]) is not int
        or type(schedule["max_search_nodes"]) is not int
        or not 0 <= schedule["search_nodes"] <= schedule["max_search_nodes"] <= 500000
        or type(schedule["max_hops"]) is not int
        or not 0 <= schedule["max_hops"] <= 3
        or type(schedule["max_orders"]) is not int
        or not 1 <= schedule["max_orders"] <= 32
        or len(schedule["trials"]) > schedule["max_orders"]
    ):
        raise ValueError("Invalid sequence budget witness")
    demands = {r["id"]: r for r in v["demands"]}
    seen = set()
    ledger = _Ledger(v, g)
    pose = {a: g["actors"][a]["initial_quaternion"] for a in v["actors"]}
    uncertainty = dict.fromkeys(v["actors"], 0.0)
    after = dict.fromkeys(v["actors"], 0)
    memory = dict.fromkeys(v["actors"], 0)
    energy = dict.fromkeys(v["actors"], 0.0)
    edges = {(e["source"], e["destination"]): e for e in v["edges"]}
    q = v["decision_sec"]
    for allocation in schedule["allocations"]:
        rid = allocation["request"]
        if rid not in demands or rid in seen:
            raise ValueError("Duplicate/unknown sequence request")
        seen.add(rid)
        r = demands[rid]
        path = allocation["path"]
        owner = allocation["owner"]
        stages = allocation["stages"]
        station = allocation["station"]
        if (
            not path
            or path[0] != owner
            or owner not in r["owners"]
            or len(path) != len(set(path))
            or not set(path) <= set(v["actors"])
            or len(path) - 1 > schedule["max_hops"]
            or station not in r["stations"]
            or len(stages) != 3 + len(path)
        ):
            raise ValueError("Invalid sequence ownership/path")
        cap = stages[1]["cell"]
        if (
            type(cap) is not int
            or not 0 <= cap < g["cells"]
            or g["goals"][rid][owner][cap] is None
        ):
            raise ValueError("Invalid capture goal")
        if not next(
            c["cells"][cap]
            for c in v["captures"]
            if c["request"] == rid and c["actor"] == owner
        ):
            raise ValueError("Capture outside forecast opportunity")
        goal = g["goals"][rid][owner][cap]
        sl = _slew_cells(g, owner, pose[owner], goal, q, uncertainty[owner])
        expected = [
            ("slew", owner, None, sl),
            ("imaging", owner, None, r["capture_cells"]),
            ("transform", owner, None, r["cpu_cells"]),
        ]
        expected += [
            ("relay", a, b, r["relay_cells"]) for a, b in itertools.pairwise(path)
        ]
        expected += [("downlink", path[-1], station, None)]
        end = None
        for index, (stage, (kind, actor, destination, width)) in enumerate(
            zip(stages, expected)
        ):
            start = stage["cell"]
            if type(start) is not int:
                raise ValueError("Invalid stage cell")
            if kind == "downlink":
                width = (
                    ground_width(r, edges.get((actor, destination), {}), start, q)
                    if 0 <= start < g["cells"]
                    else -1
                )
            if (
                stage["kind"] != kind
                or stage["actor"] != actor
                or stage.get("destination") != destination
                or stage["cells"] != width
                or start * q < r["release_sec"]
                or (start + width) * q > r["deadline_sec"] + 1e-8
                or (end is not None and (start < end or (index == 1 and start != end)))
            ):
                raise ValueError("Invalid stage order/width/deadline")
            if index == 0 and start < after[owner]:
                raise ValueError("Attitude sequence was reordered")
            edge = edges.get((actor, destination)) if destination is not None else None
            if destination is not None and (
                edge is None
                or not 0 <= start < start + width <= g["cells"]
                or not all(edge["cells"][i] for i in range(start, start + width))
            ):
                raise ValueError("Invalid sequence contact")
            if not ledger.reserve(r, stage, station):
                raise ValueError(
                    "Sequence native lane, power, bandwidth or start conflict"
                )
            end = start + width
        mem, cost = _cost(v, r, path, stages[-1]["cells"])
        if (
            mem != allocation["memory_bits"]
            or cost != allocation["energy_wh"]
            or allocation["finish_sec"] != end * q
        ):
            raise ValueError("Sequence resource witness mismatch")
        for a in v["actors"]:
            memory[a] += mem[a]
            energy[a] += cost[a]
            if (
                memory[a] > v["actors"][a]["memory_bits"]
                or energy[a] > v["actors"][a]["usable_wh"] + 1e-12
            ):
                raise ValueError("Sequence lifetime memory/energy exceeded")
        pose[owner] = goal
        uncertainty[owner] = min(
            180.0, g["actors"][owner]["rate_deg_s"] * r["capture_cells"] * q
        )
        after[owner] = cap + r["capture_cells"]
    if (
        schedule["predicted_fulfilled"] != len(seen)
        or schedule["total"] != len(demands)
        or set(schedule["unserved"]) != set(demands) - seen
    ):
        raise ValueError("Sequence coverage mismatch")
    return {"valid": True, "allocations": len(seen), "optimality_established": False}
