"""Bounded global set packing of enumerated EO/CPU/ISL/Ground chains.

Optimality applies only to this discrete, conservative forecast model. No native
future rollout is used. Interrupted enumeration/search keeps a feasible incumbent
and an honest (possibly loose) upper bound.
"""

from __future__ import annotations

import copy
import itertools
import math

from space_env.planning.joint_network import JointNetworkView, ground_width
from space_env.simulation.snapshot import digest


def _view(value):
    v = JointNetworkView.model_validate(value)
    if value["schema"] != "space-env.joint-network-view/1" or value["sha256"] != digest(
        {k: x for k, x in value.items() if k != "sha256"}
    ):
        raise ValueError("Invalid exact-planning forecast seal")
    if (
        len(v.demands) > 6
        or round(v.horizon_sec / v.decision_sec) > 48
        or v.at_sec != 0
        or v.recovery
        or any(v.stored_bits.values())
    ):
        raise ValueError(
            "Exact-small profile: initial fresh mission, <=6 requests/48 cells"
        )
    return v.model_dump(mode="json", by_alias=True)


def _span(cell, width):
    return ((1 << width) - 1) << cell


def _resources(v, r, allocation):
    """Independent admission of a candidate/returned allocation against the view."""
    actors = v["actors"]
    path, stages = allocation["path"], allocation["stages"]
    q, n = v["decision_sec"], round(v["horizon_sec"] / v["decision_sec"])
    edges = {(e["source"], e["destination"]): e["cells"] for e in v["edges"]}
    edge_rows = {(e["source"], e["destination"]): e for e in v["edges"]}
    captures = {(c["request"], c["actor"]): c["cells"] for c in v["captures"]}
    if (
        not path
        or len(path) != len(set(path))
        or any(a not in actors for a in path)
        or path[0] != allocation["owner"]
        or path[0] not in r["owners"]
        or allocation["station"] not in r["stations"]
        or len(stages) != 3 + len(path)
    ):
        raise ValueError("Invalid chain ownership/path")
    expected = [
        ("slew", path[0], None, r["slew_cells"]),
        ("imaging", path[0], None, r["capture_cells"]),
        ("transform", path[0], None, r["cpu_cells"]),
    ]
    expected += [("relay", a, b, r["relay_cells"]) for a, b in itertools.pairwise(path)]
    expected += [("downlink", path[-1], allocation["station"], r["ground_cells"])]
    masks, memory, energy = (
        {a: 0 for a in actors},
        {a: 0 for a in actors},
        {a: 0.0 for a in actors},
    )
    owner = path[0]
    memory[owner] = r["raw_bits"] + r["output_bits"] + 100000
    energy[owner] = r["owner_wh"]
    previous_end = None
    for i, (stage, (kind, actor, destination, width)) in enumerate(
        zip(stages, expected)
    ):
        start = stage["cell"]
        if kind == "downlink" and type(start) is int and 0 <= start < n:
            width = ground_width(r, edge_rows.get((actor, destination), {}), start, q)
        if (
            type(start) is not int
            or type(stage["cells"]) is not int
            or stage["cells"] != width
            or stage["kind"] != kind
            or stage["actor"] != actor
            or stage.get("destination") != destination
            or not 0 <= start < start + width <= n
            or start * q < r["release_sec"]
            or (start + width) * q > r["deadline_sec"] + 1e-8
        ):
            raise ValueError("Invalid chain stage/grid/deadline")
        if previous_end is not None and (
            start < previous_end or (i < 3 and start != previous_end)
        ):
            raise ValueError("Prefix must be contiguous; transport must be sequential")
        previous_end = start + width
        mask = _span(start, width)
        if masks[actor] & mask:
            raise ValueError("Actor conflict inside chain")
        masks[actor] |= mask
        if kind == "imaging" and not captures[r["id"], owner][start]:
            raise ValueError("Capture outside declared opportunity")
        if destination is not None:
            if not all(
                edges.get((actor, destination), [False] * n)[j]
                for j in range(start, start + width)
            ):
                raise ValueError("Transport outside declared contact")
            if kind == "relay":
                if masks[destination] & mask:
                    raise ValueError("Receiver conflict inside chain")
                masks[destination] |= mask
                memory[destination] = r["output_bits"]
                energy[actor] += r["relay_wh"]
                energy[destination] += r["receiver_wh"]
            else:
                energy[actor] += r["ground_wh"] * width / r["ground_cells"]
    if any(
        memory[a] > actors[a]["memory_bits"]
        or energy[a] > actors[a]["usable_wh"] + 1e-12
        for a in actors
    ):
        raise ValueError("Individual memory/energy infeasible")
    return masks, memory, energy


def verify_exact_schedule(value, schedule):
    """Admit resource/binding witnesses; optimality needs solver reproduction."""
    v = _view(value)
    if (
        schedule.get("sha256")
        != digest({k: x for k, x in schedule.items() if k != "sha256"})
        or schedule.get("forecast_sha256") != value["sha256"]
    ):
        raise ValueError("Exact schedule binding mismatch")
    requests = {r["id"]: r for r in v["demands"]}
    masks, memory, energy = (
        {a: 0 for a in v["actors"]},
        {a: 0 for a in v["actors"]},
        {a: 0.0 for a in v["actors"]},
    )
    stations = {
        s: [0] * round(v["horizon_sec"] / v["decision_sec"]) for s in v["stations"]
    }
    served = set()
    score = 0
    for allocation in schedule["allocations"]:
        rid = allocation["request"]
        if (
            rid not in requests
            or rid in served
            or len(allocation["path"]) - 1 > schedule["max_hops"]
        ):
            raise ValueError("Duplicate/unknown request or excessive path")
        served.add(rid)
        r = requests[rid]
        am, mm, ee = _resources(v, r, allocation)
        for actor, occupied in masks.items():
            if occupied & am[actor]:
                raise ValueError("Shared actor conflict")
            masks[actor] |= am[actor]
            memory[actor] += mm[actor]
            energy[actor] += ee[actor]
            if (
                memory[actor] > v["actors"][actor]["memory_bits"]
                or energy[actor] > v["actors"][actor]["usable_wh"] + 1e-12
            ):
                raise ValueError("Shared memory/energy conflict")
        ground = allocation["stages"][-1]
        for cell in range(ground["cell"], ground["cell"] + ground["cells"]):
            stations[allocation["station"]][cell] += 1
            if (
                stations[allocation["station"]][cell]
                > v["stations"][allocation["station"]]
            ):
                raise ValueError("Shared ground terminal conflict")
        if (
            allocation["memory_bits"] != mm
            or allocation["energy_wh"] != ee
            or allocation["finish_sec"]
            != (ground["cell"] + ground["cells"]) * v["decision_sec"]
        ):
            raise ValueError("Allocation resource witness mismatch")
        score += r["priority"] * (len(requests) + 1) + 1
    if (
        set(schedule["unserved"]) != set(requests) - served
        or schedule["total"] != len(requests)
        or schedule["predicted_fulfilled"] != len(served)
        or schedule["objective_score"] != score
        or not score
        <= schedule["upper_bound_score"]
        <= sum(r["priority"] * (len(requests) + 1) + 1 for r in requests.values())
        or schedule["optimal"] != (score == schedule["upper_bound_score"])
    ):
        raise ValueError("Invalid denominator/objective/bound witness")
    return True


def optimize_network(
    value, *, max_nodes=50000, max_hops=3, max_candidates=8192, pruning=True
):
    if (
        type(max_nodes) is not int
        or not 1 <= max_nodes <= 500000
        or type(max_hops) is not int
        or not 0 <= max_hops <= 3
        or type(max_candidates) is not int
        or not 1 <= max_candidates <= 16384
        or type(pruning) is not bool
    ):
        raise ValueError("Invalid exact-small budgets")
    v = _view(value)
    q, n = v["decision_sec"], round(v["horizon_sec"] / v["decision_sec"])
    actors = sorted(v["actors"])
    edges = {(e["source"], e["destination"]): e["cells"] for e in v["edges"]}
    edge_rows = {(e["source"], e["destination"]): e for e in v["edges"]}
    captures = {(c["request"], c["actor"]): c["cells"] for c in v["captures"]}
    requests = sorted(
        v["demands"], key=lambda r: (-r["priority"], r["deadline_sec"], r["id"])
    )
    candidates = {r["id"]: [] for r in requests}
    nodes = count = 0
    enumeration_complete = True
    enum_limit = max(1, max_nodes // 2)

    def tick(limit):
        nonlocal nodes
        if nodes >= limit:
            raise StopIteration
        nodes += 1

    def emit(r, owner, path, station, prefix):
        nonlocal count
        deadline = min(n, math.floor((r["deadline_sec"] + 1e-8) / q))

        def transport(index, stages, ready):
            nonlocal count
            last = index == len(path) - 1
            left, right = path[index], station if last else path[index + 1]
            base_width = r["ground_cells"] if last else r["relay_cells"]
            flags = edges.get((left, right), [False] * n)
            for start in range(ready, deadline - base_width + 1):
                tick(enum_limit)
                width = (
                    ground_width(r, edge_rows.get((left, right), {}), start, q)
                    if last
                    else base_width
                )
                if start + width > deadline or not all(flags[start : start + width]):
                    continue
                new = stages + [
                    {
                        "kind": "downlink" if last else "relay",
                        "actor": left,
                        "destination": right,
                        "cell": start,
                        "cells": width,
                    }
                ]
                if last:
                    if count >= max_candidates:
                        raise StopIteration
                    allocation = {
                        "request": r["id"],
                        "owner": owner,
                        "path": list(path),
                        "station": station,
                        "stages": new,
                        "finish_sec": (start + width) * q,
                    }
                    try:
                        masks, memory, energy = _resources(v, r, allocation)
                    except ValueError:
                        continue
                    allocation.update(memory_bits=memory, energy_wh=energy)
                    candidates[r["id"]].append((allocation, masks))
                    count += 1
                else:
                    transport(index + 1, new, start + width)

        transport(0, prefix, prefix[-1]["cell"] + prefix[-1]["cells"])

    try:
        for r in requests:
            for owner in sorted(r["owners"]):
                others = [a for a in actors if a != owner]
                for hops in range(min(max_hops, len(others)) + 1):
                    for tail in itertools.permutations(others, hops):
                        tick(enum_limit)
                        path = (owner, *tail)
                        if any(
                            (a, b) not in edges for a, b in itertools.pairwise(path)
                        ):
                            continue
                        for station in sorted(r["stations"]):
                            for first in range(math.ceil(r["release_sec"] / q), n):
                                tick(enum_limit)
                                cap = first + r["slew_cells"]
                                end = cap + r["capture_cells"] + r["cpu_cells"]
                                if (
                                    end >= n
                                    or end * q >= r["deadline_sec"]
                                    or not captures[r["id"], owner][cap]
                                ):
                                    continue
                                prefix = [
                                    {
                                        "kind": kind,
                                        "actor": owner,
                                        "cell": cell,
                                        "cells": width,
                                    }
                                    for kind, cell, width in (
                                        ("slew", first, r["slew_cells"]),
                                        ("imaging", cap, r["capture_cells"]),
                                        (
                                            "transform",
                                            cap + r["capture_cells"],
                                            r["cpu_cells"],
                                        ),
                                    )
                                ]
                                emit(r, owner, path, station, prefix)
    except StopIteration:
        enumeration_complete = False
    enumeration_nodes = nodes
    weights = [r["priority"] * (len(requests) + 1) + 1 for r in requests]
    upper = sum(weights)
    masks = {a: 0 for a in actors}
    memory, energy = dict.fromkeys(actors, 0), dict.fromkeys(actors, 0.0)
    terminals = {s: [0] * n for s in v["stations"]}
    best, best_score = [], 0
    search_complete = True
    # A complete direct-chain universe proves whether actor labels can be
    # permuted without changing any constraint/candidate. Partial universes or
    # ISL paths deliberately do not establish this equivalence.
    signatures = {}
    for actor in actors:
        footprint = []
        if enumeration_complete and all(
            len(a["path"]) == 1 for cs in candidates.values() for a, _ in cs
        ):
            for rid, cs in candidates.items():
                for a, am in cs:
                    if a["owner"] == actor:
                        g = a["stages"][-1]
                        footprint.append(
                            (
                                rid,
                                am[actor],
                                a["station"],
                                g["cell"],
                                g["cells"],
                                a["memory_bits"][actor],
                                a["energy_wh"][actor].hex(),
                            )
                        )
            signature = (
                v["actors"][actor]["memory_bits"],
                v["actors"][actor]["usable_wh"],
                tuple(sorted(footprint)),
            )
        else:
            signature = (actor,)
        signatures.setdefault(signature, []).append(actor)
    groups = list(signatures.values())
    transpositions = {}
    cache_limit = min(max_nodes, 16384)
    pruned_states = 0
    flat, indexed, request_bits = [], {}, {}
    actor_cells = {a: [0] * n for a in actors}
    station_cells = {s: [0] * n for s in v["stations"]}
    memory_costs = {a: {} for a in actors}
    energy_costs = {a: {} for a in actors}
    for rid, cs in candidates.items():
        indexed[rid], request_bits[rid] = [], 0
        for allocation, am in cs:
            bit = 1 << len(flat)
            flat.append((allocation, am))
            indexed[rid].append((bit, allocation, am))
            request_bits[rid] |= bit
            for a in actors:
                occupied = am[a]
                while occupied:
                    low = occupied & -occupied
                    actor_cells[a][low.bit_length() - 1] |= bit
                    occupied ^= low
                for table, cost in (
                    (memory_costs[a], allocation["memory_bits"][a]),
                    (energy_costs[a], allocation["energy_wh"][a]),
                ):
                    if cost:
                        table[cost] = table.get(cost, 0) | bit
            g = allocation["stages"][-1]
            for cell in range(g["cell"], g["cell"] + g["cells"]):
                station_cells[allocation["station"]][cell] |= bit
    mandatory_cache = {}
    mandatory_cache_limit = min(max_nodes, 4096)

    def feasible_candidates():
        invalid = 0
        for a in actors:
            occupied = masks[a]
            while occupied:
                low = occupied & -occupied
                invalid |= actor_cells[a][low.bit_length() - 1]
                occupied ^= low
            for cost, bits in memory_costs[a].items():
                if memory[a] + cost > v["actors"][a]["memory_bits"]:
                    invalid |= bits
            for cost, bits in energy_costs[a].items():
                if energy[a] + cost > v["actors"][a]["usable_wh"] + 1e-12:
                    invalid |= bits
        for station, used in terminals.items():
            for cell, count_used in enumerate(used):
                if count_used >= v["stations"][station]:
                    invalid |= station_cells[station][cell]
        return {rid: bits & ~invalid for rid, bits in request_bits.items()}

    def mandatory(bits):
        if bits in mandatory_cache:
            return mandatory_cache[bits]
        busy, ground = None, None
        todo = bits
        while todo:
            low = todo & -todo
            allocation, am = flat[low.bit_length() - 1]
            g = allocation["stages"][-1]
            gm = {
                s: _span(g["cell"], g["cells"]) if s == allocation["station"] else 0
                for s in terminals
            }
            if busy is None:
                busy, ground = am.copy(), gm
            else:
                busy = {a: busy[a] & am[a] for a in actors}
                ground = {s: ground[s] & gm[s] for s in terminals}
            if not any(busy.values()) and not any(ground.values()):
                break
            todo ^= low
        result = busy, ground
        if len(mandatory_cache) < mandatory_cache_limit:
            mandatory_cache[bits] = result
        return result

    def mandatory_bound(index, feasible):
        remaining = [
            (weights[i], *mandatory(feasible[requests[i]["id"]]))
            for i in range(index, len(requests))
            if feasible[requests[i]["id"]]
        ]
        constraints = set()
        for i, (_, am, _) in enumerate(remaining):
            for j in range(i):
                if any(am[a] & remaining[j][1][a] for a in actors):
                    constraints.add(((1 << i) | (1 << j), 1))
        for station in terminals:
            for cell in range(n):
                required = sum(
                    1 << i
                    for i, (_, _, gm) in enumerate(remaining)
                    if gm[station] & (1 << cell)
                )
                capacity = v["stations"][station] - terminals[station][cell]
                if required.bit_count() > capacity:
                    constraints.add((required, capacity))
        if not constraints:
            return sum(item[0] for item in remaining)
        bound = 0
        for subset in range(1 << len(remaining)):
            value = sum(
                item[0] for i, item in enumerate(remaining) if subset & (1 << i)
            )
            if value > bound and all(
                (subset & mask).bit_count() <= capacity
                for mask, capacity in constraints
            ):
                bound = value
        return bound

    def state_key(index):
        return (
            index,
            tuple(
                tuple(sorted((masks[a], memory[a], energy[a].hex()) for a in group))
                for group in groups
            ),
            tuple(tuple(terminals[s]) for s in sorted(terminals)),
        )

    # An optimistic aggregate actor-capacity relaxation: use only cells that
    # occur in the complete candidate universe and the least actor occupancy
    # of each request. It is a fractional knapsack, never an integer lower bound.
    from fractions import Fraction

    union = dict.fromkeys(actors, 0)
    minimum = []
    for r in requests:
        cs = candidates[r["id"]]
        minimum.append(
            min((sum(x.bit_count() for x in am.values()) for _, am in cs), default=0)
        )
        for _, am in cs:
            for a in actors:
                union[a] |= am[a]

    def capacity_bound(index):
        available = sum((union[a] & ~masks[a]).bit_count() for a in actors)
        items = sorted(
            (
                (weights[i], minimum[i])
                for i in range(index, len(requests))
                if minimum[i]
            ),
            key=lambda x: Fraction(x[0], x[1]),
            reverse=True,
        )
        bound = Fraction(0)
        for weight, cost in items:
            if available >= cost:
                bound += weight
                available -= cost
            else:
                bound += Fraction(weight * available, cost)
                break
        return bound.numerator // bound.denominator

    def search(index, selected, score):
        nonlocal best, best_score, pruned_states
        tick(max_nodes)
        if score > best_score:
            best, best_score = copy.deepcopy(selected), score
        if index == len(requests) or score + sum(weights[index:]) <= best_score:
            return
        if pruning and score + capacity_bound(index) <= best_score:
            pruned_states += 1
            return
        feasible = feasible_candidates()
        if pruning and score + mandatory_bound(index, feasible) <= best_score:
            pruned_states += 1
            return
        key = state_key(index)
        if pruning and transpositions.get(key, -1) >= score:
            pruned_states += 1
            return
        if pruning and (len(transpositions) < cache_limit or key in transpositions):
            transpositions[key] = score
        for bit, allocation, am in indexed[requests[index]["id"]]:
            tick(max_nodes)
            if not feasible[requests[index]["id"]] & bit:
                continue
            mm, ee = allocation["memory_bits"], allocation["energy_wh"]
            ground, station = allocation["stages"][-1], allocation["station"]
            cells = range(ground["cell"], ground["cell"] + ground["cells"])
            if any(
                masks[a] & am[a]
                or memory[a] + mm[a] > v["actors"][a]["memory_bits"]
                or energy[a] + ee[a] > v["actors"][a]["usable_wh"] + 1e-12
                for a in actors
            ) or any(terminals[station][j] >= v["stations"][station] for j in cells):
                continue
            previous_energy = energy.copy()
            for a in actors:
                masks[a] |= am[a]
                memory[a] += mm[a]
                energy[a] += ee[a]
            for j in cells:
                terminals[station][j] += 1
            search(index + 1, selected + [allocation], score + weights[index])
            for a in actors:
                masks[a] ^= am[a]
                memory[a] -= mm[a]
                energy[a] = previous_energy[a]
            for j in cells:
                terminals[station][j] -= 1
            if best_score == upper:
                return
        search(index + 1, selected, score)

    try:
        search(0, [], 0)
    except StopIteration:
        search_complete = False
    if enumeration_complete and search_complete:
        upper = best_score
    served = {a["request"] for a in best}
    body = {
        "schema": "space-env.joint-network-schedule/1",
        "planner": "exact_small",
        "pruning": pruning,
        "forecast_sha256": value["sha256"],
        "allocations": best,
        "unserved": {
            r["id"]: "not_selected_in_certified_model"
            if upper == best_score
            else "bounded_search_unresolved"
            for r in requests
            if r["id"] not in served
        },
        "predicted_fulfilled": len(best),
        "completed": [],
        "total": len(requests),
        "search_nodes": nodes,
        "enumeration_nodes": enumeration_nodes,
        "candidates": count,
        "equivalent_actor_groups": groups,
        "pruned_states": pruned_states,
        "cached_states": len(transpositions),
        "cache_limit": cache_limit,
        "mandatory_cache_states": len(mandatory_cache),
        "mandatory_cache_limit": mandatory_cache_limit,
        "enumeration_complete": enumeration_complete,
        "search_complete": search_complete,
        "search_budget_exhausted": not enumeration_complete or not search_complete,
        "max_search_nodes": max_nodes,
        "max_candidates": max_candidates,
        "max_hops": max_hops,
        "objective_score": best_score,
        "upper_bound_score": upper,
        "optimal": upper == best_score,
        "scope": "lexicographic priority/count optimum only for initial discrete contiguous-EO/CPU, simple-path custody, conservative lifetime memory/energy forecast model; not native or continuous mission optimality",
    }
    result = body | {"sha256": digest(body)}
    verify_exact_schedule(value, result)
    return result
