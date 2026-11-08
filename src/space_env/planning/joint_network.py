"""Bounded joint owner/path/time packing over an explicitly sealed forecast."""

from __future__ import annotations

import copy
import itertools
import math
from typing import Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.snapshot import digest


class NetworkActor(StrictModel):
    memory_bits: float = Field(gt=0, strict=True)
    usable_wh: float = Field(ge=0, strict=True)


class NetworkDemand(StrictModel):
    id: str
    owners: tuple[str, ...] = Field(min_length=1, max_length=4)
    stations: tuple[str, ...] = Field(min_length=1, max_length=4)
    release_sec: float = Field(ge=0, strict=True)
    deadline_sec: float = Field(gt=0, strict=True)
    priority: int = Field(ge=0, strict=True)
    raw_bits: int = Field(gt=0, strict=True)
    output_bits: int = Field(gt=0, strict=True)
    slew_cells: int = Field(ge=1, strict=True)
    capture_cells: int = Field(ge=1, strict=True)
    cpu_cells: int = Field(ge=1, strict=True)
    relay_cells: int = Field(ge=1, strict=True)
    ground_cells: int = Field(ge=1, strict=True)
    ground_seconds: float = Field(default=0.0, ge=0, strict=True)
    owner_wh: float = Field(gt=0, strict=True)
    relay_wh: float = Field(gt=0, strict=True)
    receiver_wh: float = Field(gt=0, strict=True)
    ground_wh: float = Field(gt=0, strict=True)


class NetworkEdge(StrictModel):
    source: str
    destination: str
    cells: tuple[StrictBool, ...] = Field(min_length=1, max_length=240)
    setup_sec: tuple[FiniteScalar, ...] = Field(default=(), max_length=240)


class CaptureSlots(StrictModel):
    request: str
    actor: str
    cells: tuple[StrictBool, ...] = Field(min_length=1, max_length=240)


class NetworkRecovery(StrictModel):
    phase: Literal["fresh", "raw", "final", "completed", "terminal", "blocked"]
    owner: str
    remaining_bits: float = Field(ge=0, strict=True)
    remaining_cpu_sec: float = Field(ge=0, strict=True)
    intact: StrictBool = True
    reason: str | None = None


class JointNetworkView(StrictModel):
    schema_id: str = Field(alias="schema")
    horizon_sec: float = Field(gt=0, le=1200, strict=True)
    decision_sec: float = Field(ge=1, le=30, strict=True)
    actors: dict[str, NetworkActor] = Field(min_length=1, max_length=4)
    stations: dict[str, StrictInt] = Field(min_length=1, max_length=4)
    demands: tuple[NetworkDemand, ...] = Field(min_length=1, max_length=16)
    edges: tuple[NetworkEdge, ...] = Field(max_length=28)
    captures: tuple[CaptureSlots, ...] = Field(max_length=64)
    at_sec: float = Field(default=0.0, ge=0, strict=True)
    recovery: dict[str, NetworkRecovery] = Field(default_factory=dict)
    stored_bits: dict[str, float] = Field(default_factory=dict)
    geometry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def graph(self):
        n = round(self.horizon_sec / self.decision_sec)
        if not 1 <= n <= 240 or abs(n * self.decision_sec - self.horizon_sec) > 1e-8:
            raise ValueError("invalid network grid")
        if set(self.actors) & set(self.stations) or any(
            type(v) is not int or not 1 <= v <= 4 for v in self.stations.values()
        ):
            raise ValueError("invalid graph nodes/terminals")
        ids = {r.id for r in self.demands}
        if any(
            (
                e.setup_sec
                and (len(e.setup_sec) != n or e.destination not in self.stations)
            )
            or any(not math.isfinite(x) or x < 0 for x in e.setup_sec)
            for e in self.edges
        ):
            raise ValueError("Invalid ground acquisition envelope")
        if (
            self.at_sec >= self.horizon_sec
            or abs(
                self.at_sec / self.decision_sec - round(self.at_sec / self.decision_sec)
            )
            > 1e-8
            or (self.recovery and set(self.recovery) != ids)
            or any(s.owner not in self.actors for s in self.recovery.values())
            or (self.stored_bits and set(self.stored_bits) != set(self.actors))
            or any(not math.isfinite(x) or x < 0 for x in self.stored_bits.values())
        ):
            raise ValueError("invalid recovery state/grid/inventory")
        if len(ids) != len(self.demands):
            raise ValueError("duplicate demand")
        for r in self.demands:
            if (
                len(set(r.owners)) != len(r.owners)
                or len(set(r.stations)) != len(r.stations)
                or not set(r.owners) <= set(self.actors)
                or not set(r.stations) <= set(self.stations)
                or not r.release_sec < r.deadline_sec <= self.horizon_sec
            ):
                raise ValueError("unknown demand nodes or invalid deadline")
        if len({(e.source, e.destination) for e in self.edges}) != len(
            self.edges
        ) or any(
            e.source not in self.actors
            or e.destination not in (set(self.actors) | set(self.stations))
            or e.source == e.destination
            or len(e.cells) != n
            for e in self.edges
        ):
            raise ValueError("invalid directed edge")
        expected = {(r.id, a) for r in self.demands for a in r.owners}
        if (
            {(c.request, c.actor) for c in self.captures} != expected
            or len(self.captures) != len(expected)
            or any(len(c.cells) != n for c in self.captures)
        ):
            raise ValueError("incomplete capture opportunities")
        return self


def ground_width(request, edge, start, decision_sec):
    setup = edge.get("setup_sec", ())
    seconds = request.get("ground_seconds") or request["ground_cells"] * decision_sec
    return max(
        request["ground_cells"],
        math.ceil((seconds + (setup[start] if setup else 0.0)) / decision_sec),
    )


def schedule_network(value, *, max_nodes=50000, max_hops=3, request_order=None):
    if (
        type(max_nodes) is not int
        or not 1 <= max_nodes <= 500000
        or type(max_hops) is not int
        or not 0 <= max_hops <= 3
    ):
        raise ValueError("invalid joint search budgets")
    v = copy.deepcopy(value)
    validated = JointNetworkView.model_validate(v)
    if v["schema"] != "space-env.joint-network-view/1" or v["sha256"] != digest(
        {k: x for k, x in v.items() if k != "sha256"}
    ):
        raise ValueError("invalid network forecast seal")
    v = validated.model_dump(mode="json", by_alias=True)
    ids = [r["id"] for r in v["demands"]]
    if request_order is not None and (
        not isinstance(request_order, (list, tuple))
        or any(type(x) is not str for x in request_order)
        or len(request_order) != len(ids)
        or set(request_order) != set(ids)
    ):
        raise ValueError("request_order must be a permutation of every request")
    ordered = (
        sorted(v["demands"], key=lambda r: request_order.index(r["id"]))
        if request_order is not None
        else sorted(
            v["demands"], key=lambda r: (-r["priority"], r["deadline_sec"], r["id"])
        )
    )
    q = v["decision_sec"]
    n = round(v["horizon_sec"] / q)
    actors = sorted(v["actors"])
    edges = {(e["source"], e["destination"]): e["cells"] for e in v["edges"]}
    edge_rows = {(e["source"], e["destination"]): e for e in v["edges"]}
    captures = {(c["request"], c["actor"]): c["cells"] for c in v["captures"]}
    occupied = {a: set() for a in actors}
    terminals = {s: [0] * n for s in v["stations"]}
    memory = {a: v["stored_bits"].get(a, 0.0) for a in actors}
    energy = dict.fromkeys(actors, 0.0)
    allocations, unserved = [], {}
    nodes = 0
    exhausted = False

    def tick():
        nonlocal nodes
        if nodes >= max_nodes:
            raise StopIteration
        nodes += 1

    def available(a, start, width):
        return not any(i in occupied[a] for i in range(start, start + width))

    def paths(owner):
        others = [a for a in actors if a != owner]
        for hops in range(min(max_hops, len(others)) + 1):
            for tail in itertools.permutations(others, hops):
                path = (owner, *tail)
                if all((a, b) in edges for a, b in itertools.pairwise(path)):
                    yield path

    def candidate(r, owner, path, station, first):
        tick()
        state = v["recovery"].get(r["id"])
        phase = state["phase"] if state else "fresh"
        sl, im, cp = r["slew_cells"], r["capture_cells"], r["cpu_cells"]
        if phase == "raw":
            sl = im = 0
            cp = math.ceil(
                (
                    state["remaining_cpu_sec"]
                    + (r["raw_bits"] + r["output_bits"]) / 1e8
                    + 1.0
                    + 1e-6
                )
                / q
            )
        elif phase == "final":
            sl = im = cp = 0
            if not state["intact"] and len(path) > 1:
                return None
        cap = first + sl
        ready = cap + im + cp
        deadline = min(n, math.floor((r["deadline_sec"] + 1e-8) / q))
        if (
            ready >= deadline
            or not available(owner, first, sl + im + cp)
            or (phase == "fresh" and not captures[r["id"], owner][cap])
        ):
            return None
        mem = dict.fromkeys(actors, 0)
        cost = dict.fromkeys(actors, 0.0)
        mem[owner] = (
            r["raw_bits"] + r["output_bits"] + 100000
            if phase == "fresh"
            else r["output_bits"] + 100000
            if phase == "raw"
            else 0
        )
        cost[owner] = r["owner_wh"] if phase != "final" else 0.0
        stages = [
            {"kind": k, "actor": owner, "cell": start, "cells": width}
            for k, start, width in (
                ("slew", first, sl),
                ("imaging", cap, im),
                ("transform", cap + im, cp),
            )
            if width
        ]
        for left, right in itertools.pairwise(path):
            width = r["relay_cells"]
            chosen = None
            for start in range(ready, deadline - width + 1):
                tick()
                if (
                    available(left, start, width)
                    and available(right, start, width)
                    and all(edges[left, right][i] for i in range(start, start + width))
                ):
                    chosen = start
                    break
            if chosen is None:
                return None
            stages.append(
                {
                    "kind": "relay",
                    "actor": left,
                    "destination": right,
                    "cell": chosen,
                    "cells": width,
                }
            )
            ready = chosen + width
            mem[right] = max(mem[right], r["output_bits"])
            cost[left] += r["relay_wh"]
            cost[right] += r["receiver_wh"]
        last = path[-1]
        chosen = None
        for start in range(ready, deadline - r["ground_cells"] + 1):
            tick()
            width = ground_width(r, edge_rows.get((last, station), {}), start, q)
            if (
                start + width <= deadline
                and available(last, start, width)
                and all(
                    edges.get((last, station), [False] * n)[i]
                    for i in range(start, start + width)
                )
                and all(
                    terminals[station][i] < v["stations"][station]
                    for i in range(start, start + width)
                )
            ):
                chosen = start
                break
        if chosen is None:
            return None
        width = ground_width(r, edge_rows[last, station], chosen, q)
        cost[last] += r["ground_wh"] * width / r["ground_cells"]
        if any(
            memory[a] + mem[a] > v["actors"][a]["memory_bits"]
            or energy[a] + cost[a] > v["actors"][a]["usable_wh"] + 1e-12
            for a in actors
        ):
            return None
        stages.append(
            {
                "kind": "downlink",
                "actor": last,
                "destination": station,
                "cell": chosen,
                "cells": width,
            }
        )
        return {
            "request": r["id"],
            "owner": owner,
            "station": station,
            "path": list(path),
            "finish_sec": (chosen + width) * q,
            "stages": stages,
            "memory_bits": mem,
            "energy_wh": cost,
        }

    for r in ordered:
        state = v["recovery"].get(r["id"])
        if state and state["phase"] in {"completed", "terminal", "blocked"}:
            if state["phase"] != "completed":
                unserved[r["id"]] = state["reason"] or "terminal_native_state"
            continue
        best = None
        try:
            owners = [state["owner"]] if state else sorted(r["owners"])
            for owner in owners:
                for path in paths(owner):
                    tick()
                    for station in sorted(r["stations"]):
                        for first in range(
                            math.ceil(max(r["release_sec"], v["at_sec"]) / q), n
                        ):
                            proposal = candidate(r, owner, path, station, first)
                            if proposal is not None:
                                key = (
                                    proposal["finish_sec"],
                                    len(path),
                                    owner,
                                    station,
                                    path,
                                )
                                if best is None or key < best[0]:
                                    best = (key, proposal)
        except StopIteration:
            exhausted = True
        if best is None:
            unserved[r["id"]] = (
                "search_budget_exhausted"
                if exhausted
                else "no_complete_resource_feasible_route"
            )
            continue
        a = best[1]
        allocations.append(a)
        for actor in actors:
            memory[actor] += a["memory_bits"][actor]
            energy[actor] += a["energy_wh"][actor]
        for stage in a["stages"]:
            span = range(stage["cell"], stage["cell"] + stage["cells"])
            occupied[stage["actor"]].update(span)
            if stage["kind"] == "relay":
                occupied[stage["destination"]].update(span)
            if stage["kind"] == "downlink":
                for i in span:
                    terminals[a["station"]][i] += 1
    body = {
        "schema": "space-env.joint-network-schedule/1",
        "forecast_sha256": v["sha256"],
        "allocations": allocations,
        "unserved": unserved,
        "predicted_fulfilled": len(allocations)
        + sum(s["phase"] == "completed" for s in v["recovery"].values()),
        "completed": [
            rid for rid, s in v["recovery"].items() if s["phase"] == "completed"
        ],
        "total": len(v["demands"]),
        "search_nodes": nodes,
        "search_budget_exhausted": exhausted,
        "max_search_nodes": max_nodes,
        "max_hops": max_hops,
        "request_order": [r["id"] for r in ordered],
        "scope": "bounded serial joint owner/path/time packing; whole-product custody; conservative total memory and energy; no global optimum or guaranteed native success",
    }
    return body | {"sha256": digest(body)}
