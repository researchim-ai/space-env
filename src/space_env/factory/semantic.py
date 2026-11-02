"""Conservative typed identity-graph hashes; not an isomorphism/physics proof."""

from __future__ import annotations

import math

from space_env.simulation.snapshot import canonical_json, digest

_IGNORE = {
    "sha256",
    "binding",
    "seed",
    "episode_seed",
    "metadata",
    "sources",
    "description",
    "parameter_status",
    "policy_origin",
    "lineage",
    "reason",
    "uri",
    "path",
    "command_id",
}
_MAPS = {
    "vehicles": "actor",
    "ephemerides": "actor",
    "orbits": "actor",
    "spacecraft_bus": "actor",
    "payloads": "actor",
    "transmitters": "actor",
    "acquisition_sensors": "actor",
    "endpoints": "actor",
    "actors": "actor",
    "roles": "role",
    "receivers": "station",
    "weather": "station",
    "targets": "target",
    "acquisition_targets": "target",
}
_COLLECTIONS = {
    "stations": "station",
    "products": "product",
    "requests": "request",
    "jobs": "job",
    "outputs": "product",
    "targets": "target",
    "acquisition_targets": "target",
    "areas_of_interest": "aoi",
    "actions": "action",
    "modes": "mode",
}
_REFS = {
    "actor": "actor",
    "actor_id": "actor",
    "station": "station",
    "station_id": "station",
    "product_id": "product",
    "request_id": "request",
    "job_id": "job",
    "target_id": "target",
    "aoi_id": "aoi",
    "raw_product_id": "product",
    "delivery_product_id": "product",
    "role": "role",
    "target_actor": "actor",
    "target_station": "station",
}
_REF_LISTS = {
    "actor_order": "actor",
    "eligible_actors": "actor",
    "actors": "actor",
    "stations": "station",
    "inputs": "product",
    "depends_on": "product",
    "request_ids": "request",
}


def fingerprint(document, *, max_nodes=100000):
    canonical_json(document)
    colors, edges, symbols = [], [], {}

    def node(color):
        if len(colors) >= max_nodes:
            raise ValueError("semantic fingerprint node budget exceeded")
        colors.append(digest(color))
        edges.append([])
        return len(colors) - 1

    def symbol(kind, label):
        key = (kind, label)
        if key not in symbols:
            symbols[key] = node(["symbol", kind])
        return symbols[key]

    def link(a, b, label):
        edges[a].append(("out:" + label, b))
        edges[b].append(("in:" + label, a))

    def visit(v, key="", context="root"):
        if isinstance(v, dict):
            parent = node("object")
            if key in _MAPS:
                kind = _MAPS[key]
                for identity, child in v.items():
                    pair = node("named-entry")
                    link(parent, pair, "entry")
                    link(pair, symbol(kind, identity), "identity")
                    link(pair, visit(child, context=kind), "value")
                return parent
            for field, child in v.items():
                if field in _IGNORE:
                    continue
                if field == "source":
                    if not ("destination" in v and isinstance(child, str)):
                        continue
                    target = symbol("actor", child)
                elif isinstance(child, str) and child:
                    if field in {"id", "name"}:
                        target = symbol(context, child)
                    elif field in _REFS:
                        target = symbol(_REFS[field], child)
                    elif field in {"destination", "destination_id"}:
                        target = symbol(
                            "actor"
                            if v.get("kind") == "relay" or "source" in v
                            else "station",
                            child,
                        )
                    else:
                        target = visit(child, field, field)
                else:
                    target = visit(child, field, _COLLECTIONS.get(field, field))
                link(parent, target, field)
            return parent
        if isinstance(v, (list, tuple)):
            if all(type(x) in {int, float} for x in v):
                return node(["numeric-array", [float(x) or 0.0 for x in v]])
            parent = node("array")
            for i, child in enumerate(v):
                target = (
                    symbol(_REF_LISTS[key], child)
                    if key in _REF_LISTS and isinstance(child, str)
                    else visit(child, context=_COLLECTIONS.get(key, context))
                )
                link(parent, target, "member" if key in _COLLECTIONS else str(i))
            return parent
        if type(v) in {int, float}:
            if not math.isfinite(v):
                raise ValueError("nonfinite semantic value")
            v = float(v)
            if v == 0:
                v = 0.0
        return node([type(v).__name__, v])

    root = visit(document)
    initial = colors[:]
    for _ in range(8):
        colors = [
            digest([initial[i], sorted((label, colors[j]) for label, j in neighbors)])
            for i, neighbors in enumerate(edges)
        ]
    return digest(
        {
            "algorithm": "typed-identity-refinement/1",
            "root": colors[root],
            "nodes": sorted(colors),
        }
    )


def mission_fingerprint(scenario, task=None):
    scenario = {
        k: v
        for k, v in scenario.items()
        if k not in {"id", "seed", "lineage", "metadata", "sources"}
    }
    task = (
        None
        if task is None
        else {k: v for k, v in task.items() if k not in {"id", "policy_origin"}}
    )
    return fingerprint({"scenario": scenario, "task": task})


def split_group(scenario):
    """Conservatively group underlying geometry, not workloads/IDs/seeds/resources."""
    d = scenario["definition"]
    ops = d.get("operations", d)
    stations = [
        {k: v for k, v in s.items() if k != "terminals"} for s in ops["stations"]
    ]
    return fingerprint(
        {
            "backend": scenario["backend"],
            "trajectory": {"orbits": d["orbits"]}
            if "orbits" in d
            else {"ephemerides": ops.get("ephemerides")},
            "stations": stations,
            "targets": d.get("acquisition_targets", d.get("targets", [])),
            "areas_of_interest": d.get("lifecycle", {}).get("areas_of_interest", []),
        }
    )
