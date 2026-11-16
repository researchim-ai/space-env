"""Current centralized RF graph and an observation-only, no-revisit baseline."""

import copy
from collections import defaultdict, deque

from space_env.industrial.campaign_policy import CampaignPolicySpec, campaign_policy
from space_env.tasks.observation import STATES


def detour_observation(runtime, catalogue):
    mission = runtime._backend.mission
    commands = [
        (a["id"], c)
        for a in catalogue
        for c in a.get("commands", [])
        if c["kind"] == "relay"
    ]
    actors = {c[k] for _, c in commands for k in ("actor_id", "destination_id")}
    views = {a: runtime.observe(a)["actors"][a] for a in sorted(actors)}
    links = []
    for action, command in commands:
        source, destination = command["actor_id"], command["destination_id"]
        if mission.dtn.link(source, destination) is None:
            continue
        budget = mission.dtn.budget(
            mission, {"source": source, "destination": destination}
        )
        healthy = all(
            not (
                views[a].get("spacecraft_bus", {}).get("safe")
                or {"bus", "radio"}
                & set(views[a].get("spacecraft_bus", {}).get("faults", []))
            )
            for a in (source, destination)
        )
        links.append(
            {
                "action_id": action,
                "source": source,
                "destination": destination,
                "product_id": command["product_id"],
                "rf_available": healthy and budget["rate_bps"] > 0,
                "rate_bps": budget["rate_bps"],
                "reason": budget["reason"] if healthy else "endpoint_unavailable",
            }
        )
    paths = {}
    for pid in sorted({c["product_id"] for _, c in commands}):
        if pid not in mission.products:
            continue
        history = [h for h in mission.dtn.custody_history if h["product_id"] == pid]
        paths[pid] = (
            [history[0]["source"]] + [h["destination"] for h in history]
            if history
            else [mission.products[pid].actor]
        )
    return {
        "measured_at": runtime.time_sec,
        "links": links,
        "custody_paths": paths,
        "receivers": {
            a: {"receiving": mission.dtn.receiving(a)} for a in sorted(actors)
        },
        "scope": "ideal current central RF/endpoint health and custody history; future resource availability and contact windows unknown",
    }


def detour_policy(observation, specification):
    """Prefer a currently reachable path without revisiting custody holders.

    Future lane occupancy/memory and ground access are not predicted. Command
    admission remains native; the main observation is ideal and centralized.
    """
    spec = CampaignPolicySpec.model_validate(specification)
    if not spec.adaptive_routing or spec.require_route_progress:
        raise ValueError(
            "detour policy requires adaptive routing without static progress gate"
        )
    view = copy.deepcopy(observation)
    graph = observation.get("detour_graph", {})
    if graph.get("measured_at") != observation["native"].get("time_sec"):
        graph = {}
    topology = observation.get("relay_topology", [])
    sinks = defaultdict(set)
    for edge in topology:
        if edge["kind"] == "downlink":
            sinks[edge["product_id"]].add(edge["actor_id"])
    links = {e["action_id"]: e for e in graph.get("links", [])}
    routes = {r["action_id"]: r for r in observation.get("relay_routes", [])}
    costs = {}
    for row in view["native"]["catalogue"]:
        if any(
            c.get("kind") not in {"relay", "wait"}
            and graph.get("receivers", {}).get(c.get("actor_id"), {}).get("receiving")
            for c in row.get("commands", [])
        ):
            row["enabled"] = False
        if row.get("command_kinds") != ["relay"] or not row.get("enabled"):
            continue
        link, route = links.get(row["id"], {}), routes.get(row["id"], {})
        if (
            not link.get("rf_available")
            or not route.get("available")
            or route.get("age_sec", 0) > spec.max_route_age_sec + 1e-9
        ):
            continue
        pid = link["product_id"]
        visited = graph.get("custody_paths", {}).get(pid, [])
        if (
            not visited
            or visited[-1] != link["source"]
            or link["destination"] in visited
        ):
            continue
        adjacency = defaultdict(set)
        for edge in links.values():
            if (
                edge["product_id"] == pid
                and edge["rf_available"]
                and edge["destination"] not in visited
            ):
                adjacency[edge["source"]].add(edge["destination"])
        queue = deque([(link["destination"], 1)])
        seen = set(visited) | {link["destination"]}
        while queue:
            actor, distance = queue.popleft()
            if actor in sinks[pid]:
                costs[row["id"]] = (pid, distance)
                break
            for successor in sorted(adjacency[actor] - seen):
                seen.add(successor)
                queue.append((successor, distance + 1))
    best = {}
    for pid, distance in costs.values():
        best[pid] = min(best.get(pid, distance), distance)
    for row in view["native"]["catalogue"]:
        if row.get("command_kinds") == ["relay"]:
            cost = costs.get(row["id"])
            row["enabled"] = bool(cost and cost[1] == best[cost[0]])
    result = campaign_policy(view, spec.model_dump())
    pending = {
        e["id"]
        for e in observation["native"]["entities"]
        if e["kind"] == "request"
        and e["values"].get("status_code")
        not in {STATES[k] for k in ("completed", "cancelled", "expired", "failed")}
    }
    if result["choices"]["@central"] == ["wait"] and any(
        r.get("command_kinds") == ["relay"] and r.get("request_ref") in pending
        for r in observation["native"]["catalogue"]
    ):
        # Link recovery need not be an exogenous task event or a command end.
        # Poll current telemetry, without reading a configured future outage end.
        result["advance_sec"] = min(result["advance_sec"], 1.0)
    return result
