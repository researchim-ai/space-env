"""Current centralized relay telemetry; no admission attempts or future rollout."""

from __future__ import annotations

import math

from space_env.industrial.route_progress import route_progress


def relay_route_observation(runtime, catalogue, *, topology_catalogue=None):
    mission = getattr(runtime._backend, "mission", None)
    if mission is None or mission.dtn is None:
        return None
    commands = [
        (row["id"], c)
        for row in catalogue
        for c in row.get("commands", [])
        if c["kind"] == "relay"
    ]
    progress = route_progress(
        catalogue if topology_catalogue is None else topology_catalogue
    )
    actors = sorted({c[k] for _, c in commands for k in ("actor_id", "destination_id")})
    views = {a: runtime.observe(a)["actors"][a] for a in actors}
    inventory = {a: runtime.inventory(a) for a in actors}
    busy = {
        r["command"]["actor_id"]
        for r in runtime.command_status()
        if r["status"] in {"accepted", "running"}
        and r["command"]["kind"] not in {"wait", "relay"}
    }
    result = []
    for action, c in commands:
        source, destination = c["actor_id"], c["destination_id"]
        if mission.dtn.link(source, destination) is None:
            continue
        view = views[destination]
        jobs = view.get("lifecycle", {}).get("jobs", [])
        workspace = sum(
            j["workspace_bits"] + sum(p["size_bits"] for p in j["outputs"])
            for j in jobs
            if j.get("reserved") and j.get("actor") == destination
        )
        free = max(
            0.0,
            inventory[destination]["memory_capacity_bits"]
            - view["stored_bits"]
            - view.get("dtn", {}).get("reserved_bits", 0.0)
            - workspace,
        )
        size = next(
            (
                p["size_bits"]
                for p in inventory[source]["products"]
                if p["id"] == c["product_id"]
            ),
            None,
        )
        budget = mission.dtn.budget(
            mission, {"source": source, "destination": destination}
        )
        reason = "available"
        for actor, label in [(source, "source"), (destination, "receiver")]:
            bus = views[actor].get("spacecraft_bus", {})
            if bus.get("safe") or {"bus", "radio"} & set(bus.get("faults", [])):
                reason = label + "_unavailable"
                break
        if reason == "available":
            if destination in busy or mission.dtn.receiving(destination):
                reason = "receiver_busy"
            elif (
                view["soc"] <= mission.dtn.config.endpoints[destination].min_soc + 1e-12
            ):
                reason = "receiver_soc"
            elif size is None or size > free + 1e-9:
                reason = "receiver_memory"
            elif budget["rate_bps"] <= 0:
                reason = budget["reason"]
        result.append(
            {
                "action_id": action,
                **progress[action],
                "source": source,
                "destination": destination,
                "product_id": c["product_id"],
                "available": reason == "available",
                "reason": reason,
                "receiver_free_bits": free,
                "rate_bps": budget["rate_bps"],
                "range_m": budget["range_m"],
                "measured_at": runtime.time_sec,
                "link_sample_sec": math.floor(
                    (runtime.time_sec + 1e-9) / mission.dtn.config.sample_sec
                )
                * mission.dtn.config.sample_sec,
            }
        )
    return result
