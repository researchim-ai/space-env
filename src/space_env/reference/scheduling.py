"""Deterministic, simulator-verified baseline scheduling; no optimality claim."""
from __future__ import annotations

import math

from space_env.reference.operations import ReferenceDownlinkMission


def plan_downlinks(spec: dict, *, decision_sec: float = 5., max_steps: int = 10000) -> dict:
    """Plan from t=0 with full known contacts/outages, then execute a fresh replay.

    Rank feasible queue heads by priority, deadline slack, deadline and IDs. A
    product cannot bypass the mission's onboard queue discipline. Decisions are
    reconsidered at readiness/contact/outage/deadline and completion boundaries.
    This is a privileged deterministic baseline, not an expert or online agent.
    """
    if isinstance(decision_sec, bool) or not math.isfinite(decision_sec) or decision_sec <= 1e-9:
        raise ValueError("decision_sec must be finite and exceed 1ns")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    mission = ReferenceDownlinkMission(spec)
    steps = []
    while mission.clock < mission.horizon_sec-1e-9 and any(mission.remaining.values()):
        if len(steps) >= max_steps:
            raise ValueError("planning step budget exhausted; no complete plan produced")
        candidates = []
        for actor in sorted(mission.actors):
            config = mission.vehicles[actor]
            for station in sorted(mission.stations):
                product = mission._product(actor, station)
                if product is None or not mission._visible(actor, station):
                    continue
                duration = mission.remaining[product.id]/config.rate_bps
                slack = product.deadline_sec-mission.clock-duration
                if slack < -1e-9:
                    continue
                sample = min((math.floor((mission.clock+1e-9)/mission.resource_step_sec)+.5)
                             * mission.resource_step_sec, mission.horizon_sec)
                net = mission._generation(actor, sample)-config.base_power_w-config.tx_power_w
                floor = config.min_tx_soc*config.battery_wh
                energy = mission.energy_wh[actor]
                if energy < floor-1e-12 or (energy <= floor+1e-12 and net < 0):
                    continue
                candidates.append(((-product.priority, slack, product.deadline_sec, actor, station),
                                   actor, station, duration))
        actions, used, durations = {}, dict.fromkeys(mission.stations, 0), []
        for _rank, actor, station, duration in sorted(candidates):
            if actor not in actions and used[station] < mission.stations[station].terminals:
                actions[actor] = station
                used[station] += 1
                durations.append(duration)
        right = min(mission.horizon_sec, mission.clock+decision_sec,
                    *[t for t in mission._boundaries if t > mission.clock+1e-9],
                    *[mission.clock+d for d in durations if d > 1e-9])
        dt = right-mission.clock
        mission.step(dt, actions)
        steps.append({"dt_sec": dt, "actions": actions})
    # Verify with newly constructed geometry/resources, not the planning state.
    replay = ReferenceDownlinkMission(spec)
    for row in steps:
        replay.step(row["dt_sec"], row["actions"])
    report = mission.report()
    if report != replay.report():
        raise RuntimeError("schedule execution verification failed")
    return {"schema": "space-env.reference-downlink-schedule/1",
            "algorithm": "priority-slack-greedy/1", "expert": False, "optimality_proven": False,
            "spec_sha256": _spec_digest(spec), "decision_sec": decision_sec,
            "plan": {"steps": steps}, "verified": True, "report": report,
            "unfulfilled": [{"id": p["id"], "status": p["status"],
                              "reason": "not_completed_on_time_by_baseline"}
                             for p in report["products"] if p["status"] != "delivered_on_time"]}


def _spec_digest(spec: dict) -> str:
    from space_env.simulation.snapshot import digest

    return digest(spec)
