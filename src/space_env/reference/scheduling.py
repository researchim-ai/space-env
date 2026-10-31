"""Deterministic, simulator-verified baseline scheduling; no optimality claim."""
from __future__ import annotations

import copy
import math

from space_env.reference.operations import ReferenceDownlinkMission


def plan_downlinks(spec: dict, *, decision_sec: float = 5., max_steps: int = 10000,
                   select_products: bool = False) -> dict:
    """Plan from t=0 with full known contacts/outages, then execute a fresh replay.

    Rank feasible products by priority, deadline slack, deadline and IDs. Default
    mode preserves onboard queue discipline; select_products permits overrides. Decisions are
    reconsidered at readiness/contact/outage/deadline and completion boundaries.
    This is a privileged deterministic baseline, not an expert or online agent.
    """
    return plan_remaining_downlinks(ReferenceDownlinkMission(spec), decision_sec=decision_sec,
                                    max_steps=max_steps, select_products=select_products)


def mission_fingerprint(mission: ReferenceDownlinkMission) -> str:
    """Bind continuation plans to exact spec, resources, delivery and history."""
    return _spec_digest({"spec": mission.spec, "report": mission.report(), "intervals": mission.intervals})


def plan_remaining_downlinks(source: ReferenceDownlinkMission, *, decision_sec: float = 5.,
                             max_steps: int = 10000, select_products: bool = True) -> dict:
    """Plan on private copies; never advance or modify the supplied mission."""
    if isinstance(decision_sec, bool) or not math.isfinite(decision_sec) or decision_sec <= 1e-9:
        raise ValueError("decision_sec must be finite and exceed 1ns")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(select_products, bool):
        raise TypeError("select_products must be boolean")
    mission = copy.deepcopy(source)
    initial_state = mission_fingerprint(source)
    steps = []
    while mission.clock < mission.horizon_sec-1e-9 and any(mission.remaining.values()):
        if len(steps) >= max_steps:
            raise ValueError("planning step budget exhausted; no complete plan produced")
        candidates = []
        for actor in sorted(mission.actors):
            config = mission.vehicles[actor]
            if not mission.transmitter_ready(actor) or mission.processing_job(actor) is not None:
                continue
            for station in sorted(mission.stations):
                if not mission.station_permitted(actor, station):
                    continue
                rate = mission.effective_rate(actor, station)
                if rate == 0:
                    continue
                queue = ([p for p in mission.products.values() if p.actor == actor
                          and mission.remaining[p.id] > 0 and p.ready_sec <= mission.clock+1e-9
                          and mission.processing_remaining[p.id] == 0
                          and not mission.retained_for(p.id)
                          and (not p.destination or p.destination == station)] if select_products
                         else [mission._product(actor, station)])
                queue = [p for p in queue if p is not None
                         and p.deadline_sec-mission.clock-mission.remaining[p.id]/rate >= -1e-9
                         and (config.max_tx_sec is None or mission.remaining[p.id]/rate
                              <= config.max_tx_sec-mission.tx_seconds[actor]+1e-10)]
                product = min(queue, key=lambda p: (-p.priority,
                              p.deadline_sec-mission.clock-mission.remaining[p.id]/rate,
                              p.deadline_sec, p.id)) if queue else None
                if product is None or not mission._visible(actor, station):
                    continue
                duration = mission.remaining[product.id]/rate
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
                                   actor, station, duration, product.id))
        actions, selections, used, durations = {}, {}, dict.fromkeys(mission.stations, 0), []
        open_used = dict.fromkeys(mission.stations, 0)
        for _rank, actor, station, duration, product_id in sorted(candidates):
            reserved, free = mission.booking_capacity(actor, station)
            admissible = reserved or (not mission.reservation_required[station] and open_used[station] < free)
            if actor not in actions and used[station] < mission.stations[station].terminals and admissible:
                actions[actor] = station
                selections[actor] = product_id
                used[station] += 1
                if not reserved:
                    open_used[station] += 1
                limits = [duration]
                config = mission.vehicles[actor]
                if config.max_tx_sec is not None:
                    limits.append(config.max_tx_sec-mission.tx_seconds[actor])
                if config.thermal:
                    limits.append(config.thermal.crossing(mission.temperature[actor], config.thermal.trip_k, True))
                durations.append(min(limits))
        right = min(mission.horizon_sec, mission.clock+decision_sec,
                    *[t for t in mission._boundaries if t > mission.clock+1e-9],
                    *[t for a in mission.actors if (t := mission.transmitter_resume_sec(a)) is not None
                      and math.isfinite(t) and t > mission.clock+1e-9],
                    *[mission.clock+mission.processing_remaining[p.id] for a in mission.actors
                      if (p := mission.processing_job(a)) is not None],
                    *[mission.clock+d for d in durations if d > 1e-9])
        dt = right-mission.clock
        mission.step(dt, actions, products=selections if select_products else None)
        row = {"dt_sec": dt, "actions": actions}
        if select_products:
            row["products"] = selections
        steps.append(row)
    # Verify from an independent copy of the initial state, not planning output.
    replay = copy.deepcopy(source)
    for row in steps:
        replay.step(row["dt_sec"], row["actions"], products=row.get("products"))
    report = mission.report()
    if report != replay.report():
        raise RuntimeError("schedule execution verification failed")
    plan = {"steps": steps, "initial_state_sha256": initial_state, "start_sec": source.clock}
    return {"schema": "space-env.reference-downlink-schedule/2",
            "algorithm": "priority-slack-explicit/1" if select_products else "priority-slack-greedy/1", "expert": False, "optimality_proven": False,
            "spec_sha256": _spec_digest(source.spec), "decision_sec": decision_sec,
            "initial_state_sha256": initial_state, "start_sec": source.clock,
            "plan": plan, "verified": True, "report": report,
            "unfulfilled": [{"id": p["id"], "status": p["status"],
                              "reason": "cancelled_by_operator" if p["status"] == "cancelled" else "not_completed_on_time_by_baseline"}
                             for p in report["products"] if p["status"] != "delivered_on_time"]}


def _spec_digest(spec: dict) -> str:
    from space_env.simulation.snapshot import digest

    return digest(spec)
