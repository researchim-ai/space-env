"""Independent subset-DP oracle for a strictly bounded nonpreemptive calendar class."""
from __future__ import annotations

import copy
import math
from fractions import Fraction
from functools import lru_cache

from space_env.reference.operations import ReferenceDownlinkMission
from space_env.reference.scheduling import mission_fingerprint, plan_downlinks

SCOPE = "single-terminal-nonpreemptive-static-zero-power/1"


def _is_nonpreemptive(report):
    outcomes = {p["id"]: p["status"] for p in report["products"]}
    last = {}
    for r in report["receipts"]:
        name = r["product_id"]
        if outcomes[name] != "delivered_on_time":
            return False
        if name in last and abs(r["start_sec"]-last[name]) > 4*max(math.ulp(r["start_sec"]), math.ulp(last[name])):
            return False
        last[name] = r["end_sec"]
    return True


def _problem(spec):
    mission = ReferenceDownlinkMission(spec)
    if len(mission.stations) != 1 or next(iter(mission.stations.values())).terminals != 1:
        raise ValueError("oracle requires exactly one single-terminal station")
    if len(mission.products) > 16:
        raise ValueError("oracle supports at most 16 products")
    if mission.horizon_sec > 3600:
        raise ValueError("oracle calendar horizon is limited to 3600 seconds")
    for v in mission.vehicles.values():
        if (v.base_power_w or v.tx_power_w or v.solar_area_m2 or v.thermal is not None
                or v.max_tx_sec is not None or v.initial_soc < v.min_tx_soc):
            raise ValueError("oracle requires zero-power, no solar/thermal/quota constraints")
    if any(p.processing_sec or p.depends_on for p in mission.products.values()):
        raise ValueError("oracle does not support processing or dependencies")
    station = next(iter(mission.stations))
    jobs = sorted(mission.products.values(), key=lambda p: p.id)
    if any(p.size_bits != int(p.size_bits) for p in jobs) or sum(int(p.size_bits) for p in jobs) > 2**53:
        raise ValueError("oracle requires integer bit sizes with total <= 2**53")
    if any(p.size_bits/mission.vehicles[p.actor].rate_bps <= 1e-9 for p in jobs):
        raise ValueError("oracle jobs must exceed executor step resolution")
    windows = {a: [{"start_sec": Fraction(w["start_sec"]), "end_sec": Fraction(w["end_sec"])}
                   for w in mission.service_windows(a, station)] for a in mission.actors}
    return mission, station, jobs, windows


@lru_cache(maxsize=256)
def _duration(bits, rate):
    return Fraction(bits)/Fraction(rate)


def _slot(job, after, windows, rate):
    duration = _duration(job.size_bits, rate)
    for window in windows[job.actor]:
        start = max(Fraction(after), Fraction(job.ready_sec), window["start_sec"])
        end = start+duration
        if end <= min(window["end_sec"], Fraction(job.deadline_sec)):
            return start, end
    return None


def _execute(mission, station, jobs, windows, order):
    clone = copy.deepcopy(mission)
    steps = []
    for i in order:
        p = jobs[i]
        slot = _slot(p, clone.clock, windows, clone.vehicles[p.actor].rate_bps)
        if slot is None:
            raise RuntimeError("oracle generated infeasible order")
        start, end = map(float, slot)
        if start > clone.clock+4*max(math.ulp(start), math.ulp(clone.clock)):
            gap = start-clone.clock
            if gap <= 1e-9:
                raise ValueError("calendar gap is below the executor's supported step resolution")
            steps.append({"dt_sec": gap, "actions": {}})
            clone.step(gap, {})
        row = {"dt_sec": end-start, "actions": {p.actor: station}, "products": {p.actor: p.id}}
        clone.step(row["dt_sec"], row["actions"], products=row["products"])
        steps.append(row)
    score = sum(int(jobs[i].size_bits) for i in order)
    report = clone.report()
    if report["on_time_bits"] != score:
        raise RuntimeError("calendar optimum disagrees with resource execution")
    return {"steps": steps, "initial_state_sha256": mission_fingerprint(mission), "start_sec": mission.clock}, report


def solve_small_downlinks(spec: dict, *, max_states: int = 65536) -> dict:
    """Prove optimum within SCOPE by earliest-finish dominance for each job subset.

    No simulator transitions are used in the DP. Geometry is supplied by the
    same reference access model; this is an independent scheduling algorithm,
    not an independent physical ephemeris or general resource-model oracle.
    """
    if isinstance(max_states, bool) or not isinstance(max_states, int) or not 1 <= max_states <= 65536:
        raise ValueError("max_states must be an integer in [1,65536]")
    mission, station, jobs, windows = _problem(spec)
    states = {0: (Fraction(0), ())}
    explored = 0
    # Successors always add a bit, so numeric mask order is topological.
    for mask in range(1 << len(jobs)):
        if mask not in states:
            continue
        end, path = states[mask]
        explored += 1
        for i, p in enumerate(jobs):
            if mask & (1 << i):
                continue
            slot = _slot(p, end, windows, mission.vehicles[p.actor].rate_bps)
            if slot is None:
                continue
            target, candidate = mask | (1 << i), (slot[1], path+(i,))
            if target not in states and len(states) >= max_states:
                raise ValueError("oracle state budget exhausted; no optimality certificate produced")
            if target not in states or candidate < states[target]:
                states[target] = candidate
    best = min(states, key=lambda mask: (-sum(int(p.size_bits) for i, p in enumerate(jobs) if mask & (1 << i)),
                                        states[mask][0], states[mask][1]))
    order = states[best][1]
    plan, report = _execute(mission, station, jobs, windows, order)
    # Matching-class priority/least-slack heuristic; no preemption or resource extras.
    pending, greedy, clock = set(range(len(jobs))), [], Fraction(0)
    while pending:
        feasible = [(i, _slot(jobs[i], clock, windows, mission.vehicles[jobs[i].actor].rate_bps)) for i in pending]
        feasible = [(i, span) for i, span in feasible if span is not None]
        if not feasible:
            break
        i, span = min(feasible, key=lambda pair: (-jobs[pair[0]].priority,
            Fraction(jobs[pair[0]].deadline_sec)-pair[1][1], jobs[pair[0]].deadline_sec, jobs[pair[0]].id))
        greedy.append(i)
        pending.remove(i)
        clock = span[1]
    greedy_plan, greedy_report = _execute(mission, station, jobs, windows, greedy)
    optimum, baseline = report["on_time_bits"], greedy_report["on_time_bits"]
    project = plan_downlinks(spec, select_products=True)
    comparable = _is_nonpreemptive(project["report"])
    project_bits = project["report"]["on_time_bits"]
    if comparable and project_bits > optimum:
        raise RuntimeError("project baseline contradicts the bounded optimum")
    return {"schema": "space-env.reference-calendar-oracle/1", "scope": SCOPE,
            "optimality_proven_in_scope": True, "general_resource_optimality": False,
            "objective": "completed_on_time_bits", "arithmetic": "exact rationals over supplied calendar floats", "states_explored": explored,
            "plan": plan, "report": report, "selected_products": [jobs[i].id for i in order],
            "baseline": {"algorithm": "nonpreemptive-priority-slack/1", "plan": greedy_plan, "report": greedy_report},
            "project_baseline": {"algorithm": project["algorithm"], "report": project["report"],
                                 "plan": project["plan"], "comparable": comparable,
                                 "regret_bits": optimum-project_bits if comparable else None,
                                 "comparison_scope": SCOPE if comparable else "preemption_or_uncompleted_transfers"},
            "regret_bits": optimum-baseline, "relative_regret": (optimum-baseline)/optimum if optimum else 0.}
