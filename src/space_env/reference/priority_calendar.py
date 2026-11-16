"""Exact weighted delivery oracle for a narrow static nonpreemptive calendar.

Geometry is shared with native access; this is independent scheduling logic,
not independently measured orbital truth or a full EO/ISL optimum.
"""

from __future__ import annotations

from fractions import Fraction

from space_env.reference.optimal import _execute, _problem, _slot
from space_env.simulation.snapshot import digest

SCOPE = "single-terminal-static-zero-power-priority/1"


def solve_priority_calendar(spec, *, max_states=65536):
    if type(max_states) is not int or not 1 <= max_states <= 65536:
        raise ValueError("state budget must be an integer in [1,65536]")
    if spec.get("radio_network") is not None or spec.get("dtn_network") is not None:
        raise ValueError("calendar certificate excludes RF and DTN networks")
    mission, station, jobs, windows = _problem(spec)
    # Same-subset earlier finish dominates later finish, regardless of objective.
    states = {0: (Fraction(0), ())}
    for mask in range(1 << len(jobs)):
        if mask not in states:
            continue
        finish, order = states[mask]
        for index, job in enumerate(jobs):
            if mask & (1 << index):
                continue
            span = _slot(job, finish, windows, mission.vehicles[job.actor].rate_bps)
            if span is None:
                continue
            target, candidate = mask | (1 << index), (span[1], order + (index,))
            if target not in states and len(states) >= max_states:
                raise ValueError("state budget exhausted; no optimality certificate")
            if target not in states or candidate < states[target]:
                states[target] = candidate

    def value(mask):
        return sum(j.priority for i, j in enumerate(jobs) if mask & (1 << i))

    best = min(
        states, key=lambda mask: (-value(mask), states[mask][0], states[mask][1])
    )
    plan, report = _execute(mission, station, jobs, windows, states[best][1])
    completed = {
        p["id"] for p in report["products"] if p["status"] == "delivered_on_time"
    }
    native_value = sum(j.priority for j in jobs if j.id in completed)
    if native_value != value(best):
        raise RuntimeError("native weighted outcome disagrees with oracle")
    pending, greedy, clock = set(range(len(jobs))), [], Fraction(0)
    while pending:
        feasible = [
            (
                i,
                _slot(
                    jobs[i], clock, windows, mission.vehicles[jobs[i].actor].rate_bps
                ),
            )
            for i in pending
        ]
        feasible = [(i, span) for i, span in feasible if span is not None]
        if not feasible:
            break
        index, span = min(
            feasible,
            key=lambda pair: (
                -jobs[pair[0]].priority,
                jobs[pair[0]].deadline_sec,
                jobs[pair[0]].id,
            ),
        )
        greedy.append(index)
        pending.remove(index)
        clock = span[1]
    greedy_plan, greedy_report = _execute(mission, station, jobs, windows, greedy)
    greedy_ids = {
        p["id"] for p in greedy_report["products"] if p["status"] == "delivered_on_time"
    }
    greedy_value = sum(j.priority for j in jobs if j.id in greedy_ids)
    return {
        "schema": "space-env.priority-calendar/1",
        "spec_sha256": digest(spec),
        "scope": SCOPE,
        "objective": "sum_on_time_product_priority",
        "optimality_proven_in_scope": True,
        "full_mission_optimality": False,
        "states": len(states),
        "max_states": max_states,
        "value": native_value,
        "selected_products": sorted(completed),
        "plan": plan,
        "report": report,
        "baseline": {
            "method": "highest_priority_nonpreemptive",
            "value": greedy_value,
            "plan": greedy_plan,
            "report": greedy_report,
        },
        "regret_priority": native_value - greedy_value,
        "assumptions": [
            "one terminal",
            "static actor calendars",
            "nonpreemptive whole products",
            "no processing, ISL, power, solar or thermal constraints",
            "shared geometry and exact rational calendar arithmetic",
        ],
    }
