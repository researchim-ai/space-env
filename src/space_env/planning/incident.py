"""One committed operator incident and phase-aware native-prefix recovery.

Native replay is used only for already executed/committed history. Search over
future suffixes is value-only. Unknown future events/recursive incidents excluded.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

from pydantic import Field

from space_env.planning.chain import (
    ChainSettings,
    collect_chain_forecast,
    profile,
    validate_view,
    verify_chain_proposal,
)
from space_env.planning.mission_bundle import strict_json
from space_env.planning.mission_plan import PlanExecution, _sealed, make_mission_plan
from space_env.runtime import AccessPolicy
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSpec


class StationDisruption(StrictModel):
    station: str = Field(min_length=1, max_length=128)
    end_sec: float = Field(gt=0, le=21600, strict=True)


class IncidentSpec(StrictModel):
    disruptions: tuple[StationDisruption, ...] = Field(default=(), max_length=8)


class RecoveryState(StrictModel):
    rate_bps: float = Field(gt=0, strict=True)
    phase: Literal["fresh", "raw", "final", "completed", "terminal"]
    remaining_bits: float = Field(ge=0, strict=True)
    remaining_cpu_sec: float = Field(ge=0, strict=True)
    reason: str | None


def recovery_states(runtime, scenario, task):
    """Owned public telemetry; never infer actual progress from a predicted trace."""
    d, _, vehicles, _, _, chains = profile(scenario, task)
    local = runtime.observe()["actors"]
    requests = {r["id"]: r for a in local.values() for r in a["lifecycle"]["requests"]}
    jobs = {j["id"]: j for a in local.values() for j in a["lifecycle"]["jobs"]}
    inventories = {
        a: {p["id"]: p for p in runtime.inventory(a)["products"]} for a in local
    }
    definitions = {r["id"]: r for r in d["lifecycle"]["requests"]}
    result = {}
    for c in chains:
        r = definitions[c["id"]]
        a = c["actor"]
        state = requests.get(c["id"], {})
        job = next(
            j for j in d["lifecycle"]["jobs"] if j["inputs"] == [r["raw_product_id"]]
        )
        progress = jobs.get(job["id"], {})
        base = {
            "rate_bps": vehicles[a].rate_bps,
            "remaining_bits": 0.0,
            "remaining_cpu_sec": float(progress.get("remaining_sec", job["work_sec"])),
            "reason": state.get("reason"),
        }
        if state.get("status") == "completed":
            phase = "completed"
        elif state.get("status") in {"failed", "cancelled", "expired"}:
            phase = "terminal"
        elif r["delivery_product_id"] in inventories[a]:
            p = inventories[a][r["delivery_product_id"]]
            if p["cancelled"] or not p["deliverable"] or p["remaining_bits"] <= 0:
                phase = "terminal"
                base["reason"] = "final_product_unavailable_or_unacknowledged"
            else:
                phase = "final"
                base["remaining_bits"] = float(p["remaining_bits"])
        elif r["raw_product_id"] in inventories[a]:
            if progress.get("status") in {"failed", "cancelled", "expired"}:
                phase = "terminal"
                base["reason"] = "transform_terminal"
            else:
                phase = "raw"
        elif state.get("attempts", 0):
            phase = "terminal"
            base["reason"] = "acquisition_attempt_consumed_without_live_raw"
        else:
            phase = "fresh"
        result[c["id"]] = {"phase": phase, **base}
    return result


def schedule_recovery(forecast, states, settings=None):
    """Greedy remaining-stage calendar. Does not call any native engine method."""
    v = strict_json(canonical_json(forecast))
    validate_view(v)
    if not _sealed(v):
        raise ValueError("invalid recovery forecast seal")
    cfg = ChainSettings.model_validate(settings or {})
    state = {k: RecoveryState.model_validate(x) for k, x in states.items()}
    chains = v["chains"]
    ids = {c["id"] for c in chains}
    if set(state) != ids:
        raise ValueError("recovery must retain every request")
    if {k for k, x in state.items() if x.phase == "completed"} != set(v["completed"]):
        raise ValueError("recovery completion/forecast mismatch")
    q = v["decision_sec"]
    h = v["horizon_sec"]
    now = v["at_sec"]
    if (
        not 0 <= now < h
        or abs(now / q - round(now / q)) > 1e-8
        or abs(h / q - round(h / q)) > 1e-8
        or h / q > 10000
        or not 1 <= len(v["actors"]) <= 4
        or not 1 <= len(v["stations"]) <= 16
    ):
        raise ValueError("recovery grid budget exceeded")
    begin = round(now / q)
    n = round(h / q)
    nodes = 0
    contacts = {(x["actor"], x["station"]): x["windows"] for x in v["contacts"]}
    acquisitions = {x["request"]: x["windows"] for x in v["acquisitions"]}
    actor_cells = {a: set() for a in v["actors"]}
    station_cells = {s: [0] * n for s in v["stations"]}
    energy = dict.fromkeys(v["actors"], 0.0)
    tx_used = dict.fromkeys(v["actors"], 0.0)
    steps = [
        {"at_sec": i * q, "advance_sec": q, "actions": []} for i in range(begin, n)
    ]
    allocations = []
    unserved = {}

    def tick():
        nonlocal nodes
        nodes += 1
        if nodes > cfg.max_search_nodes:
            raise ValueError("recovery search budget exceeded")

    def inside(windows, start, end):
        for left, right in windows:
            tick()
            if left + (
                cfg.edge_guard_sec if left > now else 0
            ) <= start and end <= right - (cfg.edge_guard_sec if right < h else 0):
                return True
        return False

    for c in sorted(chains, key=lambda c: (c["deadline_sec"], -c["priority"], c["id"])):
        st = state[c["id"]]
        if st.phase == "completed":
            continue
        if st.phase == "terminal":
            unserved[c["id"]] = "terminal_native_state"
            continue
        a, s = c["actor"], c["station"]
        sl, im, cp, tx = c["durations_sec"]
        for duration in (sl, im, cp, tx):
            if duration <= 0 or abs(duration / q - round(duration / q)) > 1e-8:
                raise ValueError("invalid recovery stage duration")
        if st.phase == "final":
            if st.remaining_bits <= 0:
                raise ValueError("final recovery needs positive residual")
            rate = st.rate_bps
            tx = math.ceil(st.remaining_bits / rate / q) * q
        before = (
            sl + im + cp if st.phase == "fresh" else cp if st.phase == "raw" else 0.0
        )
        resource = v["actors"][a]
        cost = c["energy_wh"]
        if (
            resource["available_wh"] - resource["base_wh"] - energy[a] - cost
            < resource["battery_wh"] * c["min_soc"] + 1e-10
            or resource["tx_remaining_sec"] is not None
            and tx_used[a] + tx > resource["tx_remaining_sec"]
        ):
            unserved[c["id"]] = "conservative_energy_or_tx_quota"
            continue
        chosen = None
        for first in range(max(begin, math.ceil(c["release_sec"] / q)), n):
            tick()
            start = first * q
            ready = start + before
            if ready + tx > min(h, c["deadline_sec"]):
                break
            if st.phase == "fresh" and not inside(
                acquisitions.get(c["id"], []),
                start + sl,
                start + sl + c["acquisition_sec"],
            ):
                continue
            for cell in range(round(ready / q), n - round(tx / q) + 1):
                tick()
                end = cell * q + tx
                if end > c["deadline_sec"]:
                    break
                if any(i in actor_cells[a] for i in range(first, cell + round(tx / q))):
                    break
                if any(
                    station_cells[s][i] >= v["stations"][s]
                    for i in range(cell, cell + round(tx / q))
                ):
                    continue
                if inside(contacts[a, s], cell * q, end):
                    chosen = (start, cell * q, end)
                    break
            if chosen:
                break
        if chosen is None:
            unserved[c["id"]] = "no_remaining_stage_slot"
            continue
        start, delivery, end = chosen
        stages = []
        if st.phase == "fresh":
            stages += [
                ("slew", start, sl),
                ("imaging", start + sl, c["acquisition_sec"]),
                ("transform", start + sl + im, cp),
            ]
        elif st.phase == "raw":
            stages += [("transform", start, cp)]
        stages += [("downlink", delivery, tx)]
        for kind, at, duration in stages:
            steps[round(at / q) - begin]["actions"].append(
                {"request": c["id"], "kind": kind, "duration_sec": duration}
            )
        for i in range(round(start / q), round(end / q)):
            actor_cells[a].add(i)
        for i in range(round(delivery / q), round(end / q)):
            station_cells[s][i] += 1
        energy[a] += cost
        tx_used[a] += tx
        allocations.append(
            {
                "request": c["id"],
                "phase": st.phase,
                "start_sec": start,
                "delivery_sec": delivery,
                "finish_sec": end,
                "pointing_at_sec": start + sl + c["acquisition_sec"] / 2
                if st.phase == "fresh"
                else None,
            }
        )
    body = {
        "schema": "space-env.incident-schedule/1",
        "forecast_sha256": v["sha256"],
        "states_sha256": digest(states),
        "settings": cfg.model_dump(mode="json"),
        "steps": steps,
        "allocations": allocations,
        "unserved": unserved,
        "predicted_fulfilled": len(v["completed"]) + len(allocations),
        "total": len(chains),
        "search_nodes": nodes,
        "scope": "remaining-stage conservative greedy proposal; no optimality/expert/thermal guarantee",
    }
    return body | {"sha256": digest(body)}


def _bare_steps(plan, count=None):
    return [
        {k: s[k] for k in ("at_sec", "advance_sec", "action_ids")}
        for s in plan["steps"][:count]
    ]


def recover_incident(proposal, record, incident=None, *, settings=None):
    """Operator-only one-incident branch with native-verified committed prefix."""
    verify_chain_proposal(proposal, native=True)
    return _recover_epoch(proposal, record, incident, settings=settings)


def _recover_epoch(
    proposal,
    record,
    incident=None,
    *,
    settings=None,
    history=False,
    intent_scenario=None,
    demand_updates=(),
    catalogue_additions=(),
    demand_goals=None,
):
    """Trusted kernel; journal validates ancestry before enabling history."""
    old = PlanExecution.from_checkpoint(proposal["plan"], record)
    if record["error"] is not None or old.complete or old.cursor == 0:
        raise ValueError("incident requires successful nonempty unfinished prefix")
    spec = IncidentSpec.model_validate(incident or {})
    scenario = proposal["plan"]["scenario"]
    original = proposal["source_task"]
    intent = scenario if intent_scenario is None else intent_scenario
    d, t, _, _, stations, chains = profile(intent, original)
    now = old._session._runtime.time_sec
    q = t.decision_sec
    h = d["horizon_sec"]
    if len({x.station for x in spec.disruptions}) != len(spec.disruptions) or any(
        x.station not in stations for x in spec.disruptions
    ):
        raise ValueError("incident station unknown or duplicated")
    active = sorted(
        {
            r["command"]["actor_id"]
            for r in old._session._runtime.command_status()
            if r["status"] in {"accepted", "running"}
        }
    )
    if not active and not spec.disruptions and not demand_updates:
        raise ValueError("incident needs active commands or station disruptions")
    task = copy.deepcopy(proposal["plan"]["task"])
    known_ids = {a["id"] for a in task["actions"]}
    for action in catalogue_additions:
        if action["id"] in known_ids:
            raise ValueError("demand catalogue action collision")
        known_ids.add(action["id"])
        task["actions"].append(copy.deepcopy(action))
    tag = digest(
        {
            "plan": proposal["plan"]["sha256"],
            "cursor": old.cursor,
            "incident": spec.model_dump(mode="json"),
        }
    )[:16]
    if demand_updates:
        tag = digest({"parent_tag": tag, "demand": demand_updates})[:16]
    prefix = "@incident/" + tag + "/"
    operator_role = {"actors": [], "stations": sorted(stations), "action_slots": 8}
    if history:
        if task["roles"].get("@incident") != operator_role:
            raise ValueError(
                "journal operator authority differs from committed history"
            )
    else:
        if any(a["id"].startswith("@incident/") for a in task["actions"]):
            raise ValueError("nested incident unsupported")
        if "@incident" in task["roles"]:
            raise ValueError("reserved incident role already present")
        task["roles"]["@incident"] = operator_role
    if demand_updates:
        authority = {
            "actors": [],
            "demand_actors": sorted(d["orbits"]),
            "demand_stations": sorted(stations),
            "stations": [],
            "action_slots": 4,
        }
        if "@demand" in task["roles"] and task["roles"]["@demand"] != authority:
            raise ValueError("demand authority differs from committed history")
        task["roles"]["@demand"] = authority
    task["max_joint_actions"] = max(
        task["max_joint_actions"],
        sum(r["action_slots"] for r in task["roles"].values()),
    )
    controls = []
    if active:
        ids = []
        for actor in active:
            role = next(k for k, v in task["roles"].items() if v["actors"] == [actor])
            aid = prefix + "cancel-" + actor
            ids.append(aid)
            task["actions"].append(
                {"id": aid, "kind": "cancel_actor", "role": role, "actor": actor}
            )
        controls.append({"at_sec": now, "advance_sec": q, "action_ids": ids})
    event_time = now + len(controls) * q
    if spec.disruptions:
        ids = []
        for i, x in enumerate(spec.disruptions):
            if not event_time < x.end_sec <= h:
                raise ValueError(
                    "incident outage must end after committed event within horizon"
                )
            aid = prefix + "outage-" + str(i)
            ids.append(aid)
            task["actions"].append(
                {
                    "id": aid,
                    "kind": "inject",
                    "role": "@incident",
                    "event": {
                        "type": "station_outage_added",
                        "payload": {
                            "station": x.station,
                            "start_sec": event_time,
                            "end_sec": x.end_sec,
                        },
                    },
                }
            )
        controls.append({"at_sec": event_time, "advance_sec": q, "action_ids": ids})
    if demand_updates:
        at = now + len(controls) * q
        ids = []
        for i, payload in enumerate(demand_updates):
            aid = prefix + "demand-" + str(i)
            ids.append(aid)
            task["actions"].append(
                {
                    "id": aid,
                    "kind": "inject",
                    "role": "@demand",
                    "event": {
                        "type": "request_added",
                        "payload": copy.deepcopy(payload),
                    },
                }
            )
        controls.append({"at_sec": at, "advance_sec": q, "action_ids": ids})
    stop = now + len(controls) * q
    if stop >= h:
        raise ValueError("incident leaves no future planning cell")
    task["max_actions"] = len(task["actions"]) + 1
    task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    frozen = _bare_steps(proposal["plan"], old.cursor) + controls
    waits = [
        {"at_sec": i * q, "advance_sec": q, "action_ids": ["wait"]}
        for i in range(round(stop / q), round(h / q))
    ]
    goals = proposal["plan"]["goals"] if demand_goals is None else demand_goals
    interim = make_mission_plan(
        scenario,
        task,
        frozen + waits,
        goals=goals,
        episode_seed=proposal["plan"]["episode_seed"],
    )
    committed = PlanExecution(interim)
    committed.run(decisions=old.cursor)
    physical = digest(record["episode"]["runtime"]["backend_report"])
    if (
        digest(committed.checkpoint()["episode"]["runtime"]["backend_report"])
        != physical
    ):
        raise ValueError("incident catalogue changes executed physical prefix")
    for _ in controls:
        transition = committed.step()
        if not transition["feedback"]["accepted"]:
            raise ValueError("committed incident rejected")
    commit_record = committed.checkpoint()
    runtime = committed._session._runtime
    runtime.access = AccessPolicy(truth=True, forecast=True)
    states = recovery_states(runtime, intent, original)
    completed = [k for k, v in states.items() if v["phase"] == "completed"]
    view = collect_chain_forecast(
        runtime, intent, original, completed=completed, _runtime_scenario=scenario
    )
    schedule = schedule_recovery(view, states, settings)
    pointings = {
        a["request"]: runtime.forecast_pointing(
            next(c["actor"] for c in chains if c["id"] == a["request"]),
            next(c["target"] for c in chains if c["id"] == a["request"]),
            a["pointing_at_sec"],
        )
        for a in schedule["allocations"]
        if a["phase"] == "fresh"
    }
    originals = {a["id"]: a for a in task["actions"]}
    by_id = {c["id"]: c for c in chains}
    aliases = {}
    suffix = []
    for step in schedule["steps"]:
        ids = []
        for action in step["actions"]:
            rid, kind = action["request"], action["kind"]
            c = by_id[rid]
            key = (rid, kind)
            if key not in aliases:
                new = copy.deepcopy(originals[c["actions"][kind]])
                aid = prefix + str(chains.index(c)) + "-" + kind
                new["id"] = aid
                new["commands"][0]["duration_sec"] = action["duration_sec"]
                if kind == "slew":
                    new["commands"][0]["slew"] = {
                        "frame": "GCRS",
                        "quaternion_wxyz": pointings[rid]["goal_quaternion_wxyz"],
                    }
                aliases[key] = aid
                task["actions"].append(new)
            ids.append(aliases[key])
        suffix.append(
            {
                "at_sec": step["at_sec"],
                "advance_sec": q,
                "action_ids": sorted(ids) or ["wait"],
            }
        )
    task["max_actions"] = len(task["actions"]) + 1
    plan = make_mission_plan(
        scenario,
        task,
        frozen + suffix,
        goals=goals,
        episode_seed=proposal["plan"]["episode_seed"],
    )
    check = PlanExecution(plan)
    check.run(decisions=len(frozen))
    commit_sha = digest(commit_record["episode"]["runtime"]["backend_report"])
    if digest(check.checkpoint()["episode"]["runtime"]["backend_report"]) != commit_sha:
        raise ValueError("recovery aliases change committed native prefix")
    body = {
        "schema": "space-env.incident-recovery/1",
        "source_proposal": copy.deepcopy(proposal),
        "source_record": copy.deepcopy(record),
        "incident": spec.model_dump(mode="json"),
        "plan": plan,
        "forecast": view,
        "states": states,
        "schedule": schedule,
        "pointings": pointings,
        "committed_controls": controls,
        "source_prefix_sha256": physical,
        "committed_prefix_sha256": commit_sha,
        "committed_prefix_steps": len(frozen),
        "scope": "single committed incident; public residual state; fixed original demand; no future-event oracle",
    }
    return body | {"sha256": digest(body)}


def verify_incident(value):
    p = strict_json(canonical_json(value))
    if not _sealed(p) or p.get("schema") != "space-env.incident-recovery/1":
        raise ValueError("invalid incident envelope")
    fresh = recover_incident(
        p["source_proposal"],
        p["source_record"],
        p["incident"],
        settings=p["schedule"]["settings"],
    )
    if canonical_json(fresh) != canonical_json(p):
        raise ValueError("incident native recomputation mismatch")
    return {
        "integrity_passed": True,
        "committed_prefix_native_replayed": True,
        "forecasts_recomputed": True,
        "predicted_fulfilled": p["schedule"]["predicted_fulfilled"],
        "total": p["schedule"]["total"],
        "native_feasibility_validated": False,
        "expert_teacher_qualified": False,
    }
