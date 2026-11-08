"""Causal native mission changes and phase-aware continuation, preserving source."""

from __future__ import annotations

import copy
from typing import Literal

from pydantic import Field

from space_env.api.workspace import (
    _PLAN_LOCK,
    MissionOrder,
    _metadata,
    compile_configuration,
)
from space_env.api.workspace_replan import ReplanRequest
from space_env.planning.chain import collect_chain_forecast, profile
from space_env.planning.incident import recovery_states, schedule_recovery
from space_env.runtime import AccessPolicy
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession, TaskSpec


class MissionEvent(StrictModel):
    type: Literal[
        "request_cancelled",
        "request_priority_changed",
        "request_deadline_changed",
        "station_outage_added",
        "station_recovered",
        "spacecraft_fault",
        "spacecraft_fault_cleared",
        "spacecraft_safe_mode",
        "spacecraft_recover",
    ]
    payload: dict


class MissionChange(ReplanRequest):
    additions: tuple[MissionOrder, ...] = Field(default=(), max_length=4)
    events: tuple[MissionEvent, ...] = Field(default=(), max_length=8)


def evolve_workspace(store, session_id, token, payload):
    from space_env.api.operator import OperatorError, token_digest

    request = MissionChange.model_validate(payload)
    if not request.reason.strip():
        raise ValueError("Укажите причину изменения.")
    with store.lock:
        source = store.get(session_id)
        with source.lock, source.session._lock:
            role, write, _ = source.permission(token)
            if (
                role != "@central"
                or not write
                or source.identities[token_digest(token)] != "operator"
            ):
                raise OperatorError(
                    "forbidden",
                    "Изменения миссии доступны центральному оператору.",
                    403,
                )
            source._guard()
            key = digest({"source": session_id, "evolve": request.idempotency_key})
            sha = digest(request.model_dump(mode="json"))
            for candidate in store.sessions.values():
                old = (
                    candidate.session._scenario.get("metadata", {})
                    .get("workspace", {})
                    .get("dynamic")
                )
                if old and old["key_sha256"] == key and not candidate.closed:
                    if old["request_sha256"] != sha:
                        raise OperatorError(
                            "idempotency_conflict", "Ключ уже использован.", 409
                        )
                    if not candidate.credentials:
                        raise OperatorError(
                            "evolution_already_created",
                            f"Ветка {candidate.id} сохранена; используйте файл доступа.",
                            409,
                        )
                    return {
                        "session_id": candidate.id,
                        "tokens": candidate.credentials.copy(),
                        "state": candidate.state(candidate.credentials["operator"]),
                    }
            if (
                source.control["controller"] != "operator"
                or request.expected_revision != source.session.revision
                or request.control_epoch != source.control["epoch"]
            ):
                raise OperatorError(
                    "stale_control", "Версия или управляющий канал изменились.", 409
                )
            meta = _metadata(source)
            if meta is None or source.session._done:
                raise OperatorError(
                    "unsupported_evolution",
                    "Нужна незавершённая workspace-миссия.",
                    409,
                )
            q = source.session._task.decision_sec
            if (
                abs(
                    source.session._runtime.time_sec / q
                    - round(source.session._runtime.time_sec / q)
                )
                > 1e-8
            ):
                raise OperatorError(
                    "evolution_boundary", "Нужна граница сетки решений.", 409
                )
            if len(store.sessions) >= 8:
                raise OperatorError("capacity", "Закройте ненужные ветки.", 429)
            if not _PLAN_LOCK.acquire(blocking=False):
                raise OperatorError("planner_busy", "Планировщик занят.", 429)
            try:
                return _build(store, source, meta, request, key, sha)
            finally:
                _PLAN_LOCK.release()


def _replay(scenario, task, source):
    session = TaskSession(scenario, task, seed=source.session._seed)
    for row in source.session._transitions:
        r = session.step(**row["request"])
        if r["feedback"] != row["transition"]["feedback"]:
            raise ValueError("Изменился результат исторической команды.")
    if digest(session._runtime._backend.report()) != digest(
        source.session._runtime._backend.report()
    ):
        raise ValueError("Изменилось физическое состояние префикса.")
    return session


def _build(store, source, meta, request, key, sha):
    config = copy.deepcopy(meta.get("resolved_configuration", meta["configuration"]))
    now = source.session._runtime.time_sec
    q = source.session._task.decision_sec
    active = sorted(
        {
            r["command"]["actor_id"]
            for r in source.session._runtime.command_status()
            if r["status"] in {"accepted", "running"}
        }
    )
    event_at = now + (q if active else 0)
    events = [e.model_dump(mode="json") for e in request.events]
    known_coords = {(r["lat_deg"], r["lon_deg"]) for r in config["requests"]}
    for addition in request.additions:
        if addition.actor_id is None or addition.station is None:
            raise ValueError("Новая заявка требует явного аппарата и станции.")
        if (addition.lat_deg, addition.lon_deg) not in known_coords:
            raise ValueError(
                "Динамическая заявка требует уже объявленной точки наблюдения."
            )
        if addition.release_sec < event_at:
            raise ValueError(
                f"Выпуск новой заявки должен быть не раньше {event_at} с — момента события после прерывания."
            )
        config["requests"].append(addition.model_dump(mode="json"))
    for event in events:
        if event["type"] in {"request_priority_changed", "request_deadline_changed"}:
            rid = event["payload"].get("request_id")
            rows = [r for r in config["requests"] if r["id"] == rid]
            if not rows:
                raise ValueError("Неизвестная заявка.")
            field = (
                "priority"
                if event["type"] == "request_priority_changed"
                else "deadline_sec"
            )
            rows[0][field] = event["payload"].get(field)
    cfg, intent, templates = compile_configuration(config)
    for addition in request.additions:
        r = next(
            r
            for r in intent["definition"]["lifecycle"]["requests"]
            if r["id"] == addition.id
        )
        job = next(
            j
            for j in intent["definition"]["lifecycle"]["jobs"]
            if j["id"] == "cpu-" + addition.id
        )
        events.insert(
            0, {"type": "request_added", "payload": {"request": r, "jobs": [job]}}
        )
    if len(events) > 8:
        raise ValueError("Не более восьми событий за изменение, включая новые заявки.")
    stop = event_at + (q if events else 0)
    if stop >= cfg.horizon_sec:
        raise ValueError("События не оставляют времени для продолжения.")
    scenario = copy.deepcopy(source.session._original)
    task = source.session._task.model_dump(mode="json", by_alias=True)
    tag = "@change/" + key[:16] + "/"
    controls = []
    if events:
        task["roles"]["@mission"] = {
            "actors": [],
            "stations": [s.name for s in cfg.stations],
            "demand_actors": list(scenario["actor_order"]),
            "demand_stations": [s.name for s in cfg.stations],
            "action_slots": 8,
        }
    if active:
        ids = []
        for actor in active:
            aid = tag + "cancel/" + actor
            role = next(k for k, v in task["roles"].items() if v["actors"] == [actor])
            task["actions"].append(
                {"id": aid, "role": role, "kind": "cancel_actor", "actor": actor}
            )
            ids.append(aid)
        controls.append({"at_sec": now, "advance_sec": q, "action_ids": ids})
    if events:
        ids = []
        for i, event in enumerate(events):
            aid = tag + "event/" + str(i)
            event_role = "@mission"
            if event["type"].startswith("spacecraft_"):
                actor = event["payload"].get("actor")
                event_role = next(
                    (k for k, v in task["roles"].items() if v["actors"] == [actor]),
                    None,
                )
                if event_role is None:
                    raise ValueError("Неизвестный аппарат события.")
                task["roles"][event_role]["action_slots"] = 8
            task["actions"].append(
                {"id": aid, "role": event_role, "kind": "inject", "event": event}
            )
            ids.append(aid)
        controls.append({"at_sec": event_at, "advance_sec": q, "action_ids": ids})
    task["max_joint_actions"] = max(
        task["max_joint_actions"],
        sum(r["action_slots"] for r in task["roles"].values()),
    )
    task["max_actions"] = len(task["actions"]) + 1
    task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    committed = _replay(scenario, task, source)

    def apply_steps(session):
        for i, step in enumerate(controls):
            response = session.step(
                {"@central": step["action_ids"]},
                expected_revision=session.revision,
                idempotency_key=tag + str(i),
                advance_sec=q,
            )
            if not response["feedback"]["accepted"]:
                raise ValueError(
                    "Симулятор отклонил событие; исходная ветка сохранена."
                )

    apply_steps(committed)
    runtime = committed._runtime
    runtime.access = AccessPolicy(truth=True, forecast=True)
    states = recovery_states(runtime, intent, templates)
    bus = runtime._backend.report()["spacecraft_bus"]["states"]
    chains = profile(intent, templates)[-1]
    for c in chains:
        st = states[c["id"]]
        b = bus[c["actor"]]
        required = (
            {"bus", "radio"}
            | ({"cpu"} if st["phase"] in {"fresh", "raw"} else set())
            | ({"adcs", "payload"} if st["phase"] == "fresh" else set())
        )
        if st["phase"] not in {"terminal", "completed"} and (
            b["safe"] or required & set(b["faults"])
        ):
            st.update(phase="blocked", reason="current_spacecraft_unavailability")
    completed = [k for k, v in states.items() if v["phase"] == "completed"]
    view = collect_chain_forecast(
        runtime,
        intent,
        templates,
        completed=completed,
        _runtime_scenario=committed._scenario,
    )
    settings = (request.planner or cfg.planner).model_dump(mode="json")
    recovery = schedule_recovery(
        view, states, {k: settings[k] for k in ("max_search_nodes", "edge_guard_sec")}
    )
    originals = {a["id"]: a for a in templates["actions"]}
    by_id = {c["id"]: c for c in chains}
    points = {
        a["request"]: runtime.forecast_pointing(
            by_id[a["request"]]["actor"],
            by_id[a["request"]]["target"],
            a["pointing_at_sec"],
        )
        for a in recovery["allocations"]
        if a["phase"] == "fresh"
    }
    suffix = []
    allocations = []
    stage_times = {}
    for step in recovery["steps"]:
        ids = []
        for stage in step["actions"]:
            rid, kind = stage["request"], stage["kind"]
            c = by_id[rid]
            new = copy.deepcopy(originals[c["actions"][kind]])
            aid = tag + rid + "/" + kind
            new["id"] = aid
            new["commands"][0]["duration_sec"] = stage["duration_sec"]
            if kind == "slew":
                new["commands"][0]["slew"] = {
                    "frame": "GCRS",
                    "quaternion_wxyz": points[rid]["goal_quaternion_wxyz"],
                }
            task["actions"].append(new)
            ids.append(aid)
            stage_times.setdefault(rid, {})[kind] = step["at_sec"]
        suffix.append(
            {
                "at_sec": step["at_sec"],
                "advance_sec": q,
                "action_ids": sorted(ids) or ["wait"],
            }
        )
    for a in recovery["allocations"]:
        c = by_id[a["request"]]
        allocations.append(
            {
                **a,
                "actor": c["actor"],
                "station": c["station"],
                "stage_times": stage_times[a["request"]],
            }
        )
    prefix = copy.deepcopy(meta["schedule"]["steps"][: source.session.revision])
    # Manual decisions must be represented exactly, not copied from old intent.
    at = 0.0
    prefix = []
    for row in source.session._transitions:
        r = row["request"]
        dt = r["advance_sec"] if r["advance_sec"] is not None else q
        prefix.append(
            {
                "at_sec": at,
                "advance_sec": dt,
                "action_ids": sorted(a for ids in r["choices"].values() for a in ids),
            }
        )
        at += dt
    schedule = {
        "steps": prefix + controls + suffix,
        "allocations": allocations,
        "unserved": recovery["unserved"],
        "predicted_fulfilled": recovery["predicted_fulfilled"],
        "completed": completed,
    }
    task["max_actions"] = len(task["actions"]) + 1
    task = TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
    audit = {
        "source_session_id": source.id,
        "source_checkpoint_sha256": source.session.checkpoint(compact=True)["sha256"],
        "prefix_time_sec": now,
        "committed_time_sec": stop,
        "prefix_physical_sha256": digest(source.session._runtime._backend.report()),
        "committed_physical_sha256": digest(runtime._backend.report()),
        "events": events,
        "states": states,
        "reason": request.reason,
        "key_sha256": key,
        "request_sha256": sha,
    }
    new_meta = copy.deepcopy(meta)
    new_meta.update(
        configuration=cfg.model_dump(mode="json", by_alias=True),
        resolved_configuration=cfg.model_dump(mode="json", by_alias=True),
        schedule=schedule,
        task_sha256=digest(task),
        dynamic=audit,
        unserved_reasons={
            rid: states[rid].get("reason") or code
            for rid, code in recovery["unserved"].items()
        },
        planner={
            "baseline": {"predicted_fulfilled": recovery["predicted_fulfilled"]},
            "search_nodes": recovery["search_nodes"],
            "settings": settings,
            "selected_score": [],
        },
    )
    new_meta.pop("replan", None)
    new_meta["sha256"] = digest({k: v for k, v in new_meta.items() if k != "sha256"})
    scenario["metadata"]["workspace"] = new_meta
    fresh = _replay(scenario, task, source)
    apply_steps(fresh)
    if digest(fresh._runtime._backend.report()) != audit["committed_physical_sha256"]:
        raise ValueError("Будущий каталог изменил физику события.")
    return store.restore({"record": fresh.checkpoint(compact=True)})
