"""Native-prefix forks and bounded custody-aware network continuation.

Only committed controls are executed during planning; future candidates are
value-only. Cancelled ISL fragments obey native discard/custody semantics.
"""

from __future__ import annotations

import copy

from space_env.api.network_workspace import (
    NetworkConfiguration,
    _geometry_configuration,
    collect_network_forecast,
    metadata,
)
from space_env.api.workspace import _PLAN_LOCK, compile_configuration
from space_env.api.workspace_dynamic import MissionChange, _replay
from space_env.planning.sequence_recovery import schedule_recovery
from space_env.runtime import AccessPolicy
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSpec


def evolve_network(store, session_id, token, payload):
    from space_env.api.operator import OperatorError, token_digest

    request = MissionChange.model_validate(payload)
    if not request.reason.strip() or request.planner is not None:
        raise ValueError("Нужна причина; сетевой бюджет сохраняется из конфигурации.")
    with store.lock:
        source = store.get(session_id)
        with source.lock, source.session._lock:
            role, write, _ = source.permission(token)
            if (
                role != "@central"
                or not write
                or source.identities[token_digest(token)] != "operator"
            ):
                raise OperatorError("forbidden", "Нужен центральный оператор.", 403)
            source._guard()
            key = digest(
                {"source": session_id, "network_change": request.idempotency_key}
            )
            sha = digest(request.model_dump(mode="json"))
            for candidate in store.sessions.values():
                old = (
                    candidate.session._scenario.get("metadata", {})
                    .get("network_workspace", {})
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
                    "stale_control", "Версия или управление изменились.", 409
                )
            meta = metadata(source)
            if meta is None or source.session._done:
                raise OperatorError(
                    "unsupported_evolution", "Нужна незавершённая сетевая миссия.", 409
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


def _controls(source, task, request, config, tag):
    """Materialize operator controls; never change historical definition inputs."""
    now, q = source.session._runtime.time_sec, source.session._task.decision_sec
    active = sorted(
        {
            r["command"]["actor_id"]
            for r in source.session._runtime.command_status()
            if r["status"] in {"accepted", "running"}
        }
    )
    event_at = now + (q if active else 0)
    events = [e.model_dump(mode="json") for e in request.events]
    coords = {(r["lat_deg"], r["lon_deg"]) for r in config["mission"]["requests"]}
    for addition in request.additions:
        if (
            addition.actor_id is None
            or addition.station is None
            or addition.release_sec < event_at
            or (addition.lat_deg, addition.lon_deg) not in coords
        ):
            raise ValueError(
                "Новая заявка требует явного аппарата/станции, объявленной точки и выпуска не раньше события."
            )
        config["mission"]["requests"].append(addition.model_dump(mode="json"))
    # Validate the cumulative input before allocating any branch.
    spec = NetworkConfiguration.model_validate(config)
    if request.additions:
        _, intent, _ = compile_configuration(_geometry_configuration(spec))
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
            r["raw_size_bits"] = addition.raw_bits
            job["outputs"][0]["size_bits"] = addition.output_bits
            events.insert(
                0, {"type": "request_added", "payload": {"request": r, "jobs": [job]}}
            )
    if len(events) > 8:
        raise ValueError("Не более восьми событий, включая заявки.")
    stop = event_at + (q if events else 0)
    if stop >= spec.mission.horizon_sec:
        raise ValueError("Нет времени для продолжения после событий.")
    controls = []
    if events:
        task["roles"]["@mission"] = {
            "actors": [],
            "stations": [s.name for s in spec.mission.stations],
            "demand_actors": list(source.session._original["actor_order"]),
            "demand_stations": [s.name for s in spec.mission.stations],
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
            role = "@mission"
            if event["type"].startswith("spacecraft_"):
                role = next(
                    (
                        k
                        for k, v in task["roles"].items()
                        if v["actors"] == [event["payload"].get("actor")]
                    ),
                    None,
                )
                if role is None:
                    raise ValueError("Неизвестный аппарат события.")
                task["roles"][role]["action_slots"] = 8
            task["actions"].append(
                {"id": aid, "role": role, "kind": "inject", "event": event}
            )
            ids.append(aid)
        controls.append({"at_sec": event_at, "advance_sec": q, "action_ids": ids})
    task["max_joint_actions"] = max(
        task["max_joint_actions"],
        sum(r["action_slots"] for r in task["roles"].values()),
    )
    return controls, events


def recovery_view(spec, runtime, scenario):
    """Bind forecast to observed inventories and committed lifecycle, not intent."""
    now = runtime.time_sec
    report = runtime._backend.report()
    inventories = {a: runtime.inventory(a) for a in scenario["actor_order"]}
    lifecycle = report["lifecycle"]
    states = {}
    config = spec.model_dump(mode="json", by_alias=True)
    by_id = {r["id"]: r for r in config["mission"]["requests"]}
    for rid, r in lifecycle["requests"].items():
        row = by_id[rid]
        owner = r["eligible_actors"][0]
        row.update(
            actor_id=owner,
            station=r["destination_id"],
            priority=r["priority"],
            deadline_sec=r["deadline_sec"],
        )
        products = [p for inv in inventories.values() for p in inv["products"]]
        final = next((p for p in products if p["id"] == r["delivery_product_id"]), None)
        raw = next(
            (
                p
                for p in products
                if p["id"] == r["raw_product_id"]
                and not p["cancelled"]
                and p["remaining_bits"] > 0
            ),
            None,
        )
        job = lifecycle["jobs"]["cpu-" + rid]
        state = {
            "phase": "fresh",
            "owner": final["actor"] if final else owner,
            "remaining_bits": 0.0,
            "remaining_cpu_sec": float(job["remaining_sec"]),
            "intact": True,
            "reason": r.get("reason"),
        }
        if r["status"] == "completed":
            state["phase"] = "completed"
        elif r["status"] in {"failed", "cancelled", "expired"}:
            state["phase"] = "terminal"
        elif final:
            state.update(
                phase="final",
                owner=final["actor"],
                remaining_bits=float(final["remaining_bits"]),
                intact=final["remaining_bits"] == final["size_bits"],
            )
            if (
                final["cancelled"]
                or not final["deliverable"]
                or final["remaining_bits"] <= 0
                or final["custody_pinned"]
            ):
                state.update(
                    phase="blocked", reason="final_not_ready_or_unsettled_custody"
                )
        elif raw:
            state.update(
                phase="raw",
                owner=raw["actor"],
                remaining_bits=float(raw["remaining_bits"]),
                intact=raw["remaining_bits"] == raw["size_bits"],
            )
            if raw["custody_pinned"] or not state["intact"]:
                state.update(
                    phase="blocked", reason="raw_not_intact_or_unsettled_custody"
                )
            if job["status"] in {"failed", "cancelled", "expired"}:
                state.update(phase="terminal", reason="transform_terminal")
        elif r["attempts"]:
            state.update(
                phase="terminal", reason="acquisition_attempt_consumed_without_raw"
            )
        bus = report["spacecraft_bus"]["states"][state["owner"]]
        required = (
            {"bus", "radio"}
            | ({"cpu"} if state["phase"] in {"fresh", "raw"} else set())
            | ({"adcs", "payload"} if state["phase"] == "fresh" else set())
        )
        if state["phase"] not in {"completed", "terminal"} and (
            bus["safe"] or required & set(bus["faults"])
        ):
            state.update(phase="blocked", reason="current_spacecraft_unavailability")
        states[rid] = state
    # Expired deadlines remain valid catalogue values; scheduler keeps terminal rows.
    spec = NetworkConfiguration.model_validate(config)
    view = collect_network_forecast(spec, runtime=runtime, scenario=scenario)
    # A deadline extension must not renew an existing DTN bundle lifetime.
    # Keep operator intent in config, but constrain the residual forecast by
    # actual product and bundle expiry, including products already in custody.
    bundle_expiry = (report.get("dtn") or {}).get("bundle_expiry_sec", {})
    artifacts = lifecycle["artifacts"]
    for demand in view["demands"]:
        rid = demand["id"]
        bounds = [demand["deadline_sec"]]
        for pid in ("raw-" + rid, "final-" + rid):
            if pid in bundle_expiry:
                bounds.append(bundle_expiry[pid])
            if artifacts.get(pid, {}).get("expires_sec") is not None:
                bounds.append(artifacts[pid]["expires_sec"])
        demand["deadline_sec"] = min(bounds)
        if demand["deadline_sec"] <= now and states[rid]["phase"] not in {
            "completed",
            "terminal",
        }:
            states[rid].update(
                phase="blocked", reason="native_product_or_bundle_expired"
            )
    view.update(
        at_sec=now,
        recovery=states,
        stored_bits={
            a: float(
                sum(p["remaining_bits"] for p in inv["products"])
                + (report.get("dtn") or {}).get("reserved_bits", {}).get(a, 0.0)
            )
            for a, inv in inventories.items()
        },
    )
    for a, info in view["actors"].items():
        efficiency = report["spacecraft_bus"]["models"][a]["power"][
            "discharge_efficiency"
        ]
        info["usable_wh"] = max(
            0.0,
            report["energy_wh"][a]
            - spec.mission.battery_wh * 0.2
            - (spec.mission.horizon_sec - now) / 3600 / efficiency,
        )
    for edge in view["edges"]:
        participants = [edge["source"]] + (
            [edge["destination"]] if edge["destination"] in inventories else []
        )
        if any(
            report["spacecraft_bus"]["states"][a]["safe"]
            or {"bus", "radio"} & set(report["spacecraft_bus"]["states"][a]["faults"])
            for a in participants
        ):
            edge["cells"] = [False] * len(edge["cells"])
    view["sha256"] = digest({k: v for k, v in view.items() if k != "sha256"})
    return spec, view


def _build(store, source, meta, request, key, sha):
    config = copy.deepcopy(meta["configuration"])
    scenario = copy.deepcopy(source.session._original)
    task = source.session._task.model_dump(mode="json", by_alias=True)
    tag = "@network-change/" + key[:16] + "/"
    controls, events = _controls(source, task, request, config, tag)

    def validated_task():
        task["max_actions"] = len(task["actions"]) + 1
        return TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)

    def apply(session):
        for i, step in enumerate(controls):
            response = session.step(
                {"@central": step["action_ids"]},
                expected_revision=session.revision,
                idempotency_key=tag + str(i),
                advance_sec=step["advance_sec"],
            )
            if not response["feedback"]["accepted"]:
                raise ValueError(
                    "Native runtime отклонил изменение; источник сохранён."
                )

    committed = _replay(scenario, validated_task(), source)
    apply(committed)
    runtime = committed._runtime
    runtime.access = AccessPolicy(truth=True, forecast=True)
    spec, view = recovery_view(
        NetworkConfiguration.model_validate(config), runtime, committed._scenario
    )
    schedule, sequence_recovery = schedule_recovery(
        spec, view, runtime, committed._scenario
    )
    q, h = spec.mission.decision_sec, spec.mission.horizon_sec
    suffix = [
        {"at_sec": i * q, "advance_sec": q, "action_ids": []}
        for i in range(round(runtime.time_sec / q), round(h / q))
    ]
    targets = runtime._backend.report()["lifecycle"]["requests"]
    orders = {r.id: r for r in spec.mission.requests}
    for allocation in schedule["allocations"]:
        rid = allocation["request"]
        for i, stage in enumerate(allocation["stages"]):
            aid = tag + rid + "/" + str(i) + "/" + stage["kind"]
            command = {
                "command_id": aid,
                "actor_id": stage["actor"],
                "kind": stage["kind"],
                "duration_sec": stage["cells"] * q,
            }
            if stage["kind"] == "slew":
                cap = next(s for s in allocation["stages"] if s["kind"] == "imaging")
                point = runtime.forecast_pointing(
                    stage["actor"],
                    targets[rid]["target_id"],
                    cap["cell"] * q + orders[rid].acquisition_sec / 2,
                )
                command["slew"] = {
                    "frame": "GCRS",
                    "quaternion_wxyz": point["goal_quaternion_wxyz"],
                }
            elif stage["kind"] == "imaging":
                command.update(request_id=rid, duration_sec=orders[rid].acquisition_sec)
            elif stage["kind"] == "transform":
                command["job_id"] = "cpu-" + rid
            else:
                command.update(
                    product_id="final-" + rid, destination_id=stage["destination"]
                )
                if stage["kind"] == "relay":
                    command["relay"] = {
                        "ttl_sec": orders[rid].deadline_sec - stage["cell"] * q
                    }
            task["actions"].append(
                {
                    "id": aid,
                    "role": stage["actor"],
                    "kind": "submit",
                    "request_ref": rid,
                    "commands": [command],
                }
            )
            suffix[stage["cell"] - round(runtime.time_sec / q)]["action_ids"].append(
                aid
            )
    for step in suffix:
        step["action_ids"] = sorted(step["action_ids"]) or ["wait"]
    prefix, at = [], 0.0
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
    task = validated_task()
    audit = {
        "key_sha256": key,
        "request_sha256": sha,
        "reason": request.reason,
        "source_session_id": source.id,
        "source_checkpoint_sha256": source.session.checkpoint(compact=True)["sha256"],
        "prefix_time_sec": source.session._runtime.time_sec,
        "committed_time_sec": runtime.time_sec,
        "prefix_physical_sha256": digest(source.session._runtime._backend.report()),
        "committed_physical_sha256": digest(runtime._backend.report()),
        "events": events,
        "states": view["recovery"],
        "sequence_recovery": schedule["recovery_sequence"],
    }
    new_meta = {
        **copy.deepcopy(meta),
        "configuration": spec.model_dump(mode="json", by_alias=True),
        "forecast": view,
        "schedule": schedule,
        "steps": prefix + controls + suffix,
        "task_sha256": digest(task),
        "dynamic": audit,
        "planning_method": (
            "sequence_recovery"
            if schedule["recovery_sequence"]["mode"] == "residual_then_sequence"
            else "serial_recovery"
        ),
        "sequence_recovery": sequence_recovery,
    }
    # Initial finite-goal cones do not certify new targets, poses or recovery
    # commands. Recovery binds a new suffix witness to the committed native pose.
    for initial_only in (
        "base_forecast",
        "pointing_envelope",
        "completeness",
        "sequence_geometry",
        "terminal_geometry",
        "base_schedule",
    ):
        new_meta.pop(initial_only, None)
    new_meta["sha256"] = digest({k: v for k, v in new_meta.items() if k != "sha256"})
    scenario["metadata"]["network_workspace"] = new_meta
    fresh = _replay(scenario, task, source)
    apply(fresh)
    if digest(fresh._runtime._backend.report()) != audit["committed_physical_sha256"]:
        raise ValueError("Будущий каталог изменил физическое состояние события.")
    return store.restore({"record": fresh.checkpoint(compact=True)})
