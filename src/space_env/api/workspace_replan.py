"""Explicit operator replan forks with verified physical prefix preservation."""

from __future__ import annotations

import copy

from pydantic import Field

from space_env.api.workspace import _PLAN_LOCK, _metadata, compile_configuration
from space_env.planning.chain import collect_chain_forecast, materialize_task
from space_env.planning.fleet_scheduler import FleetSettings, schedule_fleet
from space_env.runtime import AccessPolicy
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession


class ReplanRequest(StrictModel):
    expected_revision: int = Field(ge=0, strict=True)
    control_epoch: int = Field(ge=0, strict=True)
    idempotency_key: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=200)
    planner: FleetSettings | None = None


def replan_status(operator):
    s = operator.session
    if s._scenario.get("metadata", {}).get("workspace", {}).get("dynamic"):
        return (
            False,
            "Для динамической ветки используйте восстановление оставшихся этапов.",
        )
    if s._done or not s.revision:
        return False, "Нужна начатая, ещё не завершённая миссия."
    if (
        abs(
            s._runtime.time_sec / s._task.decision_sec
            - round(s._runtime.time_sec / s._task.decision_sec)
        )
        > 1e-8
    ):
        return False, "Нужна граница сетки решений."
    life = s._runtime._backend.report()["lifecycle"]
    completed = {
        rid for rid, row in life["requests"].items() if row["status"] == "completed"
    }
    touched = {
        s._by_id[a].request_ref
        for row in s._transitions
        for ids in row["request"]["choices"].values()
        for a in ids
        if a != "wait"
    }
    if touched - completed:
        return False, "Сначала завершите начатые цепочки съёмки, обработки и доставки."
    return True, "Можно пересчитать ещё не начатые заявки в отдельной ветке."


def replan_workspace(store, session_id, token, payload):
    from space_env.api.operator import OperatorError, token_digest

    request = ReplanRequest.model_validate(payload)
    if not request.reason.strip():
        raise ValueError("Укажите причину перепланирования.")
    # Same lock order as restore/create: store -> operator -> native session.
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
                    "Перепланирование доступно центральному оператору.",
                    403,
                )
            source._guard()
            key = digest({"source": session_id, "key": request.idempotency_key})
            body_sha = digest(request.model_dump(mode="json"))
            for candidate in store.sessions.values():
                old = (
                    candidate.session._scenario.get("metadata", {})
                    .get("workspace", {})
                    .get("replan")
                )
                if old and old["key_sha256"] == key and not candidate.closed:
                    if old["request_sha256"] != body_sha:
                        raise OperatorError(
                            "idempotency_conflict",
                            "Ключ уже использован с другими параметрами.",
                            409,
                        )
                    if not candidate.credentials:
                        raise OperatorError(
                            "replan_already_created",
                            f"Ветка {candidate.id} уже сохранена. Используйте сохранённый файл доступа или локальное восстановление доступа.",
                            409,
                        )
                    return {
                        "session_id": candidate.id,
                        "tokens": candidate.credentials.copy(),
                        "state": candidate.state(candidate.credentials["operator"]),
                    }
            if source.control["controller"] != "operator":
                raise OperatorError(
                    "not_controller", "Сначала верните управление оператору.", 409
                )
            if (
                request.expected_revision != source.session.revision
                or request.control_epoch != source.control["epoch"]
            ):
                raise OperatorError(
                    "stale_control",
                    "Версия миссии или epoch управления изменились.",
                    409,
                )
            if len(store.sessions) >= 8:
                raise OperatorError(
                    "capacity", "Закройте ненужную сессию перед созданием ветки.", 429
                )
            meta = _metadata(source)
            if meta is None:
                raise OperatorError(
                    "unsupported_replan",
                    "Требуется миссия из редактора конфигурации.",
                    409,
                )
            if meta.get("dynamic"):
                raise OperatorError(
                    "dynamic_replan_required",
                    "Используйте evolve для динамического каталога.",
                    409,
                )
            if source.session._done or source.session.revision == 0:
                raise OperatorError(
                    "replan_boundary", "Нужна начатая, ещё не завершённая миссия.", 409
                )
            now = source.session._runtime.time_sec
            q = source.session._task.decision_sec
            if abs(now / q - round(now / q)) > 1e-8:
                raise OperatorError(
                    "replan_boundary",
                    "Перепланирование требует границы сетки решений.",
                    409,
                )
            life = source.session._runtime._backend.report()["lifecycle"]
            completed = {
                rid
                for rid, row in life["requests"].items()
                if row["status"] == "completed"
            }
            touched = {
                source.session._by_id[aid].request_ref
                for row in source.session._transitions
                for ids in row["request"]["choices"].values()
                for aid in ids
                if aid != "wait"
            }
            if touched - completed:
                raise OperatorError(
                    "replan_boundary",
                    "Сначала завершите начатые цепочки съёмки, обработки и доставки. Перепланируются только ещё не начатые заявки.",
                    409,
                )
            if not _PLAN_LOCK.acquire(blocking=False):
                raise OperatorError("planner_busy", "Планировщик занят.", 429)
            try:
                return _build(store, source, meta, request, completed, key, body_sha)
            finally:
                _PLAN_LOCK.release()


def _build(store, source, meta, request, completed, key, body_sha):
    # Clone the checkpoint before granting forecast access; source access and
    # physical state never change, even on errors.
    checkpoint = source.session.checkpoint(compact=True)
    probe = TaskSession.from_checkpoint(checkpoint)
    cfg, _, original = compile_configuration(
        meta.get("resolved_configuration", meta["configuration"])
    )
    runtime = probe._runtime
    runtime.access = AccessPolicy(truth=True, forecast=True)
    view = collect_chain_forecast(
        runtime, probe._scenario, original, completed=completed
    )
    result = schedule_fleet(view, request.planner or cfg.planner)
    schedule = copy.deepcopy(result["schedule"])
    chains = {c["id"]: c for c in view["chains"]}
    points = {
        a["request"]: runtime.forecast_pointing(
            a["actor"], chains[a["request"]]["target"], a["pointing_at_sec"]
        )
        for a in schedule["allocations"]
    }
    derived = materialize_task(original, view, schedule, points)
    previous_actions = {
        a.id: a.model_dump(mode="json", by_alias=True)
        for a in source.session._task.actions
    }
    for i, action in enumerate(derived["actions"]):
        if action["request_ref"] in completed:
            derived["actions"][i] = copy.deepcopy(previous_actions[action["id"]])
    prefix = []
    at = 0.0
    for row in source.session._transitions:
        r = row["request"]
        dt = r["advance_sec"] if r["advance_sec"] is not None else cfg.decision_sec
        prefix.append(
            {
                "at_sec": at,
                "advance_sec": dt,
                "action_ids": sorted(a for ids in r["choices"].values() for a in ids),
            }
        )
        at += dt
    schedule["steps"] = prefix + schedule["steps"]
    schedule["allocations"] = [
        copy.deepcopy(a)
        for a in meta["schedule"]["allocations"]
        if a["request"] in completed
    ] + schedule["allocations"]
    schedule["sha256"] = digest({k: v for k, v in schedule.items() if k != "sha256"})
    audit = {
        "schema": "space-env.workspace-replan/1",
        "source_session_id": source.id,
        "source_checkpoint_sha256": checkpoint["sha256"],
        "source_plan_sha256": meta["sha256"],
        "prefix_revision": source.session.revision,
        "prefix_time_sec": at,
        "physical_sha256": digest(source.session._runtime._backend.report()),
        "completed": sorted(completed),
        "reason": request.reason,
        "key_sha256": key,
        "request_sha256": body_sha,
        "scope": "idle completed-chain boundary; fixed assignments and physics; verified prefix replay into separate branch",
    }
    new_meta = copy.deepcopy(meta)
    new_meta.update(
        schedule=schedule,
        task_sha256=digest(derived),
        replan=audit,
        unserved_reasons={
            rid: "После текущего времени не найдено место для полной цепочки с учётом окон, дедлайна и ресурсов."
            for rid in schedule["unserved"]
        },
        planner={
            k: result[k]
            for k in ("baseline", "search_nodes", "selected_score", "settings")
        },
    )
    new_meta["sha256"] = digest({k: v for k, v in new_meta.items() if k != "sha256"})
    scenario = copy.deepcopy(checkpoint["scenario"])
    scenario["metadata"]["workspace"] = new_meta
    replay = TaskSession(scenario, derived, seed=checkpoint["episode_seed"])
    for row in source.session._transitions:
        response = replay.step(**row["request"])
        if response["feedback"] != row["transition"]["feedback"]:
            raise ValueError("Перепланирование изменило результат исполненной команды.")
    if digest(replay._runtime._backend.report()) != audit["physical_sha256"]:
        raise ValueError("Перепланирование изменило физическое состояние префикса.")
    # Standard restore provides capability isolation, provenance, durable atomic
    # installation, replay checks and a new control fence; never mutates source.
    return store.restore({"record": replay.checkpoint(compact=True)})
