"""User mission configuration → fleet plan → existing native operator session."""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from space_env.planning.chain import materialize_task, profile
from space_env.planning.fleet_scheduler import FleetSettings
from space_env.reference import Epoch
from space_env.reference.acquisition import AcquisitionTarget
from space_env.reference.operations import StationOutage
from space_env.runtime.schema import StrictModel
from space_env.scaling.coupled import _compile_coupled
from space_env.scaling.independent import FixedSite, OrbitPlane
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSpec


class MissionOrder(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,48}$")
    actor_id: str | None = Field(default=None, min_length=1, max_length=64)
    station: str | None = Field(default=None, min_length=1, max_length=32)
    lat_deg: float = Field(ge=-90, le=90, strict=True)
    lon_deg: float = Field(ge=-180, le=180, strict=True)
    priority: int = Field(default=1, ge=0, le=10000, strict=True)
    release_sec: float = Field(default=0.0, ge=0, strict=True)
    deadline_sec: float = Field(gt=0, le=3600, strict=True)
    acquisition_sec: float = Field(default=2.0, gt=0, le=60, strict=True)
    processing_sec: float = Field(default=1.0, gt=0, le=120, strict=True)
    raw_bits: int = Field(default=1000000, ge=1, le=100000000, strict=True)
    output_bits: int = Field(default=500000, ge=1, le=100000000, strict=True)


class MissionConfiguration(StrictModel):
    schema_id: Literal["space-env.mission-configuration/1"] = Field(
        default="space-env.mission-configuration/1", alias="schema"
    )
    name: str = Field(default="Моя спутниковая миссия", min_length=1, max_length=80)
    epoch_utc: str = "2026-07-18T12:00:00Z"
    seed: int = Field(default=0, ge=0, lt=2**63, strict=True)
    horizon_sec: float = Field(default=600.0, ge=60, le=3600, strict=True)
    decision_sec: float = Field(default=5.0, ge=1, le=30, strict=True)
    planes: tuple[OrbitPlane, ...] = Field(min_length=1, max_length=4)
    stations: tuple[FixedSite, ...] = Field(min_length=1, max_length=16)
    requests: tuple[MissionOrder, ...] = Field(min_length=1, max_length=64)
    memory_mbit: float = Field(default=128.0, ge=1, le=10000, strict=True)
    battery_wh: float = Field(default=100.0, ge=0.01, le=10000, strict=True)
    rate_bps: float = Field(default=2000000.0, ge=1000, le=100000000, strict=True)
    outages: tuple[StationOutage, ...] = Field(default=(), max_length=32)
    planner: FleetSettings = Field(default_factory=FleetSettings)
    max_assignment_trials: int = Field(default=12, ge=1, le=32, strict=True)

    @model_validator(mode="after")
    def references(self):
        Epoch.utc(self.epoch_utc)
        if not self.name.strip():
            raise ValueError("name: задайте название миссии")
        if (
            abs(
                self.horizon_sec / self.decision_sec
                - round(self.horizon_sec / self.decision_sec)
            )
            > 1e-8
            or self.horizon_sec / self.decision_sec > 240
        ):
            raise ValueError(
                "decision_sec: горизонт должен делиться на шаг; разрешено до240 шагов. Увеличьте шаг или уменьшите горизонт."
            )
        if len({p.id for p in self.planes}) != len(self.planes):
            raise ValueError("planes: повторяются идентификаторы орбит")
        actors = {
            f"{p.id}-{i:02d}" for p in self.planes for i in range(len(p.phases_deg))
        }
        if len(actors) > 4:
            raise ValueError("planes: поддерживается от1 до4 спутников")
        sites = {s.name for s in self.stations}
        if len(sites) != len(self.stations):
            raise ValueError("stations: названия станций должны различаться")
        if len({r.id for r in self.requests}) != len(self.requests):
            raise ValueError("requests: идентификаторы задач должны различаться")
        if len({(r.lat_deg, r.lon_deg) for r in self.requests}) > 16:
            raise ValueError("requests: поддерживается до16 разных точек съёмки")
        for r in self.requests:
            if r.actor_id is not None and r.actor_id not in actors:
                raise ValueError(
                    f"requests.{r.id}.actor_id: неизвестный спутник {r.actor_id}"
                )
            if r.station is not None and r.station not in sites:
                raise ValueError(
                    f"requests.{r.id}.station: неизвестная станция {r.station}"
                )
            if not r.release_sec < r.deadline_sec <= self.horizon_sec:
                raise ValueError(
                    f"requests.{r.id}: требуется release_sec < deadline_sec <= horizon_sec"
                )
        for outage in self.outages:
            if (
                outage.station not in sites
                or not 0 <= outage.start_sec < outage.end_sec <= self.horizon_sec
            ):
                raise ValueError("outages: проверьте станцию и границы интервала")
        return self


def parse_configuration(value):
    from space_env.api.operator import OperatorError

    try:
        return MissionConfiguration.model_validate(value)
    except ValidationError as exc:
        errors = []
        for e in exc.errors(include_input=False)[:6]:
            location = " → ".join(
                str(k + 1) if isinstance(k, int) else str(k) for k in e["loc"]
            )
            message = e["msg"].removeprefix("Value error, ")
            if e["type"] == "missing":
                message = "заполните поле"
            elif e["type"] in {
                "float_type",
                "int_type",
                "float_parsing",
                "int_parsing",
            }:
                message = "нужно число"
            elif e["type"] == "extra_forbidden":
                message = "неизвестное поле"
            errors.append(f"{location}: {message}" if location else message)
        raise OperatorError("invalid_configuration", "; ".join(errors)) from exc


def default_configuration():
    return MissionConfiguration.model_validate(
        {
            "planes": [
                {
                    "id": "a",
                    "inclination_deg": 55.0,
                    "raan_deg": 0.0,
                    "phases_deg": [0.0],
                },
                {
                    "id": "b",
                    "inclination_deg": 75.0,
                    "raan_deg": 120.0,
                    "phases_deg": [0.0],
                },
            ],
            "stations": [
                {"name": "West", "lat_deg": 0.0, "lon_deg": -120.0},
                {"name": "East", "lat_deg": 0.0, "lon_deg": 0.0},
            ],
            "requests": [
                {
                    "id": "survey-west",
                    "actor_id": "a-00",
                    "station": "West",
                    "lat_deg": 0.0,
                    "lon_deg": -120.0,
                    "deadline_sec": 600.0,
                    "priority": 5,
                },
                {
                    "id": "survey-east",
                    "actor_id": "b-00",
                    "station": "East",
                    "lat_deg": 0.0,
                    "lon_deg": 0.0,
                    "deadline_sec": 600.0,
                    "priority": 3,
                },
            ],
        }
    ).model_dump(mode="json", by_alias=True)


def compile_configuration(value):
    cfg = MissionConfiguration.model_validate(value)
    if any(r.actor_id is None or r.station is None for r in cfg.requests):
        raise ValueError(
            "Автоматическое назначение требует MissionWorkspace или create_mission; compile_configuration принимает только назначенные заявки."
        )
    coords = list(dict.fromkeys((r.lat_deg, r.lon_deg) for r in cfg.requests))
    targets = [
        AcquisitionTarget(id=f"target-{i}", lat_deg=lat, lon_deg=lon)
        for i, (lat, lon) in enumerate(coords)
    ]
    args = SimpleNamespace(
        id="workspace-" + digest(cfg.model_dump(mode="json"))[:16],
        epoch_utc=cfg.epoch_utc,
        planes=cfg.planes,
        targets=targets,
        stations=cfg.stations,
        horizon_sec=cfg.horizon_sec,
        decision_sec=cfg.decision_sec,
        control_step_sec=0.1,
        rigid_step_sec=0.01,
        lifecycle_step_sec=0.5,
        controller_clock="event-driven/1",
        mode="nominal",
        min_acquired=0,
        min_fulfilled=0,
    )
    w = _compile_coupled(args)
    scenario = w["scenario"]
    d = scenario["definition"]
    task = w["task"]
    initial = [
        tuple(round(float(x), 6) for k in ("r_m", "v_m_s") for x in o["initial"][k])
        for o in d["orbits"].values()
    ]
    if len(initial) != len(set(initial)):
        raise ValueError(
            "planes: два спутника заданы в одной и той же точке с одной скоростью"
        )
    for bus in d["spacecraft_bus"].values():
        bus["attitude"].pop("rigid", None)
    for v in d["operations"]["vehicles"].values():
        v.update(
            memory_bits=cfg.memory_mbit * 1e6,
            battery_wh=cfg.battery_wh,
            rate_bps=cfg.rate_bps,
        )
    d["operations"]["outages"] = [o.model_dump(mode="json") for o in cfg.outages]
    requests = []
    jobs = []
    actions = []
    for r in cfg.requests:
        raw, final, job = "raw-" + r.id, "final-" + r.id, "cpu-" + r.id
        target = targets[coords.index((r.lat_deg, r.lon_deg))].id
        requests.append(
            {
                "id": r.id,
                "target_id": target,
                "eligible_actors": [r.actor_id],
                "release_sec": r.release_sec,
                "deadline_sec": r.deadline_sec,
                "priority": r.priority,
                "acquisition_sec": r.acquisition_sec,
                "raw_product_id": raw,
                "raw_size_bits": r.raw_bits,
                "delivery_product_id": final,
                "destination_id": r.station,
            }
        )
        jobs.append(
            {
                "id": job,
                "inputs": [raw],
                "outputs": [{"id": final, "size_bits": r.output_bits}],
                "work_sec": r.processing_sec,
                "workspace_bits": 100000,
                "power_w": 5.0,
            }
        )
        for kind, duration, fields in [
            ("slew", 12.0, {"slew": {"frame": "GCRS", "target_id": target}}),
            ("imaging", r.acquisition_sec, {"request_id": r.id}),
            ("transform", r.processing_sec, {"job_id": job}),
            (
                "downlink",
                cfg.decision_sec,
                {"product_id": final, "destination_id": r.station},
            ),
        ]:
            aid = kind + "/" + r.id
            actions.append(
                {
                    "id": aid,
                    "role": r.actor_id,
                    "kind": "submit",
                    "request_ref": r.id,
                    "commands": [
                        {
                            "command_id": aid,
                            "actor_id": r.actor_id,
                            "kind": kind,
                            "duration_sec": duration,
                            **fields,
                        }
                    ],
                }
            )
    d["lifecycle"].update(requests=requests, jobs=jobs)
    scenario["seed"] = cfg.seed
    scenario["lineage"] = (
        "workspace/"
        + digest(
            {
                "planes": [p.model_dump(mode="json") for p in cfg.planes],
                "sites": [(s.lat_deg, s.lon_deg, s.alt_m) for s in cfg.stations],
                "targets": sorted(coords),
            }
        )[:24]
    )
    task.update(actions=actions, max_actions=len(actions) + 1, policy_origin="external")
    try:
        profile(scenario, task)
    except ValueError as exc:
        if "memory budget" in str(exc):
            raise ValueError(
                "memory_mbit: памяти недостаточно для консервативного бюджета заявок; увеличьте память или сократите задачи"
            ) from exc
        raise
    return (
        cfg,
        scenario,
        TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True),
    )


_PLAN_LOCK = threading.BoundedSemaphore(1)


def create_mission(value, store):
    from space_env.api.operator import OperatorError

    if not _PLAN_LOCK.acquire(blocking=False):
        raise OperatorError(
            "planner_busy", "Планировщик занят. Дождитесь текущего расчёта.", 429
        )
    try:
        parsed = parse_configuration(value)
        from space_env.planning.assignment import assign_configuration

        cfg, scenario, task, runtime, view, result, assignment = assign_configuration(
            parsed
        )
        schedule = result["schedule"]
        chains = {c["id"]: c for c in view["chains"]}
        pointings = {
            a["request"]: runtime.forecast_pointing(
                a["actor"], chains[a["request"]]["target"], a["pointing_at_sec"]
            )
            for a in schedule["allocations"]
        }
        derived = materialize_task(task, view, schedule, pointings)
        reasons = {}
        for rid, code in schedule["unserved"].items():
            c = chains[rid]
            capture = next(x for x in view["acquisitions"] if x["request"] == rid)[
                "windows"
            ]
            contact = next(
                x
                for x in view["contacts"]
                if x["actor"] == c["actor"] and x["station"] == c["station"]
            )["windows"]
            reasons[rid] = (
                "Нет окна наблюдения точки в заданном горизонте."
                if not capture
                else "Нет окна связи с выбранной станцией."
                if not contact
                else "Недостаточен консервативный запас энергии или лимит передачи."
                if code == "conservative_energy_or_tx_quota"
                else "Не найдено место для полной цепочки с учётом дедлайна, занятости спутника и станции."
            )
        meta = {
            "schema": "space-env.workspace-plan/1",
            "configuration": parsed.model_dump(mode="json", by_alias=True),
            "resolved_configuration": cfg.model_dump(mode="json", by_alias=True),
            "assignment": assignment,
            "definition_sha256": digest(scenario["definition"]),
            "task_sha256": digest(derived),
            "schedule": schedule,
            "unserved_reasons": reasons,
            "planner": {
                k: result[k]
                for k in ("baseline", "search_nodes", "selected_score", "settings")
            },
            "scope": "kinematic pointing, metadata products and ideal geometric downlink; forecast proposal, not guaranteed service",
        }
        meta["sha256"] = digest(meta)
        scenario["metadata"] = {**scenario.get("metadata", {}), "workspace": meta}
        return store.create({"scenario": scenario, "task": derived})
    finally:
        _PLAN_LOCK.release()


def _metadata(operator):
    meta = operator.session._scenario.get("metadata", {}).get("workspace")
    if meta is None:
        return None
    if (
        meta.get("schema") != "space-env.workspace-plan/1"
        or meta.get("sha256")
        != digest({k: v for k, v in meta.items() if k != "sha256"})
        or meta["task_sha256"]
        != digest(operator.session._task.model_dump(mode="json", by_alias=True))
        or meta["definition_sha256"] != digest(operator.session._scenario["definition"])
    ):
        raise ValueError(
            "План не соответствует конфигурации или каталогу текущей миссии."
        )
    return meta


def _aligned(operator, meta):
    steps = meta["schedule"]["steps"]
    history = operator.session._transitions
    if len(history) > len(steps):
        return False
    for i, row in enumerate(history):
        request = row["request"]
        expected = steps[i]
        chosen = [a for ids in request["choices"].values() for a in ids]
        dt = (
            request["advance_sec"]
            if request["advance_sec"] is not None
            else operator.session._task.decision_sec
        )
        if (
            sorted(chosen) != sorted(expected["action_ids"])
            or abs(dt - expected["advance_sec"]) > 1e-8
        ):
            return False
    return True


def suggest(operator):
    from space_env.api.operator import OperatorError

    meta = _metadata(operator)
    if meta is None:
        return None
    if not _aligned(operator, meta):
        raise OperatorError(
            "plan_diverged",
            "Ручные действия изменили ход миссии. Продолжите вручную или перепланируйте оставшуюся работу на допустимой границе.",
            409,
        )
    i = operator.session.revision
    if i >= len(meta["schedule"]["steps"]):
        raise OperatorError("episode_done", "Миссия завершена.", 409)
    row = meta["schedule"]["steps"][i]
    return {
        "action_ids": row["action_ids"].copy(),
        "advance_sec": row["advance_sec"],
        "expected_revision": i,
        "planner": "fleet_compact",
        "note": "Запланированный шаг; результат определяет физическая среда.",
    }


def describe(operator):
    from space_env.api.workspace_replan import replan_status

    meta = _metadata(operator)
    if meta is None:
        return None
    report = operator.session._runtime._backend.report()
    life = report["lifecycle"]
    allocations = {a["request"]: a for a in meta["schedule"]["allocations"]}
    rows = []
    command_states = {
        c["command"]["command_id"]: c["status"]
        for c in operator.session._runtime.command_status()
    }
    history = operator.session._transitions
    for r in meta.get("resolved_configuration", meta["configuration"])["requests"]:
        actual = life["requests"][r["id"]]
        allocation = allocations.get(r["id"])
        rows.append(
            {
                "id": r["id"],
                "actor": r["actor_id"],
                "station": r["station"],
                "priority": r["priority"],
                "deadline_sec": r["deadline_sec"],
                "status": actual["status"],
                "reason": actual.get("reason"),
                "actual_stages": {
                    "slew": any(
                        command_states.get(f"@task/{i}/{aid}/0") == "completed"
                        for i, h in enumerate(history)
                        for ids in h["request"]["choices"].values()
                        for aid in ids
                        if aid != "wait"
                        and operator.session._by_id[aid].request_ref == r["id"]
                        and operator.session._by_id[aid].commands
                        and operator.session._by_id[aid].commands[0].kind == "slew"
                    ),
                    "imaging": life["artifacts"]
                    .get(actual["raw_product_id"], {})
                    .get("source")
                    == "acquired",
                    "transform": life["artifacts"]
                    .get(actual["delivery_product_id"], {})
                    .get("source")
                    == "transformed_out",
                    "downlink": actual["status"] == "completed",
                },
                "plan": copy.deepcopy(allocation),
                "plan_reason": meta["unserved_reasons"].get(r["id"]),
                "stages": [
                    {"kind": k, "at_sec": at}
                    for k, at in allocation["stage_times"].items()
                ]
                if allocation and "stage_times" in allocation
                else [
                    {"kind": k, "at_sec": allocation[t]}
                    for k, t in [
                        ("slew", "start_sec"),
                        ("imaging", "capture_sec"),
                        ("transform", "cpu_sec"),
                        ("downlink", "delivery_sec"),
                    ]
                ]
                if allocation
                else [],
            }
        )
    fulfilled = sum(r["status"] == "completed" for r in rows)
    replan_available, replan_note = replan_status(operator)
    return {
        "name": meta["configuration"]["name"],
        "configuration": copy.deepcopy(meta["configuration"]),
        "resolved_configuration": copy.deepcopy(
            meta.get("resolved_configuration", meta["configuration"])
        ),
        "assignment": copy.deepcopy(meta.get("assignment")),
        "replan": copy.deepcopy(meta.get("replan")),
        "dynamic": copy.deepcopy(meta.get("dynamic")),
        "replan_available": replan_available,
        "replan_note": replan_note,
        "requests": rows,
        "fulfilled": fulfilled,
        "total": len(rows),
        "planned": len(
            set(allocations)
            | set(meta.get("replan", {}).get("completed", []))
            | set(meta["schedule"].get("completed", []))
        ),
        "done": operator.session._done,
        "all_fulfilled": fulfilled == len(rows),
        "plan_aligned": _aligned(operator, meta),
        "planner": copy.deepcopy(meta["planner"]),
        "scope": meta["scope"],
    }


class MissionWorkspace:
    """Convenience SDK over the same operator coordinator used by the website."""

    def __init__(self, configuration=None, *, store=None):
        from space_env.api.operator import OperatorStore

        self.store = store if store is not None else OperatorStore()
        self.connection = create_mission(
            configuration or default_configuration(), self.store
        )
        self.operator = self.store.get(self.connection["session_id"])

    def state(self):
        return self.operator.state(self.connection["tokens"]["operator"])

    def step(self):
        token = self.connection["tokens"]["operator"]
        revision = self.operator.session.revision
        choice = self.operator.suggest(token)
        return self.operator.step(
            token,
            {
                "action_ids": choice["action_ids"],
                "advance_sec": choice.get(
                    "advance_sec", self.operator.session._task.decision_sec
                ),
                "expected_revision": choice.get("expected_revision", revision),
                "control_epoch": self.operator.control["epoch"],
                "idempotency_key": f"workspace/{self.operator.session.revision}",
            },
        )

    def run(self, *, decisions=None):
        if decisions is not None and (type(decisions) is not int or decisions < 0):
            raise ValueError("decisions must be a nonnegative integer")
        count = 0
        while not self.operator.session._done and (
            decisions is None or count < decisions
        ):
            self.step()
            count += 1
        return self.state()

    def export(self, output):
        return self.operator.export_to(self.connection["tokens"]["admin"], output)

    def replan(self, reason, *, idempotency_key, planner=None):
        """Return an independent continuation; keep this mission unchanged."""
        from space_env.api.workspace_replan import replan_workspace

        state = self.state()
        payload = {
            "reason": reason,
            "expected_revision": state["revision"],
            "control_epoch": state["control"]["epoch"],
            "idempotency_key": idempotency_key,
        }
        if planner is not None:
            payload["planner"] = planner
        connection = replan_workspace(
            self.store,
            self.connection["session_id"],
            self.connection["tokens"]["operator"],
            payload,
        )
        result = object.__new__(MissionWorkspace)
        result.store = self.store
        result.connection = connection
        result.operator = self.store.get(connection["session_id"])
        return result

    def evolve(self, reason, *, idempotency_key, additions=(), events=(), planner=None):
        """Causally apply changes and recover remaining native stages in a branch."""
        from space_env.api.workspace_dynamic import evolve_workspace

        if self.operator.session._scenario.get("metadata", {}).get("network_workspace"):
            from space_env.api.network_dynamic import evolve_network

            evolve_workspace = evolve_network

        state = self.state()
        payload = {
            "reason": reason,
            "expected_revision": state["revision"],
            "control_epoch": state["control"]["epoch"],
            "idempotency_key": idempotency_key,
            "additions": list(additions),
            "events": list(events),
        }
        if planner is not None:
            payload["planner"] = planner
        connection = evolve_workspace(
            self.store,
            self.connection["session_id"],
            self.connection["tokens"]["operator"],
            payload,
        )
        result = object.__new__(MissionWorkspace)
        result.store, result.connection = self.store, connection
        result.operator = self.store.get(connection["session_id"])
        return result

    def fleet_round(self):
        from space_env.api.fleet_round import FleetRound

        return FleetRound(self.operator)

    @classmethod
    def network(
        cls,
        configuration=None,
        *,
        store=None,
        request_order=None,
        planning_method="serial",
    ):
        """Joint EO owner/path/time proposal executed by the existing runtime."""
        from space_env.api.network_workspace import (
            create_network,
            default_network_configuration,
        )
        from space_env.api.operator import OperatorStore

        result = object.__new__(cls)
        result.store = store if store is not None else OperatorStore()
        result.connection = create_network(
            configuration
            if configuration is not None
            else default_network_configuration(),
            result.store,
            request_order=request_order,
            planning_method=planning_method,
        )
        result.operator = result.store.get(result.connection["session_id"])
        return result

    @classmethod
    def relay(cls, *, store=None):
        """Existing EO/CPU/ISL/Ground-ACK profile on the same operator runtime."""
        from space_env.api.operator import OperatorStore

        result = object.__new__(cls)
        result.store = store if store is not None else OperatorStore()
        result.connection = result.store.create({"preset": "relay-demo"})
        result.operator = result.store.get(result.connection["session_id"])
        return result

    def close(self):
        return self.store.close(
            self.connection["session_id"], self.connection["tokens"]["admin"]
        )
