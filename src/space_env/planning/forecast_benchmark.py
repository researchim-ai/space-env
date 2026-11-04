"""Frozen, paired ready-data forecast evaluation with explicit teacher gates."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from space_env.eval.statistics import assign_split
from space_env.factory.semantic import fingerprint
from space_env.planning import PlanningRun
from space_env.planning.forecast_delivery import (
    DeliveryForecast,
    DeliverySchedulerSettings,
    _profile,
    collect_delivery_forecast,
    forecast_mission_plan,
    schedule_delivery,
)
from space_env.planning.mission_bundle import (
    publish_plan_bundle,
    read_json,
    verify_plan_bundle,
)
from space_env.planning.mission_plan import PlanExecution, _sealed
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import file_digest
from space_env.tasks import TaskSpec, VerifiedEpisode


class ForecastCase(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    family: str = Field(min_length=1, max_length=128)
    scenario: dict
    task: dict
    split: Literal["train", "holdout"]
    negative: bool = Field(default=False, strict=True)
    min_fulfilled: int = Field(default=1, ge=1, le=64, strict=True)


class ForecastTeacherGates(StrictModel):
    min_completion_fraction: float = Field(default=0.1, gt=0, le=1, strict=True)
    max_rejections: int = Field(default=0, ge=0, le=10000, strict=True)
    min_cases_per_family: int = Field(default=2, ge=2, le=12, strict=True)
    min_margin: float = Field(default=0.0, ge=0, le=1, strict=True)
    require_negative_control: bool = Field(default=True, strict=True)


class ForecastBenchmarkSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    cases: tuple[ForecastCase, ...] = Field(min_length=1, max_length=12)
    baselines: tuple[
        Literal["priority", "earliest_deadline", "route_aware_rolling"], ...
    ] = ("priority", "earliest_deadline")
    settings: DeliverySchedulerSettings = Field(
        default_factory=DeliverySchedulerSettings
    )
    teacher: ForecastTeacherGates = Field(default_factory=ForecastTeacherGates)

    @model_validator(mode="after")
    def frozen_cases(self):
        if not self.baselines or len(set(self.baselines)) != len(self.baselines):
            raise ValueError("distinct reactive baselines are required")
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate forecast case ID")
        lineages = set()
        physical = set()
        for case in self.cases:
            d, t, _, products, stations, _ = _profile(case.scenario, case.task)
            lineage = case.scenario["lineage"]
            if lineage in lineages or case.split != assign_split(lineage, ""):
                raise ValueError(
                    "forecast benchmark requires distinct lineage-family split keys"
                )
            lineages.add(lineage)
            # Existing conservative typed identity graph detects renamed clones.
            task = t.model_dump(mode="json", by_alias=True)
            task.pop("id")
            identity = fingerprint({"definition": d, "task": task})
            if identity in physical:
                raise ValueError("duplicate semantic forecast case")
            physical.add(identity)
            if case.min_fulfilled > len(products):
                raise ValueError("service floor exceeds full product denominator")
            if case.negative and not all(
                any(
                    o["station"] == s
                    and o["start_sec"] == 0
                    and o["end_sec"] >= d["horizon_sec"]
                    for o in d.get("outages", [])
                )
                for s in stations
            ):
                raise ValueError(
                    "negative control requires declared full-horizon outage at every station"
                )
        return self


def forecast_teacher_report(spec, rows):
    spec = ForecastBenchmarkSpec.model_validate(spec)
    policies = ("forecast_delivery", *spec.baselines)
    expected = {(c.id, p) for c in spec.cases for p in policies}
    if (
        len(rows) != len(expected)
        or {(r.get("case"), r.get("policy")) for r in rows} != expected
    ):
        raise ValueError("teacher report requires every paired row, including failures")
    for row in rows:
        if row.get("error") is not None:
            continue
        if (
            type(row.get("total")) is not int
            or not 1 <= row["total"] <= 64
            or type(row.get("fulfilled")) is not int
            or not 0 <= row["fulfilled"] <= row["total"]
            or type(row.get("rejections")) is not int
            or row["rejections"] < 0
            or type(row.get("completion_fraction")) not in (int, float)
            or not math.isfinite(row["completion_fraction"])
            or abs(row["completion_fraction"] - row["fulfilled"] / row["total"]) > 1e-12
        ):
            raise ValueError("invalid teacher native metrics")
    table = {(r["case"], r["policy"]): r for r in rows}
    failures = []
    held = [c for c in spec.cases if c.split == "holdout" and not c.negative]
    families = {c.family for c in spec.cases if not c.negative}
    for family in sorted(families):
        if sum(c.family == family for c in held) < spec.teacher.min_cases_per_family:
            failures.append(
                {"family": family, "reason": "insufficient_distinct_positive_holdouts"}
            )
    if not held:
        failures.append({"reason": "no_positive_holdout_cases"})
    if spec.teacher.require_negative_control and not any(
        c.negative for c in spec.cases
    ):
        failures.append({"reason": "missing_negative_control"})
    for case in spec.cases:
        if case.split != "holdout" and not case.negative:
            continue
        candidate = table[case.id, "forecast_delivery"]
        for policy in policies:
            row = table[case.id, policy]
            if (
                row.get("error") is not None
                or row.get("native_replay_verified") is not True
            ):
                failures.append(
                    {
                        "case": case.id,
                        "policy": policy,
                        "reason": "missing_or_invalid_native_execution",
                    }
                )
        fraction = candidate.get("completion_fraction")
        if case.negative:
            if (
                any(table[case.id, p].get("fulfilled") != 0 for p in policies)
                or candidate.get("mission_goal_passed") is not False
            ):
                failures.append(
                    {"case": case.id, "reason": "negative_control_promoted"}
                )
            continue
        if (
            candidate.get("mission_goal_passed") is not True
            or fraction is None
            or not math.isfinite(fraction)
            or fraction < spec.teacher.min_completion_fraction
            or candidate.get("rejections", math.inf) > spec.teacher.max_rejections
        ):
            failures.append({"case": case.id, "reason": "candidate_quality_threshold"})
        if type(candidate.get("predicted_fulfilled")) is not int or candidate[
            "predicted_fulfilled"
        ] > candidate.get("fulfilled", -1):
            failures.append(
                {"case": case.id, "reason": "forecast_realization_mismatch"}
            )
        for policy in spec.baselines:
            base = table[case.id, policy].get("completion_fraction")
            if (
                base is None
                or fraction is None
                or not math.isfinite(base)
                or not math.isfinite(fraction)
                or fraction + 1e-12 < base + spec.teacher.min_margin
            ):
                failures.append(
                    {
                        "case": case.id,
                        "baseline": policy,
                        "reason": "paired_noninferiority_or_margin",
                    }
                )
    return {
        "status": "failed_declared_gates" if failures else "passed_declared_gates",
        "eligible_for_sft_selection": not failures,
        "expert_teacher_qualified": False,
        "positive_holdouts": len(held),
        "failures": failures,
        "scope": "frozen model thresholds and paired physical resources; forecast candidate has operator-authorized geometry while reactive baselines have current views only; no global expertise/optimality or flight claim",
    }


def _summary(native):
    products = native["products"]
    total = len(products)
    count = sum(p["status"] == "delivered_on_time" for p in products)
    return {"fulfilled": count, "total": total, "completion_fraction": count / total}


def run_forecast_benchmark(value, output):
    spec = ForecastBenchmarkSpec.model_validate(value)
    root = Path(output)
    if root.exists():
        raise ValueError("forecast benchmark output must not exist")
    root.mkdir(parents=True)
    frozen = spec.model_dump(mode="json")
    atomic_write(root / "spec.json", canonical_json(frozen))
    rows = []
    for number, case in enumerate(spec.cases):
        for policy in ("forecast_delivery", *spec.baselines):
            path = root / f"{number:03d}" / policy
            path.mkdir(parents=True)
            row = {
                "case": case.id,
                "policy": policy,
                "split": case.split,
                "negative": case.negative,
                "error": None,
                "native_replay_verified": False,
            }
            try:
                if policy == "forecast_delivery":
                    runtime = MissionRuntime(
                        case.scenario, access=AccessPolicy(forecast=True)
                    )
                    view = collect_delivery_forecast(runtime, case.scenario, case.task)
                    proposal = forecast_mission_plan(
                        case.scenario,
                        case.task,
                        view,
                        settings=spec.settings.model_dump(mode="json"),
                        goals={"min_fulfilled": case.min_fulfilled},
                    )
                    del runtime
                    atomic_write(
                        path / "forecast.json", canonical_json(proposal["forecast"])
                    )
                    atomic_write(
                        path / "schedule.json", canonical_json(proposal["schedule"])
                    )
                    atomic_write(path / "plan.json", canonical_json(proposal["plan"]))
                    execution = PlanExecution(proposal["plan"])
                    execution.run()
                    record = execution.save(path / "execution.json")
                    del execution
                    bundle = publish_plan_bundle(
                        proposal["plan"], record, path / "dataset"
                    )
                    cert = read_json(path / "dataset/certificate.json")
                    row.update(
                        _summary(record["episode"]["runtime"]["backend_report"]),
                        rejections=cert["rejections"],
                        mission_goal_passed=bundle["mission_goal_passed"],
                        predicted_fulfilled=proposal["schedule"]["predicted_fulfilled"],
                        native_replay_verified=True,
                        plan_sha256=proposal["plan"]["sha256"],
                        information="operator-authorized geometry/known outage forecasts",
                    )
                else:
                    run = PlanningRun(case.scenario, case.task, {"name": policy})
                    metrics = run.run()
                    record = run.save(path / "planning.json", compact=True)
                    del run
                    verified = VerifiedEpisode.from_planning_checkpoint(record)
                    row.update(
                        _summary(record["episode"]["runtime"]["backend_report"]),
                        rejections=metrics["rejections"],
                        native_replay_verified=verified.verification["quality"][
                            "passed"
                        ],
                        information="current public observations only",
                    )
                    del verified
                atomic_write(path / "row.json", canonical_json(row))
            except (ValueError, RuntimeError, KeyError, OSError) as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
                atomic_write(path / "row.json", canonical_json(row))
            rows.append(row)
            atomic_write(
                root / "progress.json",
                canonical_json(
                    {
                        "expected": len(spec.cases) * (1 + len(spec.baselines)),
                        "finished": len(rows),
                        "rows": rows,
                    }
                ),
            )
    teacher = forecast_teacher_report(spec, rows)
    files = {
        p.relative_to(root).as_posix(): file_digest(p)
        for p in root.rglob("*")
        if p.is_file()
    }
    body = {
        "schema": "space-env.forecast-benchmark/1",
        "spec_sha256": digest(frozen),
        "rows": rows,
        "teacher": teacher,
        "execution_integrity_passed": all(
            r["error"] is None and r["native_replay_verified"] is True for r in rows
        ),
        "files_sha256": files,
        "expert_teacher_qualified": False,
    }
    result = body | {"sha256": digest(body)}
    atomic_write(root / "result.json", canonical_json(result))
    return copy.deepcopy(result)


def verify_forecast_benchmark(directory, *, native=False):
    if type(native) is not bool:
        raise ValueError("native audit flag must be boolean")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("forecast benchmark requires real directory")
    result = read_json(root / "result.json")
    spec = ForecastBenchmarkSpec.model_validate(read_json(root / "spec.json"))
    if result.get("sha256") != digest(
        {k: v for k, v in result.items() if k != "sha256"}
    ):
        raise ValueError("forecast benchmark checksum mismatch")
    if (
        result.get("schema") != "space-env.forecast-benchmark/1"
        or result.get("expert_teacher_qualified") is not False
    ):
        raise ValueError("forecast benchmark schema/qualification mismatch")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("forecast benchmark symlinks forbidden")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
            if len(actual) > 512:
                raise ValueError("forecast benchmark artifact count exceeded")
    if sum((root / name).stat().st_size for name in actual) > 4294967296:
        raise ValueError("forecast benchmark artifact byte budget exceeded")
    if actual != set(result["files_sha256"]) | {"result.json"}:
        raise ValueError("forecast benchmark inventory mismatch")
    for name, sha in result["files_sha256"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or file_digest(root / name) != sha
        ):
            raise ValueError("forecast benchmark file checksum/path mismatch")
    if result["spec_sha256"] != digest(spec.model_dump(mode="json")) or canonical_json(
        result["teacher"]
    ) != canonical_json(forecast_teacher_report(spec, result["rows"])):
        raise ValueError("forecast benchmark teacher recomputation mismatch")
    rows = []
    for i, case in enumerate(spec.cases):
        for policy in ("forecast_delivery", *spec.baselines):
            path = root / f"{i:03d}" / policy
            row = read_json(path / "row.json")
            rows.append(row)
            if (
                row.get("case") != case.id
                or row.get("policy") != policy
                or row.get("split") != case.split
                or row.get("negative") is not case.negative
            ):
                raise ValueError("forecast row/case metadata mismatch")
            if row["error"] is not None:
                continue
            if policy == "forecast_delivery":
                view = DeliveryForecast(
                    canonical_json(read_json(path / "forecast.json"))
                )
                schedule = schedule_delivery(
                    view, spec.settings.model_dump(mode="json")
                )
                if canonical_json(schedule) != canonical_json(
                    read_json(path / "schedule.json")
                ):
                    raise ValueError("forecast schedule recomputation mismatch")
                plan = read_json(path / "plan.json")
                if (
                    view.view()["scenario_sha256"] != digest(case.scenario)
                    or view.view()["task_sha256"]
                    != digest(
                        TaskSpec.model_validate(case.task).model_dump(
                            mode="json", by_alias=True
                        )
                    )
                    or canonical_json(plan["scenario"]) != canonical_json(case.scenario)
                    or canonical_json(plan["task"])
                    != canonical_json(
                        TaskSpec.model_validate(case.task).model_dump(
                            mode="json", by_alias=True
                        )
                    )
                    or plan["goals"]["min_fulfilled"] != case.min_fulfilled
                    or canonical_json(plan)
                    != canonical_json(read_json(path / "dataset/plan.json"))
                    or plan["origin"] != "declared"
                    or plan["expected_native_sha256"] is not None
                    or [
                        {k: s[k] for k in ("at_sec", "advance_sec", "action_ids")}
                        for s in plan["steps"]
                    ]
                    != schedule["steps"]
                ):
                    raise ValueError("forecast view/plan/case binding mismatch")
                if native:
                    runtime = MissionRuntime(
                        case.scenario, access=AccessPolicy(forecast=True)
                    )
                    fresh = collect_delivery_forecast(runtime, case.scenario, case.task)
                    if fresh.payload != view.payload:
                        raise ValueError(
                            "forecast differs from authorized native geometry"
                        )
                audit = verify_plan_bundle(path / "dataset", native=native)
                if audit["mission_goal_passed"] is not row["mission_goal_passed"]:
                    raise ValueError("forecast candidate certificate mismatch")
                record = read_json(path / "execution.json")
                if canonical_json(record) != canonical_json(
                    read_json(path / "dataset/execution.json")
                ):
                    raise ValueError("forecast execution/bundle mismatch")
                cert = read_json(path / "dataset/certificate.json")
                if (
                    row["predicted_fulfilled"] != schedule["predicted_fulfilled"]
                    or row["rejections"] != cert["rejections"]
                ):
                    raise ValueError("forecast prediction/rejection summary mismatch")
            else:
                record = read_json(path / "planning.json")
                if (
                    not _sealed(record)
                    or not _sealed(record["episode"])
                    or not _sealed(record["episode"]["runtime"])
                    or canonical_json(record["episode"]["scenario"])
                    != canonical_json(case.scenario)
                    or record["policy"]["config"]["name"] != policy
                ):
                    raise ValueError("baseline checkpoint/case binding mismatch")
                expected_task = (
                    TaskSpec.model_validate(case.task)
                    .model_copy(update={"policy_origin": "scripted"})
                    .model_dump(mode="json", by_alias=True)
                )
                if canonical_json(record["episode"]["task"]) != canonical_json(
                    expected_task
                ) or row["rejections"] != sum(
                    not r["feedback"]["accepted"] for r in record["trace"]
                ):
                    raise ValueError("baseline task/rejection binding mismatch")
                if native:
                    verified = VerifiedEpisode.from_planning_checkpoint(record)
                    if not verified.verification["quality"]["passed"]:
                        raise ValueError("baseline quality failed")
            if any(
                row.get(k) != v
                for k, v in _summary(
                    record["episode"]["runtime"]["backend_report"]
                ).items()
            ):
                raise ValueError("forecast row native summary mismatch")
    if canonical_json(rows) != canonical_json(result["rows"]):
        raise ValueError("forecast paired rows mismatch")
    if canonical_json(read_json(root / "progress.json")) != canonical_json(
        {"expected": len(rows), "finished": len(rows), "rows": rows}
    ):
        raise ValueError("forecast progress/rows mismatch")
    integrity = all(
        r["error"] is None and r["native_replay_verified"] is True for r in rows
    )
    if result["execution_integrity_passed"] is not integrity:
        raise ValueError("forecast execution aggregate mismatch")
    return {
        "integrity_passed": True,
        "execution_integrity_passed": integrity,
        "native_execution_replayed": native,
        "teacher_status": result["teacher"]["status"],
        "expert_teacher_qualified": False,
        "sha256": result["sha256"],
    }
