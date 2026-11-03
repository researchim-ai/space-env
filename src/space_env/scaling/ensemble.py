"""Native qualification of assumed parameter variants; all attempts are retained.

This bridge reuses the isolated scale worker and its replay/quality/export gates.
It does not turn parameter assumptions into measured hardware uncertainty.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from pydantic import Field, model_serializer, model_validator

from space_env.calibration.ensemble import compile_ensemble
from space_env.calibration.schema import EnsembleSpec
from space_env.runtime.schema import StrictModel
from space_env.scaling.benchmark import ScaleSpec, Workload, run_scale
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks.schema import TaskSpec


class EnsembleQualificationSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    workload: Workload
    sensitivity: EnsembleSpec
    workers: int = Field(default=1, ge=1, le=4, strict=True)
    resume_after_decisions: int | None = Field(default=1, ge=1, le=9999, strict=True)
    resume_in_new_process: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def budget(self):
        if self.resume_in_new_process and self.resume_after_decisions is None:
            raise ValueError("new-process ensemble requires a restart point")
        task = TaskSpec.model_validate(self.workload.task)
        if "min_completion_fraction" not in self.workload.model_fields_set:
            raise ValueError(
                "ensemble requires an explicitly declared completion lower bound, including zero for integrity-only probes"
            )
        if (
            self.workload.resume_after_decisions is not None
            or self.workload.resume_in_new_process
            or self.workload.comparison_group is not None
        ):
            raise ValueError(
                "ensemble owns restart/comparison bindings; base workload must not set them"
            )
        if (
            self.resume_after_decisions is not None
            and self.resume_after_decisions >= task.max_decisions
        ):
            raise ValueError("ensemble restart must be before task decision budget")
        modes = 2 if self.resume_after_decisions is not None else 1
        count = (self.sensitivity.samples + 1) * modes
        if count > 64 or count * self.workload.repeats > 128:
            raise ValueError(
                "declared ensemble exceeds 64 workloads / 128 execution slots, including baseline and rejected members"
            )
        return self

    @model_serializer(mode="wrap")
    def legacy_defaults(self, handler):
        value = handler(self)
        if not self.resume_in_new_process:
            value.pop("resume_in_new_process", None)
        return value


def prepare_ensemble_qualification(value):
    """Freeze compilation and worker inputs before any output or simulation."""
    if len(canonical_json(value).encode()) > 2_000_000:
        raise ValueError("qualification input exceeds 2MB")
    spec = EnsembleQualificationSpec.model_validate(value)
    frozen = spec.model_dump(mode="json")
    compiled = compile_ensemble(spec.workload.scenario, spec.sensitivity)
    workloads, bindings = [], []

    def bind(label, scenario, index):
        ids = []
        modes = (
            ["warm", "cold"] if spec.resume_after_decisions is not None else ["warm"]
        )
        for mode in modes:
            w = copy.deepcopy(frozen["workload"])
            w.update(
                id=label + "-" + mode,
                scenario=copy.deepcopy(scenario),
                comparison_group=label if len(modes) == 2 else None,
                resume_after_decisions=spec.resume_after_decisions
                if mode == "cold"
                else None,
            )
            if spec.resume_in_new_process and mode == "cold":
                w["resume_in_new_process"] = True
            workloads.append(w)
            ids.append(w["id"])
        bindings.append({"member_index": index, "label": label, "workload_ids": ids})

    bind("baseline", spec.workload.scenario, None)
    for member in compiled["members"]:
        if member["status"] == "structural_candidate":
            bind(f"m{member['index']:03d}", member["scenario"], member["index"])
    scale = ScaleSpec.model_validate(
        {"id": spec.id, "workers": spec.workers, "workloads": workloads}
    ).model_dump(mode="json", by_alias=True)
    result = {
        "schema": "space-env.ensemble-qualification-prepared/1",
        "spec": frozen,
        "compiled": compiled,
        "scale_manifest": scale,
        "bindings": bindings,
        "status": "assumed_sensitivity_only",
        "industrial_readiness": False,
        "scope": "trusted diagnostic inputs; not actor observations, teacher examples or native success evidence",
    }
    if len(canonical_json(result).encode()) > 64_000_000:
        raise ValueError("prepared ensemble exceeds 64MB")
    result["sha256"] = digest(result)
    return result


def _sealed(value):
    return isinstance(value, dict) and value.get("sha256") == digest(
        {k: v for k, v in value.items() if k != "sha256"}
    )


def summarize_ensemble_qualification(prepared, scale_result):
    """Deterministic audit aggregation; hashes establish identity, not authenticity."""
    if not _sealed(prepared) or not _sealed(scale_result):
        raise ValueError("ensemble prepared/scale body checksum mismatch")
    if prepared != prepare_ensemble_qualification(prepared["spec"]):
        raise ValueError(
            "prepared inputs do not match deterministic source compilation"
        )
    scale = ScaleSpec.model_validate(prepared["scale_manifest"])
    if scale_result.get("manifest_sha256") != digest(
        scale.model_dump(mode="json", by_alias=True)
    ):
        raise ValueError("scale result is bound to a different manifest")
    by_id = {w.id: w for w in scale.workloads}
    seen = set()
    for row in scale_result["rows"]:
        if not isinstance(row, dict):
            raise TypeError("execution row must be an object")
        key = (row.get("id"), row.get("repeat"))
        if (
            key[0] not in by_id
            or type(key[1]) is not int
            or not 0 <= key[1] < by_id[key[0]].repeats
            or key in seen
        ):
            raise ValueError("unexpected, duplicate or invalid execution identity")
        seen.add(key)
        if type(row.get("passed")) is not bool:
            raise ValueError("execution row must have an explicit boolean result")
        if row["passed"] and (
            row.get("error") is not None
            or not row.get("gates")
            or any(type(v) is not bool or not v for v in row["gates"].values())
        ):
            raise ValueError("passed execution contradicts retained error/gates")
        if row.get("error") is None:
            required = {
                "native_replay",
                "quality",
                "step_latency_budget",
                "execution_budget",
                "checkpoint_budget",
                "peak_rss_budget",
                "completion",
            }
            if by_id[key[0]].max_completion_fraction is not None:
                required.add("completion_upper_bound")
            if by_id[key[0]].campaign_gates is not None:
                required.add("campaign_witnesses")
            if by_id[key[0]].geometry_gates is not None:
                required.add("geometry_delivery")
            if by_id[key[0]].resume_in_new_process:
                required.add("process_restart")
            if (
                not isinstance(row.get("gates"), dict)
                or set(row["gates"]) != required
                or any(type(v) is not bool for v in row["gates"].values())
            ):
                raise ValueError("native execution gate set is incomplete or invalid")
            if row["passed"] != all(row["gates"].values()):
                raise ValueError("execution result contradicts gate conjunction")
            if by_id[key[0]].resume_in_new_process:
                from space_env.scaling.restart import validate_observed_restart

                validate_observed_restart(row, by_id[key[0]])
            for key_name in ("native_backend_sha256", "trace_sha256"):
                if not isinstance(row.get(key_name), str) or not re.fullmatch(
                    r"[a-f0-9]{64}", row[key_name]
                ):
                    raise ValueError("native execution lacks valid semantic signatures")
        completion = row.get("metrics", {}).get("completion_fraction")
        if completion is not None and (
            isinstance(completion, bool)
            or not isinstance(completion, (int, float))
            or not math.isfinite(completion)
            or not 0 <= completion <= 1
        ):
            raise ValueError("completion fraction outside its finite domain")
    # Recompute formal comparisons from rows, never trust cached summary booleans.
    from space_env.scaling.benchmark import compare_semantics, summarize

    summary = summarize(scale, scale_result["rows"])
    comparisons = compare_semantics(scale, scale_result["rows"])
    if (
        summary != scale_result["summary"]
        or comparisons != scale_result["semantic_comparisons"]
    ):
        raise ValueError("scale summaries disagree with raw rows")
    expected_ok = (
        all(r["passed"] for r in scale_result["rows"])
        and all(v["passed"] for v in summary.values())
        and all(v["passed"] for v in comparisons.values())
    )
    if scale_result.get("ok") is not expected_ok:
        raise ValueError("scale overall result disagrees with raw rows")
    members = []
    for binding in prepared["bindings"]:
        rows = [r for r in scale_result["rows"] if r["id"] in binding["workload_ids"]]
        expected = sum(by_id[k].repeats for k in binding["workload_ids"])
        paired = comparisons.get(binding["label"])
        passed = (
            len(rows) == expected
            and all(summary[k]["passed"] for k in binding["workload_ids"])
            and (paired is None or paired["passed"])
        )
        completion = [r.get("metrics", {}).get("completion_fraction") for r in rows]
        observed = [
            v
            for v in completion
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        members.append(
            {
                **binding,
                "status": "passed" if passed else "native_failed",
                "expected_executions": expected,
                "observed_executions": len(rows),
                "passed_executions": sum(r["passed"] for r in rows),
                "worker_errors": sum(r.get("error") is not None for r in rows),
                "missing_executions": expected - len(rows),
                "completion_observed_count": len(observed),
                "completion_min_observed": min(observed) if observed else None,
                "completion_max_observed": max(observed) if observed else None,
                "restart_semantics": paired,
                "execution_rows": rows,
            }
        )
    baseline = next(m for m in members if m["member_index"] is None)
    variants = {m["member_index"]: m for m in members if m["member_index"] is not None}
    ordered = []
    for member in prepared["compiled"]["members"]:
        if member["status"] == "rejected":
            ordered.append(
                {
                    "member_index": member["index"],
                    "status": "structural_rejected",
                    "error": member["error"],
                    "observed_executions": 0,
                }
            )
        else:
            ordered.append(variants[member["index"]])
    counts = {
        s: sum(m["status"] == s for m in ordered)
        for s in ("passed", "native_failed", "structural_rejected")
    }
    attempted = prepared["compiled"]["attempted"]
    slots_per_member = (
        len(prepared["bindings"][0]["workload_ids"])
        * prepared["spec"]["workload"]["repeats"]
    )
    result = {
        "schema": "space-env.ensemble-qualification-summary/1",
        "prepared_sha256": prepared["sha256"],
        "scale_result_sha256": scale_result["sha256"],
        "baseline": baseline,
        "members": ordered,
        "attempted_variants": attempted,
        **(
            {
                "planned_process_launches": sum(
                    w.repeats * (2 if w.resume_in_new_process else 1)
                    for w in scale.workloads
                )
            }
            if any(w.resume_in_new_process for w in scale.workloads)
            else {}
        ),
        "planned_variant_execution_slots": attempted * slots_per_member,
        "not_submitted_due_to_structural_rejection": counts["structural_rejected"]
        * slots_per_member,
        "counts": counts,
        "variant_pass_fraction_all_attempts": counts["passed"] / attempted,
        "expected_submitted_executions": sum(w.repeats for w in scale.workloads),
        "observed_executions": len(scale_result["rows"]),
        "passed": baseline["status"] == "passed" and counts["passed"] == attempted,
        "industrial_readiness": False,
        "hardware_calibration_complete": False,
        "scope": "all assumed variants and declared native gates; completion includes declared negative expectations, not universal mission feasibility",
        "dataset_policy": "worker verifies replay and exports internally; no corpus or teacher-quality qualification is published here",
    }
    result["sha256"] = digest(result)
    return result


def run_ensemble_qualification(value, output):
    prepared = prepare_ensemble_qualification(value)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("ensemble output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "prepared.json", canonical_json(prepared))
    # Partial scale progress remains on disk on interruption; no successful final
    # receipt is manufactured when the underlying runner cannot finish.
    native = run_scale(prepared["scale_manifest"], output / "native")
    summary = summarize_ensemble_qualification(prepared, native)
    package = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json", ".csv", ".yaml", ".yml"}
    }
    artifacts, total = {}, 0
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if (
            path.is_symlink()
            or size > 64_000_000
            or total > 256_000_000
            or len(artifacts) >= 400
        ):
            raise ValueError("ensemble artifact budget exceeded or symlink encountered")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1_000_000), b""):
                hasher.update(chunk)
        artifacts[str(path.relative_to(output))] = hasher.hexdigest()
    result = {
        "schema": "space-env.ensemble-qualification/1",
        "summary": summary,
        "passed": summary["passed"],
        "artifacts_sha256": artifacts,
        "build": {"files_sha256": hashes, "source_data_sha256": digest(hashes)},
        "industrial_readiness": False,
        "hardware_calibration_complete": False,
        "scope": "frozen local native execution evidence; no author authentication, empirical uncertainty or arbitrary mission qualification",
    }
    result["sha256"] = digest(result)
    if len(canonical_json(result).encode()) > 32_000_000:
        raise ValueError("ensemble final receipt exceeds 32MB")
    atomic_write(output / "result.json", canonical_json(result))
    return result


def verify_ensemble_qualification(output):
    """Offline artifact/aggregation audit, explicitly not a fresh native replay."""
    output = Path(output)

    def read(name, limit):
        path = output / name
        if path.is_symlink() or path.stat().st_size > limit:
            raise ValueError("symlink or oversized ensemble artifact")
        with path.open("rb") as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            raise ValueError("oversized ensemble artifact")
        return json.loads(data)

    result = read("result.json", 32_000_000)
    if not _sealed(result):
        raise ValueError("ensemble receipt checksum mismatch")
    artifacts = result.get("artifacts_sha256")
    if (
        not isinstance(artifacts, dict)
        or not {
            "prepared.json",
            "native/result.json",
            "native/manifest.json",
            "native/preflight.json",
            "native/progress.json",
        }
        <= set(artifacts)
        or len(artifacts) > 400
    ):
        raise ValueError("missing or excessive ensemble artifact inventory")
    total = 0
    for name, expected in result["artifacts_sha256"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or any(
                (output / Path(*relative.parts[:i])).is_symlink()
                for i in range(1, len(relative.parts) + 1)
            )
        ):
            raise ValueError("unsafe ensemble artifact path")
        path = output / relative
        size = path.stat().st_size
        total += size
        if size > 64_000_000 or total > 256_000_000:
            raise ValueError("oversized ensemble artifact")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1_000_000), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected:
            raise ValueError("ensemble artifact byte checksum mismatch")
    prepared = read("prepared.json", 64_000_000)
    native = read("native/result.json", 32_000_000)
    workloads = {
        w.id: w for w in ScaleSpec.model_validate(prepared["scale_manifest"]).workloads
    }
    for row in native["rows"]:
        workload = workloads.get(row.get("id"))
        if (
            workload is not None
            and workload.resume_in_new_process
            and row.get("error") is None
        ):
            from space_env.scaling.restart import validate_packet

            progress_file = row.get("worker_progress_file", "")
            if not isinstance(progress_file, str) or not re.fullmatch(
                r"workers/[0-9]{6}/progress\.json", progress_file
            ):
                raise ValueError("invalid restart worker artifact binding")
            packet_file = str(
                Path("native") / Path(progress_file).parent / "restart.json"
            )
            if packet_file not in artifacts:
                raise ValueError(
                    "restart checkpoint is missing from artifact inventory"
                )
            packet = read(packet_file, 64_000_000)
            evidence = row.get("process_restart", {})
            expected_build = evidence.get("build", {})
            if (
                expected_build.get("source_data_sha256")
                != result["build"]["source_data_sha256"]
            ):
                raise ValueError("restart build differs from ensemble receipt")
            validate_packet(
                packet,
                workload,
                evidence.get("nonce"),
                producer_pid=evidence.get("producer_pid"),
                consumer_pid=evidence.get("consumer_pid"),
                expected_build=expected_build,
            )
            if packet["sha256"] != evidence.get("packet_sha256"):
                raise ValueError("restart checkpoint differs from execution evidence")
    manifest = prepared["scale_manifest"]
    manifest_hash = digest(manifest)
    if read("native/manifest.json", 64_000_000) != {
        "manifest": manifest,
        "sha256": manifest_hash,
    }:
        raise ValueError("persisted native manifest disagrees with prepared inputs")
    preflight = read("native/preflight.json", 16_000_000)
    progress = read("native/progress.json", 32_000_000)
    if (
        preflight.get("manifest_sha256") != manifest_hash
        or progress.get("manifest_sha256") != manifest_hash
    ):
        raise ValueError("native progress/preflight belongs to a different manifest")
    if (
        progress.get("rows") != native["rows"]
        or progress.get("finished_repeats") != len(native["rows"])
        or progress.get("expected_repeats")
        != sum(w["repeats"] for w in manifest["workloads"])
    ):
        raise ValueError("final native progress disagrees with execution rows")
    summary = summarize_ensemble_qualification(prepared, native)
    if summary != result["summary"] or result["passed"] != summary["passed"]:
        raise ValueError("ensemble receipt disagrees with native rows")
    return {
        "schema": "space-env.ensemble-artifact-audit/1",
        "verified": True,
        "qualification_passed": result["passed"],
        "receipt_sha256": result["sha256"],
        "native_reexecuted": False,
        "scope": "artifact identity and deterministic aggregation, not authentication or a fresh simulation",
    }
