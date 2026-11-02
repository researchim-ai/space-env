"""Local atomic corpus bundles; privileged audit and policy data stay separate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from space_env.factory.generate import benchmark_manifest, compile_factory
from space_env.factory.quality import (
    content_digest,
    public_view_gate,
    quality_report,
    semantic_metrics,
    sft_reasons,
    trajectory_fingerprint,
)
from space_env.factory.schema import FactorySpec
from space_env.planning.benchmark import run_benchmark, verify_evaluation
from space_env.scaling.spool import RowSpool, SpoolBudget, close_rows
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import (
    array_digest,
    atomic_jsonl,
    file_digest,
    iter_jsonl,
)
from space_env.tasks import TaskSpec, VerifiedEpisode
from space_env.tasks.tools import joint_step_schema


class _RowsBudget(ValueError):
    pass


def _collect(catalogue, evaluation, episodes, scratch):
    spec = FactorySpec.model_validate(catalogue["spec"])
    spool = RowSpool(
        scratch / "rows.sqlite",
        max_bytes=spec.quality.max_spool_bytes,
        max_row_bytes=spec.quality.max_row_bytes,
    )
    try:
        return _collect_rows(catalogue, evaluation, episodes, scratch, spool)
    except BaseException:
        spool.close()
        raise


def _collect_rows(catalogue, evaluation, episodes, scratch, spool):
    spec = FactorySpec.model_validate(catalogue["spec"])
    expected = benchmark_manifest(catalogue)
    if evaluation["manifest"] != expected or evaluation["manifest_sha256"] != digest(
        expected
    ):
        raise ValueError("evaluation is not bound to the frozen factory catalogue")
    if evaluation["sha256"] != digest(
        {k: v for k, v in evaluation.items() if k != "sha256"}
    ):
        raise ValueError("evaluation digest mismatch")
    cases = {c["id"]: c for c in catalogue["cases"]}
    policies = {p.name: p for p in spec.policies}
    expected_pairs = {
        (c["id"], p.name) for c in catalogue["cases"] for p in spec.policies
    }
    pairs = [(r["case"], r["policy"]) for r in evaluation["rows"]]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise ValueError("missing/duplicate factory evaluation rows")
    data = {
        f"data/{split}/{format}.jsonl": spool.rows(f"{split}/{format}")
        for split in ["train", "holdout", "ood"]
        for format in ["rl", "sft"]
    }
    audit, seen, valid_missions = [], spool, set()
    total_rows = 0
    for row in evaluation["rows"]:
        case = cases[row["case"]]
        item = {
            "case": case["id"],
            "policy": row["policy"],
            "split": case["split"],
            "group_sha256": case["group_sha256"],
            "error": None,
            "exports": [],
        }
        if (
            row["split"] != case["split"]
            or row["lineage"] != case["scenario"]["lineage"]
            or row["episode_seed"] != case["episode_seed"]
        ):
            raise ValueError("case assignment mismatch")
        if row["error"] or case["configuration_error"]:
            item["error"] = row["error"] or case["configuration_error"]
            audit.append(item)
            continue
        spool.begin()
        rows_before = total_rows
        try:
            filename = f"{case['id']}-{case['episode_seed']}-{row['policy']}.json"
            if row.get("artifact") != filename:
                raise ValueError("episode artifact binding mismatch")
            record = json.loads((episodes / filename).read_text())
            verified = VerifiedEpisode.from_planning_checkpoint(record)
            run = verified._planning_run
            task = copy.deepcopy(case["task"])
            task["policy_origin"] = (
                "random" if row["policy"] == "random_masked" else "scripted"
            )
            if (
                record["sha256"] != row["run_sha256"]
                or record["episode"]["sha256"] != row["episode_sha256"]
                or digest(record["trace"]) != row["decision_trace_sha256"]
            ):
                raise ValueError("episode/trace receipt mismatch")
            if (
                run.allow_operator
                or record["episode"]["scenario"] != case["scenario"]
                or record["episode"]["task"]
                != TaskSpec.model_validate(task).model_dump(mode="json", by_alias=True)
                or run.policy.config != policies[row["policy"]]
            ):
                raise ValueError("episode borrowed from another case/policy")
            if (
                record["episode"]["episode_seed"] != case["episode_seed"]
                or not row["replay_verified"]
                or semantic_metrics(run.metrics()) != semantic_metrics(row["metrics"])
            ):
                raise ValueError("native metrics/seed binding mismatch")
            item["quality"] = quality_report(run)
            if set(spec.export_roles) - set(run.session._views):
                raise ValueError("requested export role unavailable")
            reasons = sft_reasons(spec, evaluation, row)
            for role in spec.export_roles:
                trajectory = trajectory_fingerprint(run, case, role)
                for format in ["rl", "sft"]:
                    entry = {
                        "format": format,
                        "role": role,
                        "rows": 0,
                        "trajectory_sha256": trajectory,
                        "excluded": [],
                    }
                    if format == "sft":
                        entry["excluded"] = reasons[:] + (
                            [] if role == "@central" else ["SFT_requires_central_role"]
                        )
                    if entry["excluded"]:
                        item["exports"].append(entry)
                        continue
                    key = (format, trajectory)
                    if key in seen:
                        prior = seen[key]
                        if prior["split"] != case["split"]:
                            raise ValueError("duplicate trajectory crosses splits")
                        entry.update(
                            excluded=["duplicate_trajectory"], duplicate_of=prior
                        )
                        item["exports"].append(entry)
                        continue
                    path = scratch / "rows.jsonl"
                    verified.export(path, format=format, role=role)
                    for value in iter_jsonl(
                        path, max_row_bytes=spec.quality.max_row_bytes
                    ):
                        if format == "sft":
                            for message in value["messages"]:
                                if message["role"] in {"user", "tool"}:
                                    public_view_gate(json.loads(message["content"]))
                        else:
                            public_view_gate(value["observation"])
                            public_view_gate(value["next_observation"])
                        value.update(
                            schema="space-env.factory-dataset/1",
                            legacy_lineage_split=value["split"],
                            split=case["split"],
                            split_source="predeclared_factory_manifest",
                            factory_manifest_sha256=catalogue["sha256"],
                            source_case=case["id"],
                            source_policy=row["policy"],
                            geometry_group_sha256=case["group_sha256"],
                            mission_semantic_sha256=case["semantic_sha256"],
                            trajectory_sha256=trajectory,
                            planning_trace_sha256=row["decision_trace_sha256"],
                            teacher_status=evaluation["teacher"]["status"],
                            expert_qualified=False,
                            parameter_status="synthetic",
                            quality="verified_synthetic_demonstration"
                            if format == "sft"
                            else "verified_native_transition",
                            action_map={
                                "schema": "space-env.task-action-map/1",
                                "ids": ["wait", *[a.id for a in run.session._actions]],
                                "slots": run.session._task.max_joint_actions
                                if role == "@central"
                                else run.session._task.roles[role].action_slots,
                            },
                        )
                        if format == "sft":
                            value["tools"] = [
                                {"type": "function", "function": joint_step_schema()}
                            ]
                        total_rows += 1
                        if total_rows > spec.quality.max_rows:
                            raise _RowsBudget("factory row budget exceeded")
                        data[f"data/{case['split']}/{format}.jsonl"].append(value)
                        entry["rows"] += 1
                    seen[key] = {
                        "case": case["id"],
                        "policy": row["policy"],
                        "role": role,
                        "split": case["split"],
                    }
                    item["exports"].append(entry)
            valid_missions.add(case["semantic_sha256"])
            spool.finish()
        except (_RowsBudget, SpoolBudget):
            raise
        except (ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
            item["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            spool.finish(rollback=True)
            total_rows = rows_before
            for entry in item["exports"]:
                entry["rows"] = 0
                entry["excluded"].append("episode_quality_error")
        audit.append(item)
    hashes = {p: array_digest(v) for p, v in data.items()}
    counts = {
        f: sum(len(v) for p, v in data.items() if p.endswith("/" + f + ".jsonl"))
        for f in ["rl", "sft"]
    }
    errors = sum(a["error"] is not None for a in audit)
    gates = {
        "episode_errors": errors <= spec.quality.max_error_episodes,
        "minimum_rl_rows": counts["rl"] >= spec.quality.min_rl_rows,
        "minimum_sft_rows": counts["sft"] >= spec.quality.min_sft_rows,
        "unique_valid_missions": len(valid_missions)
        >= spec.quality.min_unique_missions,
    }
    report = {
        "schema": "space-env.factory-quality/1",
        "ready": all(gates.values()),
        "gates": gates,
        "episode_errors": errors,
        "episodes": len(audit),
        "row_counts": counts,
        "unique_valid_missions": len(valid_missions),
        "data_hashes": hashes,
        "audit": audit,
        "coverage": catalogue["coverage"],
        "teacher": evaluation["teacher"],
        "expert_qualified": False,
        "content_sha256": content_digest(catalogue, hashes, audit),
    }
    return data, report


def _inventory(directory):
    result = {}
    for p in sorted(directory.rglob("*")):
        if p.is_symlink():
            raise ValueError("bundle symlinks are forbidden")
        if p.is_file():
            result[p.relative_to(directory).as_posix()] = {
                "sha256": file_digest(p),
                "bytes": p.stat().st_size,
            }
    return result


def build_dataset(value, output):
    catalogue = compile_factory(value)
    spec = FactorySpec.model_validate(catalogue["spec"])
    output = Path(output)
    if output.exists():
        raise ValueError("factory output must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".factory-", dir=output.parent))
    try:
        atomic_write(stage / "manifest.json", canonical_json(catalogue))
        evaluation = run_benchmark(
            benchmark_manifest(catalogue), artifacts=stage / "episodes"
        )
        atomic_write(stage / "benchmark.json", canonical_json(evaluation))
        with tempfile.TemporaryDirectory(prefix="factory-rows-") as work:
            data, report = _collect(
                catalogue, evaluation, stage / "episodes", Path(work)
            )
            try:
                for path, rows in data.items():
                    target = stage / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    atomic_jsonl(
                        target,
                        rows,
                        max_row_bytes=spec.quality.max_row_bytes,
                        max_rows=spec.quality.max_rows,
                        max_bytes=spec.quality.max_bundle_bytes,
                    )
            finally:
                close_rows(data)
        atomic_write(stage / "report.json", canonical_json(report))
        files = _inventory(stage)
        if sum(f["bytes"] for f in files.values()) > spec.quality.max_bundle_bytes:
            raise ValueError("factory bundle byte budget exceeded")
        body = {
            "schema": "space-env.factory-bundle/1",
            "status": "ready" if report["ready"] else "quarantined",
            "manifest_sha256": catalogue["sha256"],
            "content_sha256": report["content_sha256"],
            "files": files,
        }
        receipt = {**body, "sha256": digest(body)}
        if (
            sum(f["bytes"] for f in files.values())
            + len(canonical_json(receipt).encode())
            > spec.quality.max_bundle_bytes
        ):
            raise ValueError("factory bundle byte budget exceeded including receipt")
        atomic_write(stage / "bundle.json", canonical_json(receipt))
        if output.exists():
            raise ValueError("factory output appeared during build")
        os.rename(stage, output)
        return {
            "ok": report["ready"],
            "status": receipt["status"],
            "rows": report["row_counts"],
            "episodes": report["episodes"],
            "sha256": receipt["sha256"],
            "content_sha256": receipt["content_sha256"],
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_bundle(directory, *, replay=True):
    if type(replay) is not bool:
        raise TypeError("replay must be boolean")
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("bundle root cannot be a symlink")
    if (directory / "bundle.json").is_symlink():
        raise ValueError("bundle receipt cannot be a symlink")
    receipt = json.loads((directory / "bundle.json").read_text())
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {"schema", "status", "manifest_sha256", "content_sha256", "files", "sha256"}
        or receipt["schema"] != "space-env.factory-bundle/1"
        or receipt["sha256"]
        != digest({k: v for k, v in receipt.items() if k != "sha256"})
    ):
        raise ValueError("invalid bundle receipt")
    if not isinstance(receipt["files"], dict) or len(receipt["files"]) > 4096:
        raise ValueError("invalid bundle file catalogue")
    for name in receipt["files"]:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or name == "bundle.json"
            or "\\" in name
        ):
            raise ValueError("unsafe bundle member")
    actual = _inventory(directory)
    actual.pop("bundle.json", None)
    if actual != receipt["files"]:
        raise ValueError("bundle member digest/size/inventory mismatch")
    catalogue = json.loads((directory / "manifest.json").read_text())
    if not isinstance(catalogue, dict) or not {"spec", "sha256"} <= catalogue.keys():
        raise ValueError("invalid factory catalogue")
    if (
        compile_factory(catalogue["spec"]) != catalogue
        or catalogue["sha256"] != receipt["manifest_sha256"]
    ):
        raise ValueError("factory catalogue replay mismatch")
    report = json.loads((directory / "report.json").read_text())
    if (
        not isinstance(report, dict)
        or type(report.get("ready")) is not bool
        or not {"episodes", "row_counts", "content_sha256"} <= report.keys()
    ):
        raise ValueError("invalid factory report")
    if (
        receipt["status"] != ("ready" if report["ready"] else "quarantined")
        or receipt["content_sha256"] != report["content_sha256"]
    ):
        raise ValueError("factory report binding mismatch")
    if replay:
        evaluation = json.loads((directory / "benchmark.json").read_text())
        verify_evaluation(evaluation)
        with tempfile.TemporaryDirectory(prefix="factory-verify-") as work:
            data, fresh = _collect(
                catalogue, evaluation, directory / "episodes", Path(work)
            )
            try:
                if fresh != report:
                    raise ValueError("factory quality replay mismatch")
                for name, rows in data.items():
                    checksum = hashlib.sha256()
                    for row in rows:
                        checksum.update((canonical_json(row) + "\n").encode())
                    if checksum.hexdigest() != file_digest(directory / name):
                        raise ValueError("factory training row replay mismatch")
            finally:
                close_rows(data)
    return {
        "ok": report["ready"],
        "status": receipt["status"],
        "replay_verified": bool(replay),
        "episodes": report["episodes"],
        "rows": report["row_counts"],
        "content_sha256": receipt["content_sha256"],
    }
