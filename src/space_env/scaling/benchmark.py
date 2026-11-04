"""Frozen scale workloads executed in fresh processes; measurements are not physics."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from space_env.planning.model import PolicyConfig
from space_env.runtime.schema import StrictModel
from space_env.scaling.campaign import CampaignGates
from space_env.scaling.coupled import CoupledGates
from space_env.scaling.ground import ground_window_diagnostic
from space_env.scaling.independent import GeometryDeliveryGates
from space_env.scaling.progress import read_progress
from space_env.scaling.qualification import qualification_domains
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


class Workload(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    family: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=1024)
    scenario: dict
    task: dict
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    repeats: int = Field(default=1, ge=1, le=10, strict=True)
    timeout_sec: int = Field(default=300, ge=1, le=3600, strict=True)
    max_checkpoint_bytes: int = Field(
        default=268435456, ge=1, le=4294967296, strict=True
    )
    max_peak_rss_bytes: int = Field(
        default=4294967296, ge=1, le=17179869184, strict=True
    )
    worker_threads: int = Field(default=1, ge=1, le=64, strict=True)
    max_step_p95_sec: float | None = Field(default=None, gt=0, strict=True)
    max_execution_sec: float | None = Field(default=None, gt=0, strict=True)
    min_completion_fraction: float = Field(default=0.0, ge=0.0, le=1.0, strict=True)

    max_completion_fraction: float | None = Field(default=None, ge=0, le=1, strict=True)
    campaign_gates: CampaignGates | None = None
    geometry_gates: GeometryDeliveryGates | None = None
    coupled_gates: CoupledGates | None = None
    sample_numerics: bool = Field(default=False, strict=True)
    resume_after_decisions: int | None = Field(default=None, ge=1, le=9999, strict=True)
    resume_in_new_process: bool = Field(default=False, strict=True)
    comparison_group: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")

    @model_validator(mode="after")
    def completion_bounds(self):
        if self.sample_numerics and (
            self.coupled_gates is None or len(self.scenario.get("actor_order", [])) > 4
            or self.task.get("max_decisions", 256) > 601
        ):
            raise ValueError("numerical sampling requires bounded coupled workload")
        if self.coupled_gates is not None and (
            self.scenario.get("backend") != "reference-spacecraft"
            or self.geometry_gates is not None or self.campaign_gates is not None
        ):
            raise ValueError("coupled gates require exclusive spacecraft profile")
        if self.geometry_gates is not None and (
            self.scenario.get("backend") != "reference-fixed"
            or self.campaign_gates is not None
        ):
            raise ValueError(
                "geometry gates require the independent fixed-delivery profile"
            )
        if self.resume_in_new_process and self.resume_after_decisions is None:
            raise ValueError(
                "new-process continuation requires a declared resume point"
            )
        if (
            self.max_completion_fraction is not None
            and self.max_completion_fraction < self.min_completion_fraction
        ):
            raise ValueError("completion upper bound is below the lower bound")
        if (
            self.resume_after_decisions is not None
            and self.resume_after_decisions >= self.task.get("max_decisions", 256)
        ):
            raise ValueError("resume must occur before the decision budget")
        return self

    @model_serializer(mode="wrap")
    def legacy_defaults(self, handler):
        value = handler(self)
        if not self.resume_in_new_process:
            value.pop("resume_in_new_process", None)
        if not self.sample_numerics:
            value.pop("sample_numerics", None)
        if self.coupled_gates is None:
            value.pop("coupled_gates", None)
        if self.geometry_gates is None:
            value.pop("geometry_gates", None)
        return value


class ScaleSpec(StrictModel):
    schema_id: Literal["space-env.scale-manifest/1"] = Field(
        default="space-env.scale-manifest/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=128)
    workloads: tuple[Workload, ...] = Field(min_length=1, max_length=64)
    workers: int = Field(default=1, ge=1, le=4, strict=True)

    @model_validator(mode="after")
    def unique(self):
        if (
            len({w.id for w in self.workloads}) != len(self.workloads)
            or sum(w.repeats for w in self.workloads) > 128
        ):
            raise ValueError("duplicate workload or repeat budget exceeded")
        groups = {}
        for workload in self.workloads:
            if workload.comparison_group:
                groups.setdefault(workload.comparison_group, []).append(workload)
        for items in groups.values():
            if (
                len(items) < 2
                or len(
                    {
                        digest(
                            {
                                "scenario": w.scenario,
                                "task": w.task,
                                "policy": w.policy.model_dump(mode="json"),
                            }
                        )
                        for w in items
                    }
                )
                != 1
            ):
                raise ValueError(
                    "semantic comparison requires at least two workloads with identical native/policy inputs"
                )
        return self


def run_scale(manifest, output):
    spec = ScaleSpec.model_validate(manifest)
    frozen = spec.model_dump(mode="json", by_alias=True)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("scale output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output / "manifest.json",
        canonical_json({"manifest": frozen, "sha256": digest(frozen)}),
    )
    atomic_write(
        output / "preflight.json",
        canonical_json(
            {
                "schema": "space-env.scale-preflight/1",
                "manifest_sha256": digest(frozen),
                "workloads": {
                    w.id: ground_window_diagnostic(w.model_dump(mode="json"))
                    for w in spec.workloads
                },
                "scope": "diagnostic only; no input, gate or policy modification and no worker suppression",
            }
        ),
    )
    jobs = iter(
        enumerate((w, repeat) for w in spec.workloads for repeat in range(w.repeats))
    )

    def execute(job):
        index, (workload, repeat) = job
        directory = output / "workers" / f"{index:06d}"
        directory.mkdir(parents=True)
        return _execute_workload(
            workload, repeat, progress_path=directory / "progress.json"
        )

    rows = []

    def record_progress():
        atomic_write(
            output / "progress.json",
            canonical_json(
                {
                    "schema": "space-env.scale-progress/1",
                    "manifest_sha256": digest(frozen),
                    "expected_repeats": sum(w.repeats for w in spec.workloads),
                    "finished_repeats": len(rows),
                    "rows": rows,
                    "scope": "partial observed rows; not final readiness or automatic resume",
                }
            ),
        )

    record_progress()
    if spec.workers == 1:
        for job in jobs:
            rows.append(execute(job))
            record_progress()
    else:
        with ThreadPoolExecutor(max_workers=spec.workers) as pool:
            pending = deque()
            for _ in range(spec.workers):
                job = next(jobs, None)
                if job is not None:
                    pending.append(pool.submit(execute, job))
            while pending:
                rows.append(pending.popleft().result())
                record_progress()
                job = next(jobs, None)
                if job is not None:
                    pending.append(pool.submit(execute, job))
    summary = summarize(spec, rows)
    comparisons = compare_semantics(spec, rows)
    hardware = {
        "system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "thread_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    try:
        hardware["memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf(
            "SC_PAGE_SIZE"
        )
    except (ValueError, OSError, AttributeError):
        hardware["memory_bytes"] = None
    body = {
        "schema": "space-env.scale-result/1",
        "manifest_sha256": digest(frozen),
        "hardware": hardware,
        "concurrent_workers": spec.workers,
        "rows": rows,
        "semantic_comparisons": comparisons,
        "qualification_domains": qualification_domains(spec, rows, comparisons),
        "summary": summary,
        "ok": all(r.get("passed", False) for r in rows)
        and all(v["passed"] for v in summary.values())
        and all(v["passed"] for v in comparisons.values()),
        "scope": "only the frozen workloads on this hardware; peak RSS is fresh-process lifetime high-water, not delta allocation or flight qualification",
    }
    receipt = {**body, "sha256": digest(body)}
    atomic_write(output / "result.json", canonical_json(receipt))
    return receipt


def summarize(spec, rows):
    """All repeats remain in the denominator; timing summaries exclude failures explicitly."""
    import numpy as np

    result = {}
    for workload in spec.workloads:
        group = [r for r in rows if r["id"] == workload.id]
        successful = [r for r in group if r.get("passed") and r.get("error") is None]
        signatures = {
            (r["native_backend_sha256"], r["trace_sha256"]) for r in successful
        }
        result[workload.id] = {
            "expected_repeats": workload.repeats,
            "observed_repeats": len(group),
            "passed_repeats": len(successful),
            "failed_repeats": len(group) - len(successful),
            "repeat_semantics_identical": len(signatures) == 1,
            "execution_p50_sec_successful_only": float(
                np.percentile([r["execution_sec"] for r in successful], 50)
            )
            if successful
            else None,
            "execution_p95_sec_successful_only": float(
                np.percentile([r["execution_sec"] for r in successful], 95)
            )
            if successful
            else None,
            "max_peak_rss_bytes": max(
                (r["peak_rss_bytes"] for r in successful), default=None
            ),
            "passed": len(group) == workload.repeats
            and len(successful) == workload.repeats
            and len(signatures) == 1,
        }
    return result


def compare_semantics(spec, rows):
    result = {}
    for group in sorted(
        {w.comparison_group for w in spec.workloads if w.comparison_group}
    ):
        workloads = [w for w in spec.workloads if w.comparison_group == group]
        ids = {w.id for w in workloads}
        selected = [r for r in rows if r["id"] in ids]
        expected = sum(w.repeats for w in workloads)
        successful = [r for r in selected if r.get("passed") and not r.get("error")]
        signatures = {
            (r["native_backend_sha256"], r["trace_sha256"]) for r in successful
        }
        result[group] = {
            "workloads": sorted(ids),
            "expected_repeats": expected,
            "successful_repeats": len(successful),
            "semantic_signatures": len(signatures),
            "passed": len(selected) == expected
            and len(successful) == expected
            and len(signatures) == 1,
        }
    return result


def _execute_workload(workload, repeat, *, progress_path=None):
    row = {
        "id": workload.id,
        "family": workload.family,
        "repeat": repeat,
        "error": None,
    }
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="space-env-scale-") as scratch:
            source, result = (
                Path(scratch) / "input.json",
                Path(scratch) / "result.json",
            )
            atomic_write(source, canonical_json(workload.model_dump(mode="json")))
            environment = os.environ.copy()
            environment.update(
                {
                    k: str(workload.worker_threads)
                    for k in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                }
            )
            flags = (
                ["-I"]
                if sys.flags.isolated
                else (["-s"] if sys.flags.no_user_site else [])
            )
            if workload.resume_in_new_process:
                from space_env.scaling.restart import execute_restart

                row.update(
                    execute_restart(
                        workload, source, result, environment, flags, progress_path, row
                    )
                )
                child = None
            else:
                child = subprocess.run(
                    [
                        sys.executable,
                        *flags,
                        "-m",
                        "space_env.scaling.worker",
                        str(source),
                        str(result),
                        *(
                            [str(progress_path.resolve())]
                            if progress_path is not None
                            else []
                        ),
                    ],
                    env=environment,
                    capture_output=True,
                    timeout=workload.timeout_sec,
                    check=False,
                )
            if child is not None and (child.returncode or not result.is_file()):
                raise RuntimeError(
                    f"isolated worker exit {child.returncode}: {child.stderr[-2000:].decode(errors='replace')}"
                )
            if child is not None:
                row.update(json.loads(result.read_text()))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        row.update(
            error={"type": type(exc).__name__, "message": str(exc)[:2000]},
            passed=False,
        )
    if progress_path is not None:
        row["worker_progress"] = read_progress(
            progress_path, digest(workload.model_dump(mode="json"))
        )
        row["worker_progress_file"] = str(
            Path("workers") / progress_path.parent.name / progress_path.name
        )
    row["end_to_end_wall_sec"] = time.perf_counter() - started
    return row
