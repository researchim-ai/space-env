"""Frozen scale workloads executed in fresh processes; measurements are not physics."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from space_env.planning.model import PolicyConfig
from space_env.runtime.schema import StrictModel
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


class ScaleSpec(StrictModel):
    schema_id: Literal["space-env.scale-manifest/1"] = Field(
        default="space-env.scale-manifest/1", alias="schema"
    )
    id: str = Field(min_length=1, max_length=128)
    workloads: tuple[Workload, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique(self):
        if (
            len({w.id for w in self.workloads}) != len(self.workloads)
            or sum(w.repeats for w in self.workloads) > 128
        ):
            raise ValueError("duplicate workload or repeat budget exceeded")
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
    rows = []
    for workload in spec.workloads:
        for repeat in range(workload.repeats):
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
                    atomic_write(
                        source, canonical_json(workload.model_dump(mode="json"))
                    )
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
                    child = subprocess.run(
                        [
                            sys.executable,
                            *flags,
                            "-m",
                            "space_env.scaling.worker",
                            str(source),
                            str(result),
                        ],
                        env=environment,
                        capture_output=True,
                        timeout=workload.timeout_sec,
                        check=False,
                    )
                    if child.returncode or not result.is_file():
                        raise RuntimeError(
                            f"isolated worker exit {child.returncode}: {child.stderr[-2000:].decode(errors='replace')}"
                        )
                    row.update(json.loads(result.read_text()))
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.TimeoutExpired,
            ) as exc:
                row.update(
                    error={"type": type(exc).__name__, "message": str(exc)[:2000]},
                    passed=False,
                )
            row["end_to_end_wall_sec"] = time.perf_counter() - started
            rows.append(row)
    summary = summarize(spec, rows)
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
        "rows": rows,
        "summary": summary,
        "ok": all(r.get("passed", False) for r in rows)
        and all(v["passed"] for v in summary.values()),
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
