"""One workload per fresh interpreter; no pickle/native-state deserialization."""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from space_env.planning import PlanningRun
from space_env.planning.runner import process_peak_rss_bytes
from space_env.scaling.benchmark import Workload
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks import VerifiedEpisode


def measure(value):
    w = Workload.model_validate(value)
    run = PlanningRun(w.scenario, w.task, w.policy.model_dump(mode="json"))
    started = time.perf_counter()
    metrics = run.run()
    execution_sec = time.perf_counter() - started
    tracemalloc.start()
    try:
        started = time.perf_counter()
        clone = run.session._runtime.fork()
        fork_sec = time.perf_counter() - started
        _, fork_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del clone
    started = time.perf_counter()
    record = run.checkpoint(compact=w.task.get("runtime_audit") == "compact")
    checkpoint_bytes = len(canonical_json(record).encode())
    checkpoint_sec = time.perf_counter() - started
    steps = [r["step_wall_sec"] for r in run.measurements]
    expected_trace = digest(run.trace)
    del run
    gc.collect()
    started = time.perf_counter()
    verified = VerifiedEpisode.from_planning_checkpoint(record)
    replay_sec = time.perf_counter() - started
    quality = verified.verification["quality"]
    with tempfile.TemporaryDirectory(prefix="space-env-scale-export-") as scratch:
        started = time.perf_counter()
        bundle = verified.export_bundle(Path(scratch) / "dataset")
        exports = list(bundle["files"].values())
        export_sec = time.perf_counter() - started
    peak = process_peak_rss_bytes()
    completion = metrics.get("completion_fraction")
    gates = {
        "native_replay": verified.verification["trace_sha256"] == expected_trace,
        "quality": quality["passed"],
        "step_latency_budget": w.max_step_p95_sec is None
        or bool(steps)
        and float(np.percentile(steps, 95)) <= w.max_step_p95_sec,
        "execution_budget": w.max_execution_sec is None
        or execution_sec <= w.max_execution_sec,
        "checkpoint_budget": checkpoint_bytes <= w.max_checkpoint_bytes,
        "peak_rss_budget": peak is not None and peak <= w.max_peak_rss_bytes,
        "completion": completion is not None
        and completion >= w.min_completion_fraction,
    }
    return {
        "passed": all(gates.values()),
        "worker_environment": worker_environment(),
        "gates": gates,
        "metrics": metrics,
        "decisions": len(steps),
        "step_p50_sec": float(np.percentile(steps, 50)) if steps else None,
        "step_p95_sec": float(np.percentile(steps, 95)) if steps else None,
        "execution_sec": execution_sec,
        "fork_sec": fork_sec,
        "fork_python_peak_bytes": fork_peak,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_sec": checkpoint_sec,
        "replay_sec": replay_sec,
        "export_sec": export_sec,
        "verified_export_rows": sum(e["rows"] for e in exports),
        "verified_export_rows_per_sec": sum(e["rows"] for e in exports) / export_sec,
        "peak_rss_bytes": peak,
        "native_backend_sha256": digest(record["episode"]["runtime"]["backend_report"]),
        "trace_sha256": expected_trace,
        "quality": quality,
    }


def main():
    result = measure(json.loads(Path(sys.argv[1]).read_text()))
    atomic_write(sys.argv[2], canonical_json(result))


def worker_environment():
    info = {
        "pid": os.getpid(),
        "affinity_cpus": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    for name, path in {
        "cpu_quota": "/sys/fs/cgroup/cpu.max",
        "memory_limit": "/sys/fs/cgroup/memory.max",
        "cpu_model": "/proc/cpuinfo",
    }.items():
        try:
            with open(path) as stream:
                data = stream.read(65536)
            info[name] = (
                next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in data.splitlines()
                        if line.startswith("model name")
                    ),
                    None,
                )
                if name == "cpu_model"
                else data.strip()
            )
        except OSError:
            info[name] = None
    return info


if __name__ == "__main__":
    main()
