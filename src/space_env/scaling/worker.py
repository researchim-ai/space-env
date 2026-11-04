"""One workload per fresh interpreter; no pickle/native-state deserialization."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from space_env.planning import PlanningRun
from space_env.planning.runner import process_peak_rss_bytes
from space_env.scaling.benchmark import Workload
from space_env.scaling.campaign import campaign_evidence
from space_env.scaling.progress import StageReporter
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks import VerifiedEpisode


def _run(run, progress, decisions=None, samples=None):
    if samples is not None:
        from space_env.scaling.numerics import numerical_snapshot
        if not samples or samples[-1]["time_sec"] != run.session._runtime.time_sec:
            samples.append(numerical_snapshot(run))
    if progress is None and samples is None:
        return run.run(decisions=decisions)
    count = 0
    while not run.done and (decisions is None or count < decisions):
        run.step()
        count += 1
        if samples is not None:
            samples.append(numerical_snapshot(run))
        if progress is not None:
            progress.emit(
                progress.stage,
                decisions=len(run.trace),
                physical_sec=run.session._runtime.time_sec,
            )
    return run.metrics()


def measure(value, *, progress=None, restart_packet=None, restart_nonce=None):
    def stage(name):
        if progress is not None:
            progress.emit(name)

    stage("initialize")
    w = Workload.model_validate(value)
    samples = [] if w.sample_numerics else None
    packet = None
    resume_sec = 0.0
    prefix_bytes = 0
    process_restart = None
    if w.resume_in_new_process:
        from space_env.scaling.restart import (
            MAX_PACKET_BYTES,
            read_json,
            validate_packet,
        )

        if restart_packet is None or restart_nonce is None:
            raise ValueError("new-process worker requires a bound restart packet")
        stage("resume")
        resumed_at = time.perf_counter()
        packet = read_json(
            restart_packet, min(w.max_checkpoint_bytes, MAX_PACKET_BYTES)
        )
        checkpoint = validate_packet(packet, w, restart_nonce, consumer_pid=os.getpid())
        run = PlanningRun.from_checkpoint(checkpoint)
        if run.done:
            raise ValueError("restart prefix is already terminated")
        resume_sec = time.perf_counter() - resumed_at
        prefix_bytes = Path(restart_packet).stat().st_size
        process_restart = {
            "schema": "space-env.process-continuation/1",
            "producer_pid": packet["producer_pid"],
            "consumer_pid": os.getpid(),
            "nonce": restart_nonce,
            "packet_sha256": packet["sha256"],
            "prefix_decisions": packet["prefix_decisions"],
            "build": packet["build"],
            "restore_method": "PlanningRun.from_checkpoint audited prefix replay in a fresh interpreter",
        }
    else:
        if restart_packet is not None or restart_nonce is not None:
            raise ValueError("legacy worker cannot consume a restart packet")
        run = PlanningRun(w.scenario, w.task, w.policy.model_dump(mode="json"))
    started = time.perf_counter()
    if w.resume_after_decisions is not None and not w.resume_in_new_process:
        stage("prefix_execution")
        _run(run, progress, decisions=w.resume_after_decisions, samples=samples)
        if run.done:
            raise ValueError("declared resume point reached a terminated episode")
        stage("prefix_checkpoint")
        prefix = run.checkpoint(compact=True)
        prefix_bytes = len(canonical_json(prefix).encode())
        stage("resume")
        resumed_at = time.perf_counter()
        run = PlanningRun.from_checkpoint(prefix)
        resume_sec = time.perf_counter() - resumed_at
    stage("execution")
    metrics = _run(run, progress, samples=samples)
    execution_sec = time.perf_counter() - started
    if packet is not None:
        execution_sec += packet["prefix_elapsed_sec"] + resume_sec
    stage("fork")
    tracemalloc.start()
    try:
        started = time.perf_counter()
        clone = run.session._runtime.fork()
        fork_sec = time.perf_counter() - started
        _, fork_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del clone
    stage("checkpoint")
    started = time.perf_counter()
    record = run.checkpoint(compact=w.task.get("runtime_audit") == "compact")
    checkpoint_bytes = len(canonical_json(record).encode())
    checkpoint_sec = time.perf_counter() - started
    stage("evidence")
    witness = (
        campaign_evidence(
            w.scenario, record["episode"]["runtime"]["backend_report"], w.campaign_gates
        )
        if w.campaign_gates
        else None
    )
    geometry = None
    if w.geometry_gates is not None:
        from space_env.scaling.independent import geometry_delivery_evidence

        geometry = geometry_delivery_evidence(
            w.scenario, record["episode"]["runtime"]["backend_report"], w.geometry_gates
        )
    coupled = None
    if w.coupled_gates is not None:
        from space_env.scaling.coupled import coupled_evidence
        coupled = coupled_evidence(
            w.scenario, record["episode"]["runtime"]["backend_report"], w.coupled_gates
        )
    steps = [r["step_wall_sec"] for r in run.measurements]
    expected_trace = digest(run.trace)
    del run
    gc.collect()
    started = time.perf_counter()
    stage("replay")
    verified = VerifiedEpisode.from_planning_checkpoint(record)
    replay_sec = time.perf_counter() - started
    quality = verified.verification["quality"]
    with tempfile.TemporaryDirectory(prefix="space-env-scale-export-") as scratch:
        started = time.perf_counter()
        stage("export")
        bundle = verified.export_bundle(Path(scratch) / "dataset")
        exports = list(bundle["files"].values())
        export_sec = time.perf_counter() - started
    peak = process_peak_rss_bytes()
    if packet is not None:
        peak = (
            max(peak, packet["prefix_peak_rss_bytes"])
            if peak is not None and packet["prefix_peak_rss_bytes"] is not None
            else None
        )
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
    if w.max_completion_fraction is not None:
        gates["completion_upper_bound"] = (
            completion is not None and completion <= w.max_completion_fraction
        )
    if witness is not None:
        gates["campaign_witnesses"] = witness["passed"]
    if coupled is not None:
        gates["coupled_mission"] = coupled["passed"]
    if geometry is not None:
        gates["geometry_delivery"] = geometry["passed"]
    if process_restart is not None:
        gates["process_restart"] = (
            process_restart["producer_pid"] != process_restart["consumer_pid"]
        )
    return {
        "passed": all(gates.values()),
        **({"process_restart": process_restart} if process_restart is not None else {}),
        **({"numerical_samples": {"samples": samples, "sha256": digest(samples),
              "scope": "evaluator-only current state; fresh-process restore covers continuation only"}}
           if samples is not None else {}),
        "worker_environment": worker_environment(),
        "gates": gates,
        "metrics": metrics,
        "decisions": len(steps),
        "step_p50_sec": float(np.percentile(steps, 50)) if steps else None,
        "step_p95_sec": float(np.percentile(steps, 95)) if steps else None,
        "execution_sec": execution_sec,
        "resume_sec": resume_sec,
        "prefix_checkpoint_bytes": prefix_bytes,
        "campaign_evidence": witness,
        **({"coupled_evidence": coupled} if coupled is not None else {}),
        **({"geometry_evidence": geometry} if geometry is not None else {}),
        "execution_scope": "execution includes prefix checkpoint/native resume when configured; timing is not replay evidence",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("progress", nargs="?")
    parser.add_argument("--restart-packet")
    parser.add_argument("--restart-nonce")
    args = parser.parse_args()
    value = json.loads(Path(args.input).read_text())
    progress = StageReporter(args.progress, digest(value)) if args.progress else None
    try:
        result = measure(
            value,
            progress=progress,
            restart_packet=args.restart_packet,
            restart_nonce=args.restart_nonce,
        )
        atomic_write(args.output, canonical_json(result))
        if progress is not None:
            progress.emit(progress.stage, status="completed")
    except Exception as exc:
        if progress is not None:
            try:
                progress.emit(
                    progress.stage or "initialize", status="failed", error=exc
                )
            except (OSError, ValueError):
                pass
        raise


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
