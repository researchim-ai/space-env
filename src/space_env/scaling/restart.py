"""Two sequential interpreters exchange a bounded JSON planning checkpoint.

The producer exits before the consumer is started. No forked simulator graph,
pickle, or shared cache is used. PlanningRun's existing restore replays the prefix.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from space_env.simulation.snapshot import atomic_write, canonical_json, digest

MAX_PACKET_BYTES = 64_000_000


def build_identity():
    from space_env import __version__

    package = Path(__file__).resolve().parents[1]
    files = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json", ".csv", ".yaml", ".yml"}
    }
    return {
        "source_data_sha256": digest(files),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "dependencies": {
            n: version(n) for n in ("numpy", "scipy", "pydantic", "pyerfa")
        },
    }


def read_json(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("restart artifact missing, symlink or byte budget exceeded")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("restart byte budget exceeded")
    return json.loads(raw)


def validate_packet(
    packet,
    workload,
    nonce,
    *,
    consumer_pid=None,
    producer_pid=None,
    expected_build=None,
):
    from space_env.planning.model import PolicyConfig
    from space_env.runtime.schema import ScenarioSpec
    from space_env.tasks.schema import TaskSpec

    keys = {
        "schema",
        "nonce",
        "workload_sha256",
        "build",
        "producer_pid",
        "prefix_decisions",
        "prefix_elapsed_sec",
        "prefix_peak_rss_bytes",
        "checkpoint",
        "checkpoint_sha256",
        "sha256",
    }
    if (
        not isinstance(packet, dict)
        or set(packet) != keys
        or packet["schema"] != "space-env.process-restart/1"
    ):
        raise ValueError("invalid restart packet envelope")
    if packet["sha256"] != digest({k: v for k, v in packet.items() if k != "sha256"}):
        raise ValueError("restart packet checksum mismatch")
    if (
        not isinstance(nonce, str)
        or not re.fullmatch(r"[a-f0-9]{32}", nonce)
        or packet["nonce"] != nonce
    ):
        raise ValueError("restart invocation nonce mismatch")
    if packet["workload_sha256"] != digest(workload.model_dump(mode="json")) or packet[
        "build"
    ] != (build_identity() if expected_build is None else expected_build):
        raise ValueError("restart workload/build/dependency identity mismatch")
    pid = packet["producer_pid"]
    if (
        type(pid) is not int
        or pid <= 0
        or (producer_pid is not None and pid != producer_pid)
        or (consumer_pid is not None and pid == consumer_pid)
    ):
        raise ValueError(
            "restart process identity mismatch or same-process continuation"
        )
    elapsed, peak = packet["prefix_elapsed_sec"], packet["prefix_peak_rss_bytes"]
    import math

    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
        or (peak is not None and (type(peak) is not int or peak < 0))
    ):
        raise ValueError("invalid restart timing/resource fields")
    checkpoint = packet["checkpoint"]
    if (
        not isinstance(checkpoint, dict)
        or packet["checkpoint_sha256"] != checkpoint.get("sha256")
        or checkpoint.get("sha256")
        != digest({k: v for k, v in checkpoint.items() if k != "sha256"})
    ):
        raise ValueError("restart checkpoint identity mismatch")
    if not isinstance(checkpoint.get("trace"), list):
        raise TypeError("invalid restart checkpoint trace")
    if (
        type(packet["prefix_decisions"]) is not int
        or packet["prefix_decisions"] != workload.resume_after_decisions
        or len(checkpoint.get("trace", [])) != packet["prefix_decisions"]
    ):
        raise ValueError("restart prefix decision binding mismatch")
    episode = checkpoint.get("episode")
    policy = checkpoint.get("policy")
    if not isinstance(episode, dict) or not isinstance(policy, dict):
        raise TypeError("invalid restart checkpoint episode/policy")
    expected_task = {
        **workload.task,
        "policy_origin": "random"
        if workload.policy.name == "random_masked"
        else "scripted",
    }
    if (
        ScenarioSpec.model_validate(episode.get("scenario")).model_dump(mode="json")
        != ScenarioSpec.model_validate(workload.scenario).model_dump(mode="json")
        or TaskSpec.model_validate(episode.get("task")).model_dump(mode="json")
        != TaskSpec.model_validate(expected_task).model_dump(mode="json")
        or episode.get("episode_seed") != workload.scenario.get("seed", 0)
        or PolicyConfig.model_validate(policy.get("config")) != workload.policy
        or checkpoint.get("operator_enabled") is not False
    ):
        raise ValueError(
            "restart checkpoint scenario/task/seed/policy binding mismatch"
        )
    return checkpoint


def write_prefix(value, output, nonce, progress_path=None):
    from space_env.planning import PlanningRun
    from space_env.planning.runner import process_peak_rss_bytes
    from space_env.scaling.benchmark import Workload
    from space_env.scaling.progress import StageReporter
    from space_env.scaling.worker import _run

    workload = Workload.model_validate(value)
    if not isinstance(nonce, str) or not re.fullmatch(r"[a-f0-9]{32}", nonce):
        raise ValueError("invalid prefix invocation nonce")
    if not workload.resume_in_new_process:
        raise ValueError("prefix producer requires new-process workload")
    progress = StageReporter(progress_path, digest(value)) if progress_path else None
    started = time.perf_counter()
    try:
        if progress:
            progress.emit("initialize")
        run = PlanningRun(
            workload.scenario, workload.task, workload.policy.model_dump(mode="json")
        )
        if progress:
            progress.emit("prefix_execution")
        _run(run, progress, decisions=workload.resume_after_decisions)
        if run.done:
            raise ValueError("declared resume point reached a terminated episode")
        if progress:
            progress.emit("prefix_checkpoint")
        checkpoint = run.checkpoint(compact=True)
        packet = {
            "schema": "space-env.process-restart/1",
            "nonce": nonce,
            "workload_sha256": digest(workload.model_dump(mode="json")),
            "build": build_identity(),
            "producer_pid": os.getpid(),
            "prefix_decisions": len(run.trace),
            "prefix_elapsed_sec": time.perf_counter() - started,
            "prefix_peak_rss_bytes": process_peak_rss_bytes(),
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint["sha256"],
        }
        packet["sha256"] = digest(packet)
        raw = canonical_json(packet)
        if len(raw.encode()) > min(workload.max_checkpoint_bytes, MAX_PACKET_BYTES):
            raise ValueError("restart prefix checkpoint byte budget exceeded")
        atomic_write(output, raw)
        if progress:
            progress.emit("prefix_checkpoint", status="completed")
    except Exception as exc:
        if progress:
            try:
                progress.emit(
                    progress.stage or "initialize", status="failed", error=exc
                )
            except (OSError, ValueError):
                pass
        raise


def execute_restart(workload, source, result, environment, flags, progress_path, row):
    directory = progress_path.parent if progress_path is not None else source.parent
    packet_path = directory / "restart.json"
    prefix_progress = directory / "prefix-progress.json"
    nonce = uuid.uuid4().hex
    protocol = {
        "schema": "space-env.observed-process-restart/1",
        "nonce": nonce,
        "phases": [],
        "status": "running",
        "scope": "local parent-observed sequential subprocesses, not an author signature",
    }
    row["restart_protocol"] = protocol
    deadline = time.monotonic() + workload.timeout_sec

    def launch(phase, command):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, workload.timeout_sec)
        observed = {"phase": phase, "status": "launching"}
        protocol["phases"].append(observed)
        started = time.perf_counter()
        with subprocess.Popen(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as child:
            observed.update(pid=child.pid, status="running")
            try:
                _stdout, stderr = child.communicate(
                    timeout=max(0.001, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
                observed.update(
                    status="timeout", elapsed_sec=time.perf_counter() - started
                )
                protocol["status"] = "failed"
                raise
            observed.update(
                status="exited",
                returncode=child.returncode,
                elapsed_sec=time.perf_counter() - started,
            )
            if child.returncode:
                protocol["status"] = "failed"
                raise RuntimeError(
                    f"restart {phase} worker exit {child.returncode}: {stderr[-2000:].decode(errors='replace')}"
                )
            return child.pid

    try:
        producer_pid = launch(
            "prefix",
            [
                sys.executable,
                *flags,
                "-m",
                "space_env.scaling.restart",
                str(source),
                str(packet_path),
                nonce,
                str(prefix_progress),
            ],
        )
        packet = read_json(
            packet_path, min(workload.max_checkpoint_bytes, MAX_PACKET_BYTES)
        )
        validate_packet(packet, workload, nonce, producer_pid=producer_pid)
        command = [
            sys.executable,
            *flags,
            "-m",
            "space_env.scaling.worker",
            str(source),
            str(result),
        ]
        if progress_path is not None:
            command.append(str(progress_path.resolve()))
        command.extend(["--restart-packet", str(packet_path), "--restart-nonce", nonce])
        consumer_pid = launch("continuation", command)
        observed = read_json(result, 32_000_000)
        evidence = observed.get("process_restart")
        if (
            not isinstance(evidence, dict)
            or evidence.get("producer_pid") != producer_pid
            or evidence.get("consumer_pid") != consumer_pid
            or producer_pid == consumer_pid
            or evidence.get("packet_sha256") != packet["sha256"]
            or evidence.get("nonce") != nonce
            or observed.get("gates", {}).get("process_restart") is not True
        ):
            raise ValueError(
                "continuation result disagrees with observed restart processes"
            )
        protocol.update(
            status="completed",
            prefix_checkpoint_bytes=packet_path.stat().st_size,
            packet_sha256=packet["sha256"],
        )
        return observed
    except Exception:
        protocol["status"] = "failed"
        # Keep prefix diagnostics visible if the consumer never wrote progress.
        if (
            progress_path is not None
            and not progress_path.exists()
            and prefix_progress.exists()
        ):
            atomic_write(progress_path, prefix_progress.read_text())
        raise


def validate_observed_restart(row, workload):
    """Check agreement of consumer evidence with parent-observed phase outcomes."""
    evidence, protocol = row.get("process_restart"), row.get("restart_protocol")
    if not isinstance(evidence, dict) or not isinstance(protocol, dict):
        raise TypeError("missing process restart evidence")
    phases = protocol.get("phases", [])
    if (
        protocol.get("status") != "completed"
        or not isinstance(phases, list)
        or len(phases) != 2
        or any(not isinstance(p, dict) for p in phases)
        or [p.get("phase") for p in phases] != ["prefix", "continuation"]
    ):
        raise ValueError("restart phase sequence incomplete")
    if any(
        p.get("status") != "exited"
        or type(p.get("returncode")) is not int
        or p["returncode"] != 0
        or type(p.get("pid")) is not int
        or p["pid"] <= 0
        for p in phases
    ):
        raise ValueError("restart phase exit/PID evidence invalid")
    if (
        type(evidence.get("producer_pid")) is not int
        or type(evidence.get("consumer_pid")) is not int
        or phases[0]["pid"] == phases[1]["pid"]
        or evidence.get("producer_pid") != phases[0]["pid"]
        or evidence.get("consumer_pid") != phases[1]["pid"]
        or row.get("worker_environment", {}).get("pid") != phases[1]["pid"]
    ):
        raise ValueError("restart observed and reported PIDs disagree")
    if (
        not isinstance(evidence.get("nonce"), str)
        or not re.fullmatch(r"[a-f0-9]{32}", evidence["nonce"])
        or evidence["nonce"] != protocol.get("nonce")
        or evidence.get("packet_sha256") != protocol.get("packet_sha256")
        or not isinstance(evidence.get("packet_sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", evidence["packet_sha256"])
    ):
        raise ValueError("restart packet/invocation evidence disagrees")
    if (
        type(evidence.get("prefix_decisions")) is not int
        or evidence.get("prefix_decisions") != workload.resume_after_decisions
        or row.get("gates", {}).get("process_restart") is not True
    ):
        raise ValueError("restart decision/gate binding disagrees")


if __name__ == "__main__":
    write_prefix(
        read_json(sys.argv[1], 2_000_000),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4] if len(sys.argv) > 4 else None,
    )
