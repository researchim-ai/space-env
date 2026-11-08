"""Bounded actual sequence-lane qualification, one fresh interpreter per cell."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from space_env.llm.episode import seal
from space_env.llm.episode import sources as model_sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def sources():
    """Package Python source manifest; workloads are separately frozen in plan."""
    return model_sources()


def workloads():
    """Explicit public regression ladder, not a Cartesian or flight-scale claim."""
    from space_env.api.network_workspace import NetworkConfiguration
    from space_env.llm.mission_suite import compile_suite

    base = compile_suite()["cases"][0]["configuration"]
    rows = []
    for name, actors, requests, horizon in (
        ("tiny", 1, 1, 60.0),
        ("pair", 2, 3, 150.0),
        ("reference", 3, 3, 180.0),
        ("backlog", 4, 8, 600.0),
        ("boundary", 4, 16, 1200.0),
    ):
        c = copy.deepcopy(base)
        m = c["mission"]
        m.update(name="sequence-scale-" + name, horizon_sec=horizon)
        m["planes"][0]["phases_deg"] = [15.0 * i for i in range(actors)]
        m["requests"] = []
        for i in range(requests):
            r = copy.deepcopy(base["mission"]["requests"][i % 3])
            # Small rows explicitly scale deadlines; the reference keeps its
            # published windows. Long rows cover the horizon without feasibility
            # filtering and follow a declared 0.058 deg/s along-track target grid.
            release = (
                5.0 * round((horizon - 180.0) * i / (requests - 1) / 5.0)
                if horizon > 180 and requests > 1
                else 0.0
            )
            deadline = min(
                horizon, release + r["deadline_sec"] * min(1.0, horizon / 180.0)
            )
            r.update(
                id=f"request-{i}",
                release_sec=release,
                deadline_sec=deadline,
                lon_deg=(r["lon_deg"] + 0.058 * release + 180) % 360 - 180,
            )
            m["requests"].append(r)
        c["links"] = [
            {"source": f"fleet-{i:02d}", "destination": f"fleet-{i + 1:02d}"}
            for i in range(actors - 1)
        ]
        c["max_hops"] = min(3, actors - 1)
        rows.append(
            {
                "id": name,
                "actors": actors,
                "requests": requests,
                "horizon_sec": horizon,
                "configuration": NetworkConfiguration.model_validate(c).model_dump(
                    mode="json", by_alias=True
                ),
            }
        )
    return rows


def select(selected):
    ids = [w["id"] for w in workloads()]
    if (
        not isinstance(selected, list)
        or not selected
        or any(type(s) is not str for s in selected)
    ):
        raise ValueError("Invalid selection")
    if len(set(selected)) != len(selected) or not set(selected) <= set(ids):
        raise ValueError("Invalid selection")
    return selected


def _read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 64_000_000:
        raise ValueError("Unsafe/oversized artifact")
    v = json.loads(path.read_bytes())
    if v.get("sha256") != digest({k: x for k, x in v.items() if k != "sha256"}):
        raise ValueError("Artifact seal mismatch")
    return v


def _write(path, value):
    atomic_write(path, canonical_json(value))


def measure(case, expected_sources, output):
    from space_env.llm.benchmark import reference_policy, run_agent
    from space_env.llm.episode import LLMEpisode, verify_record
    from space_env.llm.mission_suite import TASK
    from space_env.planning.runner import process_peak_rss_bytes
    from space_env.planning.sequence_lanes import (
        schedule_sequence,
        verify_sequence_schedule,
    )
    from space_env.scaling.worker import worker_environment
    from space_env.workspace import MissionWorkspace

    if sources() != expected_sources or case not in workloads():
        raise ValueError("Worker source/workload mismatch")
    phases = {}

    def timed(name, fn, *args, **kwargs):
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        phases[name] = {
            "wall_sec": time.perf_counter() - start,
            "cumulative_peak_rss_bytes": process_peak_rss_bytes(),
        }
        return value

    def scheduler(*args, **kwargs):
        return timed("planning", schedule_sequence, *args, **kwargs)

    mission = None
    try:
        start = time.perf_counter()
        # Interpose only a clock around the real scheduler; no substitute plan.
        with patch("space_env.planning.sequence_lanes.schedule_sequence", scheduler):
            mission = MissionWorkspace.network(
                case["configuration"], planning_method="sequence_lanes"
            )
        phases["compile"] = {
            "wall_sec": time.perf_counter() - start - phases["planning"]["wall_sec"],
            "cumulative_peak_rss_bytes": process_peak_rss_bytes(),
        }
        session = mission.operator.session
        meta = session._scenario["metadata"]["network_workspace"]
        timed(
            "schedule_validation",
            verify_sequence_schedule,
            meta["forecast"],
            meta["sequence_geometry"],
            meta["schedule"],
        )
        episode = LLMEpisode(
            session,
            task_text=TASK,
            group=case["id"],
            split="test",
            max_calls=512,
            policy_origin="scripted_privileged",
        )
        record = timed(
            "native_execution",
            run_agent,
            episode,
            reference_policy(
                meta["steps"], {a.id: a.role for a in session._task.actions}
            ),
        )
        timed("replay", verify_record, record)
        if sources() != expected_sources:
            raise ValueError("Source changed during worker")
        _write(output / "episode.json", record)
        return seal(
            {
                "case": case["id"],
                "status": "completed",
                "phases": phases,
                "worker_environment": worker_environment(),
                "metrics": record["metrics"],
                "planned": meta["schedule"]["predicted_fulfilled"],
                "latest_planned_finish_sec": max(
                    (a["finish_sec"] for a in meta["schedule"]["allocations"]),
                    default=0.0,
                ),
                "search_nodes": meta["schedule"]["search_nodes"],
                "search_budget_exhausted": meta["schedule"]["search_budget_exhausted"],
                "episode_sha256": record["sha256"],
                "sourcehashfinal": digest(sources()),
            }
        )
    finally:
        if mission is not None:
            mission.close()


def _limits(timeout_sec, max_rss_bytes, workers=1):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Workers must be an integer in [1, 4]")
    if (
        type(timeout_sec) not in (int, float)
        or not math.isfinite(timeout_sec)
        or not 0 < timeout_sec <= 300
    ):
        raise ValueError("Timeout must be in (0, 300] seconds")
    if (
        type(max_rss_bytes) is not int
        or not 64_000_000 <= max_rss_bytes <= 8_000_000_000
    ):
        raise ValueError("Invalid RSS budget")


def _run_cell(root, name, timeout_sec, max_rss_bytes, run_started):
    """Each parent thread supervises one isolated interpreter at a time."""
    cell = root / name
    cell.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    start = time.perf_counter()
    diagnostics = bytearray()
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "space_env.scaling.sequence",
                "--worker",
                str(root),
                name,
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        stderr = f"{type(exc).__name__}: {str(exc)[:2000]}"
        (cell / "stderr.txt").write_text(stderr)
        error = seal(
            {
                "case": name,
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc)[:2000],
                "sourcehashfinal": digest(sources()),
            }
        )
        _write(cell / "error.json", error)
        return {
            "case": name,
            "status": "error",
            "pid": None,
            "returncode": None,
            "worker_wall_sec": time.perf_counter() - start,
            "started_offset_sec": start - run_started,
            "finished_offset_sec": time.perf_counter() - run_started,
            "observed_peak_rss_bytes": 0,
            "stderr_sha256": digest(stderr),
            "error_sha256": error["sha256"],
        }

    def drain(proc=proc, diagnostics=diagnostics):
        with proc.stderr:
            while chunk := proc.stderr.read(4096):
                diagnostics.extend(chunk[: max(0, 65536 - len(diagnostics))])

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    status = None
    observed_peak = 0
    while proc.poll() is None:
        try:
            fields = Path(f"/proc/{proc.pid}/status").read_text().splitlines()
            rss = next(
                int(line.split()[1]) * 1024
                for line in fields
                if line.startswith("VmRSS:")
            )
            observed_peak = max(observed_peak, rss)
        except (OSError, StopIteration, ValueError):
            pass
        if time.perf_counter() - start >= timeout_sec:
            status = "timeout"
        elif observed_peak > max_rss_bytes:
            status = "rss_limit"
        if status is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            break
        time.sleep(0.02)
    if status is None:
        status = "completed" if proc.returncode == 0 else "error"
    reader.join(timeout=1.0)
    stderr = bytes(diagnostics).decode("utf-8", errors="replace")
    (cell / "stderr.txt").write_text(stderr)
    row = {
        "case": name,
        "stderr_sha256": digest(stderr),
        "status": status,
        "pid": proc.pid,
        "returncode": proc.returncode,
        "worker_wall_sec": time.perf_counter() - start,
        "started_offset_sec": start - run_started,
        "finished_offset_sec": time.perf_counter() - run_started,
        "observed_peak_rss_bytes": observed_peak,
    }
    if status == "completed":
        receipt = _read(cell / "receipt.json")
        row["receipt_sha256"] = receipt["sha256"]
    if (cell / "error.json").exists():
        row["error_sha256"] = _read(cell / "error.json")["sha256"]
    return row


def run(
    output, *, selected=None, timeout_sec=60.0, max_rss_bytes=1_000_000_000, workers=1
):
    """Bounded fresh workers; per-cell deadlines exclude queue waiting time."""
    selected = select(["tiny", "reference"] if selected is None else selected)
    _limits(timeout_sec, max_rss_bytes, workers)
    root = Path(output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    plan = seal(
        {
            "schema": "space-env.sequence-scale-plan/2",
            "workloads": workloads(),
            "selected": selected,
            "model_sources": sources(),
            "timeout_sec": timeout_sec,
            "max_rss_bytes": max_rss_bytes,
            "workers": workers,
            "aggregate_potential_rss_bytes": workers * max_rss_bytes,
        }
    )
    _write(root / "plan.json", plan)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # map yields in selection order, independently of completion order.
        rows = list(
            executor.map(
                lambda name: _run_cell(root, name, timeout_sec, max_rss_bytes, started),
                selected,
            )
        )
    orchestration_wall_sec = time.perf_counter() - started
    final = sources()
    report = seal(
        {
            "schema": "space-env.sequence-scale/2",
            "plan_sha256": plan["sha256"],
            "rows": rows,
            "orchestration_wall_sec": orchestration_wall_sec,
            "sourcehashfinal": digest(final),
            "source_consistent": final == plan["model_sources"],
        }
    )
    _write(root / "report.json", report)
    return report


def audit(directory):
    """Cold native replay, regenerated workload/selection/source and result binding."""
    from space_env.llm.episode import verify_record
    from space_env.llm.mission_suite import TASK
    from space_env.planning.sequence_lanes import verify_sequence_schedule

    root = Path(directory)
    plan, report = _read(root / "plan.json"), _read(root / "report.json")
    selected = select(plan["selected"])
    _limits(plan["timeout_sec"], plan["max_rss_bytes"], plan["workers"])
    if (
        plan["schema"] != "space-env.sequence-scale-plan/2"
        or report["schema"] != "space-env.sequence-scale/2"
        or plan["aggregate_potential_rss_bytes"]
        != plan["workers"] * plan["max_rss_bytes"]
        or plan["workloads"] != workloads()
        or plan["model_sources"] != sources()
        or report["plan_sha256"] != plan["sha256"]
        or not report["source_consistent"]
        or report["sourcehashfinal"] != digest(sources())
        or [r["case"] for r in report["rows"]] != selected
    ):
        raise ValueError("Source/workload/selection binding mismatch")
    elapsed = report["orchestration_wall_sec"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid orchestration measurement")
    events, pids = [], set()
    for row in report["rows"]:
        start, finish = row["started_offset_sec"], row["finished_offset_sec"]
        if (
            any(
                type(t) not in (int, float) or not math.isfinite(t)
                for t in (start, finish, row["worker_wall_sec"])
            )
            or not 0 <= start <= finish <= elapsed
            or row["worker_wall_sec"] < 0
        ):
            raise ValueError("Invalid worker interval")
        events.extend(((start, 1), (finish, -1)))
        pid = row["pid"]
        if pid is not None:
            if type(pid) is not int or pid <= 0 or pid in pids:
                raise ValueError("Invalid/duplicate worker PID")
            pids.add(pid)
    active = 0
    for _, change in sorted(events):
        active += change
        if not 0 <= active <= plan["workers"]:
            raise ValueError("Worker concurrency exceeds frozen limit")
    cases = {w["id"]: w for w in workloads()}
    passed, replayed = 0, 0
    for row in report["rows"]:
        if row["status"] not in {"completed", "timeout", "error", "rss_limit"}:
            raise ValueError("Invalid worker status")
        cell = root / row["case"]
        log = cell / "stderr.txt"
        if (
            log.is_symlink()
            or log.stat().st_size > 262144
            or digest(log.read_text()) != row["stderr_sha256"]
        ):
            raise ValueError("Worker diagnostic mismatch")
        if (cell / "error.json").exists():
            error = _read(cell / "error.json")
            if (
                row["status"] != "error"
                or error["sha256"] != row.get("error_sha256")
                or error["case"] != row["case"]
                or error["status"] != "error"
                or not isinstance(error["error_type"], str)
            ):
                raise ValueError("Worker error binding mismatch")
        elif "error_sha256" in row:
            raise ValueError("Missing worker error")
        if row["status"] != "completed":
            if (cell / "receipt.json").exists():
                raise ValueError("Finished receipt hidden by failure")
            continue
        receipt, record = _read(cell / "receipt.json"), _read(cell / "episode.json")
        meta = record["initial_checkpoint"]["scenario"]["metadata"]["network_workspace"]
        verify_record(record)
        verify_sequence_schedule(
            meta["forecast"], meta["sequence_geometry"], meta["schedule"]
        )
        expected_config = {
            "task_text": TASK,
            "group": row["case"],
            "split": "test",
            "max_calls": 512,
            "max_dialogue_bytes": 16000000,
            "policy_origin": "scripted_privileged",
        }
        if (
            receipt["sha256"] != row["receipt_sha256"]
            or receipt["case"] != row["case"]
            or receipt["status"] != "completed"
            or row["returncode"] != 0
            or receipt["worker_environment"]["pid"] != row["pid"]
            or receipt["sourcehashfinal"] != report["sourcehashfinal"]
            or record["sha256"] != receipt["episode_sha256"]
            or record["configuration"] != expected_config
            or record["metrics"] != receipt["metrics"]
            or meta["configuration"] != cases[row["case"]]["configuration"]
            or meta["planning_method"] != "sequence_lanes"
            or receipt["latest_planned_finish_sec"]
            != max(
                (a["finish_sec"] for a in meta["schedule"]["allocations"]), default=0.0
            )
            or any(
                receipt[k] != meta["schedule"][s]
                for k, s in (
                    ("planned", "predicted_fulfilled"),
                    ("search_nodes", "search_nodes"),
                    ("search_budget_exhausted", "search_budget_exhausted"),
                )
            )
        ):
            raise ValueError("Episode/receipt binding mismatch")
        if set(receipt["phases"]) != {
            "compile",
            "planning",
            "schedule_validation",
            "native_execution",
            "replay",
        }:
            raise ValueError("Missing measurement phase")
        for phase in receipt["phases"].values():
            if (
                type(phase["wall_sec"]) not in (int, float)
                or not math.isfinite(phase["wall_sec"])
                or phase["wall_sec"] < 0
                or type(phase["cumulative_peak_rss_bytes"]) is not int
                or phase["cumulative_peak_rss_bytes"] <= 0
            ):
                raise ValueError("Invalid measurement")
        replayed += 1
        passed += (
            max(p["cumulative_peak_rss_bytes"] for p in receipt["phases"].values())
            <= plan["max_rss_bytes"]
            and receipt["planned"] == record["metrics"]["fulfilled"]
            and not record["metrics"]["failed_calls"]
        )
    return {
        "valid": True,
        "selected": len(selected),
        "frozen_total": len(cases),
        "replayed": replayed,
        "passed": passed,
        "qualified": passed == len(selected),
        "full_matrix": selected == list(cases),
        "sourcehashfinal": report["sourcehashfinal"],
        "workers": plan["workers"],
        "per_worker_rss_budget_bytes": plan["max_rss_bytes"],
        "aggregate_potential_rss_bytes": plan["aggregate_potential_rss_bytes"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", nargs=2, metavar=("ROOT", "CASE"))
    p.add_argument("--output")
    p.add_argument("--audit")
    p.add_argument("--select", nargs="+")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.worker:
        root, name = Path(args.worker[0]), args.worker[1]
        plan = _read(root / "plan.json")
        select(plan["selected"])
        if name not in plan["selected"]:
            raise ValueError("Unselected worker")
        case = next(c for c in workloads() if c["id"] == name)
        try:
            receipt = measure(case, plan["model_sources"], root / name)
            _write(root / name / "receipt.json", receipt)
        except Exception as exc:
            _write(
                root / name / "error.json",
                seal(
                    {
                        "case": name,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:2000],
                        "sourcehashfinal": digest(sources()),
                    }
                ),
            )
            raise
    elif args.audit:
        print(canonical_json(audit(args.audit)))
    elif args.output:
        print(
            canonical_json(
                run(
                    args.output,
                    selected=args.select,
                    timeout_sec=args.timeout,
                    workers=args.workers,
                )
            )
        )
    else:
        p.error("Use --output or --audit")


if __name__ == "__main__":
    main()
