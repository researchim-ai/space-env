"""Bounded atomic worker diagnostics, separate from native replay evidence."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

from space_env.simulation.snapshot import atomic_write, canonical_json

MAX_PROGRESS_BYTES = 32768
STAGES = frozenset(
    {
        "initialize",
        "prefix_execution",
        "prefix_checkpoint",
        "resume",
        "execution",
        "fork",
        "checkpoint",
        "evidence",
        "replay",
        "export",
    }
)


def _resources():
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = usage.ru_maxrss
        return {
            "cpu_sec": usage.ru_utime + usage.ru_stime,
            "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
        }
    except ImportError:
        return {"cpu_sec": None, "peak_rss_bytes": None}


class StageReporter:
    def __init__(self, path, workload_sha256):
        self.path = Path(path)
        self.workload_sha256 = workload_sha256
        self.started = time.perf_counter()
        self.last_write = -math.inf
        self.stage = None
        self.stage_started = None
        self.history = []
        self.sequence = 0
        self.decisions = 0
        self.physical_sec = 0.0

    def emit(
        self, stage, *, decisions=None, physical_sec=None, status="running", error=None
    ):
        if stage not in STAGES or status not in {"running", "completed", "failed"}:
            raise ValueError("invalid worker diagnostic stage/status")
        now = time.perf_counter()
        elapsed = now - self.started
        changed = stage != self.stage
        if changed:
            if self.stage is not None:
                self.history.append(
                    {
                        "stage": self.stage,
                        "started_sec": self.stage_started,
                        "ended_sec": elapsed,
                    }
                )
            if len(self.history) > 32:
                raise ValueError("worker diagnostic stage budget exceeded")
            self.stage, self.stage_started = stage, elapsed
        if decisions is not None:
            self.decisions = decisions
        if physical_sec is not None:
            self.physical_sec = physical_sec
        if not changed and status == "running" and now - self.last_write < 1.0:
            return
        self.sequence += 1
        body = {
            "schema": "space-env.worker-progress/1",
            "workload_sha256": self.workload_sha256,
            "pid": os.getpid(),
            "sequence": self.sequence,
            "status": status,
            "stage": self.stage,
            "stage_started_sec": self.stage_started,
            "elapsed_sec": elapsed,
            "decisions": self.decisions,
            "physical_sec": self.physical_sec,
            "history": self.history,
            **_resources(),
            "scope": "partial diagnostic only; last observed stage/resources, not native outcome or replay proof",
        }
        if error is not None:
            body["error"] = {"type": type(error).__name__, "message": str(error)[:512]}
        content = canonical_json(body)
        if len(content.encode()) > MAX_PROGRESS_BYTES:
            raise ValueError("worker diagnostic byte budget exceeded")
        atomic_write(self.path, content)
        self.last_write = now


def read_progress(path, expected_sha256):
    """Malformed/missing partial diagnostics cannot replace the worker result."""
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_PROGRESS_BYTES + 1)
        if len(raw) > MAX_PROGRESS_BYTES:
            return None
        body = json.loads(raw)
        canonical_json(body)
        if (
            not isinstance(body, dict)
            or body.get("schema") != "space-env.worker-progress/1"
            or body.get("workload_sha256") != expected_sha256
            or body.get("stage") not in STAGES
            or body.get("status") not in {"running", "completed", "failed"}
            or type(body.get("pid")) is not int
            or body["pid"] < 1
            or type(body.get("sequence")) is not int
            or body["sequence"] < 1
        ):
            return None

        def nonnegative(value):
            return type(value) in {int, float} and math.isfinite(value) and value >= 0

        if (
            not nonnegative(body.get("elapsed_sec"))
            or not nonnegative(body.get("stage_started_sec"))
            or body["stage_started_sec"] > body["elapsed_sec"]
            or type(body.get("decisions")) is not int
            or body["decisions"] < 0
            or not nonnegative(body.get("physical_sec"))
            or (body.get("cpu_sec") is not None and not nonnegative(body["cpu_sec"]))
            or (
                body.get("peak_rss_bytes") is not None
                and (
                    type(body["peak_rss_bytes"]) is not int
                    or body["peak_rss_bytes"] < 0
                )
            )
            or not isinstance(body.get("history"), list)
            or len(body["history"]) > 32
        ):
            return None
        end = 0.0
        for row in body["history"]:
            if (
                not isinstance(row, dict)
                or row.get("stage") not in STAGES
                or not nonnegative(row.get("started_sec"))
                or not nonnegative(row.get("ended_sec"))
                or not end
                <= row["started_sec"]
                <= row["ended_sec"]
                <= body["stage_started_sec"]
            ):
                return None
            end = row["ended_sec"]
        return body
    except (OSError, ValueError, TypeError, OverflowError):
        return None
