"""Same-build control for owned copies, borrowed reads, capture and reward scalars."""

from __future__ import annotations

import argparse
import copy
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import space_env.runtime.mission as runtime
from space_env.planning import PlanningRun
from space_env.reference import dynamic, lifecycle
from space_env.scaling.campaign import campaign_workload
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def baseline_capture(self):
    return copy.deepcopy(
        (
            self.mission,
            self.geometry_revision,
            self.burn_commands,
            self.lifecycle,
            self.bus,
        )
    )


def measure(workload, control):
    with ExitStack() as stack:
        if control:
            stack.enter_context(patch.object(runtime, "owned_copy", copy.deepcopy))
            stack.enter_context(patch.object(lifecycle, "owned_copy", copy.deepcopy))
            stack.enter_context(
                patch.object(
                    runtime.MissionRuntime,
                    "_observe_backend",
                    lambda self, actor: self._backend.observe(actor),
                )
            )
            stack.enter_context(
                patch.object(
                    dynamic.DynamicReferenceBackend, "capture", baseline_capture
                )
            )
            stack.enter_context(
                patch.object(
                    dynamic.DynamicReferenceBackend, "task_score", lambda self: None
                )
            )
        run = PlanningRun(workload["scenario"], workload["task"], workload["policy"])
        transitions = []
        start = time.perf_counter()
        while not run.done:
            transitions.append(digest(run.step()))
        elapsed = time.perf_counter() - start
        return {
            "transitions_sha256": digest(transitions),
            "trace_sha256": digest(run.trace),
            "backend_sha256": digest(run.session._runtime._backend.report()),
            "decisions": len(transitions),
            "wall_sec": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    rows = []
    for mode in ["nominal", "ground_recovery", "ack_loss", "ground_outage", "ttl"]:
        workload = campaign_workload(
            {
                "cameras": 1,
                "relays": 1,
                "waves": 1,
                "horizon_sec": 40.0 if mode == "ground_recovery" else 20.0,
                "repeats": 1,
                "mode": mode,
            }
        )
        control = measure(workload, True)
        optimized = measure(workload, False)
        checks = {
            k: control[k] == optimized[k]
            for k in [
                "transitions_sha256",
                "trace_sha256",
                "backend_sha256",
                "decisions",
            ]
        }
        rows.append(
            {
                "mode": mode,
                "control": control,
                "optimized": optimized,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        print(mode, rows[-1]["passed"], flush=True)
    result = {
        "schema": "space-env.native-copy-parity/1",
        "ok": all(r["passed"] for r in rows),
        "scope": "same-build diagnostic controls; 2 actors, 1 request, 20/40s; shared-host single timing samples, not throughput qualification",
        "rows": rows,
    }
    result["sha256"] = digest(result)
    atomic_write(args.output, canonical_json(result))
    print(json.dumps({"ok": result["ok"], "sha256": result["sha256"]}))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
