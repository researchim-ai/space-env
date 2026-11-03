"""Diagnostic A/B of old and optimized reads within one source build.

The control temporarily restores old read materialization in this process only.
It does not change physics, tolerances, cadence, inputs or the installed package.
Wall times are single samples, not performance qualification.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from space_env.planning import PlanningRun
from space_env.reference.operations import ReferenceDownlinkMission
from space_env.scaling.campaign import campaign_workload
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks import TaskSession


def _uncached_refresh(self, runtime, projections, revision):
    return {
        role: p.build(runtime, role, revision, self._actions)
        for role, p in projections.items()
    }


def measure(workload, *, control):
    with ExitStack() as stack:
        if control:
            original = ReferenceDownlinkMission.observation

            def all_actors(self, actor=None):
                return original(self)

            stack.enter_context(
                patch.object(TaskSession, "_refresh", _uncached_refresh)
            )
            stack.enter_context(
                patch.object(ReferenceDownlinkMission, "observation", all_actors)
            )
        run = PlanningRun(workload["scenario"], workload["task"], workload["policy"])
        transitions = []
        started = time.perf_counter()
        while not run.done:
            transitions.append(digest(run.step()))
        wall = time.perf_counter() - started
        return {
            "transitions_sha256": digest(transitions),
            "trace_sha256": digest(run.trace),
            "backend_sha256": digest(run.session._runtime._backend.report()),
            "decisions": len(transitions),
            "wall_sec": wall,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    rows = []
    for mode in ["nominal", "ack_loss", "ground_outage", "ttl"]:
        workload = campaign_workload(
            {
                "cameras": 1,
                "relays": 1,
                "waves": 1,
                "horizon_sec": 20.0,
                "repeats": 1,
                "mode": mode,
            }
        )
        control = measure(workload, control=True)
        optimized = measure(workload, control=False)
        checks = {
            key: control[key] == optimized[key]
            for key in [
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
        "schema": "space-env.projection-parity/1",
        "ok": all(r["passed"] for r in rows),
        "scope": "same-build diagnostic; two actors, one request, 20s; single-sample wall times under shared host load; not a throughput gate",
        "rows": rows,
    }
    result["sha256"] = digest(result)
    atomic_write(args.output, canonical_json(result))
    if not result["ok"]:
        raise SystemExit(1)
    print(json.dumps({"ok": result["ok"], "sha256": result["sha256"]}))


if __name__ == "__main__":
    main()
