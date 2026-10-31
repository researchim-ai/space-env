"""Cancellation/replacement, deadline revision and degraded link throughput."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from demo import build_spec

from space_env.data_generators.reference_operations import (
    ReferenceOperationsRecorder,
    replay_reference_operations,
)
from space_env.reference.adaptation import adapt_reference_run
from space_env.reference.event_env import ReferenceEventDownlinkEnv
from space_env.rl.trajectories import rollout_episode


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    changes = [
        ("product_cancelled", {"product_id": "scene-b"}),
        ("onboard_product_added", {"id": "replacement", "actor": "b", "size_bits": 1e6,
                                   "ready_sec": 1, "deadline_sec": 9, "priority": 10}),
        ("product_deadline_changed", {"product_id": "scene-a", "deadline_sec": 12}),
        ("transmitter_rate_changed", {"actor": "a", "rate_bps": .5e6}),
        ("station_rate_changed", {"station": "Ground", "rate_bps": .75e6}),
    ]
    events = [{"id": f"change-{i}", "type": kind, "payload": payload, "time_sec": 1,
               "source": "mission-operator", "reason": "mission service reconfiguration"}
              for i, (kind, payload) in enumerate(changes)]
    recorder = ReferenceOperationsRecorder(spec, lineage="lifecycle-reference-demo")
    recorder.step(1, {"a": "Ground"})
    prefix = directory / "prefix.json"
    recorder.save(prefix)
    (directory / "events.json").write_text(json.dumps({"events": events}))
    adapted, schedule = adapt_reference_run(prefix, events)
    target = directory / "run.json"
    adapted.save(target)
    (directory / "schedule.json").write_text(json.dumps(schedule, allow_nan=False))
    rows = rollout_episode(ReferenceEventDownlinkEnv(spec, events, product_slots=3, dt_sec=10),
                           seed=7, policy=lambda obs: np.array([1, 1]))
    (directory / "rl.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    report = adapted.mission.report()
    replay = replay_reference_operations(target)
    discarded = sum(x["discarded_bits"] for x in report["inventory_ledger"].values())
    if report["on_time_bits"] != 3e6 or discarded != 3e6 or not replay["ok"]:
        raise RuntimeError("lifecycle mission outcome failed")
    if sum(r["reward"] for r in rows) != 3 or rows[-1]["success"]:
        raise RuntimeError("cancelled products must not count as successful delivery")
    print(json.dumps({"on_time_bits": report["on_time_bits"], "discarded_bits": discarded,
                      "products": report["products"], "rate_limits": report["rate_limits"],
                      "replay": replay, "rl_steps": len(rows), "rl_reward_mbit": sum(r["reward"] for r in rows)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
