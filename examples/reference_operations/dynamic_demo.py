"""New onboard data and a station failure during an actual GCRS mission episode."""
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
    events = [{"id": "urgent-scene", "type": "onboard_product_added", "time_sec": 1,
               "source": "payload-simulator", "reason": "preacquired scene became available",
               "payload": {"id": "urgent", "actor": "a", "size_bits": 1e6,
                           "ready_sec": 1, "deadline_sec": 6, "priority": 10}},
              {"id": "station-failure", "type": "station_outage_added", "time_sec": 1,
               "source": "fault-injector", "reason": "station unavailable until repair",
               "payload": {"station": "Ground", "start_sec": 1, "end_sec": 3}}]
    recorder = ReferenceOperationsRecorder(spec, lineage="dynamic-reference-demo")
    recorder.step(1, {"a": "Ground"})
    prefix = directory / "prefix.json"
    recorder.save(prefix)
    adapted, schedule = adapt_reference_run(prefix, events)
    run = directory / "dynamic-run.json"
    adapted.save(run)
    (directory / "schedule.json").write_text(json.dumps(schedule, allow_nan=False))
    rows = rollout_episode(ReferenceEventDownlinkEnv(spec, events, product_slots=3, dt_sec=10),
                           seed=7, policy=lambda obs: np.array([1, 1]))
    (directory / "rl.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    replay = replay_reference_operations(run)
    report = adapted.mission.report()
    if not replay["ok"] or report["on_time_bits"] != 6e6 or not rows[-1]["success"]:
        raise RuntimeError("dynamic operations verification failed")
    print(json.dumps({"on_time_bits": report["on_time_bits"], "events": len(report["events"]),
                      "replay": replay, "rl_steps": len(rows), "rl_reward_mbit": sum(r["reward"] for r in rows),
                      "inventory_ledger": report["inventory_ledger"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
