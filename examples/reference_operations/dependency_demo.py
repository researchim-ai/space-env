"""Dynamic same-actor processing DAG, retained inputs, shared station and replay."""
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
from space_env.reference.event_env import ReferenceEventDownlinkEnv
from space_env.reference.scheduling import plan_remaining_downlinks
from space_env.rl.trajectories import rollout_episode


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    spec["vehicles"]["a"]["processing_power_w"] = 12
    spec["products"][0].update(processing_sec=1, deadline_sec=20)
    spec["products"][1]["deadline_sec"] = 20
    events = [{"id": "dependent-arrival", "type": "onboard_product_added", "time_sec": .5,
        "source": "payload-simulator", "reason": "dependent processing input became available",
        "payload": {"id": "analysis-input", "actor": "a", "size_bits": 1e6, "ready_sec": .5,
                    "deadline_sec": 15, "processing_sec": 2, "workspace_bits": .5e6,
                    "priority": 9, "depends_on": ["scene-a"]}}]
    recorder = ReferenceOperationsRecorder(spec, lineage="dependency-reference-demo")
    recorder.step(.5, {})
    recorder.inject(events)
    recorder.save(directory / "prefix.json")
    schedule = plan_remaining_downlinks(recorder.mission)
    for row in schedule["plan"]["steps"]:
        recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
    path = directory / "run.json"
    recorder.save(path)
    (directory / "schedule.json").write_text(json.dumps(schedule, allow_nan=False))
    rows = rollout_episode(ReferenceEventDownlinkEnv(spec, events, dt_sec=1), seed=7,
                           policy=lambda obs: np.array([1, 1]))
    (directory / "rl.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    report, replay = recorder.mission.report(), replay_reference_operations(path)
    if report["on_time_bits"] != 6e6 or not replay["ok"] or not rows[-1]["success"]:
        raise RuntimeError("dependency scenario failed")
    child_start = min(i["start_sec"] for i in recorder.mission.processing_intervals if i["product_id"] == "analysis-input")
    parent_end = recorder.mission.processing_completed["scene-a"]
    if child_start < parent_end:
        raise RuntimeError("dependent computation started early")
    print(json.dumps({"on_time_bits": report["on_time_bits"], "cpu_seconds": report["processing"]["seconds"],
                      "parent_processed_sec": parent_end, "child_started_sec": child_start,
                      "replay": replay, "rl_steps": len(rows)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
