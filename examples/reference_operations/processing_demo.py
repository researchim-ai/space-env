"""In-place data processing, scratch memory, CPU pause, dynamic arrival and downlink."""
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
    spec["vehicles"]["b"]["processing_power_w"] = 15
    spec["products"][0].update(processing_sec=3, workspace_bits=2e6, deadline_sec=20)
    spec["products"][1].update(processing_sec=2, workspace_bits=1e6, deadline_sec=20)
    events = [{"id": "arrival", "type": "onboard_product_added", "time_sec": 1,
               "source": "payload-simulator", "reason": "raw data became available",
               "payload": {"id": "urgent", "actor": "b", "size_bits": 1e6, "ready_sec": 1,
                           "deadline_sec": 10, "priority": 9, "processing_sec": 1, "workspace_bits": .5e6}}]
    recorder = ReferenceOperationsRecorder(spec, lineage="processing-reference-demo")
    recorder.step(1, {}, processing={"a": False, "b": False})
    recorder.inject(events)
    recorder.save(directory / "prefix.json")
    schedule = plan_remaining_downlinks(recorder.mission)
    for row in schedule["plan"]["steps"]:
        recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
    path = directory / "run.json"
    recorder.save(path)
    (directory / "schedule.json").write_text(json.dumps(schedule, allow_nan=False))
    env = ReferenceEventDownlinkEnv(spec, events, dt_sec=1, cpu_control=True)
    rows = rollout_episode(env, seed=7, policy=lambda obs: np.array([1, 1]))
    (directory / "rl.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    report, replay = recorder.mission.report(), replay_reference_operations(path)
    if report["on_time_bits"] != 6e6 or not replay["ok"] or not rows[-1]["success"]:
        raise RuntimeError("processing scenario failed")
    print(json.dumps({"on_time_bits": report["on_time_bits"], "cpu_seconds": report["processing"]["seconds"],
                      "cpu_energy_wh": report["processing"]["energy_wh"], "products": report["products"],
                      "replay": replay, "rl_steps": len(rows)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
