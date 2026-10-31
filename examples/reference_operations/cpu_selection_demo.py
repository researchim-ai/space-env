"""An agent selects a short urgent CPU job before a long priority job."""
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
from space_env.reference.operations import ReferenceDownlinkMission
from space_env.reference.operations_env import ReferenceDownlinkEnv
from space_env.rl.trajectories import rollout_episode


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    spec["outages"] = []
    spec["vehicles"]["a"]["processing_power_w"] = 18
    spec["products"] = [
        {"id": "long", "actor": "a", "size_bits": 1e6, "deadline_sec": 25, "processing_sec": 10, "priority": 10},
        {"id": "short", "actor": "a", "size_bits": 1e6, "deadline_sec": 3, "processing_sec": 1}]
    baseline = ReferenceDownlinkMission(spec)
    baseline.step(13, {"a": "Ground"})
    recorder = ReferenceOperationsRecorder(spec, lineage="explicit-cpu-demo")
    recorder.step(2, {"a": "Ground"}, processing_jobs={"a": "short"})
    recorder.step(11, {"a": "Ground"}, processing_jobs={"a": "long"})
    path = directory / "run.json"
    recorder.save(path)
    rows = rollout_episode(ReferenceDownlinkEnv(spec, dt_sec=1, cpu_job_selection=True), seed=7,
        policy=lambda obs: {"radio": np.array([1, 0]), "cpu": np.array([3 if obs["clock"][0] < 2 else 2, 0])})
    (directory / "rl.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    replay = replay_reference_operations(path)
    automatic = baseline.report()["on_time_bits"]
    controlled = recorder.mission.report()["on_time_bits"]
    if automatic != 1e6 or controlled != 2e6 or not replay["ok"] or not rows[-1]["success"]:
        raise RuntimeError("CPU selection regression failed")
    print(json.dumps({"automatic_on_time_bits": automatic, "controlled_on_time_bits": controlled,
                      "cpu_seconds": recorder.mission.processing_seconds, "rl_steps": len(rows), "replay": replay}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
