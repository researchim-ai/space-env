"""Two reference-orbit vehicles deliver existing products through a shared station."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from space_env.core.frames import ecef_to_geodetic
from space_env.data_generators.reference_operations import (
    ReferenceOperationsRecorder,
    replay_reference_operations,
)
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.operations_env import ReferenceDownlinkEnv
from space_env.rl.trajectories import rollout_episode
from space_env.simulation.snapshot import digest


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    context, epoch = FrameContext(EOPSeries.packaged()), Epoch.utc("2026-07-18T12:00:00Z")
    records = {}
    for name, angle in (("a", 0), ("b", .005)):
        r = 7e6*np.array([np.cos(angle), 0, np.sin(angle)])
        v = 7546.053290107542*np.array([-np.sin(angle), 0, np.cos(angle)])
        result = propagate(ReferenceState(epoch, r, v), 30, context, sample_sec=5)
        records[name] = {**result, "sha256": digest(result)}
    lat, lon, _ = ecef_to_geodetic(context.matrix(epoch) @ np.array([7e6, 0, 0]))
    spec = {"schema": "space-env.reference-downlink-spec/1", "ephemerides": records,
            "vehicles": {name: {"memory_bits": 5e6, "battery_wh": 5, "initial_soc": .8} for name in records},
            "stations": [{"name": "Ground", "lat_deg": lat, "lon_deg": lon, "terminals": 1}],
            "products": [{"id": "scene-a", "actor": "a", "size_bits": 2e6, "deadline_sec": 10},
                         {"id": "scene-b", "actor": "b", "size_bits": 3e6, "deadline_sec": 10}],
            "outages": [{"station": "Ground", "start_sec": 2, "end_sec": 4}]}
    recorder = ReferenceOperationsRecorder(spec, lineage="two-reference-downlink-demo")
    report = recorder.step(10, {"a": "Ground", "b": "Ground"})
    path = directory / "run.json"
    recorder.save(path)
    replay = replay_reference_operations(path)
    rows = rollout_episode(ReferenceDownlinkEnv(spec, dt_sec=10), seed=7,
                           policy=lambda observation: np.array([1, 1]))
    (directory / "rl.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
    if report["on_time_bits"] != 5e6 or not replay["ok"] or not rows[-1]["success"]:
        raise RuntimeError("reference mission did not deliver/replay")
    print(json.dumps({"profile": report["profile"], "on_time_bits": report["on_time_bits"],
                      "products": report["products"], "replay": replay, "rl_steps": len(rows)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
