"""Thermal trips, cooling intervals, station handover and portable RL outcomes."""
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
from space_env.reference.scheduling import plan_downlinks
from space_env.rl.trajectories import rollout_episode


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    spec["outages"] = []
    spec["vehicles"]["a"].update(rate_bps=.5e6, max_tx_sec=5, thermal={
        "ambient_k": 280, "initial_k": 280, "capacity_j_per_k": 2,
        "conductance_w_per_k": .2, "heat_tx_w": 20, "trip_k": 300, "resume_k": 290})
    for p in spec["products"]:
        p["deadline_sec"] = 25
    spec["products"][0]["priority"] = 10
    schedule = plan_downlinks(spec, select_products=True, decision_sec=10)
    recorder = ReferenceOperationsRecorder(spec, lineage="thermal-reference-demo")
    for row in schedule["plan"]["steps"]:
        recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
    path = directory / "run.json"
    recorder.save(path)
    (directory / "spec.json").write_text(json.dumps(spec))
    (directory / "plan.json").write_text(json.dumps(schedule["plan"]))
    (directory / "schedule.json").write_text(json.dumps(schedule, allow_nan=False))
    rows = rollout_episode(ReferenceEventDownlinkEnv(spec, [], dt_sec=1), seed=7,
                           policy=lambda obs: np.array([1, 1]))
    (directory / "rl.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    report, replay = recorder.mission.report(), replay_reference_operations(path)
    if report["on_time_bits"] != 5e6 or not replay["ok"] or not rows[-1]["success"]:
        raise RuntimeError("thermal mission failed")
    health = report["transmitter_health"]["a"]
    if abs(health["thermal_balance_error_j"]) > 1e-9 or abs(health["tx_seconds"]-4) > 1e-9:
        raise RuntimeError("thermal ledger mismatch")
    print(json.dumps({"on_time_bits": report["on_time_bits"], "health": health, "replay": replay,
                      "rl_steps": len(rows), "completed_sec": report["time_sec"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
