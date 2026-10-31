"""Reserved-only station, sequential owners, outage, planning, RL and replay."""
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
        ("station_reservation_policy_changed", {"station": "Ground", "required": True}),
        ("terminal_booking_added", {"id": "b-pass", "actor": "b", "station": "Ground", "start_sec": 1, "end_sec": 7}),
        ("terminal_booking_added", {"id": "a-pass", "actor": "a", "station": "Ground", "start_sec": 7, "end_sec": 9}),
    ]
    events = [{"id": f"booking-event-{i}", "type": kind, "payload": payload, "time_sec": 1,
               "source": "ground-network-operator", "reason": "reserved station schedule"}
              for i, (kind, payload) in enumerate(changes)]
    recorder = ReferenceOperationsRecorder(spec, lineage="reference-booking-demo")
    recorder.step(1, {"a": "Ground"})
    prefix = directory / "prefix.json"
    recorder.save(prefix)
    (directory / "events.json").write_text(json.dumps({"events": events}))
    adapted, schedule = adapt_reference_run(prefix, events)
    path = directory / "run.json"
    adapted.save(path)
    (directory / "schedule.json").write_text(json.dumps(schedule, allow_nan=False))
    rows = rollout_episode(ReferenceEventDownlinkEnv(spec, events, product_slots=3, dt_sec=10),
                           seed=7, policy=lambda obs: np.array([1, 1]))
    (directory / "rl.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    report, replay = adapted.mission.report(), replay_reference_operations(path)
    if report["on_time_bits"] != 5e6 or not replay["ok"] or not rows[-1]["success"]:
        raise RuntimeError("booking scenario failed")
    print(json.dumps({"on_time_bits": report["on_time_bits"], "replay": replay,
                      "booking_utilization": report["booking_utilization"], "rl_steps": len(rows)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
