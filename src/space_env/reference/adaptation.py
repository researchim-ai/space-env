"""Recorded causal events followed by replanning and verified continuation."""
from __future__ import annotations

from pathlib import Path

from space_env.data_generators.reference_operations import ReferenceOperationsRecorder
from space_env.reference.scheduling import mission_fingerprint, plan_remaining_downlinks


def adapt_reference_run(path: str | Path, events: list[dict], *, decision_sec: float = 5.,
                        max_steps: int = 10000) -> tuple[ReferenceOperationsRecorder, dict]:
    """Return a new complete recorder and schedule; never modify source files."""
    recorder = ReferenceOperationsRecorder.from_record(path)
    recorder.inject(events)
    schedule = plan_remaining_downlinks(recorder.mission, decision_sec=decision_sec, max_steps=max_steps)
    if mission_fingerprint(recorder.mission) != schedule["initial_state_sha256"]:
        raise RuntimeError("mission changed during adaptation")
    for row in schedule["plan"]["steps"]:
        recorder.step(row["dt_sec"], row["actions"], products=row.get("products"))
    if recorder.mission.report() != schedule["report"]:
        raise RuntimeError("adaptive continuation differs from verified schedule")
    return recorder, schedule
