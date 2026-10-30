"""Portable reference downlink experiments; actions verified by actual resource replay."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from space_env.data_generators.portable_run import _differences, runtime_manifest
from space_env.reference.operations import PROFILE, ReferenceDownlinkMission
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def _runtime():
    result = runtime_manifest()
    result.update(mission_physics_profile=PROFILE, frame_contract="GCRS/ITRS fixed ephemerides")
    return result


class ReferenceOperationsRecorder:
    def __init__(self, spec: dict, *, lineage: str):
        if not isinstance(lineage, str) or not lineage.strip():
            raise ValueError("nonempty scenario lineage is required")
        self.spec, self.lineage = copy.deepcopy(spec), lineage
        self.mission = ReferenceDownlinkMission(spec)
        self.runtime = _runtime()
        self.steps = []

    def step(self, dt_sec: float, actions: dict[str, str | None]):
        row = {"dt_sec": dt_sec, "actions": copy.deepcopy(actions), "observation": self.mission.observation()}
        canonical_json(row)
        row["result"] = self.mission.step(dt_sec, actions)
        self.steps.append(row)
        return copy.deepcopy(row["result"])

    def save(self, path: str | Path):
        body = {"schema": "space-env.reference-operations-run/1", "spec": self.spec,
                "lineage": self.lineage, "runtime": self.runtime, "steps": self.steps, "final": self.mission.report()}
        record = {**body, "sha256": digest(body)}
        atomic_write(path, canonical_json(record))
        return copy.deepcopy(record)


def replay_reference_operations(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text())
    body = {k: v for k, v in record.items() if k != "sha256"}
    if record.get("schema") != "space-env.reference-operations-run/1" or record.get("sha256") != digest(body):
        raise ValueError("unsupported or corrupt reference operations run")
    if record["runtime"] != _runtime():
        return {"ok": False, "reason": "runtime_mismatch"}
    mission = ReferenceDownlinkMission(record["spec"])
    for index, row in enumerate(record["steps"]):
        differences = _differences(row["observation"], mission.observation())
        if differences:
            return {"ok": False, "reason": "observation_mismatch", "index": index, "mismatches": differences}
        actual = mission.step(row["dt_sec"], row["actions"])
        differences = _differences(row["result"], actual)
        if differences:
            return {"ok": False, "reason": "result_mismatch", "index": index, "mismatches": differences}
    differences = _differences(record["final"], mission.report())
    return {"ok": not differences, "reason": "final_mismatch" if differences else "verified", "mismatches": differences}
