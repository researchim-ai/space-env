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

    @classmethod
    def from_record(cls, path: str | Path):
        """Restore only after checksum, runtime and all transitions verify."""
        record, mission, verdict = _reconstruct(path)
        if not verdict["ok"]:
            raise ValueError(f"cannot resume reference run: {verdict['reason']}")
        recorder = cls.__new__(cls)
        recorder.spec, recorder.lineage = copy.deepcopy(record["spec"]), record["lineage"]
        recorder.mission, recorder.runtime = mission, copy.deepcopy(record["runtime"])
        recorder.steps = copy.deepcopy(record["steps"])
        return recorder

    def step(self, dt_sec: float, actions: dict[str, str | None], *, products: dict[str, str] | None = None,
             processing: dict[str, bool] | None = None, processing_jobs: dict[str, str] | None = None):
        row = {"dt_sec": dt_sec, "actions": copy.deepcopy(actions), "observation": self.mission.observation()}
        if products is not None:
            row["products"] = copy.deepcopy(products)
        if processing is not None:
            row["processing"] = copy.deepcopy(processing)
        if processing_jobs is not None:
            row["processing_jobs"] = copy.deepcopy(processing_jobs)
        canonical_json(row)
        row["result"] = self.mission.step(dt_sec, actions, products=products, processing=processing, processing_jobs=processing_jobs)
        self.steps.append(row)
        return copy.deepcopy(row["result"])

    def save(self, path: str | Path):
        schema = "space-env.reference-operations-run/2" if any(r.get("kind") == "events" for r in self.steps) else "space-env.reference-operations-run/1"
        body = {"schema": schema, "spec": self.spec,
                "lineage": self.lineage, "runtime": self.runtime, "steps": self.steps, "final": self.mission.report()}
        record = {**body, "sha256": digest(body)}
        atomic_write(path, canonical_json(record))
        return copy.deepcopy(record)

    def inject(self, events: list[dict]):
        row = {"kind": "events", "events": copy.deepcopy(events), "observation": self.mission.observation()}
        canonical_json(row)
        row["result"] = self.mission.apply_events(events)
        self.steps.append(row)
        return copy.deepcopy(row["result"])


def _reconstruct(path: str | Path):
    record = json.loads(Path(path).read_text())
    body = {k: v for k, v in record.items() if k != "sha256"}
    if record.get("schema") not in {"space-env.reference-operations-run/1", "space-env.reference-operations-run/2"} or record.get("sha256") != digest(body):
        raise ValueError("unsupported or corrupt reference operations run")
    if record["runtime"] != _runtime():
        return record, None, {"ok": False, "reason": "runtime_mismatch"}
    mission = ReferenceDownlinkMission(record["spec"])
    for index, row in enumerate(record["steps"]):
        differences = _differences(row["observation"], mission.observation())
        if differences:
            return record, None, {"ok": False, "reason": "observation_mismatch", "index": index, "mismatches": differences}
        if row.get("kind") == "events":
            if record["schema"] != "space-env.reference-operations-run/2" or set(row) != {"kind", "events", "observation", "result"}:
                raise ValueError("invalid reference event journal entry")
            actual = mission.apply_events(row["events"])
        else:
            if (not {"dt_sec", "actions", "observation", "result"} <= set(row)
                    or set(row) - {"dt_sec", "actions", "products", "processing", "processing_jobs", "observation", "result"}):
                raise ValueError("unknown reference run step fields")
            actual = mission.step(row["dt_sec"], row["actions"], products=row.get("products"), processing=row.get("processing"),
                                  processing_jobs=row.get("processing_jobs"))
        differences = _differences(row["result"], actual)
        if differences:
            return record, None, {"ok": False, "reason": "result_mismatch", "index": index, "mismatches": differences}
    differences = _differences(record["final"], mission.report())
    return record, mission, {"ok": not differences, "reason": "final_mismatch" if differences else "verified", "mismatches": differences}


def replay_reference_operations(path: str | Path) -> dict:
    return _reconstruct(path)[2]
