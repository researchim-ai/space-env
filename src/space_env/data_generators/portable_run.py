"""Self-contained command experiments with integrity checks and verified transitions.

This records direct MissionWorld commands, not arbitrary policy/tool Python code.
The observation contract is the world's declared five-field ideal telemetry.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import math
import platform
from dataclasses import asdict
from pathlib import Path

from space_env.eval.statistics import assign_split
from space_env.simulation.commands import MissionCommand
from space_env.simulation.snapshot import (
    atomic_write,
    canonical_json,
    digest,
    encode,
    world_from_payload,
    world_payload,
)
from space_env.simulation.world import MissionWorld


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_manifest() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {
        "python": platform.python_version(),
        "dependencies": {name: _version(name) for name in (
            "space-env", "numpy", "pydantic", "boinor", "astropy", "pyerfa",
            "scipy", "nrlmsise00", "sgp4", "astropy-iers-data",
        )},
        "source_sha256": digest({str(path.relative_to(root)): digest(path.read_text(encoding="utf-8"))
                                 for path in sorted(root.rglob("*.py"))}),
        "data_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(root.rglob("*"))
                        if path.is_file() and path.suffix in {".csv", ".json", ".yaml", ".yml", ".tle"}},
        "mission_physics_profile": "legacy-training/1",
        "frame_contract": "legacy-numerical-GCRF-tag; not qualified GCRS",
    }


def _observe(world: MissionWorld) -> dict:
    return {name: world.observe(name) for name in world.engines}


def _outcome(world: MissionWorld) -> dict:
    return encode({
        "time_sec": world.now_sec(), "commands": world.commands.export(),
        "vehicles": {name: {
            "state": engine.get_state_snapshot().model_dump(mode="json"),
            "products": encode(engine.capture()["products"]),
            "receipts": encode(list(engine.receipts)),
        } for name, engine in world.engines.items()},
    })


def _advance(world: MissionWorld, dt_sec: float, commands: list[MissionCommand]) -> None:
    checkpoint = world.snapshot(include_history=False)
    try:
        orders = world.accept(commands)
        world.step(dt_sec, orders)
        world.settle()
    except Exception:
        world.restore(checkpoint)
        raise


class RunRecorder:
    """Record effective telemetry/actions/outcomes from a caller-owned world.

    Do not mutate the world between steps. Such mutation is detected at replay.
    ``lineage`` must group related variants; it determines the dataset split.
    No reward/optimality or expert-quality label is inferred from successful replay.
    """

    def __init__(self, world: MissionWorld, *, lineage: str, provenance: dict | None = None):
        if not isinstance(lineage, str) or not lineage.strip():
            raise ValueError("nonempty scenario lineage is required")
        self.world = world
        self.lineage = lineage
        self.provenance = copy.deepcopy(provenance or {})
        canonical_json(self.provenance)
        self.runtime = runtime_manifest()
        self.initial = world_payload(world)
        self.steps: list[dict] = []

    def step(self, dt_sec: float, commands: list[MissionCommand]) -> dict:
        commands = list(commands)
        row = {"index": len(self.steps), "observation": _observe(self.world),
               "dt_sec": dt_sec, "commands": [asdict(command) for command in commands]}
        canonical_json(row)
        _advance(self.world, dt_sec, commands)
        row["next_observation"] = _observe(self.world)
        row["outcome"] = _outcome(self.world)
        self.steps.append(row)
        return copy.deepcopy(row)

    def save(self, path: str | Path) -> dict:
        body = {
            "schema": "space-env.run/1", "lineage": self.lineage,
            "observation_contract": "ideal-local-telemetry/1",
            "provenance": self.provenance, "runtime": self.runtime,
            "initial": self.initial, "steps": self.steps,
            "final": world_payload(self.world),
        }
        record = {**body, "sha256": digest(body)}
        atomic_write(path, canonical_json(record))
        return copy.deepcopy(record)


def _differences(expected, actual, *, path="root", limit=20) -> list[str]:
    """Exact structure/discrete values; explicit tolerance for finite floats."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return [f"{path}: keys differ"]
        result = []
        for key in expected:
            result.extend(_differences(expected[key], actual[key], path=f"{path}.{key}", limit=limit))
            if len(result) >= limit:
                break
        return result[:limit]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length differs"]
        result = []
        for i, (left, right) in enumerate(zip(expected, actual)):
            result.extend(_differences(left, right, path=f"{path}[{i}]", limit=limit))
            if len(result) >= limit:
                break
        return result[:limit]
    if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if math.isfinite(expected) and math.isfinite(actual) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-8):
            return []
    elif type(expected) is type(actual) and expected == actual:
        return []
    return [f"{path}: value differs"]


def read_run(path: str | Path) -> dict:
    import json

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "sha256"}
    if record.get("schema") != "space-env.run/1" or record.get("sha256") != digest(body):
        raise ValueError("unsupported or corrupt run artifact")
    if record.get("observation_contract") != "ideal-local-telemetry/1":
        raise ValueError("unsupported observation contract")
    if not isinstance(record.get("lineage"), str) or not record["lineage"].strip():
        raise ValueError("missing scenario lineage")
    return record


def replay_run(path: str | Path) -> dict:
    """Verify every transition, final checkpoint and complete subsystem journals.

    Refuse a changed software/backend manifest rather than silently certifying a
    cross-version replay. Hashes detect corruption, not authenticity/signatures.
    """
    return _verify_record(read_run(path))


def _verify_record(record: dict) -> dict:
    if record["runtime"] != runtime_manifest():
        return {"ok": False, "reason": "runtime_mismatch", "steps_verified": 0}
    world = world_from_payload(record["initial"])
    for index, row in enumerate(record["steps"]):
        mismatches = _differences(row["observation"], _observe(world), path=f"steps[{index}].observation")
        if row["index"] != index:
            mismatches.append(f"steps[{index}].index: sequence differs")
        if mismatches:
            return {"ok": False, "reason": "observation_mismatch", "steps_verified": index, "mismatches": mismatches}
        _advance(world, row["dt_sec"], [MissionCommand(**command) for command in row["commands"]])
        mismatches = _differences(row["next_observation"], _observe(world), path=f"steps[{index}].next_observation")
        mismatches.extend(_differences(row["outcome"], _outcome(world), path=f"steps[{index}].outcome"))
        if mismatches:
            return {"ok": False, "reason": "transition_mismatch", "steps_verified": index, "mismatches": mismatches[:20]}
    final = world_payload(world)
    # Content checksum is verified on load; numerical equivalence has its own
    # explicit tolerance and must not be replaced by checksum equality.
    expected = {key: value for key, value in record["final"].items() if key != "sha256"}
    if record["final"].get("sha256") != digest(expected):
        raise ValueError("corrupt final world snapshot")
    differences = _differences(expected, {key: value for key, value in final.items() if key != "sha256"})
    return {"ok": not differences, "reason": "final_mismatch" if differences else "verified",
            "steps_verified": len(record["steps"]), "mismatches": differences,
            "lineage": record["lineage"], "run_sha256": record["sha256"]}


def export_transitions(run_path: str | Path, output: str | Path) -> dict:
    """Atomic replay-verified behavior transitions with lineage-based split.

    These are recorded command demonstrations, not automatically expert SFT,
    noisy sensor observations, or reward-labelled RL trajectories.
    """
    record = read_run(run_path)
    verification = _verify_record(record)
    if not verification["ok"]:
        raise ValueError(f"run replay failed: {verification['reason']}")
    split = assign_split(record["lineage"], "")
    rows = [{
        "schema": "space-env.transition/1", "run_sha256": record["sha256"],
        "lineage": record["lineage"], "split": split, "index": row["index"],
        "observation_contract": record["observation_contract"],
        "provenance": record["provenance"], "observation": row["observation"],
        "action": {"dt_sec": row["dt_sec"], "commands": row["commands"]},
        "next_observation": row["next_observation"], "outcome": row["outcome"],
        "replay_verified": True,
    } for row in record["steps"]]
    content = "".join(canonical_json(row) + "\n" for row in rows)
    atomic_write(output, content)
    return {"schema": "space-env.dataset-export/1", "rows": len(rows), "split": split,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(), "run_sha256": record["sha256"]}
