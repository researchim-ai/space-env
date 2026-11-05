"""Native-verified role datasets; exogenous incident controls never agent targets."""

from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
from pathlib import Path

from space_env.planning.actor_projection import project_actor_observation
from space_env.planning.incident import verify_incident
from space_env.planning.mission_bundle import (
    publish_plan_bundle,
    read_json,
    verify_plan_bundle,
)
from space_env.planning.mission_plan import _sealed
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl, file_digest
from space_env.tasks import TaskSpec, VerifiedEpisode

MAX_BYTES = 1073741824


def role_rows(verified, recovery, role, *, sft=False):
    """Role observations/actions, team reward; no prefix/operator supervision."""
    cutoff = recovery["committed_prefix_steps"]
    task = TaskSpec.model_validate(recovery["plan"]["task"])
    forbidden = {
        a["id"] for a in recovery["plan"]["task"]["actions"] if a["kind"] == "inject"
    }
    for row in verified.iter_rows(role=role, include_rejected=True):
        if row["index"] < cutoff:
            continue
        row["schema"] = "space-env.incident-role-rl/1"
        for key in ("observation", "next_observation"):
            row[key] = project_actor_observation(
                row[key],
                task,
                role,
                catalogue_version=recovery.get("catalogue_version"),
            )
        if any(
            a["id"] in forbidden
            for key in ("observation", "next_observation")
            for a in row[key]["catalogue"]
        ):
            raise ValueError("operator incident action leaked into actor observation")
        if any(
            a in forbidden
            for values in row["action"]["choices"].values()
            for a in values
        ):
            raise ValueError("operator incident action leaked into actor target")
        row.update(
            incident_recovery_sha256=recovery["sha256"],
            reward_scope="joint_team",
            decision_protocol="role action prediction; coordinator assembles joint step",
            expert_teacher_qualified=False,
        )
        if sft:
            if not row["feedback"]["accepted"]:
                continue
            value = {
                k: v
                for k, v in row.items()
                if k not in {"observation", "next_observation", "action", "feedback"}
            }
            value["schema"] = "space-env.incident-role-sft/1"
            value["messages"] = [
                {"role": "user", "content": canonical_json(row["observation"])},
                {"role": "assistant", "content": canonical_json(row["action"])},
            ]
            yield value
        else:
            yield row


def publish_incident_data(recovery, execution, output):
    verify_incident(recovery)
    if recovery["plan"]["sha256"] != execution["plan_sha256"]:
        raise ValueError("incident dataset plan mismatch")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("incident dataset output must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        control = publish_plan_bundle(
            recovery["plan"], execution, stage / "operator_trace", max_bytes=MAX_BYTES
        )
        atomic_write(stage / "recovery.json", canonical_json(recovery))
        verified = VerifiedEpisode(execution["episode"])
        roles = sorted(
            k
            for k, v in recovery["source_proposal"]["plan"]["task"]["roles"].items()
            if v["actors"]
        )
        training = {}
        used = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
        if used >= MAX_BYTES:
            raise ValueError("incident dataset aggregate byte budget exceeded")
        for number, role in enumerate(roles):
            for format in ("rl", "sft") if control["mission_goal_passed"] else ("rl",):
                name = f"actor-{number:02d}/{format}.jsonl"
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                receipt = atomic_jsonl(
                    path,
                    role_rows(verified, recovery, role, sft=format == "sft"),
                    max_bytes=MAX_BYTES - used,
                    max_row_bytes=16_000_000,
                )
                used += receipt["bytes"]
                training[name] = {"role": role, "format": format, **receipt}
        files = {
            p.relative_to(stage).as_posix(): {
                "sha256": file_digest(p),
                "bytes": p.stat().st_size,
            }
            for p in stage.rglob("*")
            if p.is_file()
        }
        if sum(v["bytes"] for v in files.values()) > MAX_BYTES:
            raise ValueError("incident dataset aggregate byte budget exceeded")
        body = {
            "schema": "space-env.incident-role-bundle/1",
            "recovery_sha256": recovery["sha256"],
            "mission_goal_passed": control["mission_goal_passed"],
            "training_files": training,
            "files": files,
            "expert_teacher_qualified": False,
            "operator_trace_trainable_as_agent": False,
            "scope": "actor suffix RL and conditional action-prediction SFT; joint team reward; operator-compiled future-pointing catalogue; not expert qualification",
        }
        receipt = body | {"sha256": digest(body)}
        atomic_write(stage / "manifest.json", canonical_json(receipt))
        if output.exists() or output.is_symlink():
            raise ValueError("incident dataset output appeared during publication")
        stage.rename(output)
        return copy.deepcopy(receipt)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_incident_data(directory, *, native=False):
    if type(native) is not bool:
        raise ValueError("native audit flag must be boolean")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("incident role bundle requires real directory")
    manifest = read_json(root / "manifest.json")
    if (
        not _sealed(manifest)
        or manifest.get("schema") != "space-env.incident-role-bundle/1"
        or manifest.get("expert_teacher_qualified") is not False
        or manifest.get("operator_trace_trainable_as_agent") is not False
    ):
        raise ValueError("invalid incident role manifest")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("incident role symlinks forbidden")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if len(actual) > 256 or sum((root / p).stat().st_size for p in actual) > MAX_BYTES:
        raise ValueError("incident role artifact budget exceeded")
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("incident role inventory mismatch")
    for name, value in manifest["files"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or file_digest(root / name) != value["sha256"]
            or (root / name).stat().st_size != value["bytes"]
        ):
            raise ValueError("incident role file checksum/path mismatch")
    recovery = read_json(root / "recovery.json")
    if not _sealed(recovery) or recovery["sha256"] != manifest["recovery_sha256"]:
        raise ValueError("incident role recovery binding mismatch")
    control = verify_plan_bundle(root / "operator_trace", native=native)
    if control["mission_goal_passed"] is not manifest["mission_goal_passed"]:
        raise ValueError("incident role goal mismatch")
    plan = read_json(root / "operator_trace/plan.json")
    if canonical_json(plan) != canonical_json(recovery["plan"]):
        raise ValueError("incident role native plan mismatch")
    roles = sorted(
        k
        for k, v in recovery["source_proposal"]["plan"]["task"]["roles"].items()
        if v["actors"]
    )
    expected = {
        f"actor-{i:02d}/{f}.jsonl": (r, f)
        for i, r in enumerate(roles)
        for f in (("rl", "sft") if manifest["mission_goal_passed"] else ("rl",))
    }
    if set(expected) != set(manifest["training_files"]):
        raise ValueError("incident role training selection mismatch")
    for name, (role, format) in expected.items():
        meta = manifest["training_files"][name]
        if meta["role"] != role or meta["format"] != format:
            raise ValueError("incident role training metadata mismatch")
        if (
            meta["bytes"] != manifest["files"][name]["bytes"]
            or meta["file_sha256"] != manifest["files"][name]["sha256"]
        ):
            raise ValueError("incident role training file metadata mismatch")
        checksum = hashlib.sha256(b"[")
        count = 0
        with (root / name).open() as stream:
            while line := stream.readline(16_000_001):
                if len(line) > 16_000_000 or count >= 10000:
                    raise ValueError("incident role row budget exceeded")
                if count:
                    checksum.update(b",")
                checksum.update(canonical_json(read_line(line)).encode())
                count += 1
        checksum.update(b"]")
        if count != meta["rows"] or checksum.hexdigest() != meta["rows_sha256"]:
            raise ValueError("incident role row metadata mismatch")

    if native:
        verify_incident(recovery)
        record = read_json(root / "operator_trace/execution.json")
        verified = VerifiedEpisode(record["episode"])
        for name, (role, format) in expected.items():
            with (root / name).open() as stream:
                count = 0
                for row in role_rows(verified, recovery, role, sft=format == "sft"):
                    saved = stream.readline(16_000_001)
                    if len(saved) > 16_000_000:
                        raise ValueError("incident role row byte budget exceeded")
                    if not saved or canonical_json(read_line(saved)) != canonical_json(
                        row
                    ):
                        raise ValueError("incident role row native mismatch")
                    count += 1
                if (
                    stream.readline()
                    or count != manifest["training_files"][name]["rows"]
                ):
                    raise ValueError("incident role row count mismatch")
    return {
        "integrity_passed": True,
        "native_rows_recomputed": native,
        "mission_goal_passed": manifest["mission_goal_passed"],
        "training_files": len(expected),
        "expert_teacher_qualified": False,
    }


def read_line(raw):
    from space_env.planning.mission_bundle import strict_json

    return strict_json(raw)
