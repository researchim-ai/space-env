"""Atomic native-verified live-policy RL/SFT without operator supervision."""

import copy
import hashlib
import shutil
import tempfile
from pathlib import Path

from space_env.planning.actor_online import ActorCoordinator, online_rows
from space_env.planning.mission_bundle import (
    _quality,
    certificate,
    read_json,
    strict_json,
)
from space_env.planning.mission_plan import _sealed
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl, file_digest
from space_env.tasks import VerifiedEpisode

MAX_BYTES = 1073741824


def _source(controller):
    sealed = controller.as_execution()
    verified = VerifiedEpisode(sealed["execution"]["episode"])
    cert = certificate(sealed["plan"], sealed["execution"], _quality(verified))
    goal = {
        k: cert[k]
        for k in (
            "passed",
            "fulfilled",
            "total",
            "gates",
            "native_report_sha256",
            "positive_service_declared",
        )
    }
    context = {
        "plan": sealed["plan"],
        "sha256": controller._reference["sha256"],
        "committed_prefix_steps": controller._cutoff,
        "catalogue_version": controller.catalogue_version,
    }
    return verified, context, goal


def publish_actor_data(checkpoint, output):
    controller = ActorCoordinator.from_checkpoint(checkpoint)
    verified, context, goal = _source(controller)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("actor bundle output must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        atomic_write(stage / "controller.json", canonical_json(controller.checkpoint()))
        used = (stage / "controller.json").stat().st_size
        training = {}
        for i, role in enumerate(controller.roles):
            for fmt in ("rl", "sft") if goal["passed"] else ("rl",):
                if used >= MAX_BYTES:
                    raise ValueError("actor bundle byte budget exceeded")
                name = f"actor-{i:02d}/{fmt}.jsonl"
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                meta = atomic_jsonl(
                    path,
                    online_rows(verified, context, role, sft=fmt == "sft"),
                    max_bytes=MAX_BYTES - used,
                    max_row_bytes=16_000_000,
                    max_rows=10000,
                )
                used += meta["bytes"]
                training[name] = {"role": role, "format": fmt, **meta}
        files = {
            p.relative_to(stage).as_posix(): {
                "sha256": file_digest(p),
                "bytes": p.stat().st_size,
            }
            for p in stage.rglob("*")
            if p.is_file()
        }
        body = {
            "schema": "space-env.actor-policy-bundle/1",
            "controller_sha256": checkpoint["sha256"],
            "goal": goal,
            "training_files": training,
            "files": files,
            "expert_teacher_qualified": False,
            "scope": "live actor suffix; joint team reward; conditional model-goal SFT, not expert promotion",
        }
        result = body | {"sha256": digest(body)}
        atomic_write(stage / "manifest.json", canonical_json(result))
        if output.exists() or output.is_symlink():
            raise ValueError("actor output appeared during publication")
        stage.rename(output)
        return copy.deepcopy(result)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _rows(path):
    with path.open() as stream:
        count = 0
        while line := stream.readline(16_000_001):
            if len(line.encode()) > 16_000_000 or count >= 10000:
                raise ValueError("actor row byte/count budget exceeded")
            count += 1
            yield strict_json(line)


def verify_actor_data(directory, *, native=False):
    if type(native) is not bool:
        raise ValueError("native actor audit flag must be boolean")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("actor bundle requires real directory")
    manifest = read_json(root / "manifest.json")
    if (
        not _sealed(manifest)
        or manifest.get("schema") != "space-env.actor-policy-bundle/1"
        or manifest.get("expert_teacher_qualified") is not False
    ):
        raise ValueError("invalid actor manifest")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("actor bundle symlinks forbidden")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if len(actual) > 256 or sum((root / n).stat().st_size for n in actual) > MAX_BYTES:
        raise ValueError("actor bundle artifact budget exceeded")
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("actor bundle inventory mismatch")
    for name, meta in manifest["files"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or file_digest(root / name) != meta["sha256"]
            or (root / name).stat().st_size != meta["bytes"]
        ):
            raise ValueError("actor bundle checksum/path mismatch")
    checkpoint = read_json(root / "controller.json")
    if not _sealed(checkpoint) or checkpoint["sha256"] != manifest["controller_sha256"]:
        raise ValueError("actor controller binding mismatch")
    reference = checkpoint["reference"]
    origin = reference.get("origin", reference.get("source_proposal", reference))
    roles = sorted(k for k, v in origin["plan"]["task"]["roles"].items() if v["actors"])
    expected = {
        f"actor-{i:02d}/{fmt}.jsonl": (role, fmt)
        for i, role in enumerate(roles)
        for fmt in (("rl", "sft") if manifest["goal"]["passed"] else ("rl",))
    }
    if set(expected) != set(manifest["training_files"]):
        raise ValueError("actor role selection mismatch")
    for name, (role, fmt) in expected.items():
        meta = manifest["training_files"][name]
        if (
            meta["role"] != role
            or meta["format"] != fmt
            or meta["bytes"] != manifest["files"][name]["bytes"]
            or meta["file_sha256"] != manifest["files"][name]["sha256"]
        ):
            raise ValueError("actor training metadata mismatch")
        checksum = hashlib.sha256(b"[")
        count = 0
        for row in _rows(root / name):
            if count:
                checksum.update(b",")
            checksum.update(canonical_json(row).encode())
            count += 1
        checksum.update(b"]")
        if count != meta["rows"] or checksum.hexdigest() != meta["rows_sha256"]:
            raise ValueError("actor row metadata mismatch")
    if native:
        controller = ActorCoordinator.from_checkpoint(checkpoint)
        verified, context, goal = _source(controller)
        if canonical_json(goal) != canonical_json(manifest["goal"]):
            raise ValueError("actor native goal mismatch")
        for name, (role, fmt) in expected.items():
            saved = iter(_rows(root / name))
            count = 0
            for row in online_rows(verified, context, role, sft=fmt == "sft"):
                value = next(saved, None)
                if value is None or canonical_json(value) != canonical_json(row):
                    raise ValueError("actor row native mismatch")
                count += 1
            if (
                next(saved, None) is not None
                or count != manifest["training_files"][name]["rows"]
            ):
                raise ValueError("actor row native count mismatch")
    return {
        "integrity_passed": True,
        "native_rows_recomputed": native,
        "mission_goal_passed": manifest["goal"]["passed"],
        "training_files": len(expected),
        "expert_teacher_qualified": False,
    }
