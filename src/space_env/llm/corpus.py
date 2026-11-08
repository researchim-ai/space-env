"""Bounded immutable LLM corpora, causal replay and shared local split registry.

Only train episodes enter data/. Evaluation records remain privileged audit inputs.
Publication and registry registration are separate retryable local operations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from space_env.factory.registry import _register_claims
from space_env.factory.semantic import fingerprint, mission_fingerprint, split_group
from space_env.llm.episode import _json, _training_row, seal, verify_record
from space_env.llm.local_data import verify_local_bundle
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl, file_digest, iter_jsonl

SCHEMA = "space-env.llm-corpus/1"
MAX_EPISODES = 256
MAX_FILE_BYTES = 64_000_000
MAX_BUNDLE_BYTES = 512_000_000


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("invalid or oversized corpus artifact")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("corpus artifact grew past byte budget")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("corpus artifact must be an object")
    return value


def _arguments(event):
    # Failed tool calls are valuable RL data too, including malformed JSON.
    try:
        value = _json(event["arguments"])
    except (ValueError, TypeError):
        return {"invalid_arguments_text": event["arguments"]}
    return (
        value
        if isinstance(value, dict)
        else {"invalid_arguments_text": event["arguments"]}
    )


def _identity(record):
    initial = record["initial_checkpoint"]
    scenario = initial["scenario"]
    geometry = split_group(scenario)
    # Ignore caller task prose, group labels, call IDs and SDK transaction metadata.
    # This identity is LLM-specific; factory trajectory hashes have another contract.
    trajectory = fingerprint(
        {
            "mission": mission_fingerprint(scenario, initial["task"]),
            "decisions": [
                {
                    "tool": e["name"],
                    "arguments": {
                        k: v
                        for k, v in _arguments(e).items()
                        if k not in {"expected_revision", "idempotency_key"}
                    },
                    "applied": e["transition_applied"],
                    "ok": e["response"]["ok"],
                }
                for e in record["events"]
            ],
        }
    )
    return geometry, trajectory


def _analyse(records):
    groups, trajectories, rows = {}, set(), []
    data = {name: [] for name in ("rl", "sft", "wire-sft")}
    summary = {
        "episodes": len(records),
        "independent_geometries": 0,
        "train_episodes": 0,
        "evaluation_episodes": 0,
        "successful_train_episodes": 0,
        "decisions": 0,
        "all_wait_decisions": 0,
        "policy_origins": {},
    }
    for index, (record, wire) in enumerate(records):
        verify_record(record)
        geometry, trajectory = _identity(record)
        split = record["configuration"]["split"]
        if geometry in groups and groups[geometry] != split:
            raise ValueError("geometry crosses corpus splits")
        groups[geometry] = split
        if trajectory in trajectories:
            raise ValueError("duplicate semantic LLM trajectory")
        trajectories.add(trajectory)
        train = split == "train"
        summary["train_episodes" if train else "evaluation_episodes"] += 1
        summary["successful_train_episodes"] += int(
            train and record["metrics"]["success"]
        )
        origin = record["configuration"]["policy_origin"]
        summary["policy_origins"][origin] = summary["policy_origins"].get(origin, 0) + 1
        for event in record["events"]:
            choices = _arguments(event).get("choices")
            if isinstance(choices, dict) and all(
                isinstance(ids, list) for ids in choices.values()
            ):
                summary["decisions"] += 1
                summary["all_wait_decisions"] += int(
                    any(choices.values())
                    and all(a == "wait" for ids in choices.values() for a in ids)
                )
        if train:
            data["rl"].append(_training_row(record, format="rl"))
            if record["metrics"]["sft_eligible"]:
                data["sft"].append(_training_row(record, format="sft"))
            data["wire-sft"].extend(wire)
        elif wire:
            raise ValueError("evaluation episode cannot supply training wire rows")
        rows.append(
            {
                "index": index,
                "episode_sha256": record["sha256"],
                "geometry_sha256": geometry,
                "trajectory_sha256": trajectory,
                "split": split,
                "policy_origin": origin,
                "native_sft_eligible": train and record["metrics"]["sft_eligible"],
                "wire_rows": len(wire),
            }
        )
    summary["independent_geometries"] = len(groups)
    summary["rows"] = {name: len(value) for name, value in data.items()}
    return rows, summary, data


def build_corpus(inputs, output):
    """Inputs are native episode paths or provider-bound local bundle directories.

    All checks happen in an adjacent staging directory. The output must not exist.
    No external inference, automatic training, registry upload or policy fallback.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("corpus output must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".llm-corpus-", dir=output.parent))
    records, used = [], 0
    try:
        for index, item in enumerate(inputs):
            if index >= MAX_EPISODES:
                raise ValueError("corpus episode budget exceeded")
            source = Path(item)
            target = stage / "audit" / f"{index:04d}"
            target.mkdir(parents=True)
            wire = []
            if source.is_dir():
                if source.is_symlink():
                    raise ValueError("symlink bundle input is not allowed")
                # Copy an immutable bounded snapshot BEFORE auditing it (TOCTOU).
                manifest = _read(source / "manifest.json")
                names = {"manifest.json", "episode.json", "inference.json"}
                exports = manifest.get("exports")
                if not isinstance(exports, dict) or not set(exports) <= {
                    "rl.jsonl",
                    "sft.jsonl",
                    "wire-sft.jsonl",
                }:
                    raise ValueError("unsupported local training bundle exports")
                names.update(exports)
                for name in sorted(names):
                    path = source / name
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or path.stat().st_size > MAX_FILE_BYTES
                    ):
                        raise ValueError("invalid or oversized local training input")
                    used += path.stat().st_size
                    if used > MAX_BUNDLE_BYTES:
                        raise ValueError("corpus input byte budget exceeded")
                    with path.open("rb") as src, (target / name).open("xb") as dst:
                        remaining = MAX_FILE_BYTES + 1
                        while chunk := src.read(min(1024 * 1024, remaining)):
                            remaining -= len(chunk)
                            if remaining <= 0:
                                raise ValueError("corpus input grew past byte budget")
                            dst.write(chunk)
                verify_local_bundle(target)
                if "wire-sft.jsonl" in exports:
                    wire = list(
                        iter_jsonl(
                            target / "wire-sft.jsonl", max_row_bytes=MAX_FILE_BYTES
                        )
                    )
            else:
                record = _read(source)
                encoded = canonical_json(record)
                used += len(encoded.encode())
                if used > MAX_BUNDLE_BYTES:
                    raise ValueError("corpus input byte budget exceeded")
                atomic_write(target / "episode.json", encoded)
            record = _read(target / "episode.json")
            records.append((record, wire))
        if not records:
            raise ValueError("empty corpus")
        entries, summary, data = _analyse(records)
        (stage / "data").mkdir()
        for name, values in data.items():
            atomic_jsonl(
                stage / "data" / f"{name}.jsonl",
                values,
                max_bytes=MAX_BUNDLE_BYTES,
                max_row_bytes=MAX_FILE_BYTES,
            )
        files = {
            str(path.relative_to(stage)): {
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        if sum(f["bytes"] for f in files.values()) > MAX_BUNDLE_BYTES:
            raise ValueError("corpus published byte budget exceeded")
        manifest = seal(
            {
                "schema": SCHEMA,
                "entries": entries,
                "summary": summary,
                "files": files,
                "scope": "native-verified behavior, not expert or fine-tuned; audit/ is privileged",
            }
        )
        atomic_write(stage / "corpus.json", canonical_json(manifest))
        verify_corpus(stage)
        # Empty pre-existing directories must not be replaced either.
        if output.exists() or output.is_symlink():
            raise ValueError("corpus output appeared during build")
        os.rename(stage, output)
        return {
            "ok": True,
            "sha256": manifest["sha256"],
            "summary": summary,
            "registry_status": "not_registered",
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_corpus(directory):
    """Cold native replay, provider-wire binding and exact exported-row comparison."""
    directory = Path(directory).resolve()
    manifest = _read(directory / "corpus.json")
    if manifest.get("schema") != SCHEMA or manifest.get("sha256") != digest(
        {k: v for k, v in manifest.items() if k != "sha256"}
    ):
        raise ValueError("invalid corpus seal")
    entries, files = manifest.get("entries"), manifest.get("files")
    if (
        not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_EPISODES
        or not isinstance(files, dict)
    ):
        raise ValueError("invalid corpus shape")
    expected_names = {"corpus.json", *files}
    actual = {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    }
    if actual != expected_names or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("corpus file inventory mismatch or symlink")
    used = 0
    for name, receipt in files.items():
        path = directory / name
        if (
            not path.resolve().is_relative_to(directory)
            or path.stat().st_size > MAX_FILE_BYTES
        ):
            raise ValueError("unsafe corpus file")
        used += path.stat().st_size
        if used > MAX_BUNDLE_BYTES or receipt != {
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }:
            raise ValueError("corpus content mismatch or byte budget exceeded")
    records = []
    allowed = {"data/rl.jsonl", "data/sft.jsonl", "data/wire-sft.jsonl"}
    for i in range(len(entries)):
        base = directory / "audit" / f"{i:04d}"
        wire = []
        allowed.add(f"audit/{i:04d}/episode.json")
        if (base / "manifest.json").exists():
            verify_local_bundle(base)
            local = _read(base / "manifest.json")
            for name in {
                "manifest.json",
                "episode.json",
                "inference.json",
                *local["exports"],
            }:
                allowed.add(f"audit/{i:04d}/{name}")
            if "wire-sft.jsonl" in local["exports"]:
                wire = list(
                    iter_jsonl(base / "wire-sft.jsonl", max_row_bytes=MAX_FILE_BYTES)
                )
        records.append((_read(base / "episode.json"), wire))
    if set(files) != allowed:
        raise ValueError("unexpected corpus files")
    actual_entries, summary, data = _analyse(records)
    if entries != actual_entries or summary != manifest.get("summary"):
        raise ValueError("corpus semantic manifest mismatch")
    for name, rows in data.items():
        if (
            list(
                iter_jsonl(
                    directory / "data" / f"{name}.jsonl", max_row_bytes=MAX_FILE_BYTES
                )
            )
            != rows
        ):
            raise ValueError("corpus exported rows mismatch")
    return {"ok": True, "sha256": manifest["sha256"], "summary": summary}


def register_corpus(directory, registry):
    directory, registry = Path(directory).resolve(), Path(registry).resolve()
    if registry.is_relative_to(directory):
        raise ValueError("registry must be outside immutable corpus")
    verified = verify_corpus(directory)
    manifest = _read(directory / "corpus.json")
    return _register_claims(
        registry,
        content=verified["sha256"],
        manifest=verified["sha256"],
        groups={(e["geometry_sha256"], e["split"]) for e in manifest["entries"]},
        trajectories={
            (e["trajectory_sha256"], e["split"]) for e in manifest["entries"]
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("output", type=Path)
    build.add_argument("inputs", type=Path, nargs="+")
    verify = sub.add_parser("verify")
    verify.add_argument("directory", type=Path)
    register = sub.add_parser("register")
    register.add_argument("directory", type=Path)
    register.add_argument("registry", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build_corpus(args.inputs, args.output)
    elif args.command == "verify":
        result = verify_corpus(args.directory)
    else:
        result = register_corpus(args.directory, args.registry)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
