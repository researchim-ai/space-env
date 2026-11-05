"""Explicit benchmark splits over legacy native bundles; SFT fails closed.

Native bundle lineage hashes are preserved as provenance, never used to override
benchmark train/development/test groups. Source JSONL hashes are checked during
streaming so a changed input cannot produce a published output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import shutil
import tempfile
from pathlib import Path

from space_env.planning.mission_bundle import read_json, strict_json
from space_env.planning.release_benchmark import POLICIES, Suite, verify_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl, file_digest

PARTITIONS = ("train", "development", "test")
MAX_BYTES = 2147483648


def source(directory, native=False):
    root = Path(directory)
    report = read_json(root / "report.json")
    spec = Suite.model_validate(read_json(root / "spec.json"))
    paths = {}
    if report["schema"] == "space-env.release-qualification/1":
        module_spec = importlib.util.spec_from_file_location(
            "release_orchestrator",
            Path(__file__).with_name("run_release_qualification.py"),
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.verify(root, native=native)
        for part in report["parts"]:
            directory = root / "parts" / f"{part['index']:03d}" / "results"
            sub = Suite.model_validate(read_json(directory / "spec.json"))
            for i, c in enumerate(sub.cases):
                for policy in POLICIES:
                    paths[c.id, policy] = directory / f"{i:03d}" / policy / "dataset"
    else:
        verify_suite(root, native=native)
        for i, c in enumerate(spec.cases):
            for policy in POLICIES:
                paths[c.id, policy] = root / f"{i:03d}" / policy / "dataset"
    if any(
        r.get("error") or r.get("native_replay_verified") is not True
        for r in report["rows"]
    ):
        raise ValueError(
            "training export requires complete native evidence; failures cannot be dropped"
        )
    return root, report, spec, paths


def rows(context, partition, fmt):
    root, report, spec, paths = context
    if fmt == "sft" and (
        partition != "train" or report["teacher"]["passed_declared_gates"] is not True
    ):
        return
    for case in spec.cases:
        if case.split != partition:
            continue
        for policy in POLICIES:
            path = paths[case.id, policy] / (fmt + ".jsonl")
            if not path.exists():
                if fmt == "sft":
                    continue
                raise ValueError("missing native RL source")
            relative = path.relative_to(root).as_posix()
            expected = report["files"][relative]["sha256"]
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                while raw := stream.readline(33554433):
                    if len(raw) > 33554432:
                        raise ValueError("source row byte budget exceeded")
                    checksum.update(raw)
                    row = strict_json(raw)
                    if (
                        row["lineage"] != case.scenario["lineage"]
                        or row["schema"] != "space-env.task-dataset/1"
                    ):
                        raise ValueError(
                            "benchmark row belongs to another lineage/schema"
                        )
                    original_sha = digest(row)
                    row = copy.deepcopy(row)
                    row["source_schema"] = row["schema"]
                    row["legacy_lineage_split"] = row["split"]
                    row.update(
                        schema="space-env.release-" + fmt + "/1",
                        split=partition,
                        benchmark_case=case.id,
                        benchmark_group=case.group,
                        benchmark_policy=policy,
                        benchmark_source_sha256=report["sha256"],
                        source_row_sha256=original_sha,
                        expert_teacher_qualified=False,
                    )
                    yield row
            if checksum.hexdigest() != expected:
                raise ValueError("native source changed during export")


def publish(directory, output, *, native=False):
    context = source(directory, native=native)
    _root, report, spec, _ = context
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("training output must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        files = {}
        used = 0
        for partition in PARTITIONS:
            formats = (
                ("rl", "sft")
                if partition == "train" and report["teacher"]["passed_declared_gates"]
                else ("rl",)
            )
            for fmt in formats:
                path = stage / partition / (fmt + ".jsonl")
                path.parent.mkdir(exist_ok=True)
                meta = atomic_jsonl(
                    path,
                    rows(context, partition, fmt),
                    max_rows=10000,
                    max_bytes=MAX_BYTES - used,
                )
                files[path.relative_to(stage).as_posix()] = {
                    **meta,
                    "partition": partition,
                    "format": fmt,
                }
                used += meta["bytes"]
        body = {
            "schema": "space-env.release-training-splits/1",
            "source_sha256": report["sha256"],
            "spec_sha256": digest(spec.model_dump(mode="json", by_alias=True)),
            "teacher_gates_passed": report["teacher"]["passed_declared_gates"],
            "sft_exported": any(
                m["format"] == "sft" and m["rows"] for m in files.values()
            ),
            "expert_teacher_qualified": False,
            "files": files,
            "native_replay_on_export": native,
            "split_authority": "explicit frozen benchmark geometry groups; legacy lineage hash is provenance only",
        }
        body["sha256"] = digest(body)
        atomic_write(stage / "manifest.json", canonical_json(body))
        if output.exists() or output.is_symlink():
            raise ValueError("training output appeared during publish")
        stage.rename(output)
        return body
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify(directory, output, native=False):
    context = source(directory, native)
    _, report, spec, _ = context
    root = Path(output)
    manifest = read_json(root / "manifest.json")
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("training symlinks forbidden")
    if manifest["schema"] != "space-env.release-training-splits/1" or manifest[
        "sha256"
    ] != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("invalid training manifest")
    if manifest["source_sha256"] != report["sha256"] or manifest[
        "spec_sha256"
    ] != digest(spec.model_dump(mode="json", by_alias=True)):
        raise ValueError("training source binding differs")
    expected = {f"{p}/rl.jsonl" for p in PARTITIONS}
    if report["teacher"]["passed_declared_gates"]:
        expected.add("train/sft.jsonl")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected | {"manifest.json"} or set(manifest["files"]) != expected:
        raise ValueError("training inventory differs")
    if (
        manifest["teacher_gates_passed"]
        is not report["teacher"]["passed_declared_gates"]
        or manifest["expert_teacher_qualified"] is not False
    ):
        raise ValueError("unapproved teacher promotion")
    for name, meta in manifest["files"].items():
        partition, filename = name.split("/")
        fmt = filename.removesuffix(".jsonl")
        if (
            meta["partition"] != partition
            or meta["format"] != fmt
            or (root / name).stat().st_size != meta["bytes"]
            or file_digest(root / name) != meta["file_sha256"]
        ):
            raise ValueError("training file binding differs")
        count = 0
        checksum = hashlib.sha256(b"[")
        with (root / name).open("rb") as stream:
            for wanted in rows(context, partition, fmt):
                raw = stream.readline(33554433)
                if len(raw) > 33554432 or not raw or strict_json(raw) != wanted:
                    raise ValueError("training split or source row differs")
                if count:
                    checksum.update(b",")
                checksum.update(canonical_json(wanted).encode())
                count += 1
            if stream.readline(1):
                raise ValueError("extra training row")
        checksum.update(b"]")
        if count != meta["rows"] or checksum.hexdigest() != meta["rows_sha256"]:
            raise ValueError("training row digest differs")
    promoted = any(
        m["format"] == "sft" and m["rows"] for m in manifest["files"].values()
    )
    if manifest["sft_exported"] is not bool(promoted):
        raise ValueError("SFT selection differs")
    return {
        "integrity_passed": True,
        "native_audit_repeated": native,
        "teacher_gates_passed": report["teacher"]["passed_declared_gates"],
        "rows": {n: m["rows"] for n, m in manifest["files"].items()},
        "sft_exported": bool(promoted),
        "expert_teacher_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "verify"])
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    result = (
        publish(args.source, args.output, native=args.native)
        if args.command == "export"
        else verify(args.source, args.output, args.native)
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
