"""Local factory compilation, corpus build and independent verification."""

import json
from pathlib import Path

from space_env.factory.bundle import build_dataset, verify_bundle
from space_env.factory.generate import compile_factory
from space_env.factory.schema import FactorySpec
from space_env.simulation.snapshot import atomic_write, canonical_json


def run_command(args):
    if args.command == "factory-register":
        from space_env.factory.registry import register_bundle

        return register_bundle(args.bundle, args.registry)
    if args.command == "factory-verify":
        return verify_bundle(args.bundle, replay=not args.integrity_only)
    if args.command == "factory-schema":
        atomic_write(args.output, canonical_json(FactorySpec.model_json_schema()))
        return {"ok": True}
    if Path(args.spec).resolve() == Path(args.output).resolve():
        raise ValueError("factory output cannot overwrite its specification")
    spec = json.loads(Path(args.spec).read_text())
    if args.command == "factory-compile":
        result = compile_factory(spec)
        atomic_write(args.output, canonical_json(result))
        return {
            "ok": result["coverage"]["configuration_errors"] == 0,
            "cases": len(result["cases"]),
            "sha256": result["sha256"],
        }
    return build_dataset(spec, args.output)
