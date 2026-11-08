"""Generate native and provider-wire training data from a real local LLM.

Run from project root: python -m examples.llm.generate_local --configuration ... --output ...
"""

import argparse
import json
from pathlib import Path

from examples.llm.run_local_qwen import api, identity
from space_env.llm.episode import (
    LLMEpisode,
    export_training,
    seal,
    verify_record,
)
from space_env.llm.local_data import verify_local_bundle as audit
from space_env.llm.local_data import wire_sft_rows
from space_env.llm.ollama import (
    InferenceStopped,
    OllamaPolicy,
)
from space_env.planning.network_benchmark import geometry_key
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl
from space_env.workspace import MissionWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument(
        "--planning-method",
        default="serial",
        choices=(
            "serial",
            "exact_small",
            "sequence_compact",
            "pointing_bounded",
            "sequence_lanes",
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--audit", type=Path)
    parser.add_argument("--reserved-splits", type=Path)
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11519")
    args = parser.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
        return 0
    if args.configuration is None:
        parser.error("--configuration is required with --output")
    config = json.loads(args.configuration.read_text())
    group = geometry_key(config)
    if args.reserved_splits:
        reserved = json.loads(args.reserved_splits.read_text())
        if reserved["sha256"] != digest(
            {k: v for k, v in reserved.items() if k != "sha256"}
        ):
            raise ValueError("Invalid reserved split manifest")
        if group in {
            c["geometry_sha256"] for c in reserved["cases"] if c["split"] != "train"
        }:
            raise ValueError("Training geometry is reserved for evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    model_identity = identity(
        Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/qwen3.5/4b"
    )
    policy = OllamaPolicy(
        endpoint=args.endpoint,
        protocol_profile="json_fixed_v1",
        max_requests=args.max_calls,
    )
    before = next(
        m
        for m in api(policy._opener, policy.endpoint, "/api/tags")["models"]
        if m["name"] == policy.model
    )
    if before["digest"] != model_identity["manifest_sha256"]:
        raise ValueError("Model identity mismatch")
    mission = MissionWorkspace.network(config, planning_method=args.planning_method)
    try:
        if policy.fixed_decision_sec != mission.operator.session._task.decision_sec:
            raise ValueError("Fixed cadence differs from task cadence")
        episode = LLMEpisode(
            mission.operator.session,
            task_text="Deliver every mission request before its deadline; native Ground ACK is required. Coordinate all spacecraft roles.",
            group=group,
            split="train",
            max_calls=args.max_calls,
            policy_origin="external_llm",
        )
        error = None
        try:
            while not episode.done:
                call = policy(episode.messages, episode.tool_schemas())
                if call is None:
                    break
                response = episode.invoke(
                    call["name"],
                    call["arguments"],
                    call_id=f"llm-{len(episode._events)}",
                )
                print(len(episode._events), call["name"], response["ok"], flush=True)
        except InferenceStopped as exc:
            error = str(exc)
        record = episode.record()
        proof = verify_record(record)
        atomic_write(args.output / "episode.json", canonical_json(record))
        atomic_write(
            args.output / "inference.json",
            canonical_json(seal({"trace": policy.trace})),
        )
        exports = {
            "rl.jsonl": export_training(record, args.output / "rl.jsonl", format="rl")
        }
        if record["metrics"]["sft_eligible"] and error is None:
            exports["sft.jsonl"] = export_training(
                record, args.output / "sft.jsonl", format="sft"
            )
            exports["wire-sft.jsonl"] = atomic_jsonl(
                args.output / "wire-sft.jsonl",
                wire_sft_rows(record, policy.trace, model=policy.model),
                max_bytes=64000000,
            )
        after = next(
            m
            for m in api(policy._opener, policy.endpoint, "/api/tags")["models"]
            if m["name"] == policy.model
        )
        if after["digest"] != before["digest"]:
            raise ValueError("Model changed during generation")
        manifest = seal(
            {
                "schema": "space-env.local-llm-data/1",
                "configuration": config,
                "geometry_sha256": group,
                "model_identity": model_identity,
                "episode_sha256": record["sha256"],
                "verification": proof,
                "exports": exports,
                "metrics": record["metrics"],
                "inference": policy.metrics(),
                "provider_error": error,
                "scope": "real local model, explicit train geometry distinct from supplied reserved evaluation splits; native and provider-wire SFT, failed episodes RL only; no model training or expert qualification",
            }
        )
        atomic_write(args.output / "manifest.json", canonical_json(manifest))
        print(canonical_json(manifest["metrics"]), flush=True)
    finally:
        mission.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
