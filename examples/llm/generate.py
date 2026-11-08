"""Generate native-replayed SFT and RL chat demonstrations without an LLM API."""

import argparse
import json
from pathlib import Path

from space_env.llm.benchmark import reference_policy, run_agent
from space_env.llm.episode import (
    LLMEpisode,
    _training_row,
    export_training,
    seal,
    verify_record,
)
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import file_digest, iter_jsonl
from space_env.workspace import MissionWorkspace


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["sha256"] != digest(
        {k: v for k, v in manifest.items() if k != "sha256"}
    ):
        raise ValueError("Invalid manifest checksum")
    record = json.loads((directory / "episode.json").read_text())
    verify_record(record)
    if manifest["episode_sha256"] != record["sha256"]:
        raise ValueError("Episode binding mismatch")
    for fmt in ("sft", "rl"):
        path = directory / (fmt + ".jsonl")
        if file_digest(path) != manifest["files"][fmt]["file_sha256"] or list(
            iter_jsonl(path, max_row_bytes=64000000)
        ) != [_training_row(record, format=fmt)]:
            raise ValueError("Training artifact mismatch")
    return {
        "verified": True,
        "success": record["metrics"]["success"],
        "sft_rows": 1,
        "rl_rows": 1,
        "external_llm_called": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    mission = MissionWorkspace.network()
    try:
        session = mission.operator.session
        metadata = session._scenario["metadata"]["network_workspace"]
        episode = LLMEpisode(
            session,
            task_text="Acquire, process and deliver the survey product before its deadline. Completion requires Ground ACK.",
            group="default-network-demonstration",
            policy_origin="scripted_privileged",
        )
        record = run_agent(
            episode,
            reference_policy(
                metadata["steps"], {a.id: a.role for a in session._task.actions}
            ),
        )
        verify_record(record)
        atomic_write(args.output / "episode.json", canonical_json(record))
        files = {
            fmt: export_training(record, args.output / (fmt + ".jsonl"), format=fmt)
            for fmt in ("sft", "rl")
        }
        manifest = seal(
            {
                "schema": "space-env.llm-demo-bundle/1",
                "episode_sha256": record["sha256"],
                "files": files,
                "metrics": record["metrics"],
                "policy_origin": "scripted_privileged",
                "external_llm_called": False,
                "expert_qualification": False,
            }
        )
        atomic_write(args.output / "manifest.json", canonical_json(manifest))
        print(canonical_json(audit(args.output)))
    finally:
        mission.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
