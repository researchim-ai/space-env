"""Execute one declared physical task and produce replay-verified native data.

Synthetic scripted teacher, not real model inference. Public regression geometry;
use a shared registry to protect splits across multiple producers.
"""

import argparse
from pathlib import Path

from space_env.llm.benchmark import reference_policy, run_agent
from space_env.llm.corpus import build_corpus, register_corpus, verify_corpus
from space_env.llm.episode import LLMEpisode, seal, verify_record
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks import TaskSession
from space_env.tasks.modeling_suite import compile_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="momentum-nominal")
    parser.add_argument(
        "--split", choices=("train", "development", "test"), default="train"
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {c["id"]: c for c in compile_suite()["cases"]}
    if args.case not in cases:
        parser.error("Unknown case: " + args.case)
    args.output.mkdir(parents=True, exist_ok=False)
    c = cases[args.case]
    session = TaskSession(c["scenario"], c["task"])
    episode = LLMEpisode(
        session,
        task_text=c["task_text"],
        group=c["id"],
        split=args.split,
        max_calls=128,
        policy_origin="scripted_privileged",
    )
    record = run_agent(
        episode,
        reference_policy(
            c["reference_steps"], {a.id: a.role for a in session._task.actions}
        ),
    )
    verify_record(record)
    path = args.output / "episode.json"
    atomic_write(path, canonical_json(record))
    build_corpus([path], args.output / "corpus")
    registered = register_corpus(
        args.output / "corpus", args.registry or args.output / "registry.sqlite"
    )
    verification = verify_corpus(args.output / "corpus")
    report = seal(
        {
            "schema": "space-env.physical-goal-data/1",
            "case": args.case,
            "split": args.split,
            "episode_sha256": record["sha256"],
            "goal": record["metrics"]["goal"],
            "corpus": verification,
            "registration": registered,
            "model_called": False,
            "model_trained": False,
            "scope": "scripted physical-control native SFT/RL; public regression inputs, not an expert or blind benchmark; no provider wire-SFT",
        }
    )
    atomic_write(args.output / "report.json", canonical_json(report))
    print(
        canonical_json(
            {
                "success": record["metrics"]["success"],
                "corpus": verification["summary"],
                "sha256": report["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
