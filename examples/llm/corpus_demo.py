"""Native replay-backed corpus demo. Scripted policy; no model inference or training.

python -m examples.llm.corpus_demo --output /tmp/new-corpus-demo
"""

import argparse
from pathlib import Path

from space_env.llm.benchmark import reference_policy, run_agent
from space_env.llm.corpus import build_corpus, register_corpus, verify_corpus
from space_env.llm.episode import LLMEpisode
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.workspace import MissionWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    training = MissionWorkspace.network()
    try:
        session = training.operator.session
        metadata = session._scenario["metadata"]["network_workspace"]
        record = run_agent(
            LLMEpisode(
                session,
                task_text="Deliver every product to Ground.",
                group="demo-train",
                split="train",
                policy_origin="scripted_privileged",
            ),
            reference_policy(
                metadata["steps"], {a.id: a.role for a in session._task.actions}
            ),
        )
        config = metadata["configuration"]
        # A second geometry reserved before registering the training corpus.
        config["mission"]["planes"][0]["radius_m"] += 10000.0
    finally:
        training.close()
    holdout = MissionWorkspace.network(config)
    try:
        reservation = LLMEpisode(
            holdout.operator.session,
            task_text="Reserved geometry, not an evaluation score.",
            group="demo-reserved",
            split="test",
            policy_origin="scripted_public",
        ).record()
    finally:
        holdout.close()
    inputs = args.output / "inputs"
    inputs.mkdir()
    for name, value in [("train", record), ("reserved", reservation)]:
        atomic_write(inputs / (name + ".json"), canonical_json(value))
    db = args.output / "registry.sqlite"
    build_corpus([inputs / "reserved.json"], args.output / "reserved")
    register_corpus(args.output / "reserved", db)
    built = build_corpus([inputs / "train.json"], args.output / "train")
    registered = register_corpus(args.output / "train", db)
    proof = verify_corpus(args.output / "train")
    print(
        canonical_json(
            {
                "build": built,
                "registration": registered,
                "verification": proof,
                "model_called": False,
                "model_fine_tuned": False,
            }
        )
    )


if __name__ == "__main__":
    main()
