"""Fresh sequence teacher -> registered corpus -> causal SFT preparation.

Four explicit public regression cases, not a real LLM or training experiment.
Run after finishing source changes; historical source-bound corpora stay intact.
"""

import argparse
import json
from pathlib import Path

from space_env.llm.benchmark import reference_policy, run_agent
from space_env.llm.corpus import build_corpus, register_corpus, verify_corpus
from space_env.llm.episode import LLMEpisode, seal, sources
from space_env.llm.learning_experiment import prepare
from space_env.llm.mission_suite import TASK, compile_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.workspace import MissionWorkspace

SELECTED = (
    "equatorial-relay-chain",
    "equatorial-station-blackout",
    "inclined-35-relay-chain",
    "inclined-55-relay-chain",
)


def run(output, *, registry=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    suite, before = compile_suite(), sources()
    cases = {c["id"]: c for c in suite["cases"]}
    inputs = output / "inputs"
    inputs.mkdir()
    rows, paths = [], []
    atomic_write(output / "suite.json", canonical_json(suite))
    for name in SELECTED:
        case = cases[name]
        workspace = MissionWorkspace.network(
            case["configuration"], planning_method="sequence_lanes"
        )
        try:
            session = workspace.operator.session
            meta = session._scenario["metadata"]["network_workspace"]
            record = run_agent(
                LLMEpisode(
                    session,
                    task_text=TASK,
                    group=case["group"],
                    split=case["split"],
                    max_calls=64,
                    policy_origin="scripted_privileged",
                ),
                reference_policy(
                    meta["steps"], {a.id: a.role for a in session._task.actions}
                ),
            )
        finally:
            workspace.close()
        path = inputs / (name + ".json")
        atomic_write(path, canonical_json(record))
        paths.append(path)
        rows.append(
            {
                "case": name,
                "split": case["split"],
                "episode_sha256": record["sha256"],
                "metrics": record["metrics"],
            }
        )
        atomic_write(output / "progress.json", canonical_json(seal({"rows": rows})))
    corpus = output / "corpus"
    build_corpus(paths, corpus)
    verified = verify_corpus(corpus)
    registered = register_corpus(corpus, registry or output / "registry.sqlite")
    prepared = prepare([corpus], evaluation_split="development")
    wait, active = 0, 0
    for row in prepared["rows"]:
        target = json.loads(row["target"])
        non_wait = any(
            v != "wait" for k, v in target.items() if k.startswith("action_")
        )
        if non_wait:
            active += 1
        else:
            wait += 1
    if not active:
        raise ValueError(
            "No active command training targets; refusing an all-wait demo"
        )
    if sources() != before:
        raise ValueError("Sources changed during demo")
    report = seal(
        {
            "schema": "space-env.ai-advancement-demo/1",
            "model_sources": before,
            "suite_sha256": suite["sha256"],
            "selected": list(SELECTED),
            "rows": rows,
            "corpus": verified,
            "registration": registered,
            "preparation": {
                "train_decisions": len(prepared["rows"]),
                "active_command_decisions": active,
                "no_active_command_decisions": wait,
                "evaluation_tasks": len(prepared["evaluations"]),
                "evaluation_split": "development",
                "unrepresentable_decisions": prepared["unrepresentable_decisions"],
                "data_sha256": digest(prepared["rows"]),
            },
            "model_called": False,
            "model_fine_tuned": False,
            "scope": "four selected public cases, three geometry groups; scripted privileged teacher; native data/registry/causal preparation integration only, not full qualification or real model scores",
        }
    )
    atomic_write(output / "report.json", canonical_json(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, help="Use a shared registry across dataset producers"
    )
    args = parser.parse_args()
    report = run(args.output, registry=args.registry)
    print(
        canonical_json(
            {
                "sha256": report["sha256"],
                "preparation": report["preparation"],
                "model_called": False,
                "model_fine_tuned": False,
            }
        )
    )


if __name__ == "__main__":
    main()
