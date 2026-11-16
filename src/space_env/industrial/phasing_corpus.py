"""Local generated-phasing corpus with explicit regimes and retained attempts.

No LLM inference, training or CI. Checkpoints and manifests are trainer-only.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from space_env.industrial.agent import IndustrialEpisode, verify_episode
from space_env.industrial.orbits import OrbitSession, orbit_policy
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks.phasing_generation import RANGES, generated_phasing_recipe


def corpus_manifest(
    *,
    samples_per_split=2,
    actors=2,
    windows=False,
    disruptions=None,
    competition=False,
    utility=False,
    acceptance=None,
):
    if type(samples_per_split) is not int or not 1 <= samples_per_split <= 32:
        raise ValueError("samples_per_split must be an integer in 1..32")
    if type(windows) is not bool:
        raise TypeError("windows must be boolean")
    if disruptions not in {None, "recoverable", "critical"} or (
        disruptions is not None and not windows
    ):
        raise ValueError(
            "disruption profile requires windows and recoverable/critical mode"
        )
    if type(competition) is not bool or type(utility) is not bool:
        raise TypeError("competition and utility must be boolean")
    if competition and (not windows or disruptions is not None):
        raise ValueError(
            "competition requires windows and excludes disruption profiles"
        )
    if acceptance not in {None, "priority", "mandatory", "infeasible"} or (
        acceptance is not None and (not competition or not utility)
    ):
        raise ValueError("acceptance presets require a utility competition corpus")
    rows = []
    seen = set()
    for split in ("train", "development", "test"):
        for seed in range(samples_per_split):
            configuration = {
                "actors": actors,
                "generation": {
                    "split": split,
                    "seed": seed,
                    **({"windows": True} if windows else {}),
                    **({"disruptions": disruptions} if disruptions is not None else {}),
                    **({"competition": True} if competition else {}),
                },
            }
            recipe = generated_phasing_recipe(actors, configuration["generation"])
            if utility:
                configuration["utility"] = {
                    "priorities": {
                        f"slot-sat-{index:02}": 9 if index == seed % actors else 1
                        for index in range(actors)
                    }
                }
            if acceptance is not None:
                high = f"slot-sat-{seed % actors:02}"
                low = f"slot-sat-{(seed + 1) % actors:02}"
                ids = list(configuration["utility"]["priorities"])
                configuration["acceptance"] = {
                    "minimum_fraction": (1.0 if acceptance == "mandatory" else 9.0)
                    / (9 + actors - 1),
                    "mandatory": [low]
                    if acceptance == "mandatory"
                    else ids
                    if acceptance == "infeasible"
                    else [high],
                }
            identity = recipe["generation"]["physics_sha256"]
            if identity in seen:
                raise ValueError("duplicate physical instance; no silent resampling")
            seen.add(identity)
            rows.append(
                {
                    "id": f"{split}-{seed:04}",
                    "configuration": configuration,
                    "generation": recipe["generation"],
                }
            )
    regimes = copy.deepcopy(RANGES)
    if disruptions is not None:
        for domain in regimes.values():
            domain["horizons"] = tuple(t + 10 for t in domain["horizons"])
    body = {
        "schema": "space-env.phasing-corpus/1",
        "sources": sources(),
        "regimes": regimes,
        "disruptions": disruptions,
        "windowed": windows,
        "competition": competition,
        "utility": utility,
        "acceptance": acceptance,
        "cases": rows,
        "scope": "Disjoint exact physical instances and held-out horizon/thrust regimes in one common synthetic central-gravity family. Constructive targets; no feasibility-based filtering or claim of independent hardware provenance.",
    }
    return {**body, "sha256": digest(body)}


def task_brief(configuration):
    if configuration.get("acceptance"):
        return "Fulfil the published task_objective: all mandatory goals and its minimum weighted fraction. Other native goals are optional for this declared task; native all-goals success is tracked separately. Preserve declared resource constraints and use only current observations and tools."
    return "Reach the published orbital-state targets with fuel reserves preserved. Choose at most one visible manoeuvre per spacecraft. Use current ideal state, published engine constants and declared tools; no future execution or reference schedule is available."


def run_case(case, policy):
    if policy not in {"wait", "public"}:
        raise ValueError("unknown corpus policy")
    episode = IndustrialEpisode(
        OrbitSession(case["configuration"]),
        split=case["configuration"]["generation"]["split"],
        policy_origin="scripted_public",
        max_calls=24,
        task_text=task_brief(case["configuration"]),
    )
    index = 0
    while not episode.done:
        obs = episode.observe()
        action = orbit_policy(obs) if policy == "public" else {}
        episode.call(
            str(index),
            canonical_json(
                {
                    "action": action,
                    "expected_revision": obs["revision"],
                    "idempotency_key": str(index),
                }
            ),
        )
        index += 1
    return episode.record()


def run_corpus(
    output,
    *,
    samples_per_split=2,
    actors=2,
    windows=False,
    disruptions=None,
    competition=False,
    utility=False,
    acceptance=None,
):
    root = Path(output)
    if root.exists():
        raise ValueError("choose a new output directory")
    manifest = corpus_manifest(
        samples_per_split=samples_per_split,
        actors=actors,
        windows=windows,
        disruptions=disruptions,
        competition=competition,
        utility=utility,
        acceptance=acceptance,
    )
    root.mkdir(parents=True)
    atomic_write(root / "manifest.json", canonical_json(manifest))
    rows = []
    for case in manifest["cases"]:
        for policy in ("wait", "public"):
            row = {"case": case["id"], "policy": policy}
            try:
                record = run_case(case, policy)
                verification = verify_episode(record)
                name = case["id"] + "--" + policy + ".json"
                atomic_write(root / name, canonical_json(record))
                row.update(
                    status="completed",
                    record=name,
                    sha256=record["sha256"],
                    verification=verification,
                    metrics=record["metrics"],
                    invalid_calls=sum(
                        "error" in e["response"] for e in record["events"]
                    ),
                    call_budget_exhausted=record["call_budget_exhausted"],
                )
            except Exception as exc:  # noqa: BLE001 -- retain every failed attempt
                row.update(
                    status="error", error_type=type(exc).__name__, error=str(exc)[:1024]
                )
            rows.append(row)
            atomic_write(root / "progress.json", canonical_json({"rows": rows}))
    if manifest["sources"] != sources():
        raise ValueError("source changed during corpus execution")
    body = {
        "schema": "space-env.phasing-corpus-results/1",
        "manifest_sha256": manifest["sha256"],
        "rows": rows,
        "errors": sum(r["status"] == "error" for r in rows),
        "external_llm_measured": False,
        "training_measured": False,
        "flight_qualified": False,
    }
    result = {**body, "sha256": digest(body)}
    atomic_write(root / "report.json", canonical_json(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=2)
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--windows", action="store_true")
    parser.add_argument("--disruptions", choices=["recoverable", "critical"])
    parser.add_argument("--competition", action="store_true")
    parser.add_argument("--utility", action="store_true")
    parser.add_argument("--acceptance", choices=["priority", "mandatory", "infeasible"])
    args = parser.parse_args()
    result = run_corpus(
        args.output,
        samples_per_split=args.samples_per_split,
        actors=args.actors,
        windows=args.windows,
        disruptions=args.disruptions,
        competition=args.competition,
        utility=args.utility,
        acceptance=args.acceptance,
    )
    print(canonical_json({"attempts": len(result["rows"]), "errors": result["errors"]}))


if __name__ == "__main__":
    main()
