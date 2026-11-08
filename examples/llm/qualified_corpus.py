"""Build a train-only corpus from a COMPLETE source-bound qualification matrix.

All valid sequence episodes are included, including failed goals and evaluation
splits in privileged audit/. Failed runtime cells remain disclosed exclusions.
"""

import argparse
import json
from pathlib import Path

from space_env.llm.corpus import build_corpus, register_corpus, verify_corpus
from space_env.llm.episode import seal
from space_env.llm.qualification import audit_study
from space_env.simulation.snapshot import canonical_json


def build(study, output, registry, *, resume=False):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.is_relative_to(study):
        raise ValueError("Corpus must be outside qualification evidence")
    proof = audit_study(study)
    plan = json.loads((study / "study.json").read_text())
    journal = json.loads((study / "journal.json").read_text())
    finished = {
        entry["case"]: entry
        for entry in journal["attempts"]
        if entry["status"] == "completed"
    }
    inputs = []
    expected = []
    missing = []
    for case in plan["suite"]["cases"]:
        entry = finished[case["id"]]
        path = study / "cases" / case["id"] / f"attempt-{entry['attempt']:02d}"
        report = json.loads((path / "report.json").read_text())
        row = next(r for r in report["rows"] if r["method"] == "sequence_lanes")
        if row["status"] != "completed":
            missing.append(case["id"])
            continue
        inputs.append(path / row["artifact"])
        expected.append(row["episode_sha256"])
    if resume:
        verify_corpus(output)
    else:
        build_corpus(inputs, output)
    manifest = json.loads((output / "corpus.json").read_text())
    if [e["episode_sha256"] for e in manifest["entries"]] != expected:
        raise ValueError("Corpus episodes are not bound to qualification study")
    registered = register_corpus(output, registry)
    return seal(
        {
            "schema": "space-env.qualified-corpus-link/1",
            "study_report_sha256": proof["report_sha256"],
            "corpus_sha256": manifest["sha256"],
            "summary": manifest["summary"],
            "missing_verified_episodes": missing,
            "registration": {
                "registered": registered["ok"],
                "content_sha256": registered["content_sha256"],
            },
            "model_fine_tuned": False,
            "external_llm_measured": False,
            "scope": "all verified sequence episodes from complete regression matrix; eval audit only, train-only data; scripted privileged teacher, not expert or wire-SFT",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        canonical_json(
            build(args.study, args.output, args.registry, resume=args.resume)
        )
    )


if __name__ == "__main__":
    main()
