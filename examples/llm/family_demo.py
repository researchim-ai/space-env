"""Representative scripted family benchmark across all ML splits, no LLM calls."""

import argparse
from pathlib import Path

from space_env.llm.mission_suite import audit, compile_suite, run_suite
from space_env.simulation.snapshot import canonical_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="36 cases / 108 policy runs instead of representative 8 / 24",
    )
    args = parser.parse_args()
    suite = compile_suite()
    selected = (
        None
        if args.full
        else [suite["cases"][i]["id"] for i in [0, 1, 2, 3, 4, 5, 12, 24]]
    )
    report = run_suite(args.output, selected=selected)
    print(
        canonical_json(
            {
                "audit": audit(args.output),
                "suite_sha256": suite["sha256"],
                "selected": report["selected"],
                "aggregates": report["aggregates"],
                "model_called": False,
                "model_fine_tuned": False,
            }
        )
    )


if __name__ == "__main__":
    main()
