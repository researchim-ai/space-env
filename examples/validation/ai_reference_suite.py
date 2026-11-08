"""Run from the repository: PYTHONPATH=src python examples/validation/ai_reference_suite.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_env.simulation.snapshot import atomic_write
from space_env.validation.ai_reference_suite import run_suite, verify_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Also write the JSON report to this file"
    )
    args = parser.parse_args()
    report = run_suite()
    if not verify_report(report):
        raise RuntimeError("report seal or source manifest verification failed")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        atomic_write(args.output, payload)
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
