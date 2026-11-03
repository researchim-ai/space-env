"""Installed Q5-B numerical acceptance; use target venv python -I outside checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.pointing import pointing_validation_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
report = pointing_validation_report()
assert report["sha256"] == digest({k: v for k, v in report.items() if k != "sha256"})
assert report["industrial_readiness"] is False
assert report["build"]["distribution_matches_source_version"]
atomic_write(args.output, canonical_json(report))
print(
    json.dumps(
        {
            "passed": report["passed"],
            "cases": len(report["cases"]),
            "checks": len(report["checks"]),
            "sha256": report["sha256"],
        }
    )
)
raise SystemExit(0 if report["passed"] else 1)
