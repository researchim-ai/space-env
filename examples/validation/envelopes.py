"""Installed-package numerical acceptance; run outside the repository with -I."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.envelopes import analytic_envelope_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--long", action="store_true", dest="long_arcs")
parser.add_argument("--external", action="store_true")
args = parser.parse_args()
report = analytic_envelope_report(long_arcs=args.long_arcs, external=args.external)
assert report["sha256"] == digest({k: v for k, v in report.items() if k != "sha256"})
assert report["industrial_readiness"] is False
assert report["build"]["distribution_matches_source_version"]
atomic_write(args.output, canonical_json(report))
print(
    json.dumps(
        {
            "passed": report["passed"],
            "checks": len(report["analytic"]["checks"]),
            "sha256": report["sha256"],
        }
    )
)
raise SystemExit(0 if report["passed"] else 1)
