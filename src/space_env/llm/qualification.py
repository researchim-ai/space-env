"""Resumable local qualification on all frozen families and geometry groups.

One finished case per commit, no retry of finished error cells, fixed complete
matrix denominators. This is deterministic regression evidence, not blind/OOD,
flight calibration, a statistical confidence interval or a real LLM evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from space_env.llm.comparison import audit as audit_case
from space_env.llm.comparison import run as run_case
from space_env.llm.episode import seal, sources
from space_env.llm.mission_suite import compile_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest

METHODS = ("serial", "sequence_lanes")
MAX_ATTEMPTS = 3


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64000000:
        raise ValueError("Unsafe or oversized qualification artifact")
    with path.open("rb") as stream:
        raw = stream.read(64000001)
    if len(raw) > 64000000:
        raise ValueError("Oversized qualification artifact")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("sha256") != digest(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("Qualification artifact seal mismatch")
    return value


def _plan():
    return seal(
        {
            "schema": "space-env.sequence-qualification-plan/1",
            "suite": compile_suite(),
            "methods": list(METHODS),
            "model_sources": sources(),
            "max_attempts": MAX_ATTEMPTS,
            "scope": "36 frozen public regression workloads, six geometry groups; privileged reference executor; no blind test or external LLM",
        }
    )


@contextmanager
def _lease(root):
    path = root / ".writer.sqlite"
    if path.is_symlink():
        raise ValueError("Unsafe qualification writer lock")
    conn = sqlite3.connect(path, timeout=0.1)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise ValueError("Qualification writer is busy") from exc
        yield
    finally:
        conn.rollback()
        conn.close()


def _path(root, case, attempt):
    path = root / "cases" / case / f"attempt-{attempt:02d}"
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Unsafe qualification attempt path")
    return path


def _journal(root, plan):
    journal = _read(root / "journal.json")
    if (
        journal.get("schema") != "space-env.sequence-qualification-journal/1"
        or journal.get("plan_sha256") != plan["sha256"]
        or not isinstance(journal.get("attempts"), list)
    ):
        raise ValueError("Invalid qualification journal")
    ids = {c["id"] for c in plan["suite"]["cases"]}
    seen = set()
    finished = set()
    previous_status = {}
    previous = {}
    for entry in journal["attempts"]:
        name, index, status = entry["case"], entry["attempt"], entry["status"]
        if (
            name not in ids
            or type(index) is not int
            or not 1 <= index <= MAX_ATTEMPTS
            or (name, index) in seen
            or index != previous.get(name, 0) + 1
            or status not in {"running", "interrupted", "completed"}
        ):
            raise ValueError("Invalid qualification attempt sequence")
        if name in previous_status and previous_status[name] != "interrupted":
            raise ValueError("Only interrupted attempts can be superseded")
        previous_status[name] = status
        if (
            status == "interrupted"
            and (_path(root, name, index) / "report.json").exists()
        ):
            raise ValueError("Completed report hidden as interrupted attempt")
        if name in finished:
            raise ValueError("Finished cases cannot be retried")
        if status == "completed":
            finished.add(name)
        seen.add((name, index))
        previous[name] = index
    # An unjournaled directory must not silently disappear from provenance.
    expected = {_path(root, c, i).relative_to(root).as_posix() for c, i in seen}
    actual = {
        p.relative_to(root).as_posix() for p in (root / "cases").glob("*/attempt-*")
    }
    completed_paths = {
        _path(root, e["case"], e["attempt"]).relative_to(root).as_posix()
        for e in journal["attempts"]
        if e["status"] == "completed"
    }
    if not actual <= expected or not completed_paths <= actual:
        raise ValueError("Qualification attempt inventory mismatch")
    if sum(e["status"] == "running" for e in journal["attempts"]) > 1:
        raise ValueError("Ambiguous concurrent qualification attempts")
    return journal


def _save_journal(root, journal):
    atomic_write(
        root / "journal.json",
        canonical_json(seal({k: v for k, v in journal.items() if k != "sha256"})),
    )


def _child(root, plan, entry):
    path = _path(root, entry["case"], entry["attempt"])
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Symlink in qualification attempt")
    report = _read(path / "report.json")
    if (
        report["model_sources"] != plan["model_sources"]
        or report["selected"] != [entry["case"]]
        or report["methods"] != plan["methods"]
        or canonical_json(report["suite"]) != canonical_json(plan["suite"])
    ):
        raise ValueError("Qualification child input mismatch")
    if (
        entry["status"] == "completed"
        and entry.get("report_sha256") != report["sha256"]
    ):
        raise ValueError("Qualification child report changed")
    proof = audit_case(path)
    return report, proof


def summarize(plan, rows):
    cases = {c["id"]: c for c in plan["suite"]["cases"]}
    expected = {(c, m) for c in cases for m in plan["methods"]}
    mapping = {(r["case"], r["method"]): r for r in rows}
    if len(mapping) != len(rows) or not set(mapping) <= expected:
        raise ValueError("Duplicate or unexpected qualification cells")
    for (name, _), row in mapping.items():
        if (
            row["split"] != cases[name]["split"]
            or row["family"] != cases[name]["family"]
            or row["status"] not in {"completed", "error"}
        ):
            raise ValueError("Qualification cell identity mismatch")
        if row["status"] == "error":
            if row["metrics"] is not None:
                raise ValueError("Error cell has fabricated metrics")
        else:
            total = sum(
                r["priority"]
                for r in cases[name]["configuration"]["mission"]["requests"]
            )
            metrics = row["metrics"]
            count = len(cases[name]["configuration"]["mission"]["requests"])
            if (
                type(metrics.get("success")) is not bool
                or type(metrics.get("fulfilled")) is not int
                or not 0 <= metrics["fulfilled"] <= count
                or metrics.get("total") != count
                or (metrics["success"] and metrics["fulfilled"] != count)
            ):
                raise ValueError("Invalid qualification goal/count")
            value = metrics["weighted_fulfilled"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= total
            ):
                raise ValueError("Invalid qualification utility")
    axes = {"overall": {"all": list(cases)}}
    for field in ("split", "family", "group"):
        axes[field] = {
            label: [c["id"] for c in cases.values() if c[field] == label]
            for label in sorted({c[field] for c in cases.values()})
        }
    aggregates = []
    for axis, labels in axes.items():
        for label, names in labels.items():
            for method in plan["methods"]:
                cells = [mapping.get((name, method)) for name in names]
                measured = [
                    r for r in cells if r is not None and r["status"] == "completed"
                ]
                values = [
                    r["metrics"]["weighted_fulfilled"]
                    / max(
                        1,
                        sum(
                            d["priority"]
                            for d in cases[r["case"]]["configuration"]["mission"][
                                "requests"
                            ]
                        ),
                    )
                    for r in measured
                ]
                aggregates.append(
                    {
                        "axis": axis,
                        "label": label,
                        "method": method,
                        "cases": len(names),
                        "geometry_groups": len({cases[n]["group"] for n in names}),
                        "replayed": len(measured),
                        "pending": sum(r is None for r in cells),
                        "errors": sum(
                            r is not None and r["status"] == "error" for r in cells
                        ),
                        "full_goals": sum(r["metrics"]["success"] for r in measured),
                        "value_lower": sum(values) / len(names),
                        "value_upper": (sum(values) + len(names) - len(measured))
                        / len(names),
                    }
                )
    paired = {"improved": 0, "tied": 0, "regressed": 0, "unknown": 0}
    for name in cases:
        baseline = mapping.get((name, "serial"))
        candidate = mapping.get((name, "sequence_lanes"))
        if (
            not baseline
            or not candidate
            or baseline["status"] != "completed"
            or candidate["status"] != "completed"
        ):
            paired["unknown"] += 1
            continue
        delta = (
            candidate["metrics"]["weighted_fulfilled"]
            - baseline["metrics"]["weighted_fulfilled"]
        )
        paired["improved" if delta > 0 else "regressed" if delta < 0 else "tied"] += 1
    measured = [r for r in rows if r["status"] == "completed"]
    return {
        "expected_cells": len(expected),
        "recorded_cells": len(rows),
        "replayed": len(measured),
        "status": (
            "partial"
            if len(rows) < len(expected)
            else "native_matrix_replayed"
            if len(measured) == len(expected)
            and all(r["planned"] == r["metrics"]["fulfilled"] for r in measured)
            else "observed_runtime_or_forecast_issues"
        ),
        "runtime_errors": sum(r["status"] == "error" for r in rows),
        "forecast_native_count_mismatches": sum(
            r["planned"] != r["metrics"]["fulfilled"] for r in measured
        ),
        "paired": {"metric": "delivered_priority", **paired},
        "aggregates": aggregates,
        "bounds_kind": "unknown-result bounds, not statistical confidence intervals",
        "geometry_groups": len({c["group"] for c in cases.values()}),
        "blind_holdout_claim": False,
        "external_llm_measured": False,
        "model_fine_tuned": False,
    }


def _state(plan, journal, reports):
    rows = [row for report in reports for row in report["rows"]]
    completed = [
        entry for entry in journal["attempts"] if entry["status"] == "completed"
    ]
    complete = len(completed) == len(plan["suite"]["cases"])
    return seal(
        {
            "schema": "space-env.sequence-qualification/1",
            "plan_sha256": plan["sha256"],
            "journal_sha256": journal["sha256"],
            "complete": complete,
            "completed_cases": len(completed),
            "interrupted_attempts": sum(
                e["status"] == "interrupted" for e in journal["attempts"]
            ),
            "children": [
                {
                    "case": e["case"],
                    "attempt": e["attempt"],
                    "report_sha256": e["report_sha256"],
                }
                for e in completed
            ],
            "summary": summarize(plan, rows),
        }
    )


def run_study(output, *, resume=False, max_cases=None):
    if type(resume) is not bool or (
        max_cases is not None
        and (type(max_cases) is not int or not 1 <= max_cases <= 36)
    ):
        raise ValueError("Invalid qualification run options")
    root = Path(output).absolute()
    plan = _plan()
    if resume:
        if not root.is_dir() or canonical_json(
            _read(root / "study.json")
        ) != canonical_json(plan):
            raise ValueError("Cannot resume a different suite/source revision")
    else:
        root.mkdir(parents=True, exist_ok=False)
    with _lease(root):
        if not resume:
            atomic_write(root / "study.json", canonical_json(plan))
            (root / "cases").mkdir()
            _save_journal(
                root,
                {
                    "schema": "space-env.sequence-qualification-journal/1",
                    "plan_sha256": plan["sha256"],
                    "attempts": [],
                },
            )
        journal = _journal(root, plan)
        reports = []
        finished = set()
        for entry in journal["attempts"]:
            path = _path(root, entry["case"], entry["attempt"])
            if entry["status"] == "completed" or (
                entry["status"] == "running" and (path / "report.json").exists()
            ):
                report, _ = _child(root, plan, entry)
                entry.update(status="completed", report_sha256=report["sha256"])
                finished.add(entry["case"])
                reports.append(report)
            elif entry["status"] == "running":
                entry["status"] = "interrupted"
        _save_journal(root, journal)
        journal = _journal(root, plan)
        done = 0
        for case in plan["suite"]["cases"]:
            if case["id"] in finished:
                continue
            attempt = 1 + sum(e["case"] == case["id"] for e in journal["attempts"])
            if attempt > MAX_ATTEMPTS:
                raise ValueError("Qualification attempt budget exhausted")
            entry = {"case": case["id"], "attempt": attempt, "status": "running"}
            path = _path(root, case["id"], attempt)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Record before launching. A crash before directory creation is a
            # valid interrupted attempt, not permission to discard provenance.
            journal["attempts"].append(entry)
            _save_journal(root, journal)
            run_case(path, [case["id"]], methods=plan["methods"])
            report, _ = _child(root, plan, entry)
            entry.update(status="completed", report_sha256=report["sha256"])
            reports.append(report)
            _save_journal(root, journal)
            journal = _journal(root, plan)
            state = _state(plan, journal, reports)
            atomic_write(root / "progress.json", canonical_json(state))
            done += 1
            if max_cases is not None and done >= max_cases:
                break
        journal = _journal(root, plan)
        state = _state(plan, journal, reports)
        atomic_write(root / "progress.json", canonical_json(state))
        if state["complete"]:
            atomic_write(root / "report.json", canonical_json(state))
        return state


def audit_study(directory, *, require_complete=True):
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be boolean")
    root = Path(directory).resolve()
    plan = _read(root / "study.json")
    if canonical_json(plan) != canonical_json(_plan()):
        raise ValueError("Qualification source/suite mismatch")
    journal = _journal(root, plan)
    if any(e["status"] == "running" for e in journal["attempts"]):
        raise ValueError("Interrupted write: resume before audit")
    reports = []
    for entry in journal["attempts"]:
        if entry["status"] == "completed":
            report, _ = _child(root, plan, entry)
            reports.append(report)
        elif (_path(root, entry["case"], entry["attempt"]) / "report.json").exists():
            raise ValueError("Completed report hidden as interrupted attempt")
    expected = _state(plan, journal, reports)
    if canonical_json(_read(root / "progress.json")) != canonical_json(expected):
        raise ValueError("Qualification aggregate/progress mismatch")
    if require_complete:
        if not expected["complete"]:
            raise ValueError("Qualification matrix is incomplete")
        if canonical_json(_read(root / "report.json")) != canonical_json(expected):
            raise ValueError("Qualification final report mismatch")
    return {
        "valid": True,
        "complete": expected["complete"],
        "replayed": expected["summary"]["replayed"],
        "report_sha256": expected["sha256"],
        "summary": expected["summary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--audit", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit_study(args.audit, require_complete=not args.partial)
    else:
        result = run_study(args.output, resume=args.resume, max_cases=args.max_cases)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
