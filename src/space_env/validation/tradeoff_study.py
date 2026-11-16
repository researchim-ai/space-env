"""Weighted calendar certificates and separate heterogeneous EO smoke evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from space_env.core.constants import MU_EARTH
from space_env.core.frames import ecef_to_geodetic
from space_env.llm.episode import seal, sources
from space_env.planning import PlanningRun
from space_env.reference import (
    EOPSeries,
    Epoch,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.priority_calendar import solve_priority_calendar
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks.heterogeneous_missions import heterogeneous_mission
from space_env.validation.choice_benchmark import _verify_file, policy_config, witness


def calendar_suite():
    """Six predeclared synthetic cases; no outcome-dependent filtering."""
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    cases = []
    for meridian in (0.0, 20.0):
        angle = math.radians(meridian)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        ephemerides = {}
        for index, actor in enumerate(("a", "b")):
            phase = index * 0.001
            r = rotation @ np.array([7e6 * math.cos(phase), 7e6 * math.sin(phase), 0])
            v = rotation @ (
                math.sqrt(MU_EARTH / 7e6)
                * np.array([-math.sin(phase), math.cos(phase), 0])
            )
            result = propagate(
                ReferenceState(epoch, r, v), 20.0, context, sample_sec=5.0
            )
            ephemerides[actor] = {**result, "sha256": digest(result)}
        lat, lon, _ = ecef_to_geodetic(
            context.matrix(epoch) @ np.array(ephemerides["a"]["states_si"][0][:3])
        )
        for family in ("bulk_vs_pair", "release_and_gap", "infeasible"):
            rate = 280000.0 if family == "bulk_vs_pair" else 200000.0
            items = (
                [
                    (2800000, 11, 0.0, 10.0),
                    (150000, 6, 0.0, 10.0),
                    (150000, 6, 0.0, 10.0),
                ]
                if family == "bulk_vs_pair"
                else [
                    (1200000, 10, 0.0, 12.0),
                    (400000, 7, 0.0, 12.0),
                    (400000, 7, 4.0, 12.0),
                ]
            )
            spec = {
                "schema": "space-env.reference-downlink-spec/1",
                "ephemerides": copy.deepcopy(ephemerides),
                "vehicles": {
                    a: {
                        "memory_bits": 8e6,
                        "battery_wh": 1.0,
                        "rate_bps": rate,
                        "base_power_w": 0.0,
                        "tx_power_w": 0.0,
                    }
                    for a in ("a", "b")
                },
                "stations": [
                    {
                        "name": "Ground",
                        "lat_deg": float(lat),
                        "lon_deg": float(lon),
                        "terminals": 1,
                    }
                ],
                "horizon_sec": 10.0 if family == "bulk_vs_pair" else 12.0,
                "products": [
                    {
                        "id": f"product-{i}",
                        "actor": "a" if i == 0 else "b",
                        "size_bits": bits,
                        "priority": priority,
                        "ready_sec": ready,
                        "deadline_sec": deadline,
                    }
                    for i, (bits, priority, ready, deadline) in enumerate(items)
                ],
            }
            if family == "release_and_gap":
                spec["outages"] = [
                    {"station": "Ground", "start_sec": 2.0, "end_sec": 4.0}
                ]
            if family == "infeasible":
                for product in spec["products"]:
                    product.update(ready_sec=0.0, deadline_sec=1.0)
            cases.append(
                {
                    "id": family + "-meridian-" + str(int(meridian)),
                    "family": family,
                    "progenitor": "public-central-calendar/v1",
                    "spec": spec,
                }
            )
    return cases


def _execute_native(job):
    config, method = job
    row = {
        "case": f"{config['profile']}-meridian-{config['longitude_shift_deg']}",
        "configuration": config,
        "policy": method,
    }
    run = None
    try:
        case = heterogeneous_mission(config)
        run = PlanningRun(case["scenario"], case["task"], policy_config(method))
        run.run()
        native_return = sum(r["transition"]["reward"] for r in run.session._transitions)
        expectation = (
            run.metrics()["fulfilled"]
            + witness(run)["fulfilled_priority"]
            - 0.1 * run.metrics()["rejections"]
        )
        if not math.isclose(native_return, expectation, abs_tol=1e-9):
            raise RuntimeError(
                "native SLA reward does not match completed value and rejection costs"
            )
        row["native_return"] = native_return
        row.update(
            status="completed",
            record=run.checkpoint(compact=True),
            metrics=run.metrics(),
            witness=witness(run),
        )
    except Exception as exc:  # noqa: BLE001 -- keep every predeclared attempt
        row.update(status="error", error_type=type(exc).__name__, error=str(exc)[:1000])
        if run is not None:
            row["record"] = run.checkpoint(compact=True)
    return row


def _replay_native(job):
    root, row = job
    if row["status"] == "error":
        return False
    if row["status"] != "completed":
        raise ValueError("unknown native study status")
    path = (Path(root) / row["artifact"]).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("native artifact escapes study root")
    actual = _verify_file(path)
    case = heterogeneous_mission(row["configuration"])
    task = copy.deepcopy(case["task"])
    task["policy_origin"] = "scripted"
    expected = {
        "record_sha256": row["record_sha256"],
        "scenario_sha256": digest(case["scenario"]),
        "task_sha256": digest(task),
        "policy": policy_config(row["policy"]),
        "operator_enabled": False,
        "metrics": {
            k: v
            for k, v in row["metrics"].items()
            if k not in ("process_peak_rss_bytes", "decision_wall_sec")
        },
        "witness": row["witness"],
    }
    reward = (
        actual["metrics"]["fulfilled"]
        + actual["witness"]["fulfilled_priority"]
        - 0.1 * actual["metrics"]["rejections"]
    )
    if not math.isclose(row["native_return"], reward, abs_tol=1e-9):
        raise ValueError("SLA reward witness mismatch")
    if actual != expected:
        raise ValueError("heterogeneous native outcome/recipe mismatch")
    return True


def _native_jobs():
    # Two fixed representative geometries; construction tests cover all six.
    return [
        (
            {"profile": profile, "longitude_shift_deg": shift, "catalogue_seed": 0},
            method,
        )
        for profile, shift in [("balanced", -12), ("bulk_tradeoff", 12)]
        for method in ("priority", "route_aware_rolling")
    ]


def study_manifest():
    return {
        "schema": "space-env.tradeoff-manifest/1",
        "calendars": calendar_suite(),
        "native_jobs": _native_jobs(),
        "scope": "calendar optimum and EO are distinct scopes; no complete EO optimality, LLM, OOD or training claim",
    }


def run_study(output):
    root = Path(output)
    if root.exists():
        raise ValueError("study output must be new")
    root.mkdir(parents=True)
    before = sources()
    manifest = study_manifest()
    cases, jobs = manifest["calendars"], manifest["native_jobs"]
    atomic_write(root / "manifest.json", canonical_json(seal(manifest)))
    calendars = [
        {"case": case["id"], "result": solve_priority_calendar(case["spec"])}
        for case in cases
    ]
    atomic_write(root / "calendars.json", canonical_json(calendars))
    rows = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        for index, value in enumerate(pool.map(_execute_native, jobs)):
            record = value.pop("record", None)
            if record is not None:
                name = f"native-{index}.json"
                atomic_write(root / name, canonical_json(record))
                value.update(artifact=name, record_sha256=record["sha256"])
            rows.append(value)
            atomic_write(
                root / "progress.json",
                canonical_json(
                    {"completed": len(rows), "expected": len(jobs), "rows": rows}
                ),
            )
    if before != sources():
        raise ValueError("source changed during tradeoff study")
    report = seal(
        {
            "schema": "space-env.tradeoff-study/1",
            "manifest_sha256": digest(manifest),
            "sources": before,
            "calendar_results_sha256": digest(calendars),
            "native_rows": rows,
            "calendar_scope_only": True,
            "full_mission_optimality": False,
            "llm_measured": False,
            "scope": manifest["scope"],
        }
    )
    atomic_write(root / "report.json", canonical_json(report))
    return report


def audit(directory):
    root = Path(directory)
    report = json.loads((root / "report.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    if (
        report != seal({k: v for k, v in report.items() if k != "sha256"})
        or report["sources"] != sources()
    ):
        raise ValueError("tradeoff source/seal mismatch")
    if (
        manifest != seal(body)
        or canonical_json(body) != canonical_json(study_manifest())
        or digest(body) != report["manifest_sha256"]
    ):
        raise ValueError("tradeoff manifest mismatch")
    if (
        report["schema"] != "space-env.tradeoff-study/1"
        or report["full_mission_optimality"] is not False
        or report["llm_measured"] is not False
    ):
        raise ValueError("unsupported tradeoff claim")
    expected = [
        {"case": case["id"], "result": solve_priority_calendar(case["spec"])}
        for case in body["calendars"]
    ]
    if (
        expected != json.loads((root / "calendars.json").read_text())
        or digest(expected) != report["calendar_results_sha256"]
    ):
        raise ValueError("calendar certificate/native replay mismatch")
    if [
        (r["configuration"], r["policy"]) for r in report["native_rows"]
    ] != _native_jobs():
        raise ValueError("native study matrix mismatch")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        valid = list(
            pool.map(
                _replay_native,
                [(str(root.resolve()), row) for row in report["native_rows"]],
            )
        )
    return {
        "valid": True,
        "native_replayed": sum(valid),
        "native_errors_retained": len(valid) - sum(valid),
        "calendar_certificates_recomputed": len(expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
    else:
        result = run_study(args.output)
        print(
            canonical_json(
                {
                    "native_rows": len(result["native_rows"]),
                    "report": str(args.output / "report.json"),
                }
            )
        )


if __name__ == "__main__":
    main()
