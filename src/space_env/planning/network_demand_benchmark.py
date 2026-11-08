"""Separate frozen demand/resource workloads and a bounded optimization reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from space_env.api.network_workspace import (
    NetworkConfiguration,
    collect_network_forecast,
)
from space_env.planning.network_benchmark import (
    STRATEGIES,
    NetworkSuite,
    geometry_key,
    request_order,
)
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.network_evidence import (
    run_scale_case,
    seal,
    source_manifest,
    validate_row,
)

METHODS = (*STRATEGIES, "exact_small")


def demand_suite():
    """Literal new sites/orbits; no forecast/outcome calls or success filtering."""
    # split, group, inclination, RAAN, first target, station, radius
    groups = (
        ("train", "demand-equatorial", 5.0, -4.0, (0.0, -130.0), (1.0, -113.0), 7e6),
        ("train", "demand-inclined", 28.0, 8.0, (1.0, -118.0), (3.0, -101.0), 7.02e6),
        (
            "development",
            "demand-mid",
            42.0,
            -18.0,
            (1.0, -144.0),
            (5.0, -127.0),
            7.08e6,
        ),
        ("development", "demand-high", 68.0, 22.0, (1.0, -104.0), (8.0, -87.0), 7.12e6),
        ("test", "demand-south", 18.0, -34.0, (1.0, -160.0), (3.0, -143.0), 7.04e6),
        ("test", "demand-retro", 104.0, 35.0, (1.0, -91.0), (9.0, -85.0), 7.16e6),
    )
    cases = []
    for index, (split, group, inclination, raan, target, station, radius) in enumerate(
        groups
    ):
        for family in ("bulk-burst", "staggered-cpu"):
            burst = family == "bulk-burst"
            orders = (
                (
                    (11, 0.0, 120.0, 3200000, 2800000, 1.0, False),
                    (6, 0.0, 120.0, 300000, 150000, 1.0, False),
                    (6, 0.0, 120.0, 300000, 150000, 1.0, False),
                )
                if burst
                else (
                    (9, 0.0, 150.0, 2400000, 2000000, 18.0, False),
                    (5, 0.0, 130.0, 300000, 150000, 1.0, False),
                    (7, 25.0, 185.0, 600000, 350000, 3.0, True),
                    (12, 80.0, 235.0, 900000, 600000, 25.0, True),
                    (4, 105.0, 240.0, 250000, 100000, 2.0, True),
                )
            )
            requests = [
                {
                    "id": f"request-{i}",
                    "priority": priority,
                    "release_sec": release,
                    "deadline_sec": deadline,
                    "raw_bits": raw,
                    "output_bits": final,
                    "processing_sec": cpu,
                    "lat_deg": target[0] + (4.0 if later else 0.0),
                    "lon_deg": target[1] + (18.0 if later else 0.0),
                }
                for i, (
                    priority,
                    release,
                    deadline,
                    raw,
                    final,
                    cpu,
                    later,
                ) in enumerate(orders)
            ]
            config = NetworkConfiguration.model_validate(
                {
                    "mission": {
                        "name": group + " / " + family,
                        "horizon_sec": 240.0,
                        "decision_sec": 5.0,
                        "memory_mbit": 8.0 if burst or index % 2 else 4.0,
                        "planes": [
                            {
                                "id": "fleet",
                                "inclination_deg": inclination,
                                "raan_deg": raan,
                                "radius_m": radius,
                                "phases_deg": [0.0, 1.2, 2.4],
                            }
                        ],
                        "stations": [
                            {
                                "name": "Ground",
                                "lat_deg": station[0],
                                "lon_deg": station[1],
                                "terminals": 1 if burst else 1 + index % 2,
                            }
                        ],
                        "requests": requests,
                        "outages": [
                            {"station": "Ground", "start_sec": 0.0, "end_sec": 100.0},
                            {"station": "Ground", "start_sec": 121.0, "end_sec": 240.0},
                        ]
                        if burst
                        else [],
                    },
                    "links": [],
                    "max_hops": 0,
                    "max_search_nodes": 50000,
                }
            ).model_dump(mode="json", by_alias=True)
            cases.append(
                {
                    "id": group + "-" + family,
                    "group": group,
                    "split": split,
                    "condition": "station_outage" if burst else "nominal",
                    "configuration": config,
                    "geometry_sha256": geometry_key(config),
                }
            )
    return NetworkSuite.model_validate(
        {"schema": "space-env.network-benchmark-suite/1", "cases": cases}
    ).model_dump(mode="json", by_alias=True)


def workload(case, method, view):
    if method not in METHODS:
        raise ValueError("Unknown demand reference method")
    work = {
        "id": case.id + "--" + method,
        "configuration": case.configuration.model_dump(mode="json", by_alias=True),
        "recover_after": None,
        "planning_method": "exact_small" if method == "exact_small" else "serial",
    }
    if method != "exact_small":
        work["request_order"] = request_order(view, method)
    return work


def summarize(suite, rows):
    spec = NetworkSuite.model_validate(suite)
    expected = {(c.id, method) for c in spec.cases for method in METHODS}
    if (
        len(rows) != len(expected)
        or {(r["case"], r["method"]) for r in rows} != expected
    ):
        raise ValueError("Full demand comparison denominator required")
    paired = []
    for case in spec.cases:
        selected = [r for r in rows if r["case"] == case.id]
        if len({r["forecast_sha256"] for r in selected}) != 1 or any(
            r["group"] != case.group
            or r["split"] != case.split
            or type(r["result"]["total"]) is not int
            or r["result"]["total"] != len(case.configuration.mission.requests)
            for r in selected
        ):
            raise ValueError("Unequal information/group/denominator")
        oracle = next(r["result"] for r in selected if r["method"] == "exact_small")
        cert = oracle.get("optimizer") if oracle["status"] == "completed" else None
        bound = cert["upper_bound_score"] // (oracle["total"] + 1) if cert else None
        for row in selected:
            result = row["result"]
            paired.append(
                {
                    "case": case.id,
                    "method": row["method"],
                    "forecast_value_upper_bound": bound,
                    "forecast_optimum_proved": cert["optimal"] if cert else False,
                    "native_weighted_fulfilled": result.get("weighted_fulfilled")
                    if result["status"] == "completed"
                    else None,
                    "native_gap_to_forecast_bound": bound - result["weighted_fulfilled"]
                    if bound is not None and result["status"] == "completed"
                    else None,
                }
            )
    aggregates = []
    for split in ("train", "development", "test"):
        for method in METHODS:
            selected = [
                r["result"]
                for r in rows
                if r["split"] == split and r["method"] == method
            ]
            scores = [
                r["weighted_fulfilled"] / max(1, r["priority_total"])
                if r["status"] == "completed"
                else None
                for r in selected
            ]
            aggregates.append(
                {
                    "split": split,
                    "method": method,
                    "rows": len(selected),
                    "completed_rows": sum(x is not None for x in scores),
                    "mean_value_lower_bound": sum(
                        x if x is not None else 0 for x in scores
                    )
                    / len(scores),
                    "mean_value_upper_bound": sum(
                        x if x is not None else 1 for x in scores
                    )
                    / len(scores),
                }
            )
    return {
        "aggregates": aggregates,
        "paired": paired,
        "scope": "Native utility separate from a bounded discrete conservative forecast optimum; no continuous/native optimality or superiority certificate",
    }


def audit(directory):
    from space_env.planning.network_optimal import optimize_network

    root = Path(directory)

    def read(name):
        value = json.loads((root / name).read_text())
        if value.get("sha256") != digest(
            {k: v for k, v in value.items() if k != "sha256"}
        ):
            raise ValueError("Demand artifact checksum mismatch")
        return value

    report, saved = read("report.json"), read("suite.json")
    suite = {k: v for k, v in saved.items() if k != "sha256"}
    spec = NetworkSuite.model_validate(suite)
    summary = summarize(suite, report["rows"])
    if (
        report["suite_sha256"] != digest(suite)
        or report["model_sources"] != source_manifest()
        or report["harness_sha256"]
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or any(report[k] != v for k, v in summary.items())
    ):
        raise ValueError("Demand source/input/statistics mismatch")
    for case in spec.cases:
        view = read(case.id + "--forecast.json")
        for row in (r for r in report["rows"] if r["case"] == case.id):
            if (
                read(case.id + "--" + row["method"] + ".json") != seal(row)
                or row["forecast_sha256"] != view["sha256"]
            ):
                raise ValueError("Demand row/forecast mismatch")
            if row["result"]["status"] == "completed":
                validate_row(row["result"], workload(case, row["method"], view))
        exact = next(
            r["result"]
            for r in report["rows"]
            if r["case"] == case.id and r["method"] == "exact_small"
        )
        if exact["status"] == "completed":
            replay = optimize_network(
                view,
                max_nodes=case.configuration.max_search_nodes,
                max_hops=case.configuration.max_hops,
            )
            if replay["sha256"] != exact["schedule_sha256"]:
                raise ValueError("Forecast optimization not reproduced")
    return seal(
        {
            "consistent": True,
            "rows": len(report["rows"]),
            "report_sha256": report["sha256"],
            "scope": "Artifact checks and forecast solver reproduction, not native replay or independent physical calibration",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument(
        "--suite-only",
        action="store_true",
        help="Freeze inputs without executing missions",
    )
    args = parser.parse_args()
    if args.audit:
        if args.output or args.suite_only:
            parser.error("--audit is exclusive")
        print(canonical_json(audit(args.audit)))
        return 0
    if not args.output:
        parser.error("--output new-directory required")
    args.output.mkdir(parents=True, exist_ok=False)
    suite = demand_suite()
    atomic_write(args.output / "suite.json", canonical_json(seal(suite)) + "\n")
    if args.suite_only:
        return 0
    rows = []
    spec = NetworkSuite.model_validate(suite)
    for case in spec.cases:
        view = collect_network_forecast(case.configuration)
        atomic_write(
            args.output / (case.id + "--forecast.json"), canonical_json(view) + "\n"
        )
        for method in METHODS:
            result = run_scale_case(workload(case, method, view), timeout_sec=120)
            if (
                result["status"] == "completed"
                and result["forecast_sha256"] != view["sha256"]
            ):
                raise ValueError("Native planner received a different forecast")
            row = {
                "case": case.id,
                "group": case.group,
                "split": case.split,
                "method": method,
                "forecast_sha256": view["sha256"],
                "result": result,
            }
            rows.append(row)
            atomic_write(
                args.output / (case.id + "--" + method + ".json"),
                canonical_json(seal(row)) + "\n",
            )
            print(
                case.id,
                method,
                result["status"],
                result.get("weighted_fulfilled"),
                flush=True,
            )
    report = seal(
        {
            "schema": "space-env.network-demand-reference/1",
            "suite_sha256": digest(suite),
            "rows": rows,
            "model_sources": source_manifest(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **summarize(suite, rows),
        }
    )
    atomic_write(args.output / "report.json", canonical_json(report) + "\n")
    return 0 if all(r["result"]["status"] == "completed" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
