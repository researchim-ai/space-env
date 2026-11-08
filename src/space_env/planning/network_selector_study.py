"""Explicit local comparison of legacy/new selectors on a new frozen holdout."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from space_env.api.network_workspace import collect_network_forecast
from space_env.planning import network_benchmark as benchmark
from space_env.planning.network_demand_benchmark import demand_suite
from space_env.planning.network_selector import fit_regret_selector
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.network_evidence import seal, source_manifest, validate_row

RECIPES = {
    "learned-0": ("legacy", 0),
    "learned-1": ("regret_softmax", 0),
    "learned-2": ("regret_softmax", 1),
    "learned-3": ("regret_softmax", 2),
}
POLICIES = (*benchmark.STRATEGIES, *RECIPES)


def selector_suite():
    """Literal geometries fixed before outcome collection; historical suites unchanged."""
    old = demand_suite()
    cases = copy.deepcopy(old["cases"])
    replacement = {
        "demand-south": (
            "selector-inclined-33",
            33.0,
            -48.0,
            7.06e6,
            (1.0, -174.0),
            (4.0, -157.0),
        ),
        "demand-retro": (
            "selector-polar-87",
            87.0,
            48.0,
            7.10e6,
            (1.0, -78.0),
            (9.0, -61.0),
        ),
    }
    for case in cases:
        if case["split"] != "test":
            continue
        group, inc, raan, radius, target, site = replacement[case["group"]]
        case["id"] = case["id"].replace(case["group"], group)
        case["group"] = group
        mission = case["configuration"]["mission"]
        mission["name"] = case["id"]
        plane = mission["planes"][0]
        plane.update(inclination_deg=inc, raan_deg=raan, radius_m=radius)
        first = mission["requests"][0]
        delta = (target[0] - first["lat_deg"], target[1] - first["lon_deg"])
        for request in mission["requests"]:
            request["lat_deg"] += delta[0]
            request["lon_deg"] += delta[1]
        mission["stations"][0].update(lat_deg=site[0], lon_deg=site[1])
        case["geometry_sha256"] = benchmark.geometry_key(case["configuration"])
    historical = {c["geometry_sha256"] for c in old["cases"]}
    assert not historical & {
        c["geometry_sha256"] for c in cases if c["split"] == "test"
    }
    return benchmark.NetworkSuite.model_validate(
        {"schema": old["schema"], "cases": cases}
    ).model_dump(mode="json", by_alias=True)


def fit_models(suite, views, rows):
    return {
        policy: (benchmark.fit_selector if method == "legacy" else fit_regret_selector)(
            suite, views, rows, seed=seed
        )
        for policy, (method, seed) in RECIPES.items()
    }


def write(directory, name, value):
    atomic_write(directory / name, canonical_json(seal(value)) + "\n")


def read(directory, name):
    value = json.loads((directory / name).read_text())
    body = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != digest(body):
        raise ValueError("Invalid artifact checksum: " + name)
    return body


def audit(directory):
    directory = Path(directory)
    suite = read(directory, "suite.json")
    report = read(directory, "report.json")
    if (
        suite != selector_suite()
        or report["suite_sha256"] != digest(suite)
        or report["recipes"] != {k: list(v) for k, v in RECIPES.items()}
    ):
        raise ValueError("Study recipe or suite mismatch")
    spec = benchmark.NetworkSuite.model_validate(suite)
    views = {
        c.id: json.loads((directory / (c.id + "--forecast.json")).read_text())
        for c in spec.cases
    }
    rows = report["rows"]
    for c in spec.cases:
        benchmark._validate_view(views[c.id])
        for policy in POLICIES:
            matches = [r for r in rows if r["case"] == c.id and r["policy"] == policy]
            if len(matches) != 1:
                raise ValueError("Incomplete/duplicate policy matrix")
            row = matches[0]
            stored = read(directory, c.id + "--" + policy + ".json")
            if (
                stored != {k: v for k, v in row.items() if k != "policy"}
                or row["forecast_sha256"] != views[c.id]["sha256"]
            ):
                raise ValueError("Entry/forecast mismatch")
            validate_row(
                row["result"],
                benchmark.workload(c, row["strategy"], views[c.id]),
                expected_source_sha256=digest(report["model_sources"]),
            )
    train_rows = [
        {k: v for k, v in r.items() if k != "policy"}
        for r in rows
        if r["split"] == "train" and r["policy"] in benchmark.STRATEGIES
    ]
    train_views = {c.id: views[c.id] for c in spec.cases if c.split == "train"}
    # Native row source admission uses current source; historical audits require
    # the exact source tree which trained the model, not silent relabelling.
    if report["model_sources"] != source_manifest():
        raise ValueError("Reproducing training requires the recorded source revision")
    models = fit_models(suite, train_views, train_rows)
    for policy, model in models.items():
        if read(directory, policy + "--model.json") != {
            k: v for k, v in model.items() if k != "sha256"
        }:
            raise ValueError("Training reproduction mismatch")
        for row in rows:
            if row["policy"] == policy and row["strategy"] != benchmark.choose_strategy(
                model, views[row["case"]]
            ):
                raise ValueError("Policy choice mismatch")
    summary = benchmark.summarize(suite, rows, POLICIES)
    if any(report[k] != v for k, v in summary.items()):
        raise ValueError("Summary mismatch")
    return {
        "valid": True,
        "native_rows": len(rows),
        "models_reproduced": len(models),
        "superiority_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
        return 0
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    suite = selector_suite()
    write(output, "suite.json", suite)
    spec = benchmark.NetworkSuite.model_validate(suite)
    # Inputs frozen before execution; no held-out outcomes used to fit models.
    views = {c.id: collect_network_forecast(c.configuration) for c in spec.cases}
    for case, view in views.items():
        atomic_write(output / (case + "--forecast.json"), canonical_json(view) + "\n")
    rows = []

    def evaluate(case, policy, strategy):
        entry = benchmark.evaluate(
            case, strategy, views[case.id], output / (case.id + "--" + policy + ".json")
        )
        rows.append({**entry, "policy": policy})
        print(case.id, policy, entry["result"]["status"], flush=True)

    for c in spec.cases:
        if c.split == "train":
            for strategy in benchmark.STRATEGIES:
                evaluate(c, strategy, strategy)
    models = fit_models(
        suite,
        {c.id: views[c.id] for c in spec.cases if c.split == "train"},
        [{k: v for k, v in r.items() if k != "policy"} for r in rows],
    )
    for policy, model in models.items():
        atomic_write(output / (policy + "--model.json"), canonical_json(model) + "\n")
    for c in spec.cases:
        if c.split != "train":
            for strategy in benchmark.STRATEGIES:
                evaluate(c, strategy, strategy)
        for policy, model in models.items():
            evaluate(c, policy, benchmark.choose_strategy(model, views[c.id]))
    write(
        output,
        "report.json",
        {
            "schema": "space-env.network-selector-study/1",
            "suite_sha256": digest(suite),
            "recipes": {k: list(v) for k, v in RECIPES.items()},
            "rows": rows,
            "model_sources": source_manifest(),
            **benchmark.summarize(suite, rows, POLICIES),
            "scope": "84 fresh native runs; fixed three-rule contextual choices; two new held-out geometry groups; descriptive only, no industrial or statistical superiority",
        },
    )
    return 0 if all(r["result"]["status"] == "completed" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
