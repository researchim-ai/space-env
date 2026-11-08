"""Frozen geometry-group benchmark and a train-only contextual dispatcher.

The learned action selects one of three request orderings. All strategies use
the same forecast, search limits and native runtime. This is not command-level RL.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.api.network_workspace import (
    NetworkConfiguration,
    collect_network_forecast,
)
from space_env.planning.joint_network import JointNetworkView
from space_env.reference import Epoch
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.network_evidence import (
    run_scale_case,
    seal,
    source_manifest,
    validate_row,
)

STRATEGIES = ("priority", "earliest_deadline", "value_per_bit")
FEATURES = (
    "bias",
    "priority_spread",
    "mean_slack",
    "mean_output_mbit",
    "contact_fraction",
    "capture_fraction",
)


def geometry_key(configuration):
    c = NetworkConfiguration.model_validate(configuration).model_dump(
        mode="json", by_alias=True
    )["mission"]

    def angle(value):
        return round(value % 360.0, 12) % 360.0

    def site(value):
        result = {k: v for k, v in value.items() if k not in {"name", "terminals"}}
        result["lon_deg"] = (
            0.0
            if abs(result["lat_deg"]) == 90
            else (result["lon_deg"] + 180.0) % 360.0 - 180.0
        )
        return result

    orbits = []
    for p in c["planes"]:
        raan = angle(p["raan_deg"])
        for phase in p["phases_deg"]:
            # Equatorial circular states have a redundant RAAN/phase pair.
            shift = (
                raan
                if p["inclination_deg"] == 0
                else -raan
                if p["inclination_deg"] == 180
                else 0.0
            )
            orbits.append(
                {
                    "radius_m": p["radius_m"],
                    "inclination_deg": p["inclination_deg"],
                    "raan_deg": 0.0 if p["inclination_deg"] in (0.0, 180.0) else raan,
                    "phase_deg": angle(phase + shift),
                }
            )
    targets = {
        (
            r["lat_deg"],
            0.0 if abs(r["lat_deg"]) == 90 else (r["lon_deg"] + 180.0) % 360.0 - 180.0,
        )
        for r in c["requests"]
    }
    return digest(
        {
            "epoch": Epoch.utc(c["epoch_utc"]).iso(),
            "orbits": sorted(orbits, key=canonical_json),
            "stations": sorted((site(s) for s in c["stations"]), key=canonical_json),
            "targets": sorted(targets),
        }
    )


class NetworkCase(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    group: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    split: Literal["train", "development", "test"]
    condition: Literal["nominal", "station_outage"]
    geometry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    configuration: NetworkConfiguration


class NetworkSuite(StrictModel):
    schema_id: Literal["space-env.network-benchmark-suite/1"] = Field(alias="schema")
    cases: tuple[NetworkCase, ...] = Field(min_length=6, max_length=24)

    @model_validator(mode="after")
    def split_guard(self):
        ids, groups, geometries, definitions = set(), {}, {}, set()
        for c in self.cases:
            config = c.configuration.model_dump(mode="json", by_alias=True)
            if bool(c.configuration.mission.outages) != (
                c.condition == "station_outage"
            ):
                raise ValueError("Condition does not match declared outage inputs")
            if c.id in ids or c.geometry_sha256 != geometry_key(config):
                raise ValueError("Duplicate case or mismatched geometry")
            ids.add(c.id)
            for key, table in ((c.group, groups), (c.geometry_sha256, geometries)):
                if key in table and table[key] != c.split:
                    raise ValueError("Geometry group crosses ML split")
                table[key] = c.split
            config["mission"].pop("name")
            config["mission"].pop("seed")
            identity = digest(config)
            if identity in definitions:
                raise ValueError("Duplicate physical input")
            definitions.add(identity)
        if any(
            sum(s == split for s in groups.values()) < 2
            for split in ("train", "development", "test")
        ):
            raise ValueError("At least two geometry groups per split required")
        return self


def network_suite():
    """Sites/targets are literal WGS84 inputs; no orbit/forecast/outcome fitting."""
    # split, group, inclination, RAAN, phases, target(lat,lon), station(lat,lon), radius
    geometries = (
        (
            "train",
            "equatorial",
            0.0,
            0.0,
            [0.0, 15.0, 30.0],
            (0.0, -128.0),
            (0.0, -66.0),
            7e6,
        ),
        (
            "train",
            "inclined-20",
            20.0,
            5.0,
            [2.0, 16.0, 31.0],
            (2.0, -121.0),
            (10.0, -60.0),
            7.05e6,
        ),
        (
            "development",
            "inclined-35",
            35.0,
            -10.0,
            [5.0, 20.0, 35.0],
            (3.0, -133.0),
            (15.0, -70.0),
            7.1e6,
        ),
        (
            "development",
            "inclined-75",
            75.0,
            -30.0,
            [0.0, 15.0, 30.0],
            (0.0, -154.0),
            (29.0, -139.0),
            7.02e6,
        ),
        (
            "test",
            "inclined-55",
            55.0,
            15.0,
            [0.0, 12.0, 24.0],
            (2.0, -111.0),
            (18.0, -74.0),
            7.08e6,
        ),
        (
            "test",
            "retrograde-98",
            98.0,
            -15.0,
            [0.0, 15.0, 30.0],
            (0.0, -141.0),
            (30.0, -126.0),
            7.15e6,
        ),
    )
    cases = []
    for split, group, inclination, raan, phases, target, station, radius in geometries:
        for condition in ("nominal", "station_outage"):
            cfg = NetworkConfiguration.model_validate(
                {
                    "mission": {
                        "name": group + " / " + condition,
                        "horizon_sec": 180.0,
                        "decision_sec": 5.0,
                        "planes": [
                            {
                                "id": "fleet",
                                "inclination_deg": inclination,
                                "raan_deg": raan,
                                "phases_deg": phases,
                                "radius_m": radius,
                            }
                        ],
                        "stations": [
                            {
                                "name": "Ground",
                                "lat_deg": station[0],
                                "lon_deg": station[1],
                            }
                        ],
                        "requests": [
                            {
                                "id": f"request-{i}",
                                "lat_deg": target[0] + delta,
                                "lon_deg": target[1] + delta * 2.5,
                                "priority": priority,
                                "deadline_sec": deadline,
                                "processing_sec": cpu,
                                "raw_bits": raw,
                                "output_bits": output,
                            }
                            for i, (
                                delta,
                                priority,
                                deadline,
                                cpu,
                                raw,
                                output,
                            ) in enumerate(
                                (
                                    (0.0, 3, 145.0, 1.0, 1000000, 500000),
                                    (0.6, 9, 175.0, 6.0, 1800000, 900000),
                                    (-0.6, 6, 165.0, 2.0, 700000, 350000),
                                )
                            )
                        ],
                        "outages": []
                        if condition == "nominal"
                        else [
                            {"station": "Ground", "start_sec": 100.0, "end_sec": 140.0}
                        ],
                    },
                    "links": [
                        {"source": "fleet-00", "destination": "fleet-01"},
                        {"source": "fleet-01", "destination": "fleet-02"},
                        {
                            "source": "fleet-00",
                            "destination": "fleet-02",
                            "max_range_m": 5000000.0,
                        },
                    ],
                }
            ).model_dump(mode="json", by_alias=True)
            cases.append(
                {
                    "id": group + "-" + condition.replace("_", "-"),
                    "group": group,
                    "split": split,
                    "condition": condition,
                    "geometry_sha256": geometry_key(cfg),
                    "configuration": cfg,
                }
            )
    return NetworkSuite.model_validate(
        {"schema": "space-env.network-benchmark-suite/1", "cases": cases}
    ).model_dump(mode="json", by_alias=True)


def request_order(view, strategy):
    _validate_view(view)
    if strategy not in STRATEGIES:
        raise ValueError("Unknown network strategy")

    def key(r):
        if strategy == "priority":
            return -r["priority"], r["deadline_sec"], r["id"]
        if strategy == "earliest_deadline":
            return r["deadline_sec"], -r["priority"], r["id"]
        return -r["priority"] / r["output_bits"], r["deadline_sec"], r["id"]

    return [r["id"] for r in sorted(view["demands"], key=key)]


def _validate_view(view):
    JointNetworkView.model_validate(view)
    if view["sha256"] != digest({k: v for k, v in view.items() if k != "sha256"}):
        raise ValueError("Invalid forecast seal")


def features(view):
    _validate_view(view)
    demands = view["demands"]
    priorities = [r["priority"] for r in demands]
    return np.array(
        [
            1.0,
            (max(priorities) - min(priorities)) / max(1, max(priorities)),
            np.mean(
                [
                    (r["deadline_sec"] - r["release_sec"]) / view["horizon_sec"]
                    for r in demands
                ]
            ),
            np.mean([r["output_bits"] / 1e6 for r in demands]),
            np.mean([np.mean(e["cells"]) for e in view["edges"]])
            if view["edges"]
            else 0.0,
            np.mean([np.mean(c["cells"]) for c in view["captures"]]),
        ],
        dtype=float,
    )


def _training_identity(spec):
    return digest(
        [
            c.model_dump(mode="json", by_alias=True)
            for c in spec.cases
            if c.split == "train"
        ]
    )


def _reward(row):
    if row.get("status") != "completed":
        return None
    if (
        type(row.get("weighted_fulfilled")) is not int
        or type(row.get("priority_total")) is not int
        or not 0 <= row["weighted_fulfilled"] <= row["priority_total"]
    ):
        raise ValueError("Invalid native reward")
    return row["weighted_fulfilled"] / max(1, row["priority_total"])


def training_arrays(suite, train_views, rows):
    """Admit the exact train-only native payoff matrix for any selector version."""
    spec = NetworkSuite.model_validate(suite)
    train = [c for c in spec.cases if c.split == "train"]
    expected = {(c.id, strategy) for c in train for strategy in STRATEGIES}
    if (
        set(train_views) != {c.id for c in train}
        or len(rows) != len(expected)
        or {(e["case"], e["strategy"]) for e in rows} != expected
    ):
        raise ValueError("Training requires all and only train case/strategy outcomes")
    values = {}
    for e in rows:
        if (
            e["forecast_sha256"] != train_views[e["case"]]["sha256"]
            or _reward(e["result"]) is None
        ):
            raise ValueError("Training failed or forecast information mismatch")
        case = next(c for c in train if c.id == e["case"])
        validate_row(e["result"], workload(case, e["strategy"], train_views[case.id]))
        values[e["case"], e["strategy"]] = _reward(e["result"])
    x = np.stack([features(train_views[c.id]) for c in train])
    payoffs = np.array(
        [[values[c.id, strategy] for strategy in STRATEGIES] for c in train]
    )
    return spec, train, x, payoffs


def fit_selector(suite, train_views, rows, *, seed=0):
    """Original v1 training retained; new methods live in network_selector."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid training seed")
    spec, train, x, payoffs = training_arrays(suite, train_views, rows)
    weights = np.random.default_rng(seed).normal(
        0.0, 0.01, (len(FEATURES), len(STRATEGIES))
    )
    for _ in range(600):
        logits = x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        advantages = payoffs - (probabilities * payoffs).sum(axis=1, keepdims=True)
        weights += 0.2 * x.T @ (probabilities * advantages) / len(train)
    return seal(
        {
            "schema": "space-env.network-selector/1",
            "seed": seed,
            "features": list(FEATURES),
            "strategies": list(STRATEGIES),
            "weights": weights.tolist(),
            "training_input_sha256": _training_identity(spec),
            "training_cases": [c.id for c in train],
            "training_rows_sha256": digest(
                sorted(rows, key=lambda e: (e["case"], e["strategy"]))
            ),
            "iterations": 600,
            "learning_rate": 0.2,
            "scope": "full-information train-only contextual strategy selector; no command-level RL or expert teacher qualification",
        }
    )


def choose_strategy(model, view):
    if (
        model.get("sha256") != digest({k: v for k, v in model.items() if k != "sha256"})
        or model.get("features") != list(FEATURES)
        or model.get("strategies") != list(STRATEGIES)
    ):
        raise ValueError("Invalid selector artifact")
    weights = np.array(model["weights"], dtype=float)
    if (
        weights.shape != (len(FEATURES), len(STRATEGIES))
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Invalid selector weights")
    x = features(view)
    if "transform" in model:
        transform = model["transform"]
        center, scale = (
            np.asarray(transform["center"], dtype=float),
            np.asarray(transform["scale"], dtype=float),
        )
        if (
            transform.get("schema") != "space-env.train-standardization/1"
            or center.shape != x.shape
            or scale.shape != x.shape
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or center[0] != 0
            or scale[0] != 1
            or transform.get("clip") != 8.0
        ):
            raise ValueError("Invalid train-only feature transform")
        x = np.clip((x - center) / scale, -8.0, 8.0)
    return STRATEGIES[int(np.argmax(x @ weights))]


def workload(case, strategy, view):
    config = case.configuration.model_dump(mode="json", by_alias=True)
    if {r["id"] for r in view["demands"]} != {
        r["id"] for r in config["mission"]["requests"]
    }:
        raise ValueError("Forecast request denominator mismatch")
    return {
        "id": case.id + "--" + strategy,
        "configuration": config,
        "recover_after": None,
        "request_order": request_order(view, strategy),
    }


def evaluate(case, strategy, view, output):
    work = workload(case, strategy, view)
    row = run_scale_case(work, timeout_sec=120)
    if row["status"] == "completed" and row.get("forecast_sha256") != view["sha256"]:
        row = seal(
            {
                "status": "error",
                "total": len(work["configuration"]["mission"]["requests"]),
                "native_fulfilled": None,
                "error": "Unequal forecast information",
            }
        )
    entry = {
        "case": case.id,
        "group": case.group,
        "split": case.split,
        "strategy": strategy,
        "forecast_sha256": view["sha256"],
        "result": row,
    }
    atomic_write(output, canonical_json(seal(entry)) + "\n")
    return entry


def summarize(suite, rows, policies):
    spec = NetworkSuite.model_validate(suite)
    if len(set(policies)) != len(policies) or not set(STRATEGIES) <= set(policies):
        raise ValueError("Distinct policies must include every baseline")
    if any(
        p not in STRATEGIES
        and (not re.fullmatch(r"learned-[0-9]{1,10}", p) or int(p[8:]) >= 2**32)
        for p in policies
    ):
        raise ValueError("Invalid learned policy identity")
    expected = {(c.id, p) for c in spec.cases for p in policies}
    if (
        len(rows) != len(expected)
        or {(r["case"], r["policy"]) for r in rows} != expected
    ):
        raise ValueError("Comparison must retain every case/policy row")
    cases = {c.id: c for c in spec.cases}
    if any(
        len({r["forecast_sha256"] for r in rows if r["case"] == c.id}) != 1
        for c in spec.cases
    ):
        raise ValueError("Unequal forecast information")
    for row in rows:
        case = cases[row["case"]]
        if (
            row["split"] != case.split
            or row["group"] != case.group
            or row["result"]["total"] != len(case.configuration.mission.requests)
            or (row["policy"] in STRATEGIES and row["strategy"] != row["policy"])
        ):
            raise ValueError("Comparison split/group/denominator mismatch")
    aggregates = []
    for split in ("train", "development", "test"):
        for policy in policies:
            selected = [
                r for r in rows if r["split"] == split and r["policy"] == policy
            ]
            scores = [_reward(r["result"]) for r in selected]
            aggregates.append(
                {
                    "split": split,
                    "policy": policy,
                    "rows": len(scores),
                    "completed_rows": sum(s is not None for s in scores),
                    "mean_value_lower_bound": sum(
                        s if s is not None else 0 for s in scores
                    )
                    / len(scores),
                    "mean_value_upper_bound": sum(
                        s if s is not None else 1 for s in scores
                    )
                    / len(scores),
                }
            )
    paired = []
    for learned in (p for p in policies if p.startswith("learned-")):
        for baseline in STRATEGIES:
            group_scores = []
            for group in sorted({c.group for c in spec.cases if c.split == "test"}):
                diffs = []
                for c in (
                    c for c in spec.cases if c.group == group and c.split == "test"
                ):
                    a, b = [
                        next(r for r in rows if r["case"] == c.id and r["policy"] == p)
                        for p in (learned, baseline)
                    ]
                    ar, br = _reward(a["result"]), _reward(b["result"])
                    if ar is None or br is None:
                        diffs = []
                        break
                    diffs.append(ar - br)
                if diffs:
                    group_scores.append(float(np.mean(diffs)))
            all_groups = len({c.group for c in spec.cases if c.split == "test"})
            bootstrap = (
                [
                    float(np.mean(sample))
                    for sample in itertools.product(
                        group_scores, repeat=len(group_scores)
                    )
                ]
                if len(group_scores) == all_groups and all_groups <= 6
                else []
            )
            paired.append(
                {
                    "policy": learned,
                    "baseline": baseline,
                    "complete_test_groups": len(group_scores),
                    "expected_test_groups": all_groups,
                    "mean_group_advantage": float(np.mean(group_scores))
                    if len(group_scores) == all_groups
                    else None,
                    "descriptive_group_bootstrap_95": [
                        float(np.quantile(bootstrap, 0.025)),
                        float(np.quantile(bootstrap, 0.975)),
                    ]
                    if bootstrap
                    else None,
                    "superiority_established": False,
                    "scope": "few hand-authored geometry groups; descriptive paired results, not statistical or operational superiority",
                }
            )
    return {
        "aggregates": aggregates,
        "paired": paired,
        "expert_teacher_qualified": False,
    }


def audit_benchmark(directory):
    """Verify complete pairing and reproduce training without new native runs."""
    directory = Path(directory)

    def read(name):
        value = json.loads((directory / name).read_text())
        if value.get("sha256") != digest(
            {k: v for k, v in value.items() if k != "sha256"}
        ):
            raise ValueError("Benchmark artifact checksum mismatch")
        return value

    report, suite = read("report.json"), read("suite.json")
    suite = {k: v for k, v in suite.items() if k != "sha256"}
    spec = NetworkSuite.model_validate(suite)
    if (
        report["suite_sha256"] != digest(suite)
        or report["model_sources"] != source_manifest()
        or report["harness_sha256"]
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("Benchmark input/source binding mismatch")
    summary = summarize(suite, report["rows"], report["policies"])
    expected_seeds = {p[8:] for p in report["policies"] if p.startswith("learned-")}
    if (
        set(report["models"]) & set(report["training_errors"])
        or set(report["models"]) | set(report["training_errors"]) != expected_seeds
    ):
        raise ValueError("Missing/foreign model or training failure")
    views = {c.id: read(c.id + "--forecast.json") for c in spec.cases}
    for view in views.values():
        _validate_view(view)
    for row in report["rows"]:
        stored = read(row["case"] + "--" + row["policy"] + ".json")
        if {k: v for k, v in stored.items() if k != "sha256"} != {
            k: v for k, v in row.items() if k != "policy"
        } or row["forecast_sha256"] != views[row["case"]]["sha256"]:
            raise ValueError("Stored row or information mismatch")
        case = next(c for c in spec.cases if c.id == row["case"])
        if row["result"]["status"] == "completed":
            validate_row(row["result"], workload(case, row["strategy"], views[case.id]))
    if any(report[k] != v for k, v in summary.items()):
        raise ValueError("Comparison statistics mismatch")
    train_rows = [
        {k: v for k, v in r.items() if k != "policy"}
        for r in report["rows"]
        if r["split"] == "train" and r["policy"] in STRATEGIES
    ]
    for seed, model_sha in report["models"].items():
        model = read(f"selector-{seed}.json")
        reproduced = fit_selector(
            suite,
            {c.id: views[c.id] for c in spec.cases if c.split == "train"},
            train_rows,
            seed=int(seed),
        )
        if model != reproduced or model["sha256"] != model_sha:
            raise ValueError("Selector is not reproduced from train-only outcomes")
        for row in report["rows"]:
            if row["policy"] == f"learned-{seed}" and row[
                "strategy"
            ] != choose_strategy(model, views[row["case"]]):
                raise ValueError("Learned strategy inconsistent with frozen model")
    return seal(
        {
            "schema": "space-env.network-learning-audit/1",
            "consistent": True,
            "rows": len(report["rows"]),
            "reproduced_models": len(report["models"]),
            "report_sha256": report["sha256"],
            "scope": "artifact/information pairing and train-only optimizer reproducibility; no native replay, hardware truth or superiority claim",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument(
        "--suite",
        type=Path,
        help="Explicit frozen suite JSON; default v1 stays unchanged",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.audit is not None:
        if args.output is not None or args.suite is not None:
            parser.error("--audit and --output are exclusive")
        print(canonical_json(audit_benchmark(args.audit)))
        return 0
    if args.output is None:
        parser.error("--output new-directory required")
    if (
        len(args.seeds) > 3
        or len(set(args.seeds)) != len(args.seeds)
        or any(not 0 <= s < 2**32 for s in args.seeds)
    ):
        parser.error("one to three distinct seeds in 0..2**32-1")
    args.output.mkdir(parents=True, exist_ok=False)
    suite = json.loads(args.suite.read_text()) if args.suite else network_suite()
    if "sha256" in suite:
        if suite["sha256"] != digest({k: v for k, v in suite.items() if k != "sha256"}):
            raise ValueError("Invalid supplied suite checksum")
        suite = {k: v for k, v in suite.items() if k != "sha256"}
    spec = NetworkSuite.model_validate(suite)
    atomic_write(args.output / "suite.json", canonical_json(seal(suite)) + "\n")
    views, baseline_rows = {}, []
    for c in spec.cases:
        # Frozen input forecast collected before execution, identical per policy.
        views[c.id] = collect_network_forecast(c.configuration)
        atomic_write(
            args.output / (c.id + "--forecast.json"), canonical_json(views[c.id]) + "\n"
        )
    # Only train outcomes are executed before freezing models.
    for c in (c for c in spec.cases if c.split == "train"):
        for strategy in STRATEGIES:
            row = evaluate(
                c,
                strategy,
                views[c.id],
                args.output / (c.id + "--" + strategy + ".json"),
            )
            baseline_rows.append(row)
            print(c.id, strategy, row["result"]["status"], flush=True)
    models, training_errors = {}, {}
    for seed in args.seeds:
        try:
            models[seed] = fit_selector(
                suite,
                {c.id: views[c.id] for c in spec.cases if c.split == "train"},
                baseline_rows,
                seed=seed,
            )
            atomic_write(
                args.output / f"selector-{seed}.json",
                canonical_json(models[seed]) + "\n",
            )
        except ValueError as exc:
            training_errors[seed] = str(exc)
    policies = [*STRATEGIES, *(f"learned-{s}" for s in args.seeds)]
    rows = [{**r, "policy": r["strategy"]} for r in baseline_rows]
    for c in spec.cases:
        if c.split != "train":
            for strategy in STRATEGIES:
                entry = evaluate(
                    c,
                    strategy,
                    views[c.id],
                    args.output / (c.id + "--" + strategy + ".json"),
                )
                rows.append({**entry, "policy": strategy})
        for seed in args.seeds:
            policy = f"learned-{seed}"
            if seed in models:
                strategy = choose_strategy(models[seed], views[c.id])
                entry = evaluate(
                    c,
                    strategy,
                    views[c.id],
                    args.output / (c.id + "--" + policy + ".json"),
                )
            else:
                entry = {
                    "case": c.id,
                    "group": c.group,
                    "split": c.split,
                    "strategy": None,
                    "forecast_sha256": views[c.id]["sha256"],
                    "result": {
                        "status": "error",
                        "total": len(c.configuration.mission.requests),
                        "native_fulfilled": None,
                        "error": "training_unavailable",
                    },
                }
                atomic_write(
                    args.output / (c.id + "--" + policy + ".json"),
                    canonical_json(seal(entry)) + "\n",
                )
            rows.append({**entry, "policy": policy})
        print(c.id, "evaluated", flush=True)
    report = seal(
        {
            "schema": "space-env.network-learning-report/1",
            "suite_sha256": digest(suite),
            "models": {str(k): m["sha256"] for k, m in models.items()},
            "training_errors": training_errors,
            "policies": policies,
            "rows": rows,
            **summarize(suite, rows, policies),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_sources": source_manifest(),
            "scope": "fixed known-forecast initial scheduling; train-only full-information contextual selector; no unexpected online fault learning, command-level RL, industrial calibration or competitor comparison",
        }
    )
    atomic_write(args.output / "report.json", canonical_json(report) + "\n")
    return 0 if all(r["result"]["status"] == "completed" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
