"""Offline synthetic acquisition exports for calibration contracts, NOT telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build():
    cases = []
    for index, (model, unit_x, unit_y, truth, scale, sigma, max_error) in enumerate(
        [
            ("pv_efficiency", "W", "W", 0.27, 100.0, 0.05, 0.3),
            ("inverse_inertia", "N*m*s", "rad/s", 0.4, 1.0, 0.0005, 0.003),
            ("inverse_heat_capacity", "J", "K", 0.02, 100.0, 0.003, 0.018),
        ]
    ):
        rng = random.Random(1701 + index)
        rows = []
        for split, values in [
            ("train", [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]),
            ("holdout", [0.15, 0.35, 0.8]),
        ]:
            for i, value in enumerate(values):
                x = value * scale
                rows.append(
                    {
                        "id": f"{split}-{i}",
                        "experiment_id": f"independent-{split}-{i}",
                        "split": split,
                        "x": x,
                        "y": truth * x + rng.gauss(0.0, sigma),
                        "sigma_y": sigma,
                    }
                )
        conditions = "synthetic independent aggregate experiments; known net excitation, zero intercept; constant reference conditions"
        experiment = {
            "pv_efficiency": ("constant-effective-efficiency", "scalar", None),
            "inverse_inertia": ("isolated-principal-axis-impulse", "BODY", "z"),
            "inverse_heat_capacity": ("isolated-known-net-heat", "scalar", None),
        }[model]
        data = {
            "schema": "space-env.measurement-rows/1",
            "component_id": "synthetic-" + model,
            "model": model,
            "x_unit": unit_x,
            "y_unit": unit_y,
            "conditions": conditions,
            "experiment_model": experiment[0],
            "frame": experiment[1],
            "axis": experiment[2],
            "observations": rows,
        }
        spec = {
            "schema_version": "space-env.scalar-calibration/1",
            "id": "synthetic-" + model,
            "component_id": data["component_id"],
            "model": model,
            "x_unit": unit_x,
            "y_unit": unit_y,
            "conditions": conditions,
            "experiment_model": experiment[0],
            "frame": experiment[1],
            "axis": experiment[2],
            "input_error_assumption": "negligible-independent-x",
            "noise_assumption": "independent-known-gaussian-sigma-y",
            "source": {
                "id": "synthetic-acquisition-" + model,
                "kind": "synthetic_fixture",
                "uri": "synthetic://space-env/calibration/" + model,
                "content_sha256": digest(data),
                "content_scope": "space-env.measurement-rows/1 canonical JSON",
                "locator": "offline seeded normal perturbations; not independent physical measurements",
                "retrieved_on": "2026-11-03",
                "redistribution": "project synthetic data",
            },
            "observations": rows,
            "gates": {
                "parameter_low": truth * 0.5,
                "parameter_high": truth * 1.5,
                "max_train_standardized_rmse": 2.0,
                "max_holdout_standardized_rmse": 3.0,
                "max_holdout_absolute_error": max_error,
            },
        }
        cases.append(
            {"id": model, "truth": truth, "expected_passed": True, "spec": spec}
        )
    body = {
        "schema": "space-env.calibration-workflow-fixtures/1",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "synthetic workflow data only; no laboratory/flight calibration evidence",
        "cases": cases,
    }
    body["sha256"] = digest(body)
    return body


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, allow_nan=False) + "\n")
