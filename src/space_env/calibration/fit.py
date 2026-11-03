"""Known-noise, zero-intercept scalar fits with untouched experimental holdout."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from space_env.calibration.schema import MODELS, CalibrationSpec
from space_env.simulation.snapshot import digest


def measurement_payload(value):
    """Canonical prepared export; includes predeclared train/holdout assignments."""
    spec = CalibrationSpec.model_validate(value)
    return {
        "schema": "space-env.measurement-rows/1",
        "component_id": spec.component_id,
        "model": spec.model,
        "x_unit": spec.x_unit,
        "y_unit": spec.y_unit,
        "conditions": spec.conditions,
        "experiment_model": spec.experiment_model,
        "frame": spec.frame,
        "axis": spec.axis,
        "observations": [r.model_dump(mode="json") for r in spec.observations],
    }


def calibrate(value):
    spec = CalibrationSpec.model_validate(value)
    raw = measurement_payload(spec)
    if (
        spec.source.content_scope != "space-env.measurement-rows/1 canonical JSON"
        or spec.source.content_sha256 != digest(raw)
    ):
        raise ValueError(
            "measurement content digest/scope does not match canonical acquisition export"
        )
    train = [r for r in spec.observations if r.split == "train"]
    holdout = [r for r in spec.observations if r.split == "holdout"]
    x = np.array([r.x for r in train])
    y = np.array([r.y for r in train])
    sigma = np.array([r.sigma_y for r in train])
    scale = float(np.max(np.abs(x)))
    if scale == 0:
        raise ValueError("unidentifiable parameter: all training excitation is zero")
    weighted_x = (x / scale) / sigma
    information = float(weighted_x @ weighted_x)
    estimate = float((weighted_x @ (y / sigma)) / information / scale)
    standard_error = float(1 / math.sqrt(information) / scale)
    if (
        not all(math.isfinite(v) for v in (information, estimate, standard_error))
        or standard_error <= 0
    ):
        raise ValueError("ill-conditioned calibration information")
    z = NormalDist().inv_cdf(0.975)
    interval = [estimate - z * standard_error, estimate + z * standard_error]
    low, high = float(min(x)), float(max(x))

    def residuals(rows):
        return [
            {
                "id": r.id,
                "experiment_id": r.experiment_id,
                "x": r.x,
                "observed": r.y,
                "predicted": estimate * r.x,
                "error": estimate * r.x - r.y,
                "standardized_error": (estimate * r.x - r.y) / r.sigma_y,
                "within_training_input_range": low <= r.x <= high,
            }
            for r in rows
        ]

    training, testing = residuals(train), residuals(holdout)

    def rmse(rows):
        return math.sqrt(sum(r["standardized_error"] ** 2 for r in rows) / len(rows))

    gates = {
        "parameter_in_bounds": spec.gates.parameter_low
        <= estimate
        <= spec.gates.parameter_high,
        "conditional_interval_in_bounds": spec.gates.parameter_low
        <= interval[0]
        <= interval[1]
        <= spec.gates.parameter_high,
        "training_residual": rmse(training) <= spec.gates.max_train_standardized_rmse,
        "holdout_residual": rmse(testing) <= spec.gates.max_holdout_standardized_rmse,
        "holdout_absolute_error": max(abs(r["error"]) for r in testing)
        <= spec.gates.max_holdout_absolute_error,
        "holdout_within_training_domain": all(
            r["within_training_input_range"] for r in testing
        ),
    }
    passed = all(gates.values())
    synthetic = spec.source.kind == "synthetic_fixture"
    result = {
        "schema": "space-env.scalar-fit/1",
        "id": spec.id,
        "component_id": spec.component_id,
        "model": spec.model,
        "passed": passed,
        "parameter": {
            "estimate": estimate,
            "standard_error": standard_error,
            "unit": MODELS[spec.model]["parameter_unit"],
            "conditional_normal_95_interval": interval,
            "meaning": MODELS[spec.model]["meaning"],
        },
        "gates": gates,
        "predeclared_gates": spec.gates.model_dump(mode="json"),
        "train": training,
        "holdout": testing,
        "training_input_domain": [low, high],
        "train_standardized_rmse": rmse(training),
        "holdout_standardized_rmse": rmse(testing),
        "source": spec.source.model_dump(mode="json"),
        "conditions": spec.conditions,
        "experiment_model": spec.experiment_model,
        "frame": spec.frame,
        "axis": spec.axis,
        "spec_sha256": digest(spec.model_dump(mode="json")),
        "data_sha256": digest(raw),
        "fit_status": (
            "synthetic_workflow_verified"
            if synthetic
            else "declared_measurements_holdout_checked"
        )
        if passed
        else "rejected_by_gates",
        "industrial_readiness": False,
        "independent_source_authentication": False,
        "limitations": [
            "conditional known independent Gaussian y-noise; negligible x errors and correct zero-intercept model assumed",
            "no simultaneous actuator-gain/inertia or net-heat/capacity identification",
            "declared experiment IDs and source digest do not authenticate a measurement provider",
            "no automatic promotion of native parameter_status to calibrated; no extrapolation qualification",
        ],
    }
    result["algorithm_sources_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("fit.py", "schema.py")
    }
    result["train_degrees_of_freedom"] = len(train) - 1
    if spec.model != "pv_efficiency" and interval[0] > 0:
        result["derived_reciprocal"] = {
            "estimate": 1 / estimate,
            "conditional_95_interval": [1 / interval[1], 1 / interval[0]],
            "unit": "kg*m^2" if spec.model == "inverse_inertia" else "J/K",
            "scope": "monotone reciprocal transform; not a Gaussian interval in reciprocal space or automatic native calibration",
        }
    result["sha256"] = digest(result)
    return result
