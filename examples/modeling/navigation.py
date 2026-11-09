"""Frozen synthetic navigation study; no RL/LLM training or hardware calibration."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from space_env.llm.episode import seal, sources
from space_env.navigation import NavigationSession
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.modeling_suite import compile_suite


def run_study():
    scenario = next(
        c["scenario"] for c in compile_suite()["cases"] if c["id"] == "orbit-nominal"
    )
    initial = scenario["definition"]["orbits"]["a"]["initial"]
    truth = np.array([*initial["r_m"], *initial["v_m_s"]])
    covariance = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])

    def spec(seed, **extra):
        rng = np.random.default_rng(74000 + seed)
        estimate = truth + rng.multivariate_normal(np.zeros(6), covariance)
        return {
            "kind": "cartesian_position",
            "initial_estimate_si": estimate.tolist(),
            "initial_covariance_si": covariance.tolist(),
            "standard_deviation": [3.0, 3.0, 3.0],
            "seed": seed,
            **extra,
        }

    cases = []
    checkpoints = {}
    for name, extra in [
        ("nominal", {}),
        ("delay-drop", {"latency_sec": 0.25, "dropout_probability": 0.4}),
        ("bias-negative", {"true_bias": [100.0, 0.0, 0.0]}),
        (
            "range-rate",
            {
                "kind": "range_range_rate",
                "reference_actor": "b",
                "standard_deviation": [2.0, 0.02],
            },
        ),
    ]:
        sensor = spec(42, **extra)
        session = NavigationSession(scenario, {"a": sensor})
        for i in range(5):
            session.advance(1.0, expected_revision=i, idempotency_key=str(i))
        checkpoint = session.checkpoint()
        restored = NavigationSession.from_checkpoint(checkpoint)
        assert restored.observe("a") == session.observe("a")
        cases.append(
            {
                "id": name,
                "sensor": sensor,
                "observation": session.observe("a"),
                "public_measurements": session.measurement_history("a"),
                "diagnostics": session.diagnostics("a"),
                "replay_verified": True,
            }
        )
        checkpoints[name] = checkpoint
    # Fixed denominator, independent draws from the declared prior. No outcome filtering.
    rows = []
    for seed in range(64):
        try:
            session = NavigationSession(scenario, {"a": spec(seed)})
            session.advance(5.0, expected_revision=0, idempotency_key="ensemble")
            history = session.measurement_history("a")
            rows.append(
                {
                    "seed": seed,
                    "status": "completed",
                    "diagnostics": session.diagnostics("a"),
                    "first_nis": history[0]["filter"]["nis"],
                }
            )
        except Exception as exc:  # noqa: BLE001 -- retain every attempted seed
            rows.append(
                {
                    "seed": seed,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    complete = [r for r in rows if r["status"] == "completed"]
    n = 64
    nees_bounds = [float(chi2.ppf(p, 6 * n) / n) for p in [0.005, 0.995]]
    nis_bounds = [float(chi2.ppf(p, 3 * n) / n) for p in [0.005, 0.995]]
    mean_nees = (
        float(np.mean([r["diagnostics"]["nees"] for r in complete]))
        if complete
        else None
    )
    mean_nis = float(np.mean([r["first_nis"] for r in complete])) if complete else None
    gates = {
        "planned_runs": 64,
        "completed_runs": len(complete),
        "alpha_per_metric": 0.01,
        "nees_mean_bounds": nees_bounds,
        "first_nis_mean_bounds": nis_bounds,
        "mean_nees": mean_nees,
        "mean_first_nis": mean_nis,
        "passed": len(complete) == 64
        and nees_bounds[0] <= mean_nees <= nees_bounds[1]
        and nis_bounds[0] <= mean_nis <= nis_bounds[1],
        "meaning": "Approximate EKF consistency for a fixed short synthetic two-body ensemble; not flight uncertainty calibration or general observability.",
    }
    report = seal(
        {
            "schema": "space-env.navigation-study/1",
            "model_sources": sources(),
            "scenario": scenario,
            "cases": cases,
            "ensemble": rows,
            "statistical_gates": gates,
            "scope": "public synthetic inputs; coordinate fixes or ideal known-beacon range/rate; passive central gravity; no clock/light-time/media or uncertain beacon; no TaskSession/Gym/LLM integration, no model training",
        }
    )
    return report, checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.audit):
        parser.error("choose exactly one of --output or --audit")
    if args.audit:
        recorded = json.loads((args.audit / "report.json").read_text())
        if recorded.get("model_sources") != sources():
            raise ValueError("navigation study source identity mismatch")
        expected, checkpoints = run_study()
        if recorded != expected:
            raise ValueError("navigation study regeneration mismatch")
        for name, checkpoint in checkpoints.items():
            if (
                json.loads(
                    (args.audit / (name + "-trainer-checkpoint.json")).read_text()
                )
                != checkpoint
            ):
                raise ValueError("navigation checkpoint mismatch")
        print(
            canonical_json(
                {
                    "valid": True,
                    "replayed_cases": len(checkpoints),
                    "regenerated_ensemble_runs": 64,
                    "gates_passed": expected["statistical_gates"]["passed"],
                }
            )
        )
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report, checkpoints = run_study()
        atomic_write(args.output / "report.json", canonical_json(report))
        for name, checkpoint in checkpoints.items():
            atomic_write(
                args.output / (name + "-trainer-checkpoint.json"),
                canonical_json(checkpoint),
            )
        print(
            canonical_json(
                {
                    "sha256": report["sha256"],
                    "statistical_gates": report["statistical_gates"],
                }
            )
        )


if __name__ == "__main__":
    main()
