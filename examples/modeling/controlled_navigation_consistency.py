"""Fixed 64-seed native impulse/latency consistency study; synthetic, not calibration."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from space_env.llm.episode import seal, sources
from space_env.navigation.controlled_suite import compile_suite
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks import TaskSession


def run(mode="impulse"):
    before = sources()
    if mode == "finite-j2":
        from space_env.navigation.finite_suite import compile_suite as finite_cases

        original = next(c for c in finite_cases() if c["id"] == "finite-j2")
    elif mode == "impulse":
        original = next(c for c in compile_suite() if c["id"] == "powered-impulse")
    else:
        raise ValueError("unsupported consistency mode")
    p = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
    rows = []
    for seed in range(64):
        try:
            c = copy.deepcopy(original)
            c["task"].pop("goal")
            c["task"]["roles"] = {"a": {"actors": ["a"]}}
            c["task"]["navigation_sensors"].pop("b")
            c["task"]["model_requirements"] = [
                {"feature": f, "actors": ["a"]}
                for f in ["navigation_measurements", "state_estimation"]
            ]
            initial = c["scenario"]["definition"]["orbits"]["a"]["initial"]
            truth = np.array([*initial["r_m"], *initial["v_m_s"]])
            rng = np.random.default_rng(74000 + seed)
            c["task"]["navigation_sensors"]["a"].update(
                initial_estimate_si=(
                    truth + rng.multivariate_normal(np.zeros(6), p)
                ).tolist(),
                initial_covariance_si=p.tolist(),
                seed=seed,
            )
            s = TaskSession(c["scenario"], c["task"])
            result = s.step(
                {"a": ["burn" if mode == "finite-j2" else "kick"]},
                expected_revision=0,
                idempotency_key="known-control",
                advance_sec=5.0,
            )
            if not result["feedback"]["accepted"]:
                raise ValueError("known impulse rejected")
            diagnostic = s._navigation.diagnostics("a")
            history = s._navigation.measurement_history("a")
            energy = s._runtime._backend.report()["energy_ledger"]["a"]
            rows.append(
                {
                    "seed": seed,
                    "status": "completed",
                    "diagnostics": diagnostic,
                    "first_nis": history[0]["filter"]["nis"],
                    "sensor_samples_received": len(history),
                    "pending_samples": len(s._navigation._filters["a"]["pending"]),
                    "energy_consumed_j": energy["consumed_wh"] * 3600,
                    "energy_balance_error_wh": energy["balance_error_wh"],
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
    done = [r for r in rows if r["status"] == "completed"]
    n = 64
    nees_bounds = [float(chi2.ppf(q, 6 * n) / n) for q in [0.005, 0.995]]
    nis_bounds = [float(chi2.ppf(q, 3 * n) / n) for q in [0.005, 0.995]]
    mean_nees = (
        float(np.mean([r["diagnostics"]["nees"] for r in done])) if done else None
    )
    mean_nis = float(np.mean([r["first_nis"] for r in done])) if done else None
    gates = {
        "planned_runs": n,
        "completed_runs": len(done),
        "alpha_per_metric": 0.01,
        "mean_nees": mean_nees,
        "nees_bounds": nees_bounds,
        "mean_first_nis": mean_nis,
        "first_nis_bounds": nis_bounds,
        "passed": len(done) == n
        and nees_bounds[0] <= mean_nees <= nees_bounds[1]
        and nis_bounds[0] <= mean_nis <= nis_bounds[1]
        and all(
            abs(r["energy_consumed_j"] - 65.0) < 1e-8
            and abs(r["energy_balance_error_wh"]) < 1e-12
            for r in done
        ),
    }
    if sources() != before:
        raise ValueError("model sources changed during consistency study")
    return seal(
        {
            "schema": "space-env.controlled-navigation-consistency/1",
            "mode": mode,
            "model_sources": before,
            "rows": rows,
            "statistical_gates": gates,
            "scope": "64 fixed synthetic independent prior-error/sensor-noise seeds, one orbit, 5sec, latency0.5s, sensor3W; impulse mode has known 0.5m/s delta-v, finite-j2 mode has nominal 0.5N/1s finite thrust with matched J2. Shared force inputs/functions, no independent physics or hardware calibration, no learned model.",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--mode", choices=["impulse", "finite-j2"], default="impulse")
    args = parser.parse_args()
    if bool(args.output) == bool(args.audit):
        parser.error("choose output or audit")
    if args.audit:
        old = json.loads(args.audit.read_text())
        if old.get("model_sources") != sources():
            raise ValueError("controlled consistency source drift")
        if old != run(old.get("mode", "impulse")):
            raise ValueError("controlled consistency regeneration mismatch")
        print(
            canonical_json(
                {
                    "valid": True,
                    "regenerated_runs": 64,
                    "gates_passed": old["statistical_gates"]["passed"],
                }
            )
        )
    else:
        report = run(args.mode)
        atomic_write(args.output, canonical_json(report))
        print(canonical_json(report["statistical_gates"]))


if __name__ == "__main__":
    main()
