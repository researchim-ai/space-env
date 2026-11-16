"""Matched-linear Monte Carlo consistency check, not flight calibration or OOD.

The true initial state is drawn from the declared Gaussian prior. Synthetic
measurement noise is independently seeded. All attempts are retained.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from space_env.industrial.proximity import ProximitySession, ProximitySpec
from space_env.industrial.session import environment_identity
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def run(seeds=64):
    if type(seeds) is not int or not 8 <= seeds <= 512:
        raise ValueError("seed count must be an integer in [8,512]")
    prior = ProximitySpec()
    mean = np.array(prior.initial_estimate_si)
    covariance = np.array(prior.initial_covariance_si)
    profiles = {
        "nominal": {},
        "latency": {"measurement_latency_steps": 2},
        "dropout_latency": {"measurement_latency_steps": 2, "dropout_probability": 0.2},
    }
    manifest = {
        "seeds": list(range(seeds)),
        "profiles": profiles,
        "horizon_steps": 16,
        "sources": sources(),
        "environment": environment_identity(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    rows = []
    for name, extra in profiles.items():
        attempts = []
        for seed in range(seeds):
            truth = np.random.default_rng(100000 + seed).multivariate_normal(
                mean, covariance
            )
            config = {
                "initial_truth_si": truth.tolist(),
                "initial_estimate_si": mean.tolist(),
                "initial_covariance_si": covariance.tolist(),
                "mode": "hold",
                "required_observations": 128,
                "horizon_steps": 16,
                "acceleration_psd_m2_s3": 0.0,
                "seed": seed,
                **extra,
            }
            try:
                session = ProximitySession(config)
                while not session._done:
                    session.step(
                        {"kind": "coast"},
                        expected_revision=session.revision,
                        idempotency_key=str(session.revision),
                    )
                error = session._truth - session._estimate
                attempts.append(
                    {
                        "seed": seed,
                        "status": "completed",
                        "steps": session.revision,
                        "nees": float(error @ np.linalg.solve(session._cov, error)),
                        "nis": session.observe()["last_nis"],
                    }
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                attempts.append(
                    {"seed": seed, "status": "error", "error_type": type(exc).__name__}
                )
        complete = [
            r
            for r in attempts
            if r["status"] == "completed" and r["steps"] == 16 and r["nis"] is not None
        ]
        count = len(complete)
        summary = {
            "qualified_attempts": count,
            "expected_attempts": seeds,
            "attempts": attempts,
        }
        if count:
            for metric, dim in (("nees", 6), ("nis", 3)):
                average = sum(r[metric] for r in complete) / count
                interval = (
                    chi2.ppf(0.025, count * dim) / count,
                    chi2.ppf(0.975, count * dim) / count,
                )
                summary[metric] = {
                    "mean": average,
                    "mean_95_percent_interval": [float(v) for v in interval],
                    "inside_interval": bool(interval[0] <= average <= interval[1]),
                }
        rows.append({"profile": name, **summary})
    if manifest["sources"] != sources():
        raise ValueError("source changed during filter study")
    body = {
        "schema": "space-env.relative-filter-consistency/1",
        "manifest": manifest,
        "rows": rows,
        "scope": "Matched linear CW/Gaussian filter consistency only. Same truth seeds across profiles are paired, not independent cases. 95% tests can fail by chance. No real sensor calibration or nonlinear dynamics validation.",
        "flight_calibrated": False,
    }
    return {**body, "sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.seeds)
    atomic_write(args.output, canonical_json(result))
    print(
        canonical_json(
            {"profiles": len(result["rows"]), "seeds_per_profile": args.seeds}
        )
    )


if __name__ == "__main__":
    main()
