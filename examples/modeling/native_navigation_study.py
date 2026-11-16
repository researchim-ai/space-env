"""Matched central-GCRS EKF consistency; not flight sensor calibration.

Independent priors/noise for two actors; paired truth seeds across latency profiles.
All failed and early-terminated attempts stay in the declared denominator.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from space_env.industrial.native_proximity import (
    NativeProximitySession,
    native_proximity_recipe,
)
from space_env.industrial.session import environment_identity
from space_env.llm.episode import sources
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def run(seeds=32):
    if type(seeds) is not int or not 8 <= seeds <= 128:
        raise ValueError("choose 8..128 independent seeds per profile")
    initial_sources = sources()
    rows = []
    for profile in ("nominal", "latency"):
        attempts = []
        for seed in range(seeds):
            cfg = native_proximity_recipe(profile)
            cfg["scenario"]["definition"]["horizon_sec"] = 16.0
            cfg["required_observations"] = 128
            rng = np.random.default_rng(100000 + seed)
            for index, actor in enumerate(("a", "b")):
                sensor = cfg["sensors"][actor]
                mean = np.array(sensor["initial_estimate_si"])
                truth = rng.multivariate_normal(
                    mean, np.array(sensor["initial_covariance_si"])
                )
                cfg["scenario"]["definition"]["orbits"][actor]["initial"].update(
                    r_m=truth[:3].tolist(), v_m_s=truth[3:].tolist()
                )
                sensor["seed"] = 2 * seed + index
            try:
                session = NativeProximitySession(cfg)
                while not session._done:
                    session.step(
                        {"kind": "coast"},
                        expected_revision=session.revision,
                        idempotency_key=str(session.revision),
                    )
                obs = session.observe()
                true = {
                    a: session._runtime._backend.mission.ephemerides[a].state(
                        session._runtime.time_sec
                    )
                    for a in ("a", "b")
                }
                errors = {
                    a: true[a] - np.array(obs["navigation"][a]["estimate_si"])
                    for a in ("a", "b")
                }
                covariance = {
                    a: np.array(obs["navigation"][a]["covariance_si"])
                    for a in ("a", "b")
                }
                relative = errors["a"] - errors["b"]
                nis = []
                for actor in ("a", "b"):
                    history = session._navigation.measurement_history(actor)
                    latest = next(
                        (
                            p["filter"]["nis"]
                            for p in reversed(history)
                            if "filter" in p
                        ),
                        None,
                    )
                    nis.append(latest)
                attempts.append(
                    {
                        "seed": seed,
                        "status": "completed",
                        "time_sec": obs["time_sec"],
                        "safe_intervals": session.metrics()["safe_intervals"],
                        "relative_nees": float(
                            relative
                            @ np.linalg.solve(
                                covariance["a"] + covariance["b"], relative
                            )
                        ),
                        "actor_nees": [
                            float(errors[a] @ np.linalg.solve(covariance[a], errors[a]))
                            for a in ("a", "b")
                        ],
                        "actor_nis": nis,
                    }
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                attempts.append(
                    {"seed": seed, "status": "error", "error_type": type(exc).__name__}
                )
        complete = [
            a
            for a in attempts
            if a["status"] == "completed"
            and a["time_sec"] == 16.0
            and a["safe_intervals"]
            and all(n is not None for n in a["actor_nis"])
        ]
        row = {
            "profile": profile,
            "expected_attempts": seeds,
            "qualified_attempts": len(complete),
            "attempts": attempts,
        }
        if complete:
            for metric, values, dimension in [
                ("relative_nees", [a["relative_nees"] for a in complete], 6),
                ("actor_nees", [n for a in complete for n in a["actor_nees"]], 6),
                ("actor_nis", [n for a in complete for n in a["actor_nis"]], 3),
            ]:
                count = len(values)
                mean = float(np.mean(values))
                interval = [
                    float(chi2.ppf(q, count * dimension) / count)
                    for q in (0.025, 0.975)
                ]
                row[metric] = {
                    "mean": mean,
                    "mean_95_percent_interval": interval,
                    "inside_interval": interval[0] <= mean <= interval[1],
                }
        rows.append(row)
    if initial_sources != sources():
        raise ValueError("source changed during study")
    body = {
        "schema": "space-env.native-navigation-consistency/1",
        "sources": initial_sources,
        "environment": environment_identity(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds_per_profile": seeds,
        "rows": rows,
        "scope": "Matched nonlinear central-force EKF with synthetic Cartesian sensors, known ideal controls and independently sampled priors. Same truth seeds across profiles are paired; no OOD, physical sensor calibration or unknown control errors. Individual 95% tests can fail by chance.",
        "flight_calibrated": False,
    }
    return {**body, "sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(args.seeds)
    atomic_write(args.output, canonical_json(report))
    print(canonical_json({"profiles": len(report["rows"]), "attempts": 2 * args.seeds}))


if __name__ == "__main__":
    main()
