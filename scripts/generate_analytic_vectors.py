"""Offline analytical vectors. Standard library only; never import the simulator.

Run explicitly after reviewing equations/budgets. Verification never regenerates
expected values. These are independently coded equations, not flight truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MU = 3.986004418e14
SIGMA = 5.670374419e-8
C = 299792458.0
K = 1.380649e-23


def digest(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def kepler(p, t):
    """Elliptic Kepler equation by bisection, then explicit perifocal rotation."""
    a, e = p["a_m"], p["e"]
    mean = (p["mean_anomaly_rad"] + math.sqrt(MU / a**3) * t) % math.tau
    low, high = 0.0, math.tau
    for _ in range(100):
        mid = (low + high) / 2
        if mid - e * math.sin(mid) < mean:
            low = mid
        else:
            high = mid
    anomaly = (low + high) / 2
    ca, sa = math.cos(anomaly), math.sin(anomaly)
    factor = math.sqrt(1 - e * e)
    radius = a * (1 - e * ca)
    vscale = math.sqrt(MU * a) / radius
    x, y = a * (ca - e), a * factor * sa
    vx, vy = -vscale * sa, vscale * factor * ca
    co, so = math.cos(p["raan_rad"]), math.sin(p["raan_rad"])
    cw, sw = math.cos(p["arg_periapsis_rad"]), math.sin(p["arg_periapsis_rad"])
    ci, si = math.cos(p["inclination_rad"]), math.sin(p["inclination_rad"])
    u = (co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si)
    v = (-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si)
    return [x * ui + y * vi for ui, vi in zip(u, v)] + [
        vx * ui + vy * vi for ui, vi in zip(u, v)
    ]


def build():
    cases = []
    for name, a, e, inc in (
        ("elliptic-leo", 8e6, 0.1, 0.7),
        ("elliptic-high-e", 14e6, 0.5, 1.7),
    ):
        p = {
            "a_m": a,
            "e": e,
            "inclination_rad": inc,
            "raan_rad": 0.4,
            "arg_periapsis_rad": 0.3,
            "mean_anomaly_rad": 0.2,
            "mu_m3_s2": MU,
            "epoch_utc": "2026-07-18T12:00:00Z",
            "sample_sec": 60.0,
        }
        times = [0.0, 30.0, 60.0, 330.0, 600.0, 3630.0, 86400.0, 259170.0, 259200.0]
        # Initial Cartesian state is an explicit shared input, not obtained from native code.
        p["initial_si"] = kepler(p, 0.0)
        cases.append(
            {
                "id": name,
                "kind": "orbit",
                "inputs": p,
                "times_sec": times,
                "expected": [kepler(p, t) for t in times],
            }
        )
    for power in (0.0, 4.0):
        p = {
            "capacity_j_per_k": 100.0,
            "conductance_w_per_k": 0.5,
            "initial_k": [300.0, 280.0],
            "internal_w": [power, 0.0],
        }
        times = [1.0, 10.0, 60.0, 100.0]

        def temperatures(t, power=power):
            mean = 290 + power * t / 200
            delta = power + (20 - power) * math.exp(-0.01 * t)
            return [mean + delta / 2, mean - delta / 2]

        cases.append(
            {
                "id": f"conduction-{int(power)}w",
                "kind": "conduction",
                "inputs": p,
                "times_sec": times,
                "expected": [temperatures(t) for t in times],
            }
        )
    times = [1.0, 10.0, 60.0, 100.0]
    p = {
        "capacity_j_per_k": 1000.0,
        "initial_k": 350.0,
        "emissivity": 0.8,
        "area_m2": 0.5,
        "sigma_w_m2_k4": SIGMA,
    }
    cases.append(
        {
            "id": "vacuum-cooling",
            "kind": "radiation",
            "inputs": p,
            "times_sec": times,
            "expected": [
                [(350.0**-3 + 3 * 0.8 * SIGMA * 0.5 / 1000 * t) ** (-1 / 3)]
                for t in times
            ],
        }
    )
    for name, w0, motor, external in (
        ("rigid-coast", 0.1, 0.0, 0.0),
        ("rigid-wheel", 0.0, 0.001, 0.0),
        ("rigid-external", 0.0, 0.0, 0.001),
    ):
        p = {
            "inertia_kg_m2": [2.0, 3.0, 4.0],
            "wheel_inertia_kg_m2": 0.01,
            "omega_z_rad_s": w0,
            "motor_z_nm": motor,
            "external_z_nm": external,
            "max_step_sec": 0.01,
        }
        times = [1.0, 10.0, 60.0]
        expected = []
        for t in times:
            rate = w0 + (external - motor) / 4 * t
            angle = w0 * t + (external - motor) / 8 * t * t
            # Quaternion wxyz, angular velocity xyz, wheel momentum xyz.
            expected.append(
                [
                    math.cos(angle / 2),
                    0.0,
                    0.0,
                    math.sin(angle / 2),
                    0.0,
                    0.0,
                    rate,
                    0.0,
                    0.0,
                    motor * t,
                ]
            )
        cases.append(
            {
                "id": name,
                "kind": "rigid",
                "inputs": p,
                "times_sec": times,
                "expected": expected,
            }
        )
    for name, frequency, distance in (
        ("rf-s-band", 2.2e9, 700000.0),
        ("rf-x-band", 8.2e9, 2000000.0),
    ):
        p = {
            "frequency_hz": frequency,
            "range_m": distance,
            "rf_power_w": 10.0,
            "gain_dbi": 20.0,
            "feeder_loss_db": 1.0,
            "gt_db_k": 15.0,
            "polarization_loss_db": 0.5,
            "other_loss_db": 1.0,
            "fade_db": 2.0,
            "information_rate_bps": 1e6,
            "required_eb_n0_db": 4.0,
            "implementation_loss_db": 1.0,
            "c_m_s": C,
            "boltzmann_j_k": K,
        }
        fspl = 20 * math.log10(4 * math.pi * distance * frequency / C)
        eirp = 10 * math.log10(p["rf_power_w"]) + p["gain_dbi"] - p["feeder_loss_db"]
        cn0 = eirp - fspl + 15.0 - 0.5 - 1.0 - 2.0 - 10 * math.log10(K)
        margin = cn0 - 60.0 - 4.0 - 1.0
        cases.append(
            {
                "id": name,
                "kind": "rf",
                "inputs": p,
                "times_sec": [0.0],
                "expected": [[fspl, eirp, cn0, margin]],
            }
        )
    body = {
        "schema": "space-env.analytic-vectors/1",
        "provenance": {
            "generator": "scripts/generate_analytic_vectors.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independence": "independently-coded-analytic-equations/shared-parameters",
            "backend": "Python standard library math; elliptic Kepler bisection, no ODE integrator",
            "shared_inputs": [
                "SI constants",
                "initial states",
                "synthetic model parameters",
                "binary64 arithmetic",
            ],
            "scope": "numerical equation verification only; no external flight or hardware calibration",
        },
        "cases": cases,
    }
    return {**body, "sha256": digest(body)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, allow_nan=False) + "\n")
