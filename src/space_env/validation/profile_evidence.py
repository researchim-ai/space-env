"""Analytical verification evidence, explicitly separate from hardware calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from space_env.core.constants import MU_EARTH, SIGMA_SB
from space_env.core.propulsion import mass_after_thrust
from space_env.core.thermal import step_temperature_c
from space_env.reference import (
    EOPSeries,
    Epoch,
    ForceModel,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.simulation.snapshot import digest


def profile_contracts():
    return {
        "schema": "space-env.physics-contracts/1",
        "profiles": [
            {
                "id": "joint-network-workspace",
                "models": [
                    "joint acquisition owner and directed custody route selection",
                    "kinematic pointing and native EO/CPU lifecycle",
                    "sampled synthetic RF and native DTN/ground ACK",
                    "conservative memory, energy and terminal reservations",
                    "optional bounded global discrete-chain packing reference",
                ],
                "limits": [
                    "4 actors, 16 requests, 4 stations, 3 custody hops, 1200 s",
                    "default serial packing; optional exact-small <=6 requests/48 cells",
                    "exact-small optimum is conditional on its discrete conservative model, not native continuous dynamics",
                    "native-prefix recovery; cancelled partial ISL fragments may be discarded",
                    "recovery retains native owner/destination; no ISL topology mutation",
                    "zero-loss synthetic RF sampled at 0.5 s",
                    "no hardware calibration or image science",
                ],
                "evidence": [
                    "native capture/CPU/two-hop custody/ground ACK test",
                    "resource competition, search budget and role privacy tests",
                ],
            },
            {
                "id": "workspace-eo",
                "models": [
                    "J2 orbit",
                    "kinematic pointing",
                    "metadata EO",
                    "CPU work",
                    "ideal station radio",
                    "battery and memory",
                ],
                "limits": [
                    "4 actors",
                    "64 requests",
                    "3600 s",
                    "no image science",
                    "no calibrated RF",
                ],
                "evidence": [
                    "native lifecycle and conservation tests",
                    "analytical kernels apply only to named mechanisms",
                ],
            },
            {
                "id": "workspace-dynamic",
                "models": [
                    "causal native events",
                    "partial CPU and TX recovery",
                    "station outages",
                    "subsystem availability",
                ],
                "limits": [
                    "declared existing targets",
                    "fixed physical owners",
                    "decision-grid event boundary",
                    "fault flag is not degradation physics",
                ],
                "evidence": [
                    "source/commit physical prefix comparison",
                    "native residual inventory and progress",
                ],
            },
            {
                "id": "relay-reference",
                "models": [
                    "EO and CPU",
                    "ISL DTN storage/forward",
                    "ground ACK",
                    "shared memory and transmitters",
                ],
                "limits": [
                    "bundled synthetic relay scenario",
                    "idealized radio/link assumptions",
                    "not arbitrary optimized constellation routing",
                ],
                "evidence": ["native relay and ACK tests", "role round atomicity"],
            },
            {
                "id": "rigid-reference",
                "models": [
                    "rigid body",
                    "reaction wheels",
                    "RCS",
                    "coupled bus and thermal profiles",
                ],
                "limits": [
                    "opt-in per scenario",
                    "conditional integration/cadence controls",
                    "not enabled by workspace EO compiler",
                ],
                "evidence": [
                    "existing fixed-clock and refinement studies",
                    "independent force comparison is separate",
                ],
            },
        ],
        "hardware_calibrated": False,
        "open_gaps": [
            "measured mission/component telemetry",
            "independent end-to-end orbit/ADCS/sensor/RF truth",
            "uncertainty identification from measurements",
            "large constellation mission qualification",
        ],
    }


def _kepler(a, eccentricity, times):
    # Closed Kepler ellipse, not another numerical propagation of the ODE.
    n = math.sqrt(MU_EARTH / a**3)
    mean = n * times
    anomaly = mean.copy()
    for _ in range(20):
        correction = (anomaly - eccentricity * np.sin(anomaly) - mean) / (
            1 - eccentricity * np.cos(anomaly)
        )
        anomaly -= correction
        if np.max(np.abs(correction)) < 1e-14:
            break
    else:
        raise ValueError("Independent Kepler solve did not converge")
    denominator = 1 - eccentricity * np.cos(anomaly)
    x = a * (np.cos(anomaly) - eccentricity)
    y = a * math.sqrt(1 - eccentricity**2) * np.sin(anomaly)
    vx = -a * n * np.sin(anomaly) / denominator
    vy = a * n * math.sqrt(1 - eccentricity**2) * np.cos(anomaly) / denominator
    return np.stack((x, y, times * 0, vx, vy, times * 0), axis=1)


def run_evidence():
    rows = []
    context = FrameContext(EOPSeries.packaged())
    epoch = Epoch.utc("2026-07-18T12:00:00Z")
    for eccentricity in (0.0, 0.05, 0.1):
        a = 8e6
        period = 2 * math.pi * math.sqrt(a**3 / MU_EARTH)
        initial = _kepler(a, eccentricity, np.array([0.0]))[0]
        for max_step in (60.0, 15.0):
            actual = propagate(
                ReferenceState(epoch, initial[:3], initial[3:]),
                period,
                context,
                model=ForceModel(),
                sample_sec=60.0,
                max_step_sec=max_step,
            )
            states = np.array(actual["states_si"])
            expected = _kepler(a, eccentricity, np.array(actual["times_sec"]))
            position = float(
                np.max(np.linalg.norm(states[:, :3] - expected[:, :3], axis=1))
            )
            velocity = float(
                np.max(np.linalg.norm(states[:, 3:] - expected[:, 3:], axis=1))
            )
            energy = np.sum(states[:, 3:] ** 2, axis=1) / 2 - MU_EARTH / np.linalg.norm(
                states[:, :3], axis=1
            )
            drift = float(np.max(np.abs(energy - energy[0])) / abs(energy[0]))
            rows.append(
                {
                    "id": f"kepler/e={eccentricity}/step={max_step}",
                    "level": "independent_analytical_solution_shared_mu",
                    "position_error_m": position,
                    "velocity_error_m_s": velocity,
                    "relative_energy_drift": drift,
                    "limits": {
                        "position_error_m": 0.02,
                        "velocity_error_m_s": 2e-5,
                        "relative_energy_drift": 1e-9,
                    },
                    "passed": position < 0.02 and velocity < 2e-5 and drift < 1e-9,
                    "scope": "isolated Newtonian two-body orbit; J2/drag/SRP/ADCS excluded",
                }
            )
    for duration in (10.0, 1000.0):
        mass, dv = mass_after_thrust(100.0, duration, 2.0, 200.0, 99.8)
        exhaust = 200.0 * 9.80665
        expected_mass = max(99.8, 100.0 - duration * 2.0 / exhaust)
        # Independent integral series of 1/m, converges for this small fuel ratio.
        fraction = (100.0 - expected_mass) / 100.0
        expected_dv = exhaust * math.fsum(fraction**k / k for k in range(1, 20))
        rows.append(
            {
                "id": f"rocket/{duration}",
                "level": "analytical_mass_flow_and_independent_log_series",
                "mass_error_kg": abs(mass - expected_mass),
                "delta_v_error_m_s": abs(dv - expected_dv),
                "passed": abs(mass - expected_mass) < 1e-12
                and abs(dv - expected_dv) < 1e-9,
                "scope": "constant thrust/Isp, isolated mass flow, dry-mass clipping; no actuator calibration",
            }
        )
    temperature = 20.0
    for _ in range(100):
        temperature = step_temperature_c(
            temperature,
            0.1,
            q_sun_w=10.0,
            q_diss_w=3.0,
            q_heater_w=2.0,
            radiating_area_m2=0.0,
            emissivity=0.8,
            heat_capacity_j_k=100.0,
        )
    equilibrium_power = 0.8 * SIGMA_SB * ((20.0 + 273.15) ** 4 - 3.0**4)
    equilibrium = step_temperature_c(
        20.0,
        1.0,
        q_sun_w=equilibrium_power,
        q_diss_w=0.0,
        q_heater_w=0.0,
        radiating_area_m2=1.0,
        emissivity=0.8,
        heat_capacity_j_k=100.0,
    )
    rows.append(
        {
            "id": "thermal/isolated-balance",
            "level": "analytical_energy_balance",
            "adiabatic_error_k": abs(temperature - 21.5),
            "equilibrium_error_k": abs(equilibrium - 20.0),
            "passed": abs(temperature - 21.5) < 1e-10
            and abs(equilibrium - 20.0) < 1e-10,
            "scope": "lumped thermal kernel; no component fit or mission thermal uncertainty claim",
        }
    )
    body = {
        "schema": "space-env.profile-evidence/1",
        "contracts": profile_contracts(),
        "cases": rows,
        "passed": all(r["passed"] for r in rows),
        "hardware_calibrated": False,
        "scope": "analytical numerical verification; not a physical error bound for a real mission or full-profile qualification",
        "reproducibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "orbit_numerics": {
                k: actual["manifest"][k]
                for k in (
                    "scipy_version",
                    "integrator",
                    "rtol",
                    "position_atol_m",
                    "velocity_atol_m_s",
                    "sample_sec",
                )
            },
            "sources_sha256": {
                name: hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / name).read_bytes()
                ).hexdigest()
                for name in (
                    "core/constants.py",
                    "core/propulsion.py",
                    "core/thermal.py",
                    "reference/orbit.py",
                    "reference/frames.py",
                    "validation/profile_evidence.py",
                )
            },
        },
    }
    return body | {"sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new file")
    result = run_evidence()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{sum(r['passed'] for r in result['cases'])}/{len(result['cases'])} analytical cases passed; hardware_calibrated=False"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
