"""Optional native Basilisk runner for the frozen central-gravity exchange.

Requires official bsk==2.12.2 separately; no fallback engine or data download.
N axes are assigned the declared GCRS initial axes. Central gravity is autonomous;
this does not validate Basilisk time/frame transformations or perturbed dynamics.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.validation.trajectory_exchange import (
    compare_external,
    trajectory_manifest,
)


def run(manifest):
    from Basilisk.simulation import gravityEffector, spacecraft, svIntegrators
    from Basilisk.utilities import SimulationBaseClass, macros

    if manifest != trajectory_manifest(manifest["profile"]):
        raise ValueError("input differs from frozen trajectory manifest")
    rows = []
    for case in manifest["cases"]:
        simulation = SimulationBaseClass.SimBaseClass()
        process = simulation.CreateNewProcess("orbit-process")
        process.addTask(simulation.CreateNewTask("orbit-step", macros.sec2nano(30.0)))
        vehicle = spacecraft.Spacecraft()
        vehicle.ModelTag = case["id"]
        vehicle.hub.mHub = 100.0
        vehicle.hub.IHubPntBc_B = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        vehicle.hub.r_CN_NInit = case["initial_state_si"][:3]
        vehicle.hub.v_CN_NInit = case["initial_state_si"][3:]
        earth = gravityEffector.GravBodyData()
        earth.planetName = "earth_planet_data"
        earth.mu = case["mu_m3_s2"]
        earth.isCentralBody = True
        vehicle.gravField.gravBodies = spacecraft.GravBodyVector([earth])
        integrator = svIntegrators.svIntegratorRKF78(vehicle)
        integrator.relTol = 1e-12
        integrator.absTol = 1e-8
        vehicle.setIntegrator(integrator)
        simulation.AddModelToTask("orbit-step", vehicle)
        recorder = vehicle.scStateOutMsg.recorder(macros.sec2nano(300.0))
        simulation.AddModelToTask("orbit-step", recorder)
        simulation.InitializeSimulation()
        simulation.ConfigureStopTime(macros.sec2nano(case["times_sec"][-1]))
        simulation.ExecuteSimulation()
        times = (np.asarray(recorder.times()) / 1e9).tolist()
        if times != case["times_sec"]:
            raise ValueError("Basilisk output time grid mismatch")
        rows.append(
            {
                "case_id": case["id"],
                "case_sha256": digest(case),
                **{k: case[k] for k in ("frame", "epoch_utc", "time_basis", "units")},
                "times_sec": times,
                "states_si": np.column_stack(
                    (recorder.r_BN_N, recorder.v_BN_N)
                ).tolist(),
            }
        )
    return {
        "schema": "space-env.external-trajectories/1",
        "manifest_sha256": digest(manifest),
        "engine": "Basilisk",
        "engine_version": version("bsk"),
        "adapter_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "solver_settings": {
            "integrator": "svIntegratorRKF78",
            "relTol": 1e-12,
            "absTol": 1e-8,
            "task_step_sec": 30.0,
            "output_step_sec": 300.0,
            "axes": "N initialized as GCRS; no coordinate conversion",
            "scope": "autonomous point-mass gravity only; no speed ranking",
        },
        "trajectories": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    frozen = json.loads(args.manifest.read_text())
    manifest = frozen.get("trajectory_manifest", frozen)
    submission = run(manifest)
    comparison = compare_external(manifest, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, canonical_json(submission))
    print(canonical_json(comparison))


if __name__ == "__main__":
    main()
