"""Offline external J2-formula fixtures. Never imports production orbit acceleration.

Run explicitly to regenerate comparison data; these share frames/EOP/constants,
so they must not be advertised as an independent operational orbit truth.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
from boinor.core import perturbations
from scipy.integrate import solve_ivp

from space_env.core.constants import J2_EARTH, MU_EARTH, R_EARTH_EQUATOR
from space_env.reference.frames import FrameContext
from space_env.reference.orbit import _ephemerides
from space_env.reference.time import EOPSeries, Epoch
from space_env.simulation.snapshot import digest


def generate(directory: Path):
    context, epoch = (
        FrameContext(EOPSeries.packaged()),
        Epoch.utc("2026-07-18T12:00:00Z"),
    )
    external = getattr(
        perturbations.J2_perturbation, "py_func", perturbations.J2_perturbation
    )
    source = Path(inspect.getfile(perturbations))
    third_body = getattr(perturbations.third_body, "py_func", perturbations.third_body)
    for hours, initial, date, combined in (
        (24, [6.8e6, 0.0, 0.0, 0.0, 6000.0, 5000.0], "2026-07-18T12:00:00Z", False),
        (72, [7e6, 0.0, 0.0, 0.0, 5000.0, 5700.0], "2026-07-18T12:00:00Z", False),
        (24, [6.9e6, 0.0, 0.0, 500.0, 1000.0, 7600.0], "2026-07-19T12:00:00Z", True),
    ):
        epoch = Epoch.utc(date)

        def rhs(clock, state, epoch=epoch, combined=combined):
            rotation = context.matrix(epoch.shifted(float(clock)))
            fixed = np.concatenate((rotation @ state[:3], rotation @ state[3:])) / 1000
            j2 = rotation.T @ (
                external(0.0, fixed, MU_EARTH / 1e9, J2_EARTH, R_EARTH_EQUATOR / 1000)
                * 1000
            )
            central = -MU_EARTH * state[:3] / np.linalg.norm(state[:3]) ** 3
            perturbation = j2
            if combined:
                for body, mu in zip(
                    _ephemerides(epoch.shifted(float(clock))),
                    (1.32712440018e20, 4.9048695e12),
                ):
                    perturbation += (
                        third_body(
                            0.0,
                            state / 1000,
                            MU_EARTH / 1e9,
                            mu / 1e9,
                            lambda _t, body=body: body / 1000,
                        )
                        * 1000
                    )
            return np.concatenate((state[3:], central + perturbation))

        times = np.arange(0.0, hours * 3600 + 1, 3600.0)
        solution = solve_ivp(
            rhs,
            (0.0, hours * 3600),
            initial,
            method="LSODA",
            t_eval=times,
            rtol=1e-12,
            atol=[1e-5] * 3 + [1e-8] * 3,
            max_step=30.0,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        body = {
            "schema": "space-env.external-orbit-fixture/1",
            "provenance": {
                "library": "boinor",
                "version": version("boinor"),
                "source_url": "https://github.com/boinor/boinor",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "license": "MIT",
                "generator_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "independence": "external-force-formula/shared-inputs",
                "shared_inputs": [
                    "ERFA frames",
                    "EOP",
                    "SI constants",
                    "initial state",
                ],
                "integrator": f"SciPy {scipy.__version__} LSODA; rtol=1e-12; position_atol=1e-5m; velocity_atol=1e-8m/s; max_step=30s",
                "scope": "frozen external J2 and optional Sun/Moon third-body formulas; different integration family, no independent operational truth",
            },
            "inputs": {
                "epoch_utc": epoch.iso(),
                "frame": "GCRS",
                "state_si": initial,
                "eop": context.eop.manifest(),
                "forces": {"j2": True, "third_body": combined},
                "duration_sec": hours * 3600.0,
                "sample_sec": 3600.0,
                "constants": {
                    "mu_m3_s2": MU_EARTH,
                    "j2": J2_EARTH,
                    "radius_m": R_EARTH_EQUATOR,
                },
            },
            "expected": {
                "times_sec": times.tolist(),
                "states_si": solution.y.T.tolist(),
            },
        }
        if combined:
            body["provenance"]["shared_inputs"].append(
                "ERFA Sun/Moon positions via shared _ephemerides provider"
            )
            body["inputs"]["constants"].update(
                mu_sun_m3_s2=1.32712440018e20, mu_moon_m3_s2=4.9048695e12
            )
        suffix = "_third_body_other_epoch" if combined else ""
        path = directory / f"boinor_j2_{hours}h{suffix}.json"
        path.write_text(json.dumps({**body, "sha256": digest(body)}, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "src/space_env/reference/fixtures")
