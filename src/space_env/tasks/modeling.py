"""Versioned actor-level modeling requirements; configuration is not qualification.

Trainer/control-plane metadata only. Never substitute this inventory for agent
observations, physical goal evaluation or empirical comparison with other engines.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from space_env.runtime.schema import StrictModel

Feature = Literal[
    "dynamic_orbit",
    "covariance_prediction",
    "spacecraft_bus",
    "rigid_attitude",
    "body_thrust",
    "rotating_body_thrust",
    "momentum_unloading",
    "thermal_network",
    "navigation_measurements",
    "state_estimation",
    "matched_force_estimation",
    "finite_burn_estimation",
    "continuous_relative_safety",
    "docking_contact",
]

# One public registry for requirement parsing, descriptions and availability rows.
_REGISTRY = {
    "dynamic_orbit": (
        "implemented",
        "Managed native GCRS orbit arcs; declared forces and bounds only.",
    ),
    "covariance_prediction": (
        "implemented",
        "STM and covariance prediction; no measurements/filter updates.",
    ),
    "spacecraft_bus": (
        "implemented",
        "Coupled native bus; hardware source status is caller provenance.",
    ),
    "rigid_attitude": (
        "implemented",
        "Rigid body and three orthogonal ideal reaction wheels.",
    ),
    "body_thrust": (
        "implemented",
        "Paid native engine model; explicit chemical/electric profile.",
    ),
    "rotating_body_thrust": (
        "implemented",
        "Off-center rotating thrust; bounded small mass loss, constant inertia.",
    ),
    "momentum_unloading": (
        "implemented",
        "Configured native unloading; not arbitrary RCS dynamics.",
    ),
    "thermal_network": (
        "implemented",
        "Lumped thermal network, not CAD/FEA or calibrated flight hardware.",
    ),
    "navigation_measurements": (
        "implemented",
        "Opt-in declared-force sensor tasks with known inertial controls and continuous native power; ideal known beacon if present.",
    ),
    "state_estimation": (
        "implemented",
        "Opt-in declared-force EKF with nominal inertial impulses/finite thrust; explicit matched_native perturbations, no bias states or execution-error estimation.",
    ),
    "matched_force_estimation": (
        "implemented",
        "Explicit matched_native force profile on estimated state; shared native force inputs/functions, not independent validation.",
    ),
    "finite_burn_estimation": (
        "implemented",
        "Declared inertial finite thrust, deterministic nominal mass and known cancellation; no body-thrust or unknown execution errors.",
    ),
    "continuous_relative_safety": (
        "not_implemented",
        "Relative keepout is sampled; legacy encounter kernels are separate.",
    ),
    "docking_contact": (
        "not_implemented",
        "No native docking/contact dynamics in this task contract.",
    ),
}


class ModelRequirement(StrictModel):
    feature: Feature
    actors: tuple[str, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_actors(self):
        if len(set(self.actors)) != len(self.actors) or any(
            not a.strip() or len(a) > 128 for a in self.actors
        ):
            raise ValueError("model requirement actors must be unique and nonblank")
        return self


def model_registry():
    """Return a defensive copy; implemented never means independently qualified."""
    return copy.deepcopy(
        {
            "schema": "space-env.model-registry/1",
            "features": {
                k: {"implementation": v[0], "scope": v[1]} for k, v in _REGISTRY.items()
            },
            "qualification": "not assessed by this registry",
        }
    )


def model_profile(runtime, *, navigation=None):
    """Inspect instantiated native model objects, not advertised global flags.

    Includes all actors, so belongs only to trusted task setup. No current state,
    future trajectory, fuel, faults, physical success or uncertainty is exposed.
    """
    backend = runtime._backend
    actors = runtime.describe_capabilities()["actors"]
    config = getattr(backend, "config", None)
    orbits = getattr(config, "orbits", {})
    bus = getattr(backend, "bus", None)
    rigid = getattr(getattr(bus, "rigid", None), "models", {})
    bus_models = getattr(config, "spacecraft_bus", None) or {}
    engines = getattr(getattr(config, "actuation", None), "engines", {})
    unloading = getattr(getattr(config, "unloading", None), "actors", {})
    rows = {}
    for actor in actors:
        enabled = {
            "matched_force_estimation": navigation is not None
            and actor in navigation._specs
            and navigation._specs[actor].force_profile == "matched_native",
            "finite_burn_estimation": navigation is not None
            and actor in navigation._specs
            and navigation._specs[actor].control_profile == "inertial_finite_burn",
            "navigation_measurements": navigation is not None
            and actor in navigation._specs,
            "state_estimation": navigation is not None and actor in navigation._specs,
            "dynamic_orbit": actor in orbits,
            "covariance_prediction": actor in getattr(backend, "navigation_actors", []),
            "spacecraft_bus": actor in bus_models,
            "rigid_attitude": actor in rigid,
            "body_thrust": actor in engines,
            "rotating_body_thrust": actor in engines
            and engines[actor].rigid_coupling is not None,
            "momentum_unloading": actor in unloading,
            "thermal_network": actor in bus_models
            and bus_models[actor].thermal_network is not None,
        }
        rows[actor] = {
            name: ("enabled" if enabled.get(name, False) else "not_configured")
            if state == "implemented"
            else "not_implemented"
            for name, (state, _) in _REGISTRY.items()
        }
    return {
        "schema": "space-env.model-profile/1",
        "backend": runtime.describe_capabilities()["backend"],
        "actors": rows,
        "qualification": "configuration availability only; no fidelity or superiority claim",
        "access": "trainer configuration metadata, never an actor observation",
    }


def validate_requirements(requirements, runtime, *, navigation=None):
    if not requirements:
        return
    profile = model_profile(runtime, navigation=navigation)
    errors = []
    for req in requirements:
        for actor in req.actors:
            state = profile["actors"].get(actor, {}).get(req.feature, "unknown_actor")
            if state != "enabled":
                errors.append({"actor": actor, "feature": req.feature, "status": state})
    if errors:
        raise ValueError(
            "Unsupported modeling requirements: " + json.dumps(errors, sort_keys=True)
        )


def inspect_task(scenario, task):
    """Instantiate and validate a task without advancing, submitting or forecasting."""
    from space_env.tasks.session import TaskSession

    session = TaskSession(scenario, task)
    return {
        "schema": "space-env.task-model-compatibility/1",
        "valid": True,
        "profile": session.modeling_profile(),
        "requirements": [
            r.model_dump(mode="json") for r in session._task.model_requirements
        ],
        "goal_kinds": sorted({c.kind for c in session._task.goal.conditions})
        if session._task.goal
        else [],
        "time_sec": session._runtime.time_sec,
        "scope": "task construction passed; no simulation, physical qualification or policy success asserted",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.scenario) != bool(args.task):
        parser.error("--scenario and --task are required together")
    value = (
        model_registry()
        if args.scenario is None
        else inspect_task(
            json.loads(args.scenario.read_text()), json.loads(args.task.read_text())
        )
    )
    # Never overwrite input recipes, including aliases.
    if any(
        args.output.resolve() == p.resolve()
        or (args.output.exists() and args.output.samefile(p))
        for p in (args.scenario, args.task)
        if p is not None
    ):
        parser.error("output must not alias an input")
    from space_env.simulation.snapshot import atomic_write, canonical_json

    atomic_write(args.output, canonical_json(value))
    print(json.dumps({"schema": value["schema"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
