"""Public mission families and lazy recipes over existing decision executors.

Discovery contains no hidden events, reference plans or privileged configuration.
Families are task groupings, not interchangeable physics or composition promises.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def _entries():
    families = [
        {
            "id": "earth_observation",
            "title": "Earth observation operations",
            "objective": "Fulfil released observation requests and deliver their products to ground before their deadlines, prioritising valuable requests while respecting native execution constraints.",
            "skills": [
                "task_assignment",
                "payload_processing",
                "data_delivery",
                "replanning",
            ],
            "kind": "operations",
            "fidelity": "Native EO/CPU/ISL/ground execution with finite action templates and a central union of current actor observations; synthetic scene conditions.",
            "limitations": [
                "Not a calibrated image or radiance generator",
                "Finite action catalogue; no arbitrary natural-language task compiler",
                "Short synthetic regression missions, not qualified operational scale",
            ],
            "profiles": ["nominal", "new_demand", "station_outage"],
        },
        {
            "id": "communications",
            "title": "Constellation data delivery",
            "objective": "Deliver released data streams to ground on time using native ground links and inter-satellite custody transfers. Preserve onboard products and obey native resource constraints when conditions change.",
            "skills": [
                "routing",
                "terminal_scheduling",
                "resource_management",
                "replanning",
            ],
            "kind": "native_service",
            "fidelity": "One native spacecraft runtime for orbit, attitude, RF budgets, bus resources and DTN acknowledgements. Application bytes are supplied, not newly imaged.",
            "limitations": [
                "Central ideal-local telemetry, not distributed partial-observation control",
                "Sampled link budgets, not RF waveforms",
                "Wrapper stream deadlines limited to 3600 seconds from origin",
            ],
            "profiles": [
                "nominal",
                "outage",
                "receiver_failure",
                "cancel_inflight",
                "pointing",
                "low_memory",
                "prefix",
            ],
        },
        {
            "id": "proximity",
            "title": "Relative navigation and inspection",
            "objective": "Use estimated relative state to complete the published positional inspection or retreat objective, respecting keepout and delta-v limits. Aborting an inspection is not successful inspection.",
            "skills": [
                "estimated_navigation",
                "impulsive_control",
                "inspection",
                "safe_retreat",
            ],
            "kind": "native_proximity",
            "fidelity": "Two nonlinear central-gravity GCRS orbits, synthetic noisy Cartesian-position sensors and EKFs, native impulsive propellant accounting.",
            "limitations": [
                "Positional inspection, not camera recognition or contact dynamics",
                "Central-force coasts only; conditional interval safety with a numerical allowance",
                "Independent filter errors assumed; joint sensor/bus/RF profile not qualified",
            ],
            "profiles": ["nominal", "latency", "blind", "crossing", "retreat"],
        },
        {
            "id": "fault_management",
            "title": "Mission continuity under faults",
            "objective": "Continue fulfilling released observation requests despite faults. Diagnose current symptoms, use bounded recovery attempts, isolate unavailable hardware and preserve useful work on healthy spacecraft.",
            "skills": [
                "diagnosis",
                "bounded_recovery",
                "isolation",
                "mission_continuity",
            ],
            "kind": "operations",
            "fidelity": "Native operational execution plus ideal ground BIT/reset services with charged mission time; hidden scheduled synthetic failures and repairs.",
            "limitations": [
                "Not calibrated hardware fault detection or onboard flight software",
                "Only declared radio/CPU reset interfaces",
                "No guarantee that all requests remain feasible after failure",
            ],
            "profiles": ["resettable", "permanent", "external_repair"],
        },
        {
            "id": "orbit_and_deployment",
            "title": "Bounded constellation phasing",
            "objective": "Reach every published spacecraft orbital-state target at the declared epoch while preserving the fuel reserve. Use currently visible native action templates.",
            "skills": [
                "joint_manoeuvre_scheduling",
                "finite_thrust",
                "fuel_management",
            ],
            "kind": "native_orbits",
            "fidelity": "Central ideal-state TaskSession; nonlinear central-gravity finite-thrust execution and native resource surrogate over 40-100 seconds.",
            "limitations": [
                "Local phasing, not multi-day constellation deployment",
                "No continuous conjunction guarantee",
                "Finite predeclared burn templates, not free-form trajectory optimisation",
                "No coupling to service or contact dynamics",
            ],
            "profiles": [
                "phasing_2",
                "phasing_4",
                "phasing_8",
                "generated_train",
                "generated_development",
                "generated_test",
                "windowed_train",
                "windowed_development",
                "windowed_test",
                "dynamic_recoverable_train",
                "dynamic_recoverable_development",
                "dynamic_recoverable_test",
                "dynamic_critical_train",
                "dynamic_critical_development",
                "dynamic_critical_test",
                "utility_recoverable_train",
                "utility_recoverable_development",
                "utility_recoverable_test",
                "utility_critical_train",
                "utility_critical_development",
                "utility_critical_test",
                "priority_train",
                "priority_development",
                "priority_test",
                "contract_priority_train",
                "contract_priority_development",
                "contract_priority_test",
                "contract_mandatory_train",
                "contract_mandatory_development",
                "contract_mandatory_test",
                "contract_infeasible_train",
                "contract_infeasible_development",
                "contract_infeasible_test",
            ],
        },
    ]
    return families


def mission_families():
    """Detached public metadata; never builds a recipe, checkpoint or runtime."""
    return {
        "schema": "space-env.agent-mission-families/1",
        "families": copy.deepcopy(_entries()),
        "missions": [
            {
                "id": f["id"] + "/" + p,
                "family": f["id"],
                "profile": p,
                "kind": f["kind"],
            }
            for f in _entries()
            for p in f["profiles"]
        ],
        "cross_cutting": [
            "energy",
            "memory",
            "thermal",
            "fuel",
            "time",
            "tool_use",
            "replay",
            "sft_rl_export",
        ],
        "scope": "Cross-cutting resources apply only where the selected executor models them. Profile count is not independent task diversity or a benchmark ranking.",
        "roadmap": [
            {
                "id": "on_orbit_servicing",
                "status": "not_available_here",
                "missing": "Contact, docking, manipulation and fluid-transfer dynamics",
            },
            {
                "id": "distributed_control",
                "status": "not_available_here",
                "missing": "Per-agent information boundaries and delayed coordination within a shared executor",
            },
        ],
        "qualification": {
            "flight_qualified": False,
            "hardware_calibrated": False,
            "trained_agent_measured": False,
        },
    }


def _select(mission_id):
    if not isinstance(mission_id, str):
        raise TypeError("mission id must be a catalogue string")
    for family in _entries():
        for profile in family["profiles"]:
            if mission_id == family["id"] + "/" + profile:
                return family, profile
    raise ValueError("unknown or unavailable mission: " + mission_id)


def _fault_configuration(profile):
    fault = {
        "at_sec": 4.0,
        "event": {
            "type": "spacecraft_fault",
            "payload": {"actor": "camera-00", "subsystem": "radio"},
        },
    }
    config = {"events": [fault]}
    if profile in {"permanent", "external_repair"}:
        config["permanent_faults"] = [["camera-00", "radio"]]
    if profile == "external_repair":
        config["events"].append(
            {
                "at_sec": 28.0,
                "event": {
                    "type": "spacecraft_fault_cleared",
                    "payload": {"actor": "camera-00", "subsystem": "radio"},
                },
            }
        )
    return config


def mission_recipe(mission_id):
    """Privileged setup for operators/trainers; never send this to the agent."""
    family, profile = _select(mission_id)
    if family["id"] == "communications":
        from space_env.industrial.native_service import native_service_recipe

        configuration = native_service_recipe(profile)
    elif family["id"] == "proximity":
        from space_env.industrial.native_proximity import native_proximity_recipe

        configuration = native_proximity_recipe(profile)
    elif family["id"] == "orbit_and_deployment":
        if profile.startswith(
            (
                "generated_",
                "windowed_",
                "dynamic_",
                "utility_",
                "priority_",
                "contract_",
            )
        ):
            generation = {"split": profile.rsplit("_", 1)[1], "seed": 0}
            if profile.startswith(
                ("windowed_", "dynamic_", "utility_", "priority_", "contract_")
            ):
                generation["windows"] = True
            if profile.startswith(("dynamic_", "utility_")):
                generation["disruptions"] = profile.split("_")[1]
            if profile.startswith(("priority_", "contract_")):
                generation["competition"] = True
            configuration = {"actors": 2, "generation": generation}
            if profile.startswith(("utility_", "priority_", "contract_")):
                configuration["utility"] = {
                    "priorities": {"slot-sat-00": 9, "slot-sat-01": 1}
                }
            if profile.startswith("contract_"):
                mode = profile.split("_")[1]
                configuration["acceptance"] = {
                    "minimum_fraction": 0.1 if mode == "mandatory" else 0.9,
                    "mandatory": ["slot-sat-01"]
                    if mode == "mandatory"
                    else ["slot-sat-00", "slot-sat-01"]
                    if mode == "infeasible"
                    else ["slot-sat-00"],
                }
        else:
            configuration = {"actors": int(profile.rsplit("_", 1)[1])}
    elif family["id"] == "fault_management":
        configuration = _fault_configuration(profile)
    elif profile == "new_demand":
        configuration = {"deferred_request_at_sec": 12.0}
    elif profile == "station_outage":
        configuration = {
            "events": [
                {
                    "at_sec": 8.0,
                    "event": {
                        "type": "station_outage_added",
                        "payload": {
                            "station": "Ground",
                            "start_sec": 8.0,
                            "end_sec": 28.0,
                        },
                    },
                }
            ]
        }
    else:
        configuration = {}
    # The brief is family-level: its text and identity do not reveal the future
    # fault/release variant, or permit matching an agent prompt to an event table.
    brief = "\n".join(
        [
            (
                "Fulfil the published task_objective: all mandatory goals and its minimum weighted fraction. Remaining native goals are optional for this task; native all-goals success is tracked separately."
                if configuration.get("acceptance")
                else family["objective"]
            ),
            "Model scope: " + family["fidelity"],
            "Limitations: " + "; ".join(family["limitations"]),
            "Use only the current observation and declared tools. Future requests or faults are not available to you. Judge feasibility as observations change.",
        ]
    )
    body = {
        "schema": "space-env.agent-mission-recipe/1",
        "id": mission_id,
        "family": family["id"],
        "kind": family["kind"],
        "task_text": brief,
        "configuration": configuration,
    }
    return {**body, "sha256": digest(body)}


def make_session(mission_id):
    return _session(mission_recipe(mission_id))


def _session(recipe):
    from space_env.industrial import (
        NativeProximitySession,
        NativeServiceSession,
        OperationsSession,
        OrbitSession,
    )

    return {
        "operations": OperationsSession,
        "native_service": NativeServiceSession,
        "native_proximity": NativeProximitySession,
        "native_orbits": OrbitSession,
    }[recipe["kind"]](recipe["configuration"])


def make_episode(mission_id, *, split=None, policy_origin="unspecified", max_calls=256):
    from space_env.industrial.agent import IndustrialEpisode

    recipe = mission_recipe(mission_id)
    if split is None:
        split = (recipe["configuration"].get("generation") or {}).get("split", "train")
    return IndustrialEpisode(
        _session(recipe),
        split=split,
        policy_origin=policy_origin,
        max_calls=max_calls,
        task_text=recipe["task_text"],
    )


def make_env(mission_id, *, max_calls=256):
    from space_env.industrial.gym import IndustrialEnv

    recipe = mission_recipe(mission_id)
    return IndustrialEnv(
        recipe["kind"],
        recipe["configuration"],
        max_calls=max_calls,
        task_text=recipe["task_text"],
    )


def baseline_action(kind, observation, *, policy="wait"):
    """Scripted current-observation policies; no LLM inference or optimality claim."""
    if kind not in {
        "operations",
        "native_service",
        "native_proximity",
        "native_orbits",
    } or policy not in {"wait", "public"}:
        raise ValueError("unknown executor or policy")
    if policy == "wait":
        return {"kind": "coast"} if kind == "native_proximity" else {}
    from space_env.industrial.native_proximity import native_proximity_policy
    from space_env.industrial.native_service import native_service_policy
    from space_env.industrial.operations import operations_policy
    from space_env.industrial.orbits import orbit_policy

    return {
        "operations": operations_policy,
        "native_service": native_service_policy,
        "native_proximity": native_proximity_policy,
        "native_orbits": orbit_policy,
    }[kind](observation)


def run_mission(mission_id, *, policy="wait", max_calls=128, split=None):
    family, _ = _select(mission_id)
    if policy not in {"wait", "public"}:
        raise ValueError("unknown policy")
    if split is None:
        split = (
            mission_recipe(mission_id)["configuration"].get("generation") or {}
        ).get("split", "test")
    episode = make_episode(
        mission_id, split=split, policy_origin="scripted_public", max_calls=max_calls
    )
    index = 0
    while not episode.done:
        observation = episode.observe()
        episode.call(
            str(index),
            canonical_json(
                {
                    "expected_revision": observation["revision"],
                    "idempotency_key": str(index),
                    "action": baseline_action(
                        family["kind"], observation, policy=policy
                    ),
                }
            ),
        )
        index += 1
    return episode.record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission")
    parser.add_argument("--policy", choices=["wait", "public"], default="wait")
    parser.add_argument("--max-calls", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    if args.mission:
        from space_env.industrial.agent import verify_episode

        result = run_mission(args.mission, policy=args.policy, max_calls=args.max_calls)
        verify_episode(result)
    else:
        result = mission_families()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, canonical_json(result) + "\n")


if __name__ == "__main__":
    main()
