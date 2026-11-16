"""Deterministic bounded phasing distributions with explicit held-out regimes.

Constructive targets from a private reference control, never feasibility filtered.
Split separation is by declared horizon/thrust ranges, not just random seed names.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import numpy as np
from pydantic import Field, model_serializer, model_validator

from space_env.runtime import dynamic_reference_scenario
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest
from space_env.tasks.deployment import deployment_recipe
from space_env.tasks.modeling_suite import _orbit_target
from space_env.tasks.schema import TaskSpec

RANGES = {
    "train": {"horizons": (40, 50), "thrust_centinewtons": (60, 120)},
    "development": {"horizons": (60, 70), "thrust_centinewtons": (125, 150)},
    "test": {"horizons": (80, 90), "thrust_centinewtons": (155, 180)},
}


class PhasingGeneration(StrictModel):
    split: Literal["train", "development", "test"]
    seed: int = Field(ge=0, lt=2**32, strict=True)
    windows: bool = Field(default=False, strict=True)
    disruptions: Literal["recoverable", "critical"] | None = None
    competition: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def disruption_scope(self):
        if self.competition and (not self.windows or self.disruptions is not None):
            raise ValueError(
                "competition requires windows and excludes disruption profiles"
            )
        if self.disruptions is not None and not self.windows:
            raise ValueError("disruptions require windowed phasing")
        return self

    @model_serializer(mode="wrap")
    def optional_windows(self, handler):
        value = handler(self)
        if not self.competition:
            value.pop("competition", None)
        if self.disruptions is None:
            value.pop("disruptions", None)
        if not self.windows:
            value.pop("windows", None)
        return value


def generated_phasing_recipe(actors, generation):
    spec = PhasingGeneration.model_validate(generation)
    if spec.competition:
        from space_env.tasks.phasing_competition import competitive_recipe

        return competitive_recipe(actors, spec)
    if spec.disruptions is not None:
        from space_env.tasks.phasing_disruptions import disrupted_recipe

        return disrupted_recipe(actors, spec)
    if spec.windows:
        from space_env.tasks.phasing_windows import windowed_recipe

        return windowed_recipe(actors, spec)
    # The original recipe enforces fleet bounds. Only its validated spacecraft
    # scaffold is reused; all goals and templates below are regenerated.
    base = deployment_recipe(actors)
    domain = RANGES[spec.split]

    def draw(label, size):
        payload = f"phasing-generation/v1:{spec.split}:{spec.seed}:{label}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest(), "big") % size

    horizon = float(domain["horizons"][draw("horizon", 2)])
    definition = base["scenario"]["definition"]
    definition["horizon_sec"] = horizon
    actions, conditions, limits, chosen, parameters = [], [], {}, [], []
    for index, (actor, orbit) in enumerate(definition["orbits"].items()):
        angle = (draw(actor + ":angle", 2001) - 1000) / 100000.0
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for key in ("r_m", "v_m_s"):
            orbit["initial"][key] = (rotation @ orbit["initial"][key]).tolist()
        p = orbit["propulsion"]
        p["propellant_kg"] = 7.0 + draw(actor + ":fuel", 31) / 10
        lower, upper = domain["thrust_centinewtons"]
        thrust = (lower + draw(actor + ":thrust", upper - lower + 1)) / 100
        firing = float(10 + 5 * draw(actor + ":duration", 3))
        direction = np.array(orbit["initial"]["v_m_s"])
        direction /= np.linalg.norm(direction)
        direction *= -1 if draw(actor + ":sign", 2) else 1
        target = _orbit_target(
            orbit, direction, thrust=thrust, firing=firing, end=horizon
        )
        good_slot = draw(actor + ":slot", 2)
        for slot in range(2):
            name = (
                "burn-"
                + hashlib.sha256(
                    f"{spec.split}:{spec.seed}:{actor}:{slot}".encode()
                ).hexdigest()[:20]
            )
            actions.append(
                {
                    "id": name,
                    "role": actor,
                    "kind": "submit",
                    "commands": [
                        {
                            "command_id": name,
                            "actor_id": actor,
                            "kind": "finite_burn",
                            "duration_sec": firing,
                            "burn": {
                                "frame": "GCRS",
                                "direction_gcrs": (
                                    direction * (1 if slot == good_slot else -1)
                                ).tolist(),
                                "thrust_n": thrust,
                            },
                        }
                    ],
                }
            )
            if slot == good_slot:
                chosen.append(name)
        conditions.append(
            {
                "id": "slot-" + actor,
                "kind": "orbit_state",
                "actor": actor,
                "at_sec": horizon,
                "position_m": target[:3],
                "velocity_m_s": target[3:],
                "position_tolerance_m": 0.01,
                "velocity_tolerance_m_s": 0.00001,
            }
        )
        used = thrust * firing / (p["isp_s"] * 9.80665)
        limits[actor] = {"min_propellant_kg": p["propellant_kg"] - 1.1 * used}
        parameters.append(
            {
                "actor": actor,
                "thrust_n": thrust,
                "firing_sec": firing,
                "initial_propellant_kg": p["propellant_kg"],
                "plane_rotation_rad": angle,
            }
        )
    scenario = dynamic_reference_scenario(
        definition, id="generated-local-phasing", lineage="generated-local-phasing/v1"
    )
    task = TaskSpec.model_validate(
        {
            "id": "generated-local-phasing",
            "roles": base["task"]["roles"],
            "actions": actions,
            "decision_sec": 5.0,
            "max_decision_sec": 5.0,
            "max_decisions": 24,
            "max_joint_actions": actors,
            "goal": {
                "deadline_sec": horizon,
                "conditions": conditions,
                "limits": limits,
            },
        }
    ).model_dump(mode="json", by_alias=True)
    return {
        "scenario": scenario,
        "task": task,
        "reference_steps": [{"action_ids": chosen, "advance_sec": 5.0}]
        + [
            {"action_ids": ["wait"], "advance_sec": 5.0}
            for _ in range(int(horizon / 5) - 1)
        ],
        "generation": {
            "schema": "space-env.phasing-generation/1",
            "split": spec.split,
            "seed": spec.seed,
            "actors": actors,
            "horizon_sec": horizon,
            "parameters": parameters,
            "physics_sha256": digest(
                {
                    "definition": definition,
                    "goal": task["goal"],
                    "controls": sorted(
                        [
                            {
                                k: v
                                for k, v in a["commands"][0].items()
                                if k != "command_id"
                            }
                            for a in actions
                        ],
                        key=lambda c: digest(c),
                    ),
                }
            ),
        },
        "scope": "Constructive synthetic local central-gravity phasing; held-out horizon/thrust regimes, no independent physical validation or arbitrary trajectory optimisation.",
    }
