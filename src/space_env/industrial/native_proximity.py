"""Two nonlinear native central GCRS orbits and two sensor-driven EKFs.

Relative RTN goals use native truth only in the evaluator. Public navigation
contains noisy estimates; safety intervals are checked after real execution.
No docking, maneuver-error, arbitrary-force or flight certification is implied.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from pydantic import Field, model_validator

from space_env.core.relative import relative_state_rtn, rtn_basis, rtn_state_to_eci
from space_env.industrial.proximity import ProximityAction, Vec3, proximity_policy
from space_env.industrial.safety import central_pericenter_m, check_central_interval
from space_env.industrial.session import DecisionSession
from space_env.navigation.models import SensorSpec
from space_env.navigation.session import NavigationSession
from space_env.reference.orbit import ForceModel
from space_env.runtime import dynamic_reference_scenario
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import encode


class NativeProximitySpec(StrictModel):
    scenario: dict
    sensors: dict[str, SensorSpec] = Field(min_length=2, max_length=2)
    chaser: str = "a"
    chief: str = "b"
    mode: str = "inspection"
    waypoint_m: Vec3 = (100.0, 0.0, 0.0)
    tolerance_m: float = Field(default=10.0, gt=0, le=1000, strict=True)
    maximum_speed_m_s: float = Field(default=1.0, gt=0, le=100, strict=True)
    keepout_m: float = Field(default=20.0, gt=0, le=10000, strict=True)
    retreat_radius_m: float = Field(default=150.0, gt=0, le=10000, strict=True)
    required_observations: int = Field(default=3, ge=1, le=128, strict=True)
    decision_sec: float = Field(default=2.0, ge=0.01, le=10, strict=True)
    max_impulse_m_s: float = Field(default=0.5, gt=0, le=10, strict=True)
    delta_v_budget_m_s: float = Field(default=5.0, gt=0, le=100, strict=True)
    max_interval_evaluations: int = Field(default=4097, ge=2, le=65537, strict=True)

    @model_validator(mode="after")
    def contract(self):
        if (
            self.mode not in {"inspection", "hold", "retreat"}
            or self.chaser == self.chief
            or set(self.sensors) != {self.chaser, self.chief}
        ):
            raise ValueError(
                "two distinct measured actors and declared proximity mode required"
            )
        if any(
            s.kind != "cartesian_position"
            or s.reference_actor is not None
            or s.control_profile != "impulse"
            for s in self.sensors.values()
        ):
            raise ValueError(
                "native proximity requires two position sensors and impulse controls"
            )
        if (
            self.keepout_m >= self.retreat_radius_m
            or math.hypot(*self.waypoint_m) - self.tolerance_m <= self.keepout_m
        ):
            raise ValueError("goal and retreat must lie outside keepout")
        return self


class NativeProximitySession(DecisionSession):
    kind = "native_proximity"

    def __init__(self, configuration):
        self.spec = NativeProximitySpec.model_validate(copy.deepcopy(configuration))
        self._start(self.spec.model_dump(mode="json"))
        self._navigation = NavigationSession(
            self.spec.scenario,
            {a: s.model_dump(mode="json") for a, s in self.spec.sensors.items()},
        )
        self._runtime = self._navigation._runtime
        config = self._runtime._backend.config
        if (
            set(config.orbits) != {self.spec.chaser, self.spec.chief}
            or config.spacecraft_bus is not None
            or config.actuation is not None
            or config.unloading is not None
            or self._runtime.horizon_sec > self.spec.decision_sec * 512
        ):
            raise ValueError(
                "native proximity is a bounded two-actor central impulse profile"
            )
        for actor, orbit in config.orbits.items():
            integration = orbit.integration
            if (
                integration.position_atol_m > 1e-4
                or integration.velocity_atol_m_s > 1e-7
                or integration.rtol > 1e-10
                or integration.sample_sec > 10.0
            ):
                raise ValueError(
                    "native proximity requires the declared fine integration profile"
                )
            forces = ForceModel(**orbit.forces)
            if any((forces.j2, forces.drag, forces.srp, forces.third_body)):
                raise ValueError(
                    "interval proof does not cover perturbed force profiles"
                )
            state = self._runtime._backend.mission.ephemerides[actor].state(0.0)
            if central_pericenter_m(state) < 6500000:
                raise ValueError(
                    "initial orbit pericenter outside native proximity envelope"
                )
        initial = self._relative_truth(self._runtime)
        if np.linalg.norm(initial[:3]) <= self.spec.keepout_m:
            raise ValueError("initial relative state violates keepout")
        self._used_dv, self._count = 0.0, 0
        self._aborting = self.spec.mode == "retreat"
        self._safe, self._retreated, self._success = True, False, False
        self._intervals = []

    def _relative_truth(self, runtime):
        mission = runtime._backend.mission
        own = mission.ephemerides[self.spec.chaser].state(runtime.time_sec)
        chief = mission.ephemerides[self.spec.chief].state(runtime.time_sec)
        return relative_state_rtn(own[:3], own[3:], chief[:3], chief[3:])

    def observe(self):
        nav = {
            a: self._navigation._view(self._runtime, self._navigation._filters, a)
            for a in self.spec.sensors
        }
        own = np.array(nav[self.spec.chaser]["estimate_si"])
        chief = np.array(nav[self.spec.chief]["estimate_si"])
        relative = relative_state_rtn(own[:3], own[3:], chief[:3], chief[3:])
        return {
            "schema": "space-env.native-proximity-observation/1",
            "revision": self.revision,
            "chief_actor": self.spec.chief,
            "chaser_actor": self.spec.chaser,
            "time_sec": self._runtime.time_sec,
            "horizon_sec": self._runtime.horizon_sec,
            "navigation": nav,
            "relative_estimate_rtn_si": relative.tolist(),
            "relative_gcrs_covariance_si": (
                np.array(nav[self.spec.chaser]["covariance_si"])
                + np.array(nav[self.spec.chief]["covariance_si"])
            ).tolist(),
            "covariance_assumption": "independent Cartesian filters, no cross-covariance; RTN covariance is not supplied",
            "delta_v_remaining_m_s": self.spec.delta_v_budget_m_s - self._used_dv,
            "aborting": self._aborting,
            "inspection_acknowledgements": self._count,
            "task": {
                k: copy.deepcopy(self._configuration[k])
                for k in (
                    "mode",
                    "waypoint_m",
                    "tolerance_m",
                    "maximum_speed_m_s",
                    "keepout_m",
                    "retreat_radius_m",
                    "decision_sec",
                    "max_impulse_m_s",
                    "required_observations",
                )
            },
        }

    def _apply(self, value):
        action = ProximityAction.model_validate(value)
        dv = np.array(action.delta_v_rtn_m_s)
        amount = float(np.linalg.norm(dv))
        if (
            amount > self.spec.max_impulse_m_s + 1e-12
            or self._used_dv + amount > self.spec.delta_v_budget_m_s + 1e-12
        ):
            raise ValueError("native impulse budget exceeded")
        if self._aborting and action.kind == "inspect":
            raise ValueError("inspection forbidden after abort")
        candidate = self._runtime.fork()
        filters = copy.deepcopy(self._navigation._filters)
        start = candidate.time_sec
        if amount:
            chief = np.array(
                self._navigation._view(self._runtime, filters, self.spec.chief)[
                    "estimate_si"
                ]
            )
            inertial = rtn_basis(chief[:3], chief[3:]) @ dv
            candidate.submit(
                [
                    {
                        "command_id": f"native-proximity/{self.revision}",
                        "actor_id": self.spec.chaser,
                        "kind": "impulse",
                        "duration_sec": 0.0,
                        "impulse": {"frame": "GCRS", "delta_v_m_s": inertial.tolist()},
                    }
                ],
                expected_revision=candidate.revision,
                idempotency_key=f"native-proximity-submit/{self.revision}",
            )
        states = {
            a: candidate._backend.mission.ephemerides[a].state(start).copy()
            for a in self.spec.sensors
        }
        for state in states.values():
            if central_pericenter_m(state) < 6500000:
                raise ValueError("impulse would leave central safety envelope")
        duration = min(self.spec.decision_sec, candidate.horizon_sec - start)
        self._navigation._advance_native(
            candidate,
            filters,
            duration,
            lambda dt: candidate.advance(
                dt,
                expected_revision=candidate.revision,
                idempotency_key=f"native-proximity-advance/{self.revision}/{candidate.revision}",
            ),
        )
        providers = candidate._backend.mission.ephemerides
        witness = check_central_interval(
            states[self.spec.chaser],
            states[self.spec.chief],
            lambda t: providers[self.spec.chaser].state(start + t)[:3],
            lambda t: providers[self.spec.chief].state(start + t)[:3],
            duration,
            self.spec.keepout_m,
            max_evaluations=self.spec.max_interval_evaluations,
        )
        self._runtime = candidate
        self._navigation._runtime = candidate
        self._navigation._filters = filters
        self._intervals.append(witness)
        self._safe &= witness["status"] == "safe"
        self._used_dv += amount
        self._aborting |= action.kind == "abort"
        truth = self._relative_truth(candidate)
        valid = (
            np.linalg.norm(truth[:3] - self.spec.waypoint_m) <= self.spec.tolerance_m
            and np.linalg.norm(truth[3:]) <= self.spec.maximum_speed_m_s
        )
        gained = (
            self._safe
            and valid
            and not self._aborting
            and (action.kind == "inspect" or self.spec.mode == "hold")
        )
        self._count += int(gained)
        self._retreated = bool(
            self._safe
            and self._aborting
            and np.linalg.norm(truth[:3]) >= self.spec.retreat_radius_m
            and truth[:3] @ truth[3:] >= 0
        )
        self._success = bool(
            self._safe
            and (
                self._retreated
                if self.spec.mode == "retreat"
                else not self._aborting
                and self._count >= self.spec.required_observations
            )
        )
        self._done = bool(
            not self._safe
            or self._success
            or self._retreated
            or candidate.time_sec >= candidate.horizon_sec - 1e-9
        )
        return {
            "reward": float(gained) - amount - float(not self._safe),
            "terminated": self._done,
            "elapsed_sec": duration,
            "feedback": {"accepted": True, "code": "executed"},
        }

    def metrics(self):
        orbit = self._runtime._backend.mission.ephemerides[self.spec.chaser]
        propellant = orbit.mass(self._runtime.time_sec) - orbit.dry_mass_kg
        return {
            "success": self._success,
            "terminal": self._done,
            "safe_retreat": self._retreated,
            "safe_intervals": bool(self._safe),
            "inspection_acknowledgements": self._count,
            "delta_v_used_m_s": self._used_dv,
            "propellant_kg": propellant,
            "time_sec": self._runtime.time_sec,
            "native_nonlinear_orbits": True,
            "force_scope": "central gravity; impulse-separated coasts",
        }

    def _witness(self):
        return {
            "native": self._runtime.checkpoint(compact=True),
            "filters": encode(self._navigation._filters),
            "relative_truth_rtn_si": self._relative_truth(self._runtime).tolist(),
            "intervals": copy.deepcopy(self._intervals),
            "metrics": self.metrics(),
        }


def native_proximity_recipe(profile="nominal"):
    from space_env.tasks.modeling_suite import _inputs

    if profile not in {"nominal", "latency", "blind", "crossing", "retreat"}:
        raise ValueError("unknown native proximity profile")
    source = _inputs()["chemical"]["definition"]
    definition = {
        k: copy.deepcopy(source[k])
        for k in ("schema", "eop", "operations", "orbits", "horizon_sec")
    }
    definition["horizon_sec"] = 30.0
    definition["operations"]["products"] = []
    for vehicle in definition["operations"]["vehicles"].values():
        vehicle["solar_mode"] = "ideal_sun_tracking"
    chief = definition["orbits"]["b"]
    chief["initial"] = copy.deepcopy(definition["orbits"]["a"]["initial"])
    relative = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if profile == "crossing":
        relative = [100.0, 0.0, 0.0, -10.0, 0.0, 0.0]
    if profile == "retreat":
        relative = [145.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    x = rtn_state_to_eci(relative, chief["initial"]["r_m"], chief["initial"]["v_m_s"])
    definition["orbits"]["a"]["initial"].update(
        r_m=x[:3].tolist(), v_m_s=x[3:].tolist()
    )
    sensors = {}
    for index, (actor, orbit) in enumerate(definition["orbits"].items()):
        orbit["forces"] = {}
        state = [*orbit["initial"]["r_m"], *orbit["initial"]["v_m_s"]]
        state[0] += 1.0
        sensors[actor] = {
            "kind": "cartesian_position",
            "initial_estimate_si": state,
            "initial_covariance_si": np.diag(
                [4.0, 4.0, 4.0, 0.01, 0.01, 0.01]
            ).tolist(),
            "standard_deviation": [1.0, 1.0, 1.0],
            "cadence_sec": 1.0,
            "latency_sec": 1.5 if profile == "latency" else 0.0,
            "dropout_probability": 1.0
            if profile == "blind"
            else 0.2
            if profile == "latency"
            else 0.0,
            "continuous_power_w": 3.0,
            "seed": index,
        }
    cfg = {
        "scenario": dynamic_reference_scenario(
            definition,
            id="native-proximity-" + profile,
            lineage="native-relative-central/v1",
        ),
        "sensors": sensors,
    }
    if profile == "crossing":
        cfg.update(max_impulse_m_s=3.0, delta_v_budget_m_s=6.0)
    if profile == "retreat":
        cfg["mode"] = "retreat"
    return NativeProximitySpec.model_validate(cfg).model_dump(mode="json")


def native_proximity_policy(observation):
    chief = next(
        v
        for a, v in observation["navigation"].items()
        if a == observation["chief_actor"]
    )
    state = np.array(chief["estimate_si"])
    radius = np.linalg.norm(state[:3])
    n = float(np.linalg.norm(np.cross(state[:3], state[3:])) / radius**2)
    p = np.array(observation["relative_gcrs_covariance_si"])
    conservative = np.diag(
        [max(np.linalg.eigvalsh(p[:3, :3]))] * 3
        + [max(np.linalg.eigvalsh(p[3:, 3:]))] * 3
    )
    task = observation["task"]
    return proximity_policy(
        {
            "estimate_si": observation["relative_estimate_rtn_si"],
            "covariance_si": conservative.tolist(),
            "revision": observation["revision"],
            "delta_v_remaining_m_s": observation["delta_v_remaining_m_s"],
            "aborting": observation["aborting"],
            "task": {
                **task,
                "mean_motion_rad_s": n,
                "step_sec": task["decision_sec"],
                "horizon_steps": math.ceil(
                    observation["horizon_sec"] / task["decision_sec"]
                ),
            },
        }
    )
