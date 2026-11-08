"""Declarative, current-native mission goals. No callbacks or future rollout.

Goal satisfaction is an endpoint condition, not a flight qualification. Budgets
are measured from scenario origin, so starting from a prefix cannot renew fuel.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from space_env.reference.spacecraft_bus import quaternion_angle
from space_env.runtime.schema import FiniteScalar, StrictModel

Vec3 = tuple[FiniteScalar, FiniteScalar, FiniteScalar]
Quat = tuple[FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar]


class _Condition(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    completion: Literal["maintain", "achieve"] = "maintain"
    not_before_sec: float = Field(default=0.0, ge=0, strict=True)


class DeliveryGoal(_Condition):
    kind: Literal["delivery"]
    request_ids: tuple[str, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.request_ids)) != len(self.request_ids) or any(
            not x.strip() for x in self.request_ids
        ):
            raise ValueError("goal request IDs must be unique and nonblank")
        return self


class _ActorGoal(_Condition):
    actor: str = Field(min_length=1, max_length=128)
    command_after_sec: float = Field(default=0.0, ge=0, strict=True)
    require_command: bool = Field(default=True, strict=True)


class AttitudeGoal(_ActorGoal):
    kind: Literal["attitude"]
    frame: Literal["GCRS"] = "GCRS"
    quaternion_wxyz: Quat
    tolerance_deg: float = Field(gt=0, le=180, strict=True)
    max_rate_deg_s: float = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def unit(self):
        if not math.isclose(
            math.hypot(*self.quaternion_wxyz), 1, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError("goal quaternion must be unit body-to-GCRS wxyz")
        return self


class MomentumGoal(_ActorGoal):
    kind: Literal["momentum"]
    frame: Literal["BODY"] = "BODY"
    target_nms: Vec3 = (0.0, 0.0, 0.0)
    tolerance_nms: float = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def bounded(self):
        if not math.isfinite(math.hypot(*self.target_nms)):
            raise ValueError("momentum target norm must be finite")
        return self


class ImpulseGoal(_ActorGoal):
    kind: Literal["body_impulse"]
    frame: Literal["GCRS"] = "GCRS"
    target_ns: Vec3
    tolerance_ns: float = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def nonzero(self):
        norm = math.hypot(*self.target_ns)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("body impulse target norm must be finite and nonzero")
        return self


class OrbitGoal(_ActorGoal):
    kind: Literal["orbit_state"]
    completion: Literal["maintain", "achieve"] = "achieve"
    frame: Literal["GCRS"] = "GCRS"
    at_sec: float = Field(gt=1e-9, le=259200, strict=True)
    time_tolerance_sec: float = Field(default=1e-6, gt=0, le=1, strict=True)
    position_m: Vec3
    velocity_m_s: Vec3
    position_tolerance_m: float = Field(gt=0, strict=True)
    velocity_tolerance_m_s: float = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def applicability(self):
        r, v = math.hypot(*self.position_m), math.hypot(*self.velocity_m_s)
        if not math.isfinite(r) or not math.isfinite(v) or not 0 < r <= 1e13 or v > 1e7:
            raise ValueError("orbit target outside bounded Earth-mission applicability")
        if self.not_before_sec > self.at_sec + self.time_tolerance_sec:
            raise ValueError("orbit target release misses its epoch window")
        return self


Condition = Annotated[
    DeliveryGoal | AttitudeGoal | MomentumGoal | ImpulseGoal | OrbitGoal,
    Field(discriminator="kind"),
]


class GoalLimits(StrictModel):
    max_energy_j: float | None = Field(default=None, ge=0, strict=True)
    min_propellant_kg: float | None = Field(default=None, ge=0, strict=True)
    require_healthy: bool = Field(default=False, strict=True)


class MissionGoal(StrictModel):
    schema_id: Literal["space-env.mission-goal/1"] = Field(
        default="space-env.mission-goal/1", alias="schema"
    )
    conditions: tuple[Condition, ...] = Field(min_length=1, max_length=16)
    deadline_sec: float = Field(gt=1e-9, le=259200, strict=True)
    limits: dict[str, GoalLimits] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def unique(self):
        if len({c.id for c in self.conditions}) != len(self.conditions):
            raise ValueError("goal condition IDs must be unique")
        if any(c.not_before_sec >= self.deadline_sec for c in self.conditions):
            raise ValueError("goal release must precede deadline")
        if any(
            isinstance(c, OrbitGoal) and c.at_sec > self.deadline_sec
            for c in self.conditions
        ):
            raise ValueError("orbit target epoch exceeds goal deadline")
        if any(
            isinstance(c, _ActorGoal) and c.command_after_sec >= self.deadline_sec
            for c in self.conditions
        ):
            raise ValueError("goal command epoch must precede deadline")
        return self


def _local(runtime, actor):
    return runtime.observe(actor)["actors"][actor]


def validate_goal(goal, runtime, controlled_actors):
    """Reject unsupported hardware/goals before allocating an episode history."""
    if goal.deadline_sec > runtime.horizon_sec:
        raise ValueError("goal deadline exceeds native horizon")
    actors = set(controlled_actors)
    report = runtime._backend.report()
    requests = (report.get("lifecycle") or {}).get("requests", {})
    for condition in goal.conditions:
        if isinstance(condition, DeliveryGoal):
            if set(condition.request_ids) - requests.keys():
                raise ValueError("goal references unknown native request")
            continue
        if condition.actor not in actors:
            raise ValueError("goal actor must be controlled by a task role")
        obs = _local(runtime, condition.actor)
        bus = obs.get("spacecraft_bus", {})
        if isinstance(condition, AttitudeGoal) and "quaternion_wxyz" not in bus:
            raise ValueError("attitude goal requires native spacecraft bus")
        if isinstance(condition, MomentumGoal) and "rigid_attitude" not in bus:
            raise ValueError("momentum goal requires native rigid wheel state")
        if isinstance(condition, ImpulseGoal) and condition.actor not in (
            (report.get("lifecycle") or {}).get("actuation") or {}
        ).get("states", {}):
            raise ValueError("impulse goal requires native body engine")
        if isinstance(condition, OrbitGoal) and "orbit" not in obs:
            raise ValueError("orbit goal requires current dynamic GCRS telemetry")
    for actor, limits in goal.limits.items():
        if actor not in actors:
            raise ValueError("goal limit actor must be controlled")
        obs = _local(runtime, actor)
        if limits.min_propellant_kg is not None and "orbit" not in obs:
            raise ValueError("fuel limit requires native dynamic mass telemetry")
        if limits.max_energy_j is not None and actor not in report.get(
            "energy_ledger", {}
        ):
            raise ValueError("energy limit requires native energy ledger")
        if limits.require_healthy and "spacecraft_bus" not in obs:
            raise ValueError("health limit requires native bus")


def evaluate_goal(goal, runtime, *, previous_failures=(), previous_achievements=()):
    """Evaluate actual current state and completed commands, not planned intent."""
    now = runtime.time_sec
    report = runtime._backend.report()
    local_cache = {}

    def local_actor(actor):
        if actor not in local_cache:
            local_cache[actor] = _local(runtime, actor)
        return local_cache[actor]

    commands = runtime.command_status()
    lifecycle = report.get("lifecycle") or {}
    detailed = {
        **(report.get("spacecraft_bus") or {}).get("operations", {}),
        **(lifecycle.get("unloading") or {}).get("operations", {}),
        **(lifecycle.get("actuation") or {}).get("operations", {}),
    }
    failures = set(previous_failures)
    achieved = set(previous_achievements)
    rows = []
    for index, c in enumerate(goal.conditions):
        row = {
            "index": index,
            "id": c.id,
            "kind": c.kind,
            "actor": getattr(c, "actor", None),
            "recipe": c.model_dump(mode="json"),
            "met": False,
            "errors": {},
        }
        if isinstance(c, DeliveryGoal):
            requests = report["lifecycle"]["requests"]
            row["met"] = all(
                requests[r]["status"] == "completed" for r in c.request_ids
            )
        else:
            local = local_actor(c.actor)
            bus = local.get("spacecraft_bus", {})
            kinds = {
                "attitude": {"slew"},
                "momentum": {"momentum_unload"},
                "body_impulse": {"body_burn"},
                "orbit_state": {"finite_burn", "body_burn", "impulse"},
            }[c.kind]
            completed = [
                r
                for r in commands
                if r["command"]["actor_id"] == c.actor
                and r["command"]["kind"] in kinds
                and r["status"] == "completed"
                and r["start_sec"] >= c.command_after_sec
                and detailed.get(r["command"]["command_id"], {"status": "completed"})[
                    "status"
                ]
                == "completed"
            ]
            proof = bool(completed) or not c.require_command
            row["command_completed"] = bool(completed)
            if isinstance(c, AttitudeGoal):
                angle = quaternion_angle(bus["quaternion_wxyz"], c.quaternion_wxyz)
                rigid = bus.get("rigid_attitude")
                rate = (
                    math.degrees(math.hypot(*rigid["omega_body_rad_s"]))
                    if rigid
                    else abs(bus["angular_rate_deg_s"])
                )
                busy = any(
                    r["command"]["actor_id"] == c.actor
                    and r["status"] in {"accepted", "running"}
                    and r["command"]["kind"]
                    in {"slew", "imaging", "wheel_torque", "momentum_unload"}
                    for r in commands
                )
                row["errors"] = {"attitude_deg": angle, "rate_deg_s": rate}
                row["met"] = (
                    proof
                    and not busy
                    and angle <= c.tolerance_deg
                    and rate <= c.max_rate_deg_s
                    and bus["settle_remaining_sec"] <= 1e-9
                )
            elif isinstance(c, MomentumGoal):
                errors = [
                    x - y
                    for x, y in zip(
                        bus["rigid_attitude"]["wheel_momentum_nms"], c.target_nms
                    )
                ]
                row["errors"] = {"momentum_nms": errors}
                busy = any(
                    r["command"]["actor_id"] == c.actor
                    and r["status"] in {"accepted", "running"}
                    and r["command"]["kind"]
                    in {"wheel_torque", "momentum_unload", "slew", "body_burn"}
                    for r in commands
                )
                row["met"] = (
                    proof
                    and not busy
                    and max(abs(x) for x in errors) <= c.tolerance_nms
                )
            elif isinstance(c, ImpulseGoal):
                operations = report["lifecycle"]["actuation"]["operations"]
                ids = {
                    r["command"]["command_id"]
                    for r in commands
                    if r["command"]["actor_id"] == c.actor
                    and r["command"]["kind"] == "body_burn"
                    and r["start_sec"] >= c.command_after_sec
                }
                impulses = [
                    op["impulse_vector_gcrs_ns"]
                    for cid, op in operations.items()
                    if cid in ids
                ]
                errors = [
                    math.fsum(v[i] for v in impulses) - c.target_ns[i] for i in range(3)
                ]
                row["errors"] = {"impulse_ns": errors}
                busy = any(
                    r["command"]["actor_id"] == c.actor
                    and r["command"]["kind"] == "body_burn"
                    and r["status"] in {"accepted", "running"}
                    for r in commands
                )
                row["met"] = (
                    proof and not busy and math.hypot(*errors) <= c.tolerance_ns
                )
            elif isinstance(c, OrbitGoal):
                state = local["orbit"]["state_si"]
                r = [state[i] - c.position_m[i] for i in range(3)]
                v = [state[i + 3] - c.velocity_m_s[i] for i in range(3)]
                row["errors"] = {"position_m": r, "velocity_m_s": v}
                row["met"] = (
                    proof
                    and abs(now - c.at_sec) <= c.time_tolerance_sec
                    and math.hypot(*r) <= c.position_tolerance_m
                    and math.hypot(*v) <= c.velocity_tolerance_m_s
                )
                if now > c.at_sec + c.time_tolerance_sec and c.id not in achieved:
                    failures.add("orbit_window_missed:" + c.id)
        row["met"] = bool(row["met"] and now >= c.not_before_sec)
        row["current_met"] = row["met"]
        if c.completion == "achieve":
            if row["met"]:
                achieved.add(c.id)
            row["met"] = c.id in achieved
        rows.append(row)
    resources = []
    for index, (actor, limits) in enumerate(sorted(goal.limits.items())):
        obs = local_actor(actor)
        energy = report.get("energy_ledger", {}).get(actor, {}).get("consumed_wh")
        propellant = obs.get("orbit", {}).get("propellant_kg")
        current_bus = obs.get("spacecraft_bus")
        resources.append(
            {
                "index": index,
                "actor": actor,
                "recipe": limits.model_dump(mode="json"),
                "energy_used_j": None if energy is None else energy * 3600,
                "propellant_kg": propellant,
                "healthy": None
                if current_bus is None
                else not (current_bus["safe"] or current_bus["faults"]),
            }
        )
        if (
            limits.max_energy_j is not None
            and report["energy_ledger"][actor]["consumed_wh"] * 3600
            > limits.max_energy_j + 1e-9
        ):
            failures.add("energy_budget:" + actor)
        if (
            limits.min_propellant_kg is not None
            and obs["orbit"]["propellant_kg"] < limits.min_propellant_kg - 1e-12
        ):
            failures.add("fuel_reserve:" + actor)
        bus = obs.get("spacecraft_bus", {})
        events = (report.get("spacecraft_bus") or {}).get("events", [])
        past_fault = any(
            e.get("payload", {}).get("actor") == actor
            and e.get("type") in {"spacecraft_fault", "spacecraft_safe_mode"}
            for e in events
        )
        if limits.require_healthy and (
            bus.get("safe") or bus.get("faults") or past_fault
        ):
            failures.add("health:" + actor)
    if now > goal.deadline_sec + 1e-9:
        failures.add("deadline")
    success = all(r["met"] for r in rows) and not failures
    return {
        "schema": "space-env.mission-goal-status/1",
        "time_sec": now,
        "deadline_sec": goal.deadline_sec,
        "conditions": rows,
        "completed": sum(r["met"] for r in rows),
        "total": len(rows),
        "success": success,
        "failures": sorted(failures),
        "achieved": sorted(achieved),
        "resources": resources,
        "terminal": success or bool(failures) or now >= goal.deadline_sec - 1e-9,
    }


GOAL_FEATURES = (
    "met",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "velocity_x_m_s",
    "velocity_y_m_s",
    "velocity_z_m_s",
    "target_q_w",
    "target_q_x",
    "target_q_y",
    "target_q_z",
    "momentum_x_nms",
    "momentum_y_nms",
    "momentum_z_nms",
    "attitude_error_deg",
    "rate_deg_s",
    "impulse_x_ns",
    "impulse_y_ns",
    "impulse_z_ns",
    "time_remaining_sec",
    "kind_code",
    "achieve_once",
    "require_command",
    "not_before_sec",
    "command_after_sec",
    "target_at_sec",
    "time_tolerance_sec",
    "position_tolerance_m",
    "velocity_tolerance_m_s",
    "tolerance_deg",
    "max_rate_deg_s",
    "tolerance_nms",
    "tolerance_ns",
    "actor_index",
)

GOAL_KINDS = {
    "delivery": 1,
    "attitude": 2,
    "momentum": 3,
    "body_impulse": 4,
    "orbit_state": 5,
}
GOAL_RESOURCE_FEATURES = (
    "actor_index",
    "energy_budget_j",
    "energy_used_j",
    "energy_margin_j",
    "fuel_reserve_kg",
    "propellant_kg",
    "fuel_margin_kg",
    "healthy_required",
    "healthy",
)


def goal_features(row, deadline, now, actor_index=None):
    e, recipe = row["errors"], row["recipe"]
    values = {
        "met": float(row["met"]),
        "time_remaining_sec": max(0.0, deadline - now),
        "kind_code": GOAL_KINDS[row["kind"]],
        "achieve_once": float(recipe["completion"] == "achieve"),
    }
    for source, columns in (
        ("position_m", ("position_x_m", "position_y_m", "position_z_m")),
        ("velocity_m_s", ("velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s")),
        ("momentum_nms", ("momentum_x_nms", "momentum_y_nms", "momentum_z_nms")),
        ("impulse_ns", ("impulse_x_ns", "impulse_y_ns", "impulse_z_ns")),
    ):
        if source in e:
            values.update(zip(columns, e[source]))
    if "quaternion_wxyz" in recipe:
        values.update(
            zip(
                ("target_q_w", "target_q_x", "target_q_y", "target_q_z"),
                recipe["quaternion_wxyz"],
            )
        )
    for source, column in (
        ("attitude_deg", "attitude_error_deg"),
        ("rate_deg_s", "rate_deg_s"),
    ):
        if source in e:
            values[column] = e[source]
    for source in (
        "require_command",
        "not_before_sec",
        "command_after_sec",
        "time_tolerance_sec",
        "position_tolerance_m",
        "velocity_tolerance_m_s",
        "tolerance_deg",
        "max_rate_deg_s",
        "tolerance_nms",
        "tolerance_ns",
    ):
        if source in recipe:
            values[source] = float(recipe[source])
    if "at_sec" in recipe:
        values["target_at_sec"] = recipe["at_sec"]
    if actor_index is not None:
        values["actor_index"] = actor_index
    return [values.get(k, 0.0) for k in GOAL_FEATURES], [
        int(k in values) for k in GOAL_FEATURES
    ]


def goal_resource_features(row, actor_index):
    recipe = row["recipe"]
    energy, fuel = row["energy_used_j"], row["propellant_kg"]
    budget, reserve = recipe["max_energy_j"], recipe["min_propellant_kg"]
    values = [
        actor_index,
        budget,
        energy,
        None if budget is None or energy is None else budget - energy,
        reserve,
        fuel,
        None if reserve is None or fuel is None else fuel - reserve,
        float(recipe["require_healthy"]),
        None if row["healthy"] is None else float(row["healthy"]),
    ]
    return [0.0 if x is None else x for x in values], [
        int(x is not None) for x in values
    ]
