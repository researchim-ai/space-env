"""Central ideal-state adapter for bounded native multi-spacecraft phasing.

Uses TaskSession's reward, goal evaluation and execution without a second physics
or resource ledger. Does not provide distributed control or collision guarantees.
"""

from __future__ import annotations

import copy

import numpy as np
from pydantic import Field, model_serializer

from space_env.core.constants import MU_EARTH
from space_env.industrial.acceptance import (
    TaskAcceptanceSpec,
    acceptance_bound,
    evaluate_acceptance,
    validate_acceptance,
)
from space_env.industrial.dispatch import DispatchEvent, subtract_interval
from space_env.industrial.session import DecisionSession
from space_env.industrial.utility import (
    GoalUtilitySpec,
    evaluate_utility,
    resolve_weights,
)
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession
from space_env.tasks.deployment import deployment_recipe
from space_env.tasks.phasing_generation import (
    PhasingGeneration,
    generated_phasing_recipe,
)


class OrbitSpec(StrictModel):
    actors: int = Field(default=2, ge=2, le=8, strict=True)
    generation: PhasingGeneration | None = None
    dispatch_events: tuple[DispatchEvent, ...] = Field(default=(), max_length=32)
    utility: GoalUtilitySpec | None = None
    acceptance: TaskAcceptanceSpec | None = None

    @model_serializer(mode="wrap")
    def optional_events(self, handler):
        value = handler(self)
        if self.acceptance is None:
            value.pop("acceptance", None)
        if self.utility is None:
            value.pop("utility", None)
        if not self.dispatch_events:
            value.pop("dispatch_events", None)
        return value


class OrbitAction(StrictModel):
    action_ids: tuple[str, ...] = Field(default=(), max_length=8)


class OrbitSession(DecisionSession):
    kind = "native_orbits"

    def __init__(self, configuration=None):
        self.spec = OrbitSpec.model_validate(
            {} if configuration is None else copy.deepcopy(configuration)
        )
        self._start(self.spec.model_dump(mode="json"))
        recipe = (
            deployment_recipe(self.spec.actors)
            if self.spec.generation is None
            else generated_phasing_recipe(self.spec.actors, self.spec.generation)
        )
        self._native = TaskSession(recipe["scenario"], recipe["task"])
        self._windows = copy.deepcopy(recipe.get("launch_windows"))
        self._capacity = recipe.get("dispatch_capacity", self.spec.actors)
        self._launched = set()
        self._admission_rejections = 0
        self._dynamics = {
            "mu_m3_s2": MU_EARTH,
            "isp_s": {
                actor: orbit["propulsion"]["isp_s"]
                for actor, orbit in recipe["scenario"]["definition"]["orbits"].items()
            },
        }
        self._return = 0.0
        self._rejections = 0
        self._truncated = False
        events = [
            *self.spec.dispatch_events,
            *[
                DispatchEvent.model_validate(e)
                for e in recipe.get("dispatch_events", [])
            ],
        ]
        capacity_times = [e.at_sec for e in events if e.kind == "capacity"]
        if len(capacity_times) != len(set(capacity_times)):
            raise ValueError("ambiguous simultaneous dispatch capacities")
        horizon = self._native._runtime.horizon_sec
        if len(events) > 32 or len(
            {digest(e.model_dump(mode="json")) for e in events}
        ) != len(events):
            raise ValueError("duplicate or excessive dispatch events")
        if events and self._windows is None:
            raise ValueError("dispatch events require published launch windows")
        if any(
            e.at_sec >= horizon
            or (e.actor is not None and e.actor not in self._dynamics["isp_s"])
            or (e.end_sec is not None and e.end_sec > horizon)
            for e in events
        ):
            raise ValueError("dispatch event outside mission scope")
        self._dispatch_events = tuple(
            sorted(events, key=lambda e: (e.at_sec, digest(e.model_dump(mode="json"))))
        )
        self._released_count = 0
        self._dispatch_updates = []
        self._release_dispatch_events()
        self._weights = None
        self._utility_state = None
        self._agent_return = 0.0
        self._raw_native_return = 0.0
        self._dispatch_bound = copy.deepcopy(recipe.get("dispatch_bound"))
        if self.spec.utility is not None:
            if self._native.describe()["objective"] != "delivery":
                raise ValueError(
                    "utility mode requires the declared delivery-cost objective"
                )
            self._weights = resolve_weights(
                self.spec.utility, self._native.describe()["goal"]["conditions"]
            )
            self._utility_state = evaluate_utility(
                self._native.goal_status(), self._weights
            )
            if self._utility_state["violations"]:
                raise ValueError(
                    "initial utility state violates declared safety constraints"
                )

        if self.spec.acceptance is not None and self._weights is None:
            raise ValueError("task acceptance requires explicit utility mode")
        self._acceptance_state = None
        self._reward_potential = (
            None if self._utility_state is None else self._utility_state["score"]
        )
        if self.spec.acceptance is not None:
            validate_acceptance(self.spec.acceptance, self._weights)
            self._acceptance_state = evaluate_acceptance(
                self.spec.acceptance,
                self._native.goal_status(),
                self._utility_state,
                finished=False,
            )

    @property
    def episode_budget_penalty(self):
        return 1.0 if self.spec.acceptance is not None else 0.0

    @property
    def dataset_split(self):
        return None if self.spec.generation is None else self.spec.generation.split

    def observe(self):
        view = self._native.observe("@central")
        view.pop("binding", None)
        view["native_revision"] = view["revision"]
        view["revision"] = self.revision
        view["schema"] = "space-env.native-orbits-observation/1"
        view["objective"] = self._native.describe()["goal"]
        view["dynamics"] = copy.deepcopy(self._dynamics)
        view["fidelity"] = (
            "Central ideal current state; native finite thrust, bounded 40-100 second central-gravity local phasing. No continuous collision guarantee or service coupling."
        )
        if self._windows is not None:
            view["launch_windows"] = copy.deepcopy(self._windows)
            view["dispatch_capacity"] = self._capacity
            view["launched_actors"] = sorted(self._launched)
            view["one_burn_per_actor"] = True
            for index, row in enumerate(view["catalogue"]):
                if row["id"] == "wait":
                    continue
                actor = row["commands"][0]["actor_id"]
                reason = (
                    "already_launched"
                    if actor in self._launched
                    else "launch_window_closed"
                    if not self._window_open(row["id"], view["time_sec"])
                    else "dispatch_unavailable"
                    if self._capacity == 0
                    else None
                )
                if reason:
                    if row["enabled"]:
                        row["reason"] = reason
                    row["enabled"] = False
                    row["window_reason"] = reason
                    view["action_mask"][index] = False
        if self._windows is not None:
            view["dispatch_updates"] = copy.deepcopy(self._dispatch_updates)
            by_actor = {}
            for row in view["catalogue"]:
                if row["id"] != "wait":
                    by_actor.setdefault(row["commands"][0]["actor_id"], []).extend(
                        self._windows[row["id"]]
                    )
            view["planning_status"] = {
                "no_future_dispatch_windows": sorted(
                    actor
                    for actor, intervals in by_actor.items()
                    if actor not in self._launched
                    and not any(right > view["time_sec"] for _, right in intervals)
                ),
                "scope": "Published dispatch availability only; not a proof of physical mission infeasibility",
            }
        published_bound = self._current_dispatch_bound()
        if published_bound is not None:
            view["dispatch_bound"] = published_bound
            if self._weights is not None:
                count_limit = published_bound["maximum_completed_conditions"]
                view["dispatch_bound"]["maximum_weighted_fraction"] = sum(
                    sorted(self._weights.values(), reverse=True)[:count_limit]
                ) / sum(self._weights.values())
        if self._weights is not None:
            view["reward_protocol"] = {
                "schema": "space-env.goal-utility/1",
                "condition_weights": copy.deepcopy(self._weights),
                "score": "weighted completion fraction if safe; -2 after any latched declared safety violation",
                "reward": "change in score minus 0.1 per rejected native/dispatch batch",
                "success": "unchanged native all-goals success",
                "safety_scope": "declared native fuel, energy, health, keepout and unrecognised failure codes; not a flight safety shield",
            }
        if self.spec.acceptance is not None:
            view["task_objective"] = {
                "schema": "space-env.task-acceptance/1",
                **self.spec.acceptance.model_dump(mode="json"),
                "completion": "natural native termination, safe resources, threshold and all mandatory goals",
                "mandatory_miss_penalty": 1.0,
            }
            bound = acceptance_bound(
                self.spec.acceptance, self._weights, published_bound
            )
            if bound is not None:
                view["acceptance_bound"] = bound
            view["reward_protocol"]["task_failure"] = (
                "At native termination, subtract 1 if mandatory goals are missing, unless safety is already latched; episode call-budget truncation also costs 1."
            )
        view["binding"] = digest(view)
        return view

    def _current_dispatch_bound(self):
        if self._dispatch_bound is None:
            return None
        result = copy.deepcopy(self._dispatch_bound)
        future = (
            self._capacity
            if self._native._runtime.time_sec <= result["dispatch_epoch_sec"]
            else 0
        )
        result["maximum_completed_conditions"] = min(
            self.spec.actors, len(self._launched) + future
        )
        result["scope"] = (
            "Upper bound under current published dispatch rules; future released updates can change it"
        )
        return result

    def _release_dispatch_events(self):
        now = self._native._runtime.time_sec
        while (
            self._released_count < len(self._dispatch_events)
            and self._dispatch_events[self._released_count].at_sec <= now
        ):
            event = self._dispatch_events[self._released_count]
            if event.kind == "capacity":
                self._capacity = event.capacity
            else:
                catalogue = self._native.observe("@central")["catalogue"]
                for row in catalogue:
                    if (
                        row["id"] != "wait"
                        and row["commands"][0]["actor_id"] == event.actor
                    ):
                        self._windows[row["id"]] = subtract_interval(
                            self._windows[row["id"]], event.start_sec, event.end_sec
                        )
            self._dispatch_updates.append(
                event.model_dump(mode="json", exclude_none=True)
            )
            self._released_count += 1

    def _window_open(self, action_id, time_sec):
        return any(left <= time_sec < right for left, right in self._windows[action_id])

    def _apply(self, value):
        action = OrbitAction.model_validate(value)
        visible = {row["id"] for row in self.observe()["catalogue"]}
        if set(action.action_ids) - visible:
            raise ValueError("unknown or hidden orbit action")
        admission_reason = None
        selected = list(action.action_ids)
        actors = {
            row["id"]: row["commands"][0]["actor_id"]
            for row in self.observe()["catalogue"]
            if row["id"] != "wait"
        }
        if self._windows is not None:
            if len([name for name in selected if name != "wait"]) > self._capacity:
                admission_reason = "dispatch_capacity"
            elif any(
                name != "wait" and actors[name] in self._launched for name in selected
            ):
                admission_reason = "already_launched"
            elif any(
                name != "wait"
                and not self._window_open(name, self._native._runtime.time_sec)
                for name in selected
            ):
                admission_reason = "launch_window_closed"
        transition = self._native.step(
            {"@central": [] if admission_reason else selected},
            expected_revision=self._native.revision,
            idempotency_key="orbit-" + str(self.revision),
        )
        raw_native_reward = transition["reward"]
        self._raw_native_return += raw_native_reward
        if admission_reason:
            self._admission_rejections += 1
            transition["reward"] -= 0.1
            transition["feedback"] = {"accepted": False, "code": admission_reason}
        elif transition["feedback"]["accepted"] and self._windows is not None:
            self._launched.update(actors[name] for name in selected if name != "wait")
        self._return += transition["reward"]
        self._rejections += int(
            not admission_reason and not transition["feedback"]["accepted"]
        )
        self._done = transition["terminated"] or transition["truncated"]
        self._truncated = transition["truncated"]
        self._release_dispatch_events()
        breakdown = None
        if self._weights is not None:
            previous_score = self._reward_potential
            previous_base_score = self._utility_state["score"]
            self._utility_state = evaluate_utility(
                self._native.goal_status(),
                self._weights,
                previous_violations=self._utility_state["violations"],
            )
            self._reward_potential = self._utility_state["score"]
            mandatory_penalty = 0.0
            if self.spec.acceptance is not None:
                self._acceptance_state = evaluate_acceptance(
                    self.spec.acceptance,
                    self._native.goal_status(),
                    self._utility_state,
                    finished=self._done,
                    truncated=self._truncated,
                )
                if (
                    self._done
                    and (self._acceptance_state["missing_mandatory"] or self._truncated)
                    and not self._utility_state["violations"]
                ):
                    mandatory_penalty = -1.0
                    self._reward_potential += mandatory_penalty
            delta = self._reward_potential - previous_score
            penalty = -0.1 if not transition["feedback"]["accepted"] else 0.0
            transition["reward"] = delta + penalty
            breakdown = {
                "raw_native_reward": raw_native_reward,
                "utility_delta": self._utility_state["score"] - previous_base_score,
                "rejection_penalty": penalty,
                **(
                    {
                        "mandatory_penalty": mandatory_penalty
                        - (previous_score - previous_base_score)
                    }
                    if self.spec.acceptance is not None
                    else {}
                ),
            }
        self._agent_return += transition["reward"]
        response = {
            key: copy.deepcopy(transition[key])
            for key in ("reward", "elapsed_sec", "terminated", "truncated", "feedback")
        }
        if breakdown is not None:
            response["reward_breakdown"] = breakdown
        return response

    def metrics(self):
        goal = self._native.goal_status()
        return {
            "success": goal["success"],
            "terminal": self._done,
            "native_truncated": self._truncated,
            "goal": goal,
            "native_return": self._return,
            "native_rejections": self._rejections,
            "time_sec": self._native._runtime.time_sec,
            "actors": self.spec.actors,
            "continuous_collision_qualified": False,
            **(
                {
                    "task_acceptance": copy.deepcopy(self._acceptance_state),
                    "reward_potential": self._reward_potential,
                }
                if self.spec.acceptance is not None
                else {}
            ),
            **(
                {
                    "utility": {
                        **copy.deepcopy(self._utility_state),
                        "decision_return": self._agent_return,
                        "raw_native_return": self._raw_native_return,
                    }
                }
                if self._weights is not None
                else {}
            ),
            **(
                {"released_dispatch_updates": self._released_count}
                if self._dispatch_events
                else {}
            ),
            **(
                {
                    "dispatch_rejections": self._admission_rejections,
                    "launched_actors": sorted(self._launched),
                }
                if self._windows is not None
                else {}
            ),
        }

    def _witness(self):
        return {
            "native": self._native.checkpoint(compact=True),
            "metrics": self.metrics(),
            **(
                {
                    "windows": copy.deepcopy(self._windows),
                    "capacity": self._capacity,
                    "released_updates": copy.deepcopy(self._dispatch_updates),
                }
                if self._dispatch_events
                else {}
            ),
        }


def _endpoint(initial, mass, isp, mu, command, start, epoch, now):
    """Public-input coarse RK4, including a coast before finite thrust."""
    state = initial.copy()
    clock = now
    force = command["burn"]["thrust_n"]
    duration = command["duration_sec"]
    vector = np.array(command["burn"]["direction_gcrs"])
    flow = force / (isp * 9.80665)
    for stop, thrust in ((start, 0.0), (start + duration, force), (epoch, 0.0)):
        if stop <= clock:
            continue
        count = max(1, int(np.ceil((stop - clock) / 0.25)))
        dt = (stop - clock) / count

        def derivative(t, y, thrust=thrust):
            r = y[:3]
            remaining = mass - flow * max(0.0, min(t - start, duration))
            return np.concatenate(
                (y[3:], -mu * r / np.linalg.norm(r) ** 3 + thrust * vector / remaining)
            )

        for _ in range(count):
            k1 = derivative(clock, state)
            k2 = derivative(clock + dt / 2, state + dt * k1 / 2)
            k3 = derivative(clock + dt / 2, state + dt * k2 / 2)
            k4 = derivative(clock + dt, state + dt * k3)
            state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            clock += dt
        clock = stop
    return state


def orbit_policy(observation):
    """Public-state endpoint search over one burn and published start epochs.

    Windowed mode replans from current observations; no private schedule or
    native forecast API. Shared dispatch slots are a synthetic start constraint.
    """
    now = observation["time_sec"]
    windows = observation.get("launch_windows")
    if windows is None and now != 0:
        return {}
    targets = {c["actor"]: c for c in observation["objective"]["conditions"]}
    states = {
        e["id"]: e["values"] for e in observation["entities"] if e["kind"] == "actor"
    }
    options = {}
    for row, enabled in zip(
        observation["catalogue"], observation["action_mask"], strict=True
    ):
        if row["id"] == "wait" or (
            not enabled and row.get("reason") != "launch_window_closed"
        ):
            continue
        command = row["commands"][0]
        actor = command["actor_id"]
        if actor in observation.get("launched_actors", []):
            continue
        options.setdefault(actor, []).append((row["id"], command))
    ready = []
    for actor, choices in sorted(options.items()):
        if windows is None and len(choices) == 1:
            priority = (
                -observation.get("reward_protocol", {})
                .get("condition_weights", {})
                .get(targets[actor]["id"], 0)
            )
            mandatory = observation.get("task_objective", {}).get("mandatory", [])
            rank = 0 if not mandatory or targets[actor]["id"] in mandatory else 1
            ready.append((rank, priority, 0.0, choices[0][0]))
            continue
        current = states[actor]
        initial = np.array(
            [
                current[k]
                for k in ("r_x_m", "r_y_m", "r_z_m", "v_x_m_s", "v_y_m_s", "v_z_m_s")
            ]
        )
        target = targets[actor]
        scored = []
        for name, command in choices:
            starts = (
                [now]
                if windows is None
                else [
                    max(now, left)
                    for left, right in windows[name]
                    if max(now, left) < right
                ]
            )
            for start in starts:
                if start + command["duration_sec"] > target["at_sec"]:
                    continue
                state = _endpoint(
                    initial,
                    current["mass_kg"],
                    observation["dynamics"]["isp_s"][actor],
                    observation["dynamics"]["mu_m3_s2"],
                    command,
                    start,
                    target["at_sec"],
                    now,
                )
                scored.append(
                    (
                        float(np.linalg.norm(state[:3] - target["position_m"])),
                        start,
                        name,
                    )
                )
        if scored:
            error, start, name = min(scored)
            if abs(start - now) < 1e-8:
                priority = (
                    -observation.get("reward_protocol", {})
                    .get("condition_weights", {})
                    .get(target["id"], 0)
                )
                mandatory = observation.get("task_objective", {}).get("mandatory", [])
                rank = 0 if not mandatory or target["id"] in mandatory else 1
                ready.append((rank, priority, error, name))
    capacity = observation.get("dispatch_capacity", len(ready))
    return {"action_ids": [name for _, _, _, name in sorted(ready)[:capacity]]}
