"""Causal baselines and a bounded current-window combinatorial heuristic.

No policy in this module imports or calls a mission runtime or scenario oracle.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from space_env.planning.model import PlanningInput, PolicyConfig
from space_env.simulation.snapshot import digest
from space_env.tasks.observation import STATES

VERSION = "space-env.baseline/1"


def conflicts(left, right):
    return any(
        a == b and (x == y or "actor" in {x, y}) for a, x in left for b, y in right
    )


def _acked_releases(value, entities, owns):
    """Only owned, observed ground ACKs; never TX completion or future success."""
    result = {}
    active = value["active"]
    for action in value["observation"]["catalogue"]:
        actors = action.get("actors", [])
        if (
            action["kind"] != "cancel_actor"
            or not action["enabled"]
            or len(actors) != 1
        ):
            continue
        actor = actors[0]
        role = value["actions"][action["id"]]["role"]
        if actor not in value["roles"].get(role, {}).get("actors", []):
            continue
        controls = [c for c in active if c["actor"] == actor]
        if not controls or any(c["kind"] != "downlink" for c in controls):
            continue
        if any(c["kind"] == "relay" and c["destination"] == actor for c in active):
            continue
        acknowledged = True
        for control in controls:
            product = control.get("product_id", "")
            values = entities.get("product:" + product, {}).get("values", {})
            if (
                values.get("status_code") != STATES["delivered"]
                or values.get("remaining_bits") != 0.0
                or ("actor:" + actor, "product:" + product) not in owns
            ):
                acknowledged = False
                break
        if acknowledged:
            result[action["id"]] = actor
    return result


def _ready_ground_product(command, entities, owns, now):
    key = "product:" + command.get("product_id", "")
    if ("actor:" + command["actor_id"], key) not in owns:
        return False
    values = entities.get(key, {}).get("values", {})
    return (
        values.get("remaining_bits", 0.0) > 0
        and values.get("remaining_work_sec", 0.0) <= 0
        and values.get("ready_sec", 0.0) <= now
        and values.get("status_code")
        not in {
            STATES[k]
            for k in ("delivered", "cancelled", "expired", "failed", "await_ack")
        }
    )


class BaselinePolicy:
    def __init__(self, config=None):
        self.config = PolicyConfig.model_validate(config or {})
        self._rng = np.random.default_rng(self.config.seed)
        self._cooldown = {}
        self._route_cooldown = {}
        self._stalls = {}
        self._revision = None
        self._last_binding = None
        self._last_decision = None

    def checkpoint(self):
        body = {
            "schema": VERSION,
            "config": self.config.model_dump(mode="json"),
            "rng": copy.deepcopy(self._rng.bit_generator.state),
            "cooldown": dict(self._cooldown),
            "route_cooldown": dict(self._route_cooldown),
            "stalls": copy.deepcopy(self._stalls),
            "revision": self._revision,
            "last_binding": self._last_binding,
            "last_decision": copy.deepcopy(self._last_decision),
        }
        return {**body, "sha256": digest(body)}

    @classmethod
    def from_checkpoint(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "config",
            "rng",
            "cooldown",
            "route_cooldown",
            "stalls",
            "revision",
            "last_binding",
            "last_decision",
            "sha256",
        }:
            raise ValueError("invalid policy checkpoint fields")
        body = {k: v for k, v in value.items() if k != "sha256"}
        if value.get("schema") != VERSION or digest(body) != value.get("sha256"):
            raise ValueError("invalid policy checkpoint")
        p = cls(value["config"])
        p._rng.bit_generator.state = copy.deepcopy(value["rng"])
        p._cooldown, p._stalls = (
            copy.deepcopy(value["cooldown"]),
            copy.deepcopy(value["stalls"]),
        )
        p._route_cooldown = copy.deepcopy(value["route_cooldown"])
        p._revision, p._last_binding = value["revision"], value["last_binding"]
        p._last_decision = copy.deepcopy(value["last_decision"])
        if p.checkpoint() != value:
            raise ValueError("noncanonical policy checkpoint")
        return p

    def decide(self, public: PlanningInput):
        if not isinstance(public, PlanningInput):
            raise TypeError("policies accept only PlanningInput values")
        before = self.checkpoint()
        try:
            return self._decide(public)
        except Exception:
            restored = type(self).from_checkpoint(before)
            self.__dict__.update(restored.__dict__)
            raise

    def _decide(self, public):
        value, binding = public.view(), public.sha256
        obs = value["observation"]
        revision, now = obs["revision"], obs["time_sec"]
        if revision == self._revision:
            if binding != self._last_binding:
                raise ValueError("changed public input at the same policy revision")
            return copy.deepcopy(self._last_decision)
        if self._revision is not None and revision != self._revision + 1:
            raise ValueError("policy must observe consecutive task revisions")
        if (
            value["feedback"]
            and not value["feedback"]["accepted"]
            and self._last_decision
        ):
            for id in self._last_decision["action_ids"]:
                self._cooldown[id] = revision + self.config.cooldown_steps
        self._cooldown = {k: v for k, v in self._cooldown.items() if v > revision}
        entities = {e["key"]: e for e in obs["entities"]}
        by_index = {e["index"]: e for e in obs["entities"]}
        stalled_actors = set()
        seen = set()
        for e in obs["entities"]:
            if e["kind"] != "transfer":
                continue
            key, progress = e["key"], e["values"].get("progress", 0.0)
            seen.add(key)
            old = self._stalls.get(key, [None, 0])
            count = old[1] + 1 if old[0] == progress else 0
            self._stalls[key] = [progress, count]
            if count >= self.config.stall_steps and e["values"].get("status_code") in {
                STATES["capacity_blocked"],
                STATES["link_unavailable"],
            }:
                for r in obs["relations"]:
                    if r["kind"] == "source" and r["destination"] == e["index"]:
                        stalled_actors.add(by_index[r["source"]]["id"])
        self._stalls = {k: v for k, v in self._stalls.items() if k in seen}
        self._route_cooldown = {
            k: v for k, v in self._route_cooldown.items() if v > revision
        }
        for control in value["active"]:
            if control["kind"] == "relay" and control["actor"] in stalled_actors:
                key = digest([control["actor"], control["destination"]])
                self._route_cooldown[key] = revision + self.config.cooldown_steps
        owns = (
            {
                (
                    by_index.get(r["source"], {}).get("key"),
                    by_index.get(r["destination"], {}).get("key"),
                )
                for r in obs["relations"]
                if r["kind"] == "owns"
            }
            if self.config.reclaim_acked_downlinks
            else set()
        )
        releases = (
            _acked_releases(value, entities, owns)
            if self.config.reclaim_acked_downlinks
            else {}
        )
        release_actors = {
            actor for id, actor in releases.items() if id not in self._cooldown
        }
        candidates = []
        for a in sorted(obs["catalogue"], key=lambda a: a["id"]):
            commands = a.get("commands", [])
            conditional = (
                self.config.reclaim_acked_downlinks
                and not a["enabled"]
                and a.get("reason") == "actor_busy"
                and bool(commands)
                and all(
                    c["kind"] == "downlink"
                    and c["actor_id"] in release_actors
                    and _ready_ground_product(c, entities, owns, now)
                    for c in commands
                )
            )
            if (
                a["id"] == "wait"
                or not (a["enabled"] or conditional)
                or a["id"] in self._cooldown
            ):
                continue
            if a["kind"] == "submit" and not commands:
                continue
            if (
                a["kind"] == "cancel_actor"
                and not set(a.get("actors", [])) & stalled_actors
                and a["id"] not in releases
            ):
                continue
            if a["kind"] == "inject" and a.get("operator_event", {}).get(
                "type"
            ) not in {"spacecraft_recover", "station_recovered"}:
                continue
            info = value["actions"][a["id"]]
            claims = [tuple(c) for c in info["claims"]]
            products, stations = set(), []
            priorities, deadlines = [], []
            assumptions = [
                "single-action mask is not joint admission or future feasibility"
            ]
            groups = info.get("command_claims", [info["claims"]])
            if any(
                conflicts(groups[i], groups[j])
                for i in range(len(groups))
                for j in range(i)
            ):
                continue
            utility = 0.0
            possible = True
            for c in commands:
                kind, actor = c["kind"], c["actor_id"]
                refs = [
                    ("product", c.get("product_id")),
                    ("request", c.get("request_id")),
                ]
                if c.get("job_id"):
                    job = entities.get("job:" + c["job_id"])
                    if job:
                        refs += [
                            ("product", by_index[r["source"]]["id"])
                            for r in obs["relations"]
                            if r["kind"] == "requires"
                            and r["destination"] == job["index"]
                        ]
                for kind_ref, id_ref in refs:
                    e = entities.get(kind_ref + ":" + str(id_ref), {}).get("values", {})
                    if e:
                        priorities.append(e.get("priority", 1.0))
                        deadlines.append(
                            e.get("deadline_sec", now + self.config.horizon_sec)
                        )
                if c.get("product_id"):
                    products.add(c["product_id"])
                utility += {
                    "downlink": 100.0,
                    "relay": 40.0,
                    "transform": 30.0,
                    "processing": 30.0,
                    "imaging": 20.0,
                }.get(kind, 1.0)
                if kind == "relay":
                    dest = c["destination_id"]
                    if (
                        self.config.name == "route_aware_rolling"
                        and digest([actor, dest]) in self._route_cooldown
                    ):
                        possible = False
                    claims.append(
                        (dest, "actor")
                    )  # Conservative whole receiver exclusion.
                    receiver = entities.get("actor:" + dest, {}).get("values", {})
                    product = entities.get(
                        "product:" + c.get("product_id", ""), {}
                    ).get("values", {})
                    needed = product.get("size_bits")
                    free = (
                        receiver.get("capacity_bits", 0.0)
                        - receiver.get("stored_bits", 0.0)
                        - receiver.get("reserved_bits", 0.0)
                    )
                    if (
                        "capacity_bits" in receiver
                        and needed is not None
                        and free < needed - 1e-6
                    ):
                        possible = False
                    elif not receiver or needed is None:
                        assumptions.append(
                            "receiver memory unknown; relay remains conditional"
                        )
                    if receiver.get("safe"):
                        possible = False
                    utility += 10.0 * any(
                        r["kind"] == "contact"
                        and by_index[r["source"]]["key"] == "actor:" + dest
                        for r in obs["relations"]
                    )
                    assumptions.append(
                        "no future RF/TTL success or custody credit assumed"
                    )
                if kind == "downlink":
                    stations.append(c["destination_id"])
                    assumptions.append(
                        "ground completion requires native ACK; terminal booking/RF conditional"
                    )
                if (
                    entities.get("actor:" + actor, {}).get("values", {}).get("soc", 1.0)
                    <= 0.05
                ):
                    utility *= 0.1
            if conditional:
                assumptions.append(
                    "native admission conditional on selected observed-ACK cancellation"
                )
            if a["id"] in releases:
                assumptions.append(
                    "release only controls whose owned product has observed ground ACK"
                )
            if not commands:
                utility = 500.0
            priority = max(priorities, default=1.0)
            deadline = min(deadlines, default=now + self.config.horizon_sec)
            slack = (
                deadline - now - max((c["duration_sec"] for c in commands), default=0.0)
            )
            score = (
                utility
                * max(priority, 1e-6)
                / (1.0 + max(0.0, slack) / self.config.horizon_sec)
            )
            if self.config.name == "priority":
                score = priority
            elif self.config.name == "earliest_deadline":
                score = 1.0 / (1.0 + max(0.0, deadline - now))
            if possible or self.config.name != "route_aware_rolling":
                candidates.append(
                    {
                        "id": a["id"],
                        "role": info["role"],
                        **(
                            {
                                "release_actor": releases.get(a["id"]),
                                "cancel_actor": next(iter(a.get("actors", [])), None)
                                if a["kind"] == "cancel_actor"
                                else None,
                                "requires_release": sorted(
                                    {c["actor_id"] for c in commands}
                                )
                                if conditional
                                else [],
                            }
                            if self.config.reclaim_acked_downlinks
                            else {}
                        ),
                        "claims": claims,
                        "products": sorted(products),
                        "stations": stations,
                        "station_users": [
                            [c["destination_id"], c["actor_id"]]
                            for c in commands
                            if c["kind"] == "downlink"
                        ],
                        "score": score,
                        "deadline": deadline,
                        "priority": priority,
                        "assumptions": assumptions,
                    }
                )
        candidates.sort(
            key=lambda c: (not bool(c.get("release_actor")), -c["score"], c["id"])
        )
        omitted = max(0, len(candidates) - self.config.max_candidates)
        candidates = candidates[: self.config.max_candidates]
        occupied, station_users = [], {}
        for a in value["active"]:
            occupied += [tuple(c) for c in a["claims"]]
            if a["kind"] == "relay":
                occupied.append((a["destination"], "actor"))
            if a["kind"] == "downlink":
                station_users.setdefault(a["destination"], set()).add(a["actor"])

        def feasible(selected, c):
            if len(selected) >= value["limits"]["joint"]:
                return False
            if (
                sum(s["role"] == c["role"] for s in selected)
                >= value["roles"][c["role"]]["action_slots"]
            ):
                return False
            released = {s["release_actor"] for s in selected if s.get("release_actor")}
            if not set(c.get("requires_release", [])) <= released:
                return False
            if c.get("cancel_actor") and any(
                s.get("cancel_actor") == c["cancel_actor"] for s in selected
            ):
                return False
            active_controls = [a for a in value["active"] if a["actor"] not in released]
            effective_occupied = occupied
            if released:
                effective_occupied = [
                    tuple(claim) for a in active_controls for claim in a["claims"]
                ]
                effective_occupied += [
                    (a["destination"], "actor")
                    for a in active_controls
                    if a["kind"] == "relay"
                ]
            # Native TaskSession still rechecks the real state after cancellation.
            active_claims = (
                []
                if not next(a for a in obs["catalogue"] if a["id"] == c["id"]).get(
                    "commands"
                )
                else effective_occupied
            )
            if conflicts(
                c["claims"],
                active_claims
                + [
                    v
                    for s in selected
                    if not s.get("release_actor")
                    for v in s["claims"]
                ],
            ):
                return False
            if set(c["products"]) & {p for s in selected for p in s["products"]}:
                return False
            for station in set(c["stations"]):
                ground = value["stations"].get(station, {})
                rate_cap = ground.get("rate_cap_bps")
                if ground.get("outage") or (rate_cap is not None and rate_cap <= 0):
                    return False
                cap = ground.get("terminals")
                reserved = {b["actor"] for b in ground.get("bookings", [])}
                proposed = {a for st, a in c["station_users"] if st == station}
                used = (
                    reserved
                    | (station_users.get(station, set()) - released)
                    | proposed
                    | {
                        a
                        for s in selected
                        for st, a in s["station_users"]
                        if st == station
                    }
                )
                if ground.get("reservation_required") and proposed - reserved:
                    return False
                if cap is None or len(used) > cap:
                    return False
            return True

        selected, nodes, exhausted = [], 0, False
        if self.config.name != "wait":
            order = candidates[:]
            if self.config.name == "random_masked":
                self._rng.shuffle(order)
                order = order[: int(self._rng.integers(0, len(order) + 1))]
            if self.config.reclaim_acked_downlinks:
                order.sort(key=lambda c: not bool(c.get("release_actor")))
            for c in order:
                if feasible(selected, c):
                    selected.append(c)
            if self.config.name == "route_aware_rolling":
                best = selected[:]
                best_score = sum(c["score"] for c in best)
                stack = [(0, [], 0.0)]
                while stack and nodes < self.config.max_nodes:
                    i, current, score = stack.pop()
                    nodes += 1
                    key = tuple(sorted(c["id"] for c in current))
                    if score > best_score + 1e-12 or (
                        abs(score - best_score) <= 1e-12
                        and key < tuple(sorted(c["id"] for c in best))
                    ):
                        best, best_score = current[:], score
                    if i == len(candidates):
                        continue
                    stack.append((i + 1, current, score))
                    c = candidates[i]
                    if feasible(current, c):
                        stack.append((i + 1, current + [c], score + c["score"]))
                exhausted = bool(stack)
                selected = best
        dt = self.config.decision_sec or value["limits"]["decision_sec"]
        if dt > value["limits"]["max_decision_sec"]:
            raise ValueError("policy interval exceeds task maximum")
        decision = {
            "schema": "space-env.planning-decision/1",
            "action_ids": sorted(c["id"] for c in selected) or ["wait"],
            "advance_sec": dt,
            "reason": self.config.name,
            "search_nodes": nodes,
            "search_budget_exhausted": exhausted,
            "omitted_candidates": omitted,
            "estimated_utility": math.fsum(c["score"] for c in selected),
            "assumptions": sorted({a for c in selected for a in c["assumptions"]}),
            "scope": "current-window heuristic; replan each epoch; no physical feasibility/optimality guarantee",
        }
        self._revision, self._last_binding, self._last_decision = (
            revision,
            binding,
            copy.deepcopy(decision),
        )
        return decision
