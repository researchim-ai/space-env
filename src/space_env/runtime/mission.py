"""Revisioned command runtime with verified prefix checkpoints and explicit views."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from space_env.data_generators.portable_run import _differences, runtime_manifest
from space_env.runtime.adapters import make_backend
from space_env.runtime.schema import AccessPolicy, RuntimeCommand, ScenarioSpec
from space_env.simulation.owned import owned_copy
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


class RuntimeContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self):
        return {"ok": False, "error": {"code": self.code, "message": str(self)}}


def _synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


@runtime_checkable
class Runtime(Protocol):
    def describe_capabilities(self) -> dict: ...
    def observe(self, actor: str | None = None, *, view: str = "agent") -> dict: ...
    def submit(
        self, commands: list, *, expected_revision: int, idempotency_key: str
    ) -> dict: ...
    def advance(
        self, dt_sec: float, *, expected_revision: int, idempotency_key: str
    ) -> dict: ...
    def checkpoint(self) -> dict: ...
    def restore(self, payload: dict, *, expected_revision: int): ...
    def iter_events(self, cursor: int = 0, limit: int = 100) -> dict: ...


class MissionRuntime:
    def __init__(
        self,
        scenario: dict | ScenarioSpec,
        *,
        access: AccessPolicy | None = None,
        audit_mode: str = "full",
    ):
        if audit_mode not in {"full", "compact"}:
            raise ValueError("unknown runtime audit mode")
        self._audit_mode = audit_mode
        self._lock = RLock()
        self._original = copy.deepcopy(
            scenario.model_dump(mode="json")
            if isinstance(scenario, ScenarioSpec)
            else scenario
        )
        try:
            parsed = ScenarioSpec.model_validate(scenario)
            self._backend = make_backend(parsed)
        except (ValueError, KeyError) as exc:
            raise RuntimeContractError("invalid_scenario", str(exc)) from exc
        except ImportError as exc:
            raise RuntimeContractError("missing_dependency", str(exc)) from exc
        self._spec = parsed.model_copy(
            update={"definition": copy.deepcopy(self._backend.definition)}
        )
        missing = set(parsed.required_capabilities) - self._backend.capabilities
        if missing:
            raise RuntimeContractError(
                "unsupported_capability", f"missing capabilities: {sorted(missing)}"
            )
        self.access = AccessPolicy.model_validate(access or {})
        self.revision = 0
        self._commands: dict[str, dict] = {}
        self._operations: list[dict] = []
        self._cache: dict[str, dict] = {}
        self._runtime = runtime_manifest()
        self._runtime.update(
            mission_physics_profile=parsed.physics_profile,
            frame_contract="GCRS/ITRS"
            if parsed.backend.startswith("reference-")
            else "legacy numerical GCRF tag",
        )

    @property
    @_synchronized
    def time_sec(self):
        return self._backend.now()

    @property
    @_synchronized
    def horizon_sec(self):
        """Declared physical horizon; no outcome/report computation."""
        return self._backend.horizon()

    @_synchronized
    def command_status(self):
        """Owned current command records without computing native outcome metrics."""
        return copy.deepcopy(list(self._commands.values()))

    @_synchronized
    def fork(self):
        """Control-plane transaction fork; preserves backend alias/RNG contracts.

        This does not replay history or advance physics. Agent tools do not expose
        the fork or its truth/control-plane checkpoint permissions.
        """
        self._require_truth()
        result = object.__new__(type(self))
        values = copy.deepcopy(
            {
                k: v
                for k, v in self.__dict__.items()
                if k not in {"_lock", "_backend", "_operations", "_cache"}
            }
        )
        result.__dict__.update(values)
        # Committed evidence is append-only. Public readers return copies.
        result._operations = self._operations.copy()
        result._cache = self._cache.copy()
        result._lock = RLock()
        result._backend = copy.copy(self._backend)
        result._backend.restore(self._backend.capture())
        return result

    @_synchronized
    def command_claims(self, command):
        """Declared resource ownership, not state-dependent command admission."""
        c = RuntimeCommand.model_validate(command)
        if c.actor_id not in self._spec.actor_order:
            raise RuntimeContractError("unknown_actor", "unknown actor")
        claims = getattr(self._backend, "command_claims", lambda c: ("actor",))
        return {"actor_id": c.actor_id, "resources": list(claims(c))}

    @_synchronized
    def request_authority(self, request_id):
        """Operator-only ownership of currently declared demand, even pre-release.

        This is not a future-event query and is never part of actor observations.
        """
        if not self.access.truth:
            raise RuntimeContractError(
                "permission_denied", "operator demand authority required"
            )
        if not isinstance(request_id, str) or not request_id.strip():
            raise RuntimeContractError(
                "invalid_request", "request_id must be a nonblank string"
            )
        lifecycle = getattr(self._backend, "lifecycle", None)
        request = lifecycle.requests.get(request_id) if lifecycle is not None else None
        if request is None:
            return None
        return {
            "eligible_actors": list(request.eligible_actors),
            "destination_id": request.destination_id,
        }

    @_synchronized
    def inventory(self, actor):
        """Current locally owned reference inventory; no future state query."""
        if actor not in self._spec.actor_order:
            raise RuntimeContractError("unknown_actor", "unknown actor")
        mission = getattr(self._backend, "mission", None)
        if mission is None:
            raise RuntimeContractError(
                "unsupported_capability", "reference inventory required"
            )
        life = getattr(self._backend, "lifecycle", None)
        return {
            "actor_id": actor,
            "time_sec": self.time_sec,
            "memory_capacity_bits": mission.vehicles[actor].memory_bits,
            "battery_wh": mission.vehicles[actor].battery_wh,
            "transmitter": {
                "ready": mission.transmitter_ready(actor),
                "seconds": mission.tx_seconds[actor],
                "temperature_k": mission.temperature[actor]
                if mission.vehicles[actor].thermal is not None
                else None,
            },
            "products": [
                {
                    **p.model_dump(mode="json"),
                    "remaining_bits": mission.remaining[p.id],
                    "processing_remaining_sec": mission.processing_remaining[p.id],
                    "cancelled": p.id in mission.cancelled,
                    "acknowledged": mission.acknowledged_time(p.id) is not None,
                    "custody_pinned": mission.dtn is not None
                    and mission.dtn.pinned(p.id),
                    "deliverable": life is None or life.deliverable(p.id),
                    "retained": bool(mission.retained_for(p.id))
                    or (life is not None and bool(life.retained(p.id))),
                }
                for p in mission.products.values()
                if p.actor == actor
            ],
        }

    @_synchronized
    def pointing_observation(self, actor, *, target_id=None, station_id=None):
        """Ideal current pointing error; no visibility or future success promise."""
        if actor not in self._spec.actor_order:
            raise RuntimeContractError("unknown_actor", "unknown actor")
        from space_env.core.attitude import boresight
        from space_env.reference.pointing import resolve_slew_goal
        from space_env.reference.spacecraft_bus import angle_deg

        try:
            command = RuntimeCommand.model_validate(
                {
                    "command_id": "pointing-read",
                    "actor_id": actor,
                    "kind": "slew",
                    "duration_sec": 1.0,
                    "slew": {
                        "frame": "GCRS",
                        "target_id": target_id,
                        "station_id": station_id,
                    },
                }
            )
            goal = resolve_slew_goal(self._backend, command)
            state = self._backend.bus.states[actor]
        except (ValueError, AttributeError, KeyError) as exc:
            raise RuntimeContractError("invalid_request", str(exc)) from exc
        geometry = None
        if target_id is not None and actor in self._backend.config.acquisition_sensors:
            from space_env.reference.acquisition import acquisition_geometry

            geometry = acquisition_geometry(
                self._backend.mission.ephemerides[actor],
                self._backend.acquisition_targets[target_id],
                self._backend.config.acquisition_sensors[actor],
                self.time_sec,
            )
        rigid = self._backend.bus.rigid
        body_rate = (
            math.hypot(*rigid.states[actor].omega_body_rad_s)
            if actor in rigid.models
            else None
        )
        return {
            "body_rate_rad_s": body_rate,
            "payload_max_body_rate_rad_s": (
                rigid.models[actor].payload_max_body_rate_rad_s
                if actor in rigid.models
                else None
            ),
            "geometry": geometry,
            "payload_tolerance_deg": self._backend.bus.models[
                actor
            ].attitude.payload_tolerance_deg,
            "actor_id": actor,
            "time_sec": self.time_sec,
            "target_id": target_id,
            "station_id": station_id,
            "goal_quaternion_wxyz": goal,
            "boresight_error_deg": angle_deg(
                boresight(state["quaternion_wxyz"]), boresight(goal)
            ),
            "settle_remaining_sec": state["settle_remaining_sec"],
            "scope": "current ideal geometry; future dwell not evaluated",
        }

    @_synchronized
    def station_observation(self, station):
        """Current ideal operator telemetry; scheduled future outages are omitted."""
        mission = getattr(self._backend, "mission", None)
        if mission is None or station not in mission.stations:
            raise RuntimeContractError("invalid_request", "unknown reference station")
        return {
            "id": station,
            "time_sec": self.time_sec,
            "terminals": mission.stations[station].terminals,
            "outage": any(
                o.station == station and o.start_sec <= self.time_sec < o.end_sec
                for o in mission.outages
            ),
            "reservation_required": mission.reservation_required[station],
            "rate_cap_bps": mission.station_rates[station],
            "bookings": [
                {"id": b.id, "actor": b.actor, "end_sec": b.end_sec}
                for b in sorted(mission.bookings.values(), key=lambda b: b.id)
                if b.station == station
                and b.start_sec
                <= self.time_sec
                < min(b.end_sec, mission.booking_cancelled.get(b.id, b.end_sec))
            ],
            "external_revisions": {
                k: {"revision": v["revision"], "status": v["record"]["status"]}
                for k, v in sorted(mission.external_reservations.items())
                if v["record"]["station"] == station
            },
        }

    def describe_capabilities(self):
        reference = self._spec.backend.startswith("reference-")
        return {
            "schema": "space-env.runtime-capabilities/1",
            "backend": self._spec.backend,
            "physics_profile": self._spec.physics_profile,
            "frame": "GCRS" if reference else "legacy-numerical-GCRF-tag",
            "time_contract": "elapsed seconds in declared backend; reference uses two-part TAI epoch",
            "actors": list(self._spec.actor_order),
            "supported": sorted(self._backend.capabilities),
            "fidelity": self._spec.fidelity.model_dump(),
            "access": self.access.model_dump(),
            "observation": self._spec.observation,
            "seed": self._spec.seed,
            "geometry_revision": getattr(self._backend, "geometry_revision", 0),
            "navigation_actors": list(getattr(self._backend, "navigation_actors", [])),
            "acquisition_actors": list(
                getattr(self._backend, "acquisition_actors", [])
            ),
            "seed_role": "experiment metadata; backend RNG state embedded",
            "limits": self._spec.limits.model_dump(),
            "concurrency": getattr(
                self._backend, "concurrency", "one active control window per actor"
            ),
            "checkpoint_strategy": "verified-prefix-replay/1",
        }

    @_synchronized
    def observe(self, actor=None, *, view="agent"):
        if actor is not None and actor not in self._spec.actor_order:
            raise RuntimeContractError("unknown_actor", "unknown actor")
        if view == "truth":
            self._require_truth()
            if actor is not None:
                raise RuntimeContractError(
                    "unsupported_view", "truth is a control-plane aggregate"
                )
            return {
                "revision": self.revision,
                "time_sec": self.time_sec,
                "backend": self._backend.truth(),
            }
        if view != "agent":
            raise RuntimeContractError(
                "unsupported_view", "view must be agent or truth"
            )
        names = [actor] if actor is not None else self._spec.actor_order
        return {
            "schema": "space-env.runtime-observation/1",
            "revision": self.revision,
            "time_sec": self.time_sec,
            "profile": self._spec.physics_profile,
            "view": "ideal-local/1",
            "actors": {a: owned_copy(self._observe_backend(a)) for a in names},
        }

    def _observe_backend(self, actor):
        # Called only by locked observe(), followed immediately by an owned copy.
        reader = getattr(self._backend, "_observe_borrowed", self._backend.observe)
        return reader(actor)

    @_synchronized
    def _task_score(self):
        """Trainer-only scalar accounting; never an agent observation."""
        self._require_truth()
        reader = getattr(self._backend, "task_score", None)
        return None if reader is None else reader()

    @_synchronized
    def forecast(self, actor: str, station: str):
        if not self.access.forecast:
            raise RuntimeContractError(
                "permission_denied", "privileged forecast access is disabled"
            )
        if "forecast.geometry" not in self._backend.capabilities:
            raise RuntimeContractError(
                "unsupported_capability", "backend has no geometry forecast"
            )
        if actor not in self._spec.actor_order:
            raise RuntimeContractError("unknown_actor", "unknown actor")
        try:
            windows = self._backend.forecast(actor, station)
        except (ValueError, KeyError) as exc:
            raise RuntimeContractError("invalid_request", str(exc)) from exc
        return {
            "revision": self.revision,
            "geometry_revision": getattr(self._backend, "geometry_revision", 0),
            "binding": self._forecast_binding(actor),
            "scope": getattr(
                self._backend,
                "forecast_scope",
                "known geometry/resources; conditional on accepted maneuvers; no future operator events",
            ),
            "windows": windows,
        }

    def _forecast_binding(self, actor):
        if not hasattr(self._backend, "geometry_token"):
            raise RuntimeContractError(
                "unsupported_capability", "backend has no geometry binding"
            )
        return {
            "scenario_sha256": digest(self._spec.model_dump(mode="json")),
            "trajectory_sha256": self._backend.geometry_token(actor),
            "actor_id": actor,
            "revision": self.revision,
            "geometry_revision": getattr(self._backend, "geometry_revision", 0),
        }

    def _event_forecast(
        self, capability, method, actor, *, start_sec, end_sec, settings, **arguments
    ):
        if not self.access.forecast:
            raise RuntimeContractError(
                "permission_denied", "privileged forecast access is disabled"
            )
        if capability not in self._backend.capabilities:
            raise RuntimeContractError(
                "unsupported_capability", f"backend has no {capability}"
            )
        if actor not in self._spec.actor_order:
            raise RuntimeContractError("unknown_actor", "unknown actor")
        begin = self.time_sec if start_sec is None else start_sec
        end = self._backend.horizon() if end_sec is None else end_sec
        if (
            any(
                isinstance(t, bool)
                or not isinstance(t, (int, float))
                or not math.isfinite(t)
                for t in (begin, end)
            )
            or not 0 <= begin < end <= self._backend.horizon()
        ):
            raise RuntimeContractError(
                "invalid_request", "forecast interval must fit mission coverage"
            )
        try:
            result = getattr(self._backend, method)(
                actor, start_sec=begin, end_sec=end, settings=settings, **arguments
            )
        except (ValueError, KeyError) as exc:
            raise RuntimeContractError("invalid_request", str(exc)) from exc
        return {
            "revision": self.revision,
            "geometry_revision": getattr(self._backend, "geometry_revision", 0),
            "binding": self._forecast_binding(actor),
            "result": result,
            **(
                {"trajectory_scope": self._backend.forecast_scope}
                if hasattr(self._backend, "forecast_scope")
                else {}
            ),
        }

    @_synchronized
    def forecast_acquisition(
        self, actor, target_id, *, start_sec=None, end_sec=None, settings=None
    ):
        return self._event_forecast(
            "forecast.acquisition.geometry",
            "forecast_acquisition",
            actor,
            start_sec=start_sec,
            end_sec=end_sec,
            settings=settings,
            target_id=target_id,
        )

    @_synchronized
    def forecast_pointing(self, actor, target_id, at_sec):
        """Privileged future geometric goal, not attitude/admission prediction."""
        if not self.access.forecast:
            raise RuntimeContractError(
                "permission_denied", "privileged forecast access is disabled"
            )
        if "forecast.pointing.geometry" not in self._backend.capabilities:
            raise RuntimeContractError(
                "unsupported_capability", "backend has no pointing geometry forecast"
            )
        if (
            actor not in self._spec.actor_order
            or type(at_sec) not in (int, float)
            or not math.isfinite(at_sec)
            or not self.time_sec <= at_sec <= self.horizon_sec
        ):
            raise RuntimeContractError(
                "invalid_request",
                "pointing forecast requires known actor and future coverage",
            )
        from space_env.reference.spacecraft_bus import pointing_quaternion

        try:
            target = self._backend.acquisition_targets[target_id]
            provider = self._backend.mission.ephemerides[actor]
            goal = pointing_quaternion(provider, target, at_sec).tolist()
        except (KeyError, AttributeError, ValueError) as exc:
            raise RuntimeContractError(
                "invalid_request", "unknown or unsupported pointing target"
            ) from exc
        return {
            "binding": self._forecast_binding(actor),
            "actor_id": actor,
            "target_id": target_id,
            "at_sec": float(at_sec),
            "goal_quaternion_wxyz": goal,
            "scope": "future geometric GCRS snapshot; no attitude tracking, dwell, power or feasibility guarantee",
        }

    @_synchronized
    def forecast_eclipses(self, actor, *, start_sec=None, end_sec=None, settings=None):
        return self._event_forecast(
            "forecast.eclipse",
            "forecast_eclipses",
            actor,
            start_sec=start_sec,
            end_sec=end_sec,
            settings=settings,
        )

    def _require_truth(self):
        if not self.access.truth:
            raise RuntimeContractError(
                "permission_denied", "truth/control-plane artifact access is disabled"
            )

    @_synchronized
    def report(self):
        """Current control-plane status, no future trajectory or physical truth journal."""
        return {
            "schema": "space-env.runtime-report/1",
            "revision": self.revision,
            "time_sec": self.time_sec,
            "horizon_sec": self._backend.horizon(),
            "profile": self._spec.physics_profile,
            "objective": self._spec.objective,
            "metrics": self._backend.metrics(),
            "commands": copy.deepcopy(list(self._commands.values())),
        }

    @_synchronized
    def _mutation(self, op, arguments, expected_revision, key, apply):
        if not isinstance(key, str) or not key.strip():
            raise RuntimeContractError(
                "invalid_key", "nonempty idempotency key required"
            )
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise RuntimeContractError(
                "invalid_revision", "revision must be a nonnegative integer"
            )
        body = {"op": op, "arguments": copy.deepcopy(arguments)}
        sha = digest(body)
        if key in self._cache:
            if self._cache[key]["body_sha256"] != sha:
                raise RuntimeContractError(
                    "idempotency_conflict", "key already used for another body"
                )
            return copy.deepcopy(self._cache[key]["response"])
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision != self.revision
        ):
            raise RuntimeContractError(
                "stale_revision", "expected_revision differs from current revision"
            )
        if len(self._operations) >= self._spec.limits.max_operations:
            raise RuntimeContractError(
                "operation_budget", "runtime operation budget exhausted"
            )
        previous = (self.revision, copy.deepcopy(self._commands))
        backend_state = self._backend.capture()
        before = self.observe()
        try:
            effect = apply()
            self.revision += 1
            response = {
                "ok": True,
                "operation": op,
                "revision": self.revision,
                "time_sec": self.time_sec,
                "effect": effect,
            }
            row = {
                "request": body,
                "expected_revision": expected_revision,
                "idempotency_key": key,
                "observation": before,
                "response": response,
                "next_observation": self.observe(),
                "report": self.report(),
                "backend_report": self._backend.report(),
            }
            encoded = canonical_json(row)
            if self._audit_mode == "compact":
                row = self._compact_row(
                    row,
                    evidence_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                )
            del encoded
            committed = owned_copy(row)
            cached_response = copy.deepcopy(response)
            public_response = copy.deepcopy(response)
        except Exception as exc:
            self.revision, self._commands = previous
            self._backend.restore(backend_state)
            if isinstance(exc, (ValueError, KeyError)) and not isinstance(
                exc, RuntimeContractError
            ):
                raise RuntimeContractError("invalid_request", str(exc)) from exc
            raise
        self._operations.append(committed)
        self._cache[key] = {"body_sha256": sha, "response": cached_response}
        return public_response

    def submit(self, commands: list, *, expected_revision: int, idempotency_key: str):
        if not isinstance(commands, list) or not commands:
            raise RuntimeContractError(
                "invalid_command", "submit requires a nonempty command list"
            )
        try:
            parsed = [
                c if isinstance(c, RuntimeCommand) else RuntimeCommand.model_validate(c)
                for c in commands
            ]
        except ValidationError as exc:
            raise RuntimeContractError("invalid_command", str(exc)) from exc
        rows = [c.model_dump(mode="json") for c in parsed]

        def apply():
            if len(self._commands) + len(parsed) > self._spec.limits.max_commands:
                raise RuntimeContractError("command_budget", "command budget exhausted")
            ids = set()
            claims = getattr(self._backend, "command_claims", lambda c: ("actor",))
            occupied = {}
            for row in self._commands.values():
                if row["status"] in {"accepted", "running"}:
                    command = RuntimeCommand.model_validate(row["command"])
                    occupied.setdefault(command.actor_id, set()).update(claims(command))
            for c in parsed:
                if "commands." + c.kind not in self._backend.capabilities:
                    raise RuntimeContractError(
                        "unsupported_capability", f"unsupported command: {c.kind}"
                    )
                if c.actor_id not in self._spec.actor_order:
                    raise RuntimeContractError("unknown_actor", "unknown actor")
                if c.forecast is not None and c.forecast.model_dump(
                    mode="json"
                ) != self._forecast_binding(c.actor_id):
                    raise RuntimeContractError(
                        "stale_forecast",
                        "forecast binding differs from current mission/actor/trajectory/revision",
                    )
                if c.command_id in self._commands or c.command_id in ids:
                    raise RuntimeContractError(
                        "duplicate_command", "command ID already exists"
                    )
                requested = set(claims(c))
                if requested & occupied.get(c.actor_id, set()):
                    raise RuntimeContractError(
                        "actor_busy" if requested == {"actor"} else "resource_busy",
                        "actor has a conflicting active control window/resource lane",
                    )
                occupied.setdefault(c.actor_id, set()).update(requested)
                if self.time_sec + c.duration_sec > self._backend.horizon():
                    raise RuntimeContractError(
                        "horizon", "command window exceeds horizon"
                    )
                self._backend.validate_command(c)
                ids.add(c.command_id)
            if hasattr(self._backend, "validate_batch"):
                self._backend.validate_batch(
                    parsed,
                    [
                        r["command"]
                        for r in self._commands.values()
                        if r["status"] in {"accepted", "running"}
                    ],
                )
            for c in parsed:
                if hasattr(self._backend, "admit"):
                    self._backend.admit(c)
                self._commands[c.command_id] = {
                    "command": c.model_dump(mode="json"),
                    "status": "completed" if c.kind == "impulse" else "accepted",
                    "start_sec": self.time_sec,
                    "until_sec": self.time_sec + c.duration_sec,
                }
            return {
                "commands": [
                    copy.deepcopy(self._commands[c.command_id]) for c in parsed
                ]
            }

        return self._mutation(
            "submit", {"commands": rows}, expected_revision, idempotency_key, apply
        )

    def advance(self, dt_sec: float, *, expected_revision: int, idempotency_key: str):
        if (
            isinstance(dt_sec, bool)
            or not isinstance(dt_sec, (int, float))
            or not math.isfinite(dt_sec)
            or dt_sec <= 1e-9
        ):
            raise RuntimeContractError(
                "invalid_duration", "advance must exceed 1ns and be finite"
            )
        dt_sec = float(dt_sec)

        def apply():
            if (
                dt_sec > self._spec.limits.max_advance_sec
                or self.time_sec + dt_sec > self._backend.horizon()
            ):
                raise RuntimeContractError(
                    "horizon", "advance exceeds horizon or configured limit"
                )
            end = self.time_sec + dt_sec
            while self.time_sec < end - 1e-9:
                active = [
                    r
                    for r in self._commands.values()
                    if r["status"] in {"accepted", "running"}
                ]
                right = min([end, *[r["until_sec"] for r in active]])
                native = self._backend.advance(right - self.time_sec, active)
                for row in active:
                    status = native.get(row["command"]["command_id"])
                    if status in {"failed", "rejected", "cancelled", "completed"}:
                        row["status"] = status
                    else:
                        row["status"] = (
                            "completed"
                            if self.time_sec >= row["until_sec"] - 1e-9
                            else "running"
                        )
            return {
                "advanced_sec": dt_sec,
                "metrics": self._backend.metrics(),
                "commands": copy.deepcopy(list(self._commands.values())),
            }

        return self._mutation(
            "advance", {"dt_sec": dt_sec}, expected_revision, idempotency_key, apply
        )

    def cancel(self, command_id: str, *, expected_revision: int, idempotency_key: str):
        if not isinstance(command_id, str) or not command_id.strip():
            raise RuntimeContractError(
                "invalid_command", "nonempty command ID required"
            )

        def apply():
            row = self._commands.get(command_id)
            if row is None or row["status"] not in {"accepted", "running"}:
                raise RuntimeContractError(
                    "not_active", "command is unknown or no longer active"
                )
            self._backend.cancel(command_id)
            row["status"] = "cancelled"
            return {"command": copy.deepcopy(row)}

        return self._mutation(
            "cancel",
            {"command_id": command_id},
            expected_revision,
            idempotency_key,
            apply,
        )

    def inject(
        self, events: list[dict], *, expected_revision: int, idempotency_key: str
    ):
        events = copy.deepcopy(events)

        def apply():
            if "events.inject" not in self._backend.capabilities:
                raise RuntimeContractError(
                    "unsupported_capability", "backend has no causal event injection"
                )
            self._backend.inject(events)
            return {"event_ids": [e["id"] for e in events]}

        return self._mutation(
            "inject", {"events": events}, expected_revision, idempotency_key, apply
        )

    @_synchronized
    def iter_events(self, cursor=0, limit=100):
        self._require_truth()
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not 0 <= cursor <= len(self._operations)
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise RuntimeContractError(
                "invalid_cursor", "invalid journal cursor or page size"
            )
        end = min(cursor + limit, len(self._operations))
        return {
            "events": copy.deepcopy(self._operations[cursor:end]),
            "next_cursor": end,
            "has_more": end < len(self._operations),
        }

    @staticmethod
    def _compact_row(row, *, evidence_sha256=None):
        return {
            "request": row["request"],
            "expected_revision": row["expected_revision"],
            "idempotency_key": row["idempotency_key"],
            "response": row["response"],
            "evidence_sha256": digest(row)
            if evidence_sha256 is None
            else evidence_sha256,
        }

    def _artifact(self, compact=False):
        compact = compact or self._audit_mode == "compact"
        body = {
            "schema": "space-env.runtime-checkpoint/2"
            if compact
            else "space-env.runtime-checkpoint/1",
            "original_scenario": self._original,
            "effective_scenario": self._spec.model_dump(mode="json"),
            "runtime": self._runtime,
            "operations": [self._compact_row(r) for r in self._operations]
            if compact and self._audit_mode == "full"
            else self._operations,
            "report": self.report(),
            "backend_report": self._backend.report(),
        }
        return {**owned_copy(body), "sha256": digest(body)}

    @_synchronized
    def checkpoint(self, *, compact=False):
        if type(compact) is not bool:
            raise TypeError("compact must be boolean")
        self._require_truth()
        return self._artifact(compact)

    def save(self, path: str | Path, *, compact=False):
        payload = self.checkpoint(compact=compact)
        atomic_write(path, canonical_json(payload))
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict, *, access: AccessPolicy | None = None):
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "original_scenario",
            "effective_scenario",
            "runtime",
            "operations",
            "report",
            "backend_report",
            "sha256",
        }:
            raise RuntimeContractError(
                "corrupt_checkpoint", "invalid checkpoint fields"
            )
        body = {k: v for k, v in payload.items() if k != "sha256"}
        if payload.get("schema") not in {
            "space-env.runtime-checkpoint/1",
            "space-env.runtime-checkpoint/2",
        } or payload.get("sha256") != digest(body):
            raise RuntimeContractError(
                "corrupt_checkpoint", "unsupported or corrupt runtime checkpoint"
            )
        compact = payload["schema"] == "space-env.runtime-checkpoint/2"
        runtime = cls(
            payload["original_scenario"],
            access=access,
            audit_mode="compact" if compact else "full",
        )
        if runtime._runtime != payload["runtime"]:
            raise RuntimeContractError(
                "runtime_mismatch", "runtime/source/data versions differ"
            )
        if _differences(
            runtime._spec.model_dump(mode="json"), payload["effective_scenario"]
        ):
            raise RuntimeContractError(
                "scenario_mismatch", "effective scenario differs"
            )
        if (
            not isinstance(payload["operations"], list)
            or len(payload["operations"]) > runtime._spec.limits.max_operations
        ):
            raise RuntimeContractError("corrupt_checkpoint", "invalid operation count")
        for i, expected in enumerate(payload["operations"]):
            fields_expected = (
                {
                    "request",
                    "expected_revision",
                    "idempotency_key",
                    "response",
                    "evidence_sha256",
                }
                if compact
                else {
                    "request",
                    "expected_revision",
                    "idempotency_key",
                    "observation",
                    "response",
                    "next_observation",
                    "report",
                    "backend_report",
                }
            )
            if not isinstance(expected, dict) or set(expected) != fields_expected:
                raise RuntimeContractError("corrupt_checkpoint", "invalid journal row")
            req = expected["request"]
            fields = {
                "submit": {"commands"},
                "advance": {"dt_sec"},
                "cancel": {"command_id"},
                "inject": {"events"},
            }
            if (
                not isinstance(req, dict)
                or set(req) != {"op", "arguments"}
                or not isinstance(req["op"], str)
                or req["op"] not in fields
                or not isinstance(req["arguments"], dict)
                or set(req["arguments"]) != fields[req["op"]]
            ):
                raise RuntimeContractError(
                    "corrupt_checkpoint", "invalid journal request"
                )
            kwargs = {
                "expected_revision": expected["expected_revision"],
                "idempotency_key": expected["idempotency_key"],
            }
            if req["op"] not in {"submit", "advance", "cancel", "inject"}:
                raise RuntimeContractError(
                    "corrupt_checkpoint", "unsupported journal operation"
                )
            arguments = req["arguments"]
            if req["op"] == "submit":
                runtime.submit(**arguments, **kwargs)
            elif req["op"] == "advance":
                runtime.advance(**arguments, **kwargs)
            elif req["op"] == "cancel":
                runtime.cancel(**arguments, **kwargs)
            else:
                runtime.inject(**arguments, **kwargs)
            if len(runtime._operations) != i + 1 or _differences(
                runtime._operations[-1], expected
            ):
                raise RuntimeContractError(
                    "replay_mismatch", f"runtime transition {i} differs"
                )
        if _differences(runtime.report(), payload["report"]) or _differences(
            runtime._backend.report(), payload["backend_report"]
        ):
            raise RuntimeContractError("replay_mismatch", "runtime final state differs")
        return runtime

    @classmethod
    def load(cls, path: str | Path, *, access: AccessPolicy | None = None):
        return cls.from_checkpoint(json.loads(Path(path).read_text()), access=access)

    @_synchronized
    def restore(self, payload: dict, *, expected_revision: int):
        self._require_truth()
        if self.revision or self._operations:
            raise RuntimeContractError(
                "restore_requires_fresh_runtime",
                "load a checkpoint into a fresh runtime; live revision rewind is forbidden",
            )
        if isinstance(expected_revision, bool) or expected_revision != self.revision:
            raise RuntimeContractError("stale_revision", "restore revision differs")
        restored = self.from_checkpoint(payload, access=self.access)
        if restored._spec.model_dump(mode="json") != self._spec.model_dump(mode="json"):
            raise RuntimeContractError(
                "scenario_mismatch", "cannot restore another mission"
            )
        self.__dict__.update(
            {k: v for k, v in restored.__dict__.items() if k != "_lock"}
        )

    def agent(self, actor: str):
        if actor not in self._spec.actor_order:
            raise RuntimeContractError("unknown_actor", "unknown actor")
        return AgentView(self, actor)


class AgentView:
    """Tool-facing actor view; not a sandbox for untrusted Python code."""

    def __init__(self, runtime, actor):
        self._runtime, self.actor = runtime, actor

    def observe(self):
        return self._runtime.observe(self.actor)

    def submit(self, command, *, expected_revision, idempotency_key):
        c = RuntimeCommand.model_validate(command)
        if c.actor_id != self.actor:
            raise RuntimeContractError(
                "permission_denied", "agent cannot command another actor"
            )
        return self._runtime.submit(
            [c], expected_revision=expected_revision, idempotency_key=idempotency_key
        )

    def cancel(self, command_id, *, expected_revision, idempotency_key):
        row = self._runtime._commands.get(command_id)
        if row is None or row["command"]["actor_id"] != self.actor:
            raise RuntimeContractError(
                "permission_denied", "agent cannot cancel another actor's command"
            )
        return self._runtime.cancel(
            command_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        )
