"""Bounded local operator sessions on the same causal TaskSession contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import tempfile
import threading
import zipfile
from pathlib import Path

from pydantic import Field, model_validator

from space_env.api.operator_events import CursorConflict, EventPageRequest, page_history
from space_env.api.operator_storage import OperatorRepository, StorageConflict
from space_env.core.frames import ecef_to_geodetic, geodetic_to_ecef
from space_env.runtime.schema import StrictModel
from space_env.scaling.workloads import fleet_workload
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSession, TaskTools, VerifiedEpisode
from space_env.tasks.observation import STATES


class OperatorError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


class OpenRequest(StrictModel):
    preset: str | None = None
    scenario: dict | None = None
    task: dict | None = None

    @model_validator(mode="after")
    def choice(self):
        if self.preset is not None:
            if (
                self.preset not in {"fleet-demo", "relay-demo"}
                or self.scenario is not None
                or self.task is not None
            ):
                raise ValueError("choose a known preset or an explicit scenario/task")
        elif self.scenario is None or self.task is None:
            raise ValueError("scenario and task are both required")
        return self


class StepRequest(StrictModel):
    action_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    expected_revision: int = Field(ge=0, strict=True)
    idempotency_key: str = Field(min_length=1, max_length=128)
    advance_sec: float | None = Field(default=None, gt=1e-9, le=3600, strict=True)
    control_epoch: int | None = Field(default=None, ge=0, strict=True)


def _interactive_bounds(scenario, task):
    if (
        len(scenario.get("actor_order", [])) > 16
        or len(task.get("actions", [])) > 256
        or task.get("max_decisions", 256) > 256
        or task.get("max_entities", 256) > 4096
        or task.get("max_relations", 1024) > 16384
    ):
        raise ValueError(
            "interactive profile: <=16 actors, 256 actions/decisions, 4096 entities, 16384 relations"
        )


def token_digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class HandoverRequest(StrictModel):
    controller: str = Field(pattern=r"^(operator|agent)$")
    expected_revision: int = Field(ge=0, strict=True)
    expected_control_epoch: int = Field(ge=0, strict=True)
    idempotency_key: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=512)


class OperatorSession:
    def __init__(self, task_session):
        self.session = task_session
        self.id = secrets.token_urlsafe(18)
        self.lock = threading.RLock()
        self.closed = False
        self.poisoned = False
        self.repository = None
        self.generation = 0
        self.owned_revision = task_session.revision
        self.control = {"controller": "operator", "epoch": 0}
        self.control_events = []
        self.handover_cache = {}
        self.source_import = None
        self.planner_state = None
        self.identities = {}
        self.tokens = {}
        self.credentials = {}
        for name, role, write, admin in [
            ("operator", "@central", True, False),
            ("viewer", "@central", False, False),
            ("agent", "@central", True, False),
            ("admin", "@central", False, True),
            *[("role:" + r, r, True, False) for r in self.session._task.roles],
        ]:
            token = secrets.token_urlsafe(32)
            key = token_digest(token)
            self.tokens[key] = (role, write, admin)
            self.identities[key] = "agent" if name == "agent" else "operator"
            self.credentials[name] = token
        self.frames, self.events = [], []
        for row in self.session._transitions:
            choices = row["request"]["choices"]
            transition = row["transition"]
            self.events.append(
                {
                    "controller": "external_unknown",
                    "control_epoch": 0,
                    "revision": transition["revision"],
                    "time_sec": transition["next_observation"]["time_sec"],
                    "role": next(iter(choices)) if len(choices) == 1 else "@central",
                    "action_ids": [i for ids in choices.values() for i in ids],
                    "feedback": copy.deepcopy(transition["feedback"]),
                    "elapsed_sec": transition["elapsed_sec"],
                }
            )
        self._append_frame()

    def permission(self, token):
        if self.closed:
            raise OperatorError("not_found", "operator session is closed", 404)
        if not isinstance(token, str) or token_digest(token) not in self.tokens:
            raise OperatorError(
                "unauthorized", "valid session bearer token required", 401
            )
        return self.tokens[token_digest(token)]

    def _guard(self):
        if self.poisoned:
            raise OperatorError(
                "storage_uncertain",
                "restart/recover the durable store before continuing",
                503,
            )
        self.session._assert_runtime()
        if self.session.revision != self.owned_revision:
            raise OperatorError(
                "external_mutation",
                "native session changed outside its operator coordinator",
                409,
            )
        if self.repository is not None:
            try:
                saved = self.repository.load(self.id)
            except (OSError, sqlite3.Error, ValueError) as exc:
                raise OperatorError(
                    "storage_unavailable",
                    "operator repository is unavailable or corrupt",
                    503,
                ) from exc
            if saved is None or saved["generation"] != self.generation:
                raise OperatorError(
                    "storage_conflict",
                    "durable session changed; refresh the session handle",
                    409,
                )

    def _dump(self):
        return {
            "schema": "space-env.operator-snapshot/1",
            "checkpoint": self.session.checkpoint(compact=True),
            "frames": copy.deepcopy(self.frames),
            "events": copy.deepcopy(self.events),
            "control": copy.deepcopy(self.control),
            "control_events": copy.deepcopy(self.control_events),
            "handover_cache": copy.deepcopy(self.handover_cache),
            "source_import": copy.deepcopy(self.source_import),
            "planner_state": copy.deepcopy(self.planner_state),
            "tokens": copy.deepcopy(self.tokens),
            "identities": copy.deepcopy(self.identities),
            "closed": self.closed,
        }

    def _load_meta(self, body):
        for key in [
            "frames",
            "events",
            "control",
            "control_events",
            "handover_cache",
            "source_import",
            "tokens",
            "identities",
            "closed",
        ]:
            setattr(self, key, copy.deepcopy(body[key]))
        self.tokens = {k: tuple(v) for k, v in self.tokens.items()}
        self.planner_state = copy.deepcopy(body.get("planner_state"))
        self.owned_revision = self.session.revision

    def _persist(self, kind):
        body = self._dump()
        try:
            self.generation = self.repository.save(
                self.id, body, self.generation, kind, active=not self.closed
            )
        except (OSError, sqlite3.Error):
            # A lost commit acknowledgment is not permission to execute twice.
            try:
                saved = self.repository.load(self.id, include_closed=True)
                if (
                    saved
                    and saved["generation"] == self.generation + 1
                    and saved["sha256"] == digest(body)
                ):
                    self.generation = saved["generation"]
                    return
            except (OSError, sqlite3.Error, ValueError):
                self.poisoned = True
            raise

    def _apply(self, kind, operation):
        self._guard()
        before = self._dump() if self.repository is not None else None
        old_generation = self.generation
        try:
            result = operation()
            self.owned_revision = self.session.revision
            if self.repository is not None and digest(before) != digest(self._dump()):
                self._persist(kind)
            return result
        except Exception as exc:
            if before is not None:
                try:
                    self.session = TaskSession.from_checkpoint(before["checkpoint"])
                    self._load_meta(before)
                    self.generation = old_generation
                except Exception:  # noqa: BLE001 — uncertain rollback must fence all writes.
                    self.poisoned = True
            if isinstance(exc, StorageConflict):
                raise OperatorError("storage_conflict", str(exc), 409) from exc
            if isinstance(exc, (OSError, sqlite3.Error)):
                raise OperatorError(
                    "storage_unavailable",
                    "durable commit failed; step was not acknowledged",
                    503,
                ) from exc
            raise

    def handover(self, token, payload):
        with self.lock, self.session._lock:
            if not self.permission(token)[2]:
                raise OperatorError(
                    "forbidden", "admin capability required for control transfer", 403
                )
            request = HandoverRequest.model_validate(payload)
            if not request.reason.strip():
                raise ValueError("handover reason cannot be blank")
            key = token_digest(request.idempotency_key)
            body = request.model_dump(mode="json")
            if key in self.handover_cache:
                old = self.handover_cache[key]
                if old["request_sha256"] != digest(body):
                    raise OperatorError(
                        "idempotency_conflict", "handover key already used", 409
                    )
                return copy.deepcopy(old["response"])
            if (
                request.expected_revision != self.session.revision
                or request.expected_control_epoch != self.control["epoch"]
            ):
                raise OperatorError(
                    "stale_control", "revision or control epoch changed", 409
                )
            if len(self.control_events) >= 1024:
                raise ValueError("control journal budget exhausted")
            if request.controller == self.control["controller"]:
                raise ValueError("requested controller already owns the session")

            def apply():
                self.control = {
                    "controller": request.controller,
                    "epoch": self.control["epoch"] + 1,
                }
                event = {
                    "kind": "handover",
                    "revision": self.session.revision,
                    "time_sec": self.session._runtime.time_sec,
                    "controller": request.controller,
                    "epoch": self.control["epoch"],
                    "reason": request.reason,
                }
                self.control_events.append(event)
                result = {
                    "ok": True,
                    "revision": self.session.revision,
                    "control": copy.deepcopy(self.control),
                }
                self.handover_cache[key] = {
                    "request_sha256": digest(body),
                    "response": copy.deepcopy(result),
                }
                return result

            return self._apply("handover", apply)

    def archive(self, token):
        with self.lock, self.session._lock:
            self.checkpoint(token)
            body = self._dump()
            for key in ["tokens", "identities", "closed"]:
                body.pop(key)
            body["schema"] = "space-env.operator-archive/1"
            return {**body, "sha256": digest(body)}

    def _append_frame(self):
        self.frames.append(navigation_frame(self.session))
        self.frames = self.frames[-257:]

    def _visible_events(self, role):
        events = []
        for event in self.events:
            ids = [
                i
                for i in event["action_ids"]
                if i == "wait"
                or self.session._by_id[i].role == role
                or role == "@central"
            ]
            if role == "@central" or ids and event["role"] == role:
                events.append({**event, "action_ids": ids})
        return events

    def read_events(self, token, payload):
        with self.lock, self.session._lock:
            self._guard()
            role, _, _ = self.permission(token)
            request = EventPageRequest.model_validate(payload)
            if request.channel == "journal":
                ordered = [
                    ((event["revision"], 0, 0), {"kind": "decision", "event": event})
                    for event in self._visible_events(role)
                ]
                ordered += [
                    (
                        (event["revision"], 1, event["epoch"]),
                        {"kind": "control", "event": event},
                    )
                    for event in self.control_events
                ]
                entries = [
                    value for _, value in sorted(ordered, key=lambda item: item[0])
                ]
            else:
                allowed = (
                    set(self.session._scenario["actor_order"])
                    if role == "@central"
                    else set(self.session._task.roles[role].actors)
                )
                entries = [
                    {
                        "kind": "navigation",
                        "frame": {
                            **frame,
                            "actors": {
                                actor: value
                                for actor, value in frame["actors"].items()
                                if actor in allowed
                            },
                        },
                    }
                    for frame in self.frames
                ]
            try:
                return page_history(
                    entries,
                    session_id=self.id,
                    token=token,
                    request=request,
                    revision=self.session.revision,
                    control_epoch=self.control["epoch"],
                )
            except CursorConflict as exc:
                raise OperatorError("cursor_conflict", str(exc), 409) from exc

    def state(self, token):
        with self.lock, self.session._lock:
            self._guard()
            role, write, admin = self.permission(token)
            observation = self.session.observe(role)
            from space_env.api.workspace import describe

            workspace = (
                describe(self)
                if role == "@central"
                and self.identities[token_digest(token)] == "operator"
                else None
            )
            from space_env.api.network_workspace import describe as describe_network

            network = (
                describe_network(self)
                if role == "@central"
                and self.identities[token_digest(token)] == "operator"
                else None
            )
            allowed = (
                set(self.session._scenario["actor_order"])
                if role == "@central"
                else set(self.session._task.roles[role].actors)
            )
            frames = [
                {**f, "actors": {a: v for a, v in f["actors"].items() if a in allowed}}
                for f in self.frames
            ]
            events = self._visible_events(role)
            definition = self.session._scenario["definition"]
            stations = definition.get("operations", definition)["stations"]
            visible = {
                e["id"] for e in observation["entities"] if e["kind"] == "station"
            }
            return copy.deepcopy(
                {
                    **({"workspace": workspace} if workspace is not None else {}),
                    **({"network": network} if network is not None else {}),
                    **(
                        {
                            "rolling_planner": {
                                "active": self.planner_state is not None,
                                "revision": self.planner_state["policy"]["revision"]
                                if self.planner_state
                                else None,
                                "last_choice_executed": self.planner_state["executed"]
                                if self.planner_state
                                else False,
                                "scope": "current public observations only; durable history, no forecast",
                            }
                        }
                        if role == "@central"
                        and self.identities[token_digest(token)] == "operator"
                        else {}
                    ),
                    "schema": "space-env.operator-state/1",
                    "physics_profile": self.session._scenario["physics_profile"],
                    "task_id": self.session._task.id,
                    "action_slots": self.session._task.max_joint_actions
                    if role == "@central"
                    else self.session._task.roles[role].action_slots,
                    "status_labels": {str(v): k for k, v in STATES.items()},
                    "session_id": self.id,
                    "role": role,
                    "permissions": {
                        "step": write
                        and self.identities[token_digest(token)]
                        == self.control["controller"],
                        "admin": admin,
                    },
                    "control": copy.deepcopy(self.control),
                    "control_events": copy.deepcopy(self.control_events),
                    "durable": self.repository is not None,
                    "generation": self.generation,
                    "revision": self.session.revision,
                    "time_sec": self.session._runtime.time_sec,
                    "horizon_sec": self.session._runtime.horizon_sec,
                    "decision_sec": self.session._task.decision_sec,
                    "max_decision_sec": self.session._task.max_decision_sec,
                    "done": self.session._done,
                    "observation": observation,
                    "events": events,
                    "frames": frames,
                    "stations": [
                        {
                            "name": s["name"],
                            "lat_deg": s["lat_deg"],
                            "lon_deg": s["lon_deg"],
                            "r_itrs_m": geodetic_to_ecef(
                                s["lat_deg"], s["lon_deg"], s.get("alt_m", 0.0)
                            ).tolist(),
                        }
                        for s in stations
                        if role == "@central" or s["name"] in visible
                    ],
                    "scope": "ideal current navigation; exact reference GCRS→ITRS; past sampled frames only; no estimator/forecast/flight qualification",
                }
            )

    def step(self, token, payload):
        with self.lock, self.session._lock:
            return self._apply("step", lambda: self._step(token, payload))

    def _step(self, token, payload):
        with self.lock, self.session._lock:
            role, write, _ = self.permission(token)
            if not write:
                raise OperatorError(
                    "forbidden", "this token cannot advance the mission", 403
                )
            request = StepRequest.model_validate(payload)
            identity = self.identities[token_digest(token)]
            if identity != self.control["controller"]:
                raise OperatorError(
                    "not_controller",
                    "handover is required before this controller can act",
                    409,
                )
            epoch = request.control_epoch
            if (
                (epoch is None and self.control["epoch"] != 0)
                or epoch is not None
                and epoch != self.control["epoch"]
            ):
                raise OperatorError(
                    "stale_control", "current control_epoch is required", 409
                )
            for id in request.action_ids:
                if id != "wait" and id not in self.session._by_id:
                    raise ValueError("unknown task action")
                if (
                    id != "wait"
                    and (identity != "operator" or role != "@central")
                    and self.session._by_id[id].kind == "inject"
                ):
                    raise OperatorError(
                        "forbidden", "operator events are not agent actions", 403
                    )
                if (
                    id != "wait"
                    and role != "@central"
                    and self.session._by_id[id].role != role
                ):
                    raise OperatorError("forbidden", "action is outside this role", 403)
            previous = self.session.revision
            # Stage policy state separately: a rejected/stale API call must not
            # mutate its history. Actual native rejection is recorded normally.
            from space_env.api import rolling_planner

            policy = (
                rolling_planner.stage(self)[0]
                if self.planner_state is not None
                else None
            )
            arguments = request.model_dump(exclude={"control_epoch"})
            if self.control["epoch"] or role != "@central":
                arguments["idempotency_key"] = "control/" + digest(
                    {
                        "epoch": self.control["epoch"],
                        "identity": identity,
                        "role": role,
                        "key": request.idempotency_key,
                    }
                )
            result = TaskTools(self.session, role).step(**arguments)
            if self.session.revision != previous:
                if policy is not None:
                    rolling_planner.save(
                        self,
                        policy,
                        executed=True,
                        request=self.session._transitions[-1]["request"],
                    )
                self.events.append(
                    {
                        "controller": identity,
                        "control_epoch": self.control["epoch"],
                        "revision": self.session.revision,
                        "time_sec": self.session._runtime.time_sec,
                        "role": role,
                        "action_ids": list(request.action_ids),
                        "feedback": result["feedback"],
                        "elapsed_sec": result["elapsed_sec"],
                    }
                )
                self._append_frame()
            return result

    def suggest(self, token):
        with self.lock, self.session._lock:
            self._guard()
            role, write, _ = self.permission(token)
            if (
                role != "@central"
                or not write
                or self.identities[token_digest(token)] != "operator"
            ):
                raise OperatorError(
                    "forbidden",
                    "suggestions require central operator capability; agent channel cannot query operator plans",
                    403,
                )
            if self.session._done:
                raise OperatorError("episode_done", "mission has ended", 409)
            from space_env.api.network_workspace import suggest as suggest_network
            from space_env.api.workspace import suggest

            network = suggest_network(self)
            if network is not None:
                return network

            planned = suggest(self)
            if planned is not None:
                return planned
            from space_env.api import rolling_planner

            def propose():
                policy, decision = rolling_planner.stage(self)
                rolling_planner.save(self, policy)
                return decision

            return self._apply("rolling_suggest", propose)

    def checkpoint(self, token):
        with self.lock, self.session._lock:
            self._guard()
            if not self.permission(token)[2]:
                raise OperatorError(
                    "forbidden", "control-plane artifact requires admin token", 403
                )
            return self.session.checkpoint(compact=True)

    def verify(self, token):
        with self.lock:
            record = self.checkpoint(token)
            restored = TaskSession.from_checkpoint(record)
            return {
                "ok": True,
                "episode_sha256": record["sha256"],
                "decisions": restored.revision,
            }

    def export(self, token):
        # Compatibility endpoint retains its 16 MiB bound and bytes result.
        with tempfile.TemporaryDirectory(prefix="operator-export-") as directory:
            path = Path(directory) / "training.zip"
            self.export_to(token, path, max_bytes=16 * 1024 * 1024)
            return path.read_bytes()

    def export_to(self, token, output, *, max_bytes=32 * 1024 * 1024):
        from space_env.api.operator_artifacts import BoundedWriter

        if type(max_bytes) is not int or not 1 <= max_bytes <= 32 * 1024 * 1024:
            raise ValueError("invalid operator export budget")
        output = Path(output)
        with self.lock:
            verified = VerifiedEpisode(self.checkpoint(token))
            with tempfile.TemporaryDirectory(
                prefix="operator-export-", dir=output.parent
            ) as temp:
                path = Path(temp) / "bundle"
                verified.export_bundle(path, max_bytes=max_bytes)
                audit = {
                    "schema": "space-env.operator-provenance/1",
                    "episode_sha256": self.session.checkpoint(compact=True)["sha256"],
                    "source_import": self.source_import,
                    "steps": [
                        {
                            k: e[k]
                            for k in ["revision", "controller", "control_epoch", "role"]
                        }
                        for e in self.events
                    ],
                    "control_events": self.control_events,
                    "note": "capability identities, not verified humans or expert labels",
                }
                encoded = canonical_json(audit)
                if (
                    sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                    + len(encoded.encode())
                    > max_bytes
                ):
                    raise ValueError("operator export byte budget exceeded")
                (path / "operator-provenance.json").write_text(encoded)
                created = False
                try:
                    with output.open("xb") as file:
                        created = True
                        os.chmod(output, 0o600)
                        bounded = BoundedWriter(file, max_bytes)
                        with zipfile.ZipFile(bounded, "w") as archive:
                            for member in sorted(path.rglob("*")):
                                if member.is_file():
                                    info = zipfile.ZipInfo(
                                        member.relative_to(path).as_posix(),
                                        date_time=(1980, 1, 1, 0, 0, 0),
                                    )
                                    info.compress_type = zipfile.ZIP_DEFLATED
                                    with (
                                        member.open("rb") as source,
                                        archive.open(info, "w") as target,
                                    ):
                                        shutil.copyfileobj(source, target, length=65536)
                        file.flush()
                        os.fsync(file.fileno())
                except Exception:
                    if created:
                        output.unlink(missing_ok=True)
                    raise
        return output


def navigation_frame(session):
    runtime = session._runtime
    actors = {}
    for name, provider in runtime._backend.mission.ephemerides.items():
        r = provider.state(runtime.time_sec)[:3].tolist()
        itrs = provider.context.matrix(provider.epoch.shifted(runtime.time_sec)) @ r
        lat, lon, alt = ecef_to_geodetic(itrs)
        actors[name] = {
            "r_gcrs_m": r,
            "r_itrs_m": itrs.tolist(),
            "lat_deg": float(lat),
            "lon_deg": float(lon),
            "alt_m": float(alt),
        }
    return {
        "revision": session.revision,
        "time_sec": runtime.time_sec,
        "actors": actors,
    }


def operator_snapshot_keys():
    return {
        "schema",
        "checkpoint",
        "frames",
        "events",
        "control",
        "control_events",
        "handover_cache",
        "source_import",
        "planner_state",
        "tokens",
        "identities",
        "closed",
    }


def validate_snapshot(body):
    required = operator_snapshot_keys()
    if (
        not isinstance(body, dict)
        or set(body) not in (required, required - {"planner_state"})
        or body["schema"] != "space-env.operator-snapshot/1"
        or type(body["closed"]) is not bool
    ):
        raise ValueError("invalid durable operator snapshot")
    cp = body["checkpoint"]
    from space_env.api import rolling_planner

    rolling_planner.validate_shape(body.get("planner_state"), cp)
    _interactive_bounds(cp["scenario"], cp["task"])
    if cp["task"]["policy_origin"] != "external":
        raise ValueError("operator snapshot must retain external provenance")
    n = len(cp["transitions"])
    if len(body["events"]) != n or len(body["control_events"]) > 1024:
        raise ValueError("operator journal length mismatch")
    frames = body["frames"]
    if (
        not isinstance(frames, list)
        or not 1 <= len(frames) <= 257
        or frames[-1]["revision"] != n
    ):
        raise ValueError("invalid navigation history")
    revisions = [f["revision"] for f in frames]
    if any(
        type(i) is not int or not 0 <= i <= n for i in revisions
    ) or revisions != sorted(set(revisions)):
        raise ValueError("invalid navigation revisions")
    allowed = {"@central", *cp["task"]["roles"]}
    tokens = body["tokens"]
    identities = body["identities"]
    if not isinstance(tokens, dict) or set(tokens) != set(identities) or not tokens:
        raise ValueError("invalid capability hashes")
    for key, value in tokens.items():
        if (
            len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
            or len(value) != 3
        ):
            raise ValueError("invalid stored capability")
        role, write, admin = value
        if (
            role not in allowed
            or type(write) is not bool
            or type(admin) is not bool
            or admin
            and (write or role != "@central")
            or identities[key] not in {"operator", "agent"}
        ):
            raise ValueError("invalid capability permissions")
        if identities[key] == "agent" and (role != "@central" or not write or admin):
            raise ValueError("invalid agent capability")
    if not any(v[2] for v in tokens.values()) or not any(
        identities[k] == "agent" for k in tokens
    ):
        raise ValueError("missing durable owner capabilities")
    control = body["control"]
    if (
        not isinstance(control, dict)
        or set(control) != {"controller", "epoch"}
        or control["controller"] not in {"operator", "agent"}
        or type(control["epoch"]) is not int
        or control["epoch"] != len(body["control_events"])
    ):
        raise ValueError("invalid control fence")
    imported = body["source_import"]
    if imported is not None and (
        not isinstance(imported, dict)
        or set(imported)
        != {
            "episode_sha256",
            "policy_origin",
            "revision",
            "takeover",
            "task",
            "compact",
        }
        or type(imported["revision"]) is not int
        or not 0 <= imported["revision"] <= n
        or type(imported["takeover"]) is not bool
        or type(imported["compact"]) is not bool
        or imported["policy_origin"] not in {"external", "scripted", "random"}
        or imported["task"]["policy_origin"] != imported["policy_origin"]
    ):
        raise ValueError("invalid imported-prefix provenance")
    owner = "operator"
    last_revision = -1
    times = {0: 0.0}
    for index, row in enumerate(cp["transitions"], 1):
        # Compact rows do not contain transitions; physical times checked after replay.
        if "transition" in row:
            times[index] = row["transition"]["next_observation"]["time_sec"]
    for i, event in enumerate(body["control_events"], 1):
        if (
            set(event)
            != {"kind", "revision", "time_sec", "controller", "epoch", "reason"}
            or type(event["epoch"]) is not int
            or event["epoch"] != i
            or type(event["revision"]) is not int
            or not last_revision <= event["revision"] <= n
            or not isinstance(event["reason"], str)
            or not event["reason"].strip()
        ):
            raise ValueError("invalid handover journal")
        if event["controller"] not in {"operator", "agent"} or event["kind"] not in {
            "handover",
            "recovery",
            "checkpoint_takeover",
        }:
            raise ValueError("invalid handover kind")
        if event["kind"] == "handover" and event["controller"] == owner:
            raise ValueError("handover did not change ownership")
        if event["kind"] == "recovery" and event["controller"] != owner:
            raise ValueError("recovery changed ownership")
        if event["kind"] == "checkpoint_takeover" and (
            i != 1
            or body["source_import"] is None
            or event["revision"] != body["source_import"]["revision"]
        ):
            raise ValueError("invalid checkpoint takeover fence")
        owner = event["controller"]
        last_revision = event["revision"]
    if control["controller"] != owner:
        raise ValueError("control head mismatch")
    for key, item in body["handover_cache"].items():
        response = item["response"]
        epoch = response["control"]["epoch"]
        if (
            len(key) != 64
            or len(item["request_sha256"]) != 64
            or type(epoch) is not int
            or not 1 <= epoch <= len(body["control_events"])
        ):
            raise ValueError("invalid handover receipt")
        event = body["control_events"][epoch - 1]
        if response != {
            "ok": True,
            "revision": event["revision"],
            "control": {"controller": event["controller"], "epoch": epoch},
        }:
            raise ValueError("handover receipt mismatch")


def validate_history(operator, body):
    from space_env.api import rolling_planner

    rolling_planner.validate_history(operator, body.get("planner_state"))
    session = operator.session
    times = {0: 0.0}
    for row, event in zip(session._transitions, body["events"], strict=True):
        transition = row["transition"]
        choices = row["request"]["choices"]
        expected = {
            "revision": transition["revision"],
            "time_sec": transition["next_observation"]["time_sec"],
            "role": next(iter(choices)) if len(choices) == 1 else "@central",
            "action_ids": [i for ids in choices.values() for i in ids],
            "feedback": transition["feedback"],
            "elapsed_sec": transition["elapsed_sec"],
        }
        if canonical_json(
            {k: v for k, v in event.items() if k not in {"controller", "control_epoch"}}
        ) != canonical_json(expected):
            raise ValueError("operator/native operation journal mismatch")
        times[expected["revision"]] = expected["time_sec"]
        imported = body["source_import"]
        if imported and expected["revision"] <= imported["revision"]:
            identity = "imported_agent" if imported["takeover"] else "external_unknown"
            epoch = 0
        else:
            eligible = [
                c
                for c in body["control_events"]
                if c["revision"] < expected["revision"]
            ]
            identity = eligible[-1]["controller"] if eligible else "operator"
            epoch = eligible[-1]["epoch"] if eligible else 0
        if (
            event["controller"] != identity
            or type(event["control_epoch"]) is not int
            or event["control_epoch"] != epoch
        ):
            raise ValueError("operator step provenance/fence mismatch")
    for event in body["control_events"]:
        if canonical_json(event["time_sec"]) != canonical_json(
            times[event["revision"]]
        ):
            raise ValueError("handover changed physical time")
    imported = body["source_import"]
    if imported:
        source = TaskSession(
            body["checkpoint"]["scenario"],
            imported["task"],
            seed=body["checkpoint"]["episode_seed"],
        )
        for row in session._transitions[: imported["revision"]]:
            source.step(**row["request"])
        if (
            source.checkpoint(compact=imported["compact"])["sha256"]
            != imported["episode_sha256"]
        ):
            raise ValueError("imported source checkpoint mismatch")
    indexed = {f["revision"]: f for f in body["frames"]}
    if len(indexed) == 1:
        if canonical_json(navigation_frame(session)) != canonical_json(
            body["frames"][-1]
        ):
            raise ValueError("current navigation mismatch")
        return
    cp = body["checkpoint"]
    probe = TaskSession(cp["scenario"], cp["task"], seed=cp["episode_seed"])
    if 0 in indexed and canonical_json(navigation_frame(probe)) != canonical_json(
        indexed[0]
    ):
        raise ValueError("initial navigation mismatch")
    for row in session._transitions:
        probe.step(**row["request"])
        if probe.revision in indexed and canonical_json(
            navigation_frame(probe)
        ) != canonical_json(indexed[probe.revision]):
            raise ValueError("historical navigation replay mismatch")


class OperatorStore:
    def __init__(self, path=None):
        self.sessions = {}
        self.lock = threading.RLock()
        self.repository = OperatorRepository(path) if path is not None else None
        if self.repository is not None:
            for id, saved in self.repository.active():
                self.sessions[id] = self._hydrate(id, saved)

    def _hydrate(self, id, saved):
        body = saved["state"]
        if body["closed"] == saved["active"]:
            raise ValueError("durable lifecycle mismatch")
        validate_snapshot(body)
        result = OperatorSession(TaskSession.from_checkpoint(body["checkpoint"]))
        validate_history(result, body)
        result.id = id
        result._load_meta(body)
        result.credentials = {}
        result.repository = self.repository
        result.generation = saved["generation"]
        return result

    def _install(self, operator, kind):
        operator.repository = self.repository
        if self.repository is not None:
            try:
                operator._persist(kind)
            except StorageConflict as exc:
                raise OperatorError("capacity", str(exc), 429) from exc
            except (OSError, sqlite3.Error) as exc:
                raise OperatorError(
                    "storage_unavailable", "session creation was not acknowledged", 503
                ) from exc
        self.sessions[operator.id] = operator
        return {
            "session_id": operator.id,
            "tokens": operator.credentials.copy(),
            "state": operator.state(operator.credentials["operator"]),
        }

    def create(self, payload):
        with self.lock:
            if self.repository is not None:
                for key in list(self.sessions):
                    if self.repository.generation(key) is None:
                        self.sessions[key].closed = True
                        del self.sessions[key]
            if len(self.sessions) >= 8:
                raise OperatorError(
                    "capacity", "close a session before opening another", 429
                )
            request = OpenRequest.model_validate(payload)
            if request.preset == "relay-demo":
                fixture = json.loads(
                    (
                        Path(__file__).resolve().parents[1]
                        / "configs"
                        / "operator_relay.json"
                    ).read_text()
                )
                scenario, task = fixture["scenario"], fixture["task"]
            elif request.preset:
                w = fleet_workload(
                    {
                        "id": "operator-demo",
                        "actors": 4,
                        "products": 12,
                        "planes": 2,
                        "horizon_sec": 600.0,
                        "decision_sec": 30.0,
                        "repeats": 1,
                    }
                )
                scenario, task = w["scenario"], w["task"]
                task["roles"] = {a: {"actors": [a]} for a in scenario["actor_order"]}
                for action in task["actions"]:
                    action["role"] = action["commands"][0]["actor_id"]
            else:
                scenario, task = request.scenario, copy.deepcopy(request.task)
            task.update(policy_origin="external", runtime_audit="compact")
            _interactive_bounds(scenario, task)
            operator = OperatorSession(TaskSession(scenario, task))
            return self._install(operator, "create")

    def restore(self, payload):
        with self.lock:
            if self.repository is not None:
                for key in list(self.sessions):
                    if self.repository.generation(key) is None:
                        self.sessions[key].closed = True
                        del self.sessions[key]
            if len(self.sessions) >= 8:
                raise OperatorError("capacity", "close a session first", 429)
            if set(payload) == {"archive"}:
                archive = copy.deepcopy(payload["archive"])
                if (
                    not isinstance(archive, dict)
                    or archive.get("schema") != "space-env.operator-archive/1"
                    or archive.get("sha256")
                    != digest({k: v for k, v in archive.items() if k != "sha256"})
                ):
                    raise ValueError("invalid operator archive")
                allowed_archive = set(operator_snapshot_keys()) - {
                    "tokens",
                    "identities",
                    "closed",
                } | {"sha256"}
                if set(archive) not in (
                    allowed_archive,
                    allowed_archive - {"planner_state"},
                ):
                    raise ValueError("invalid portable archive fields")
                archive.pop("sha256")
                archive["schema"] = "space-env.operator-snapshot/1"
                operator = OperatorSession(
                    TaskSession.from_checkpoint(archive["checkpoint"])
                )
                body = operator._dump()
                body.update(archive)
                validate_snapshot(body)
                validate_history(operator, body)
                operator._load_meta(body)
                return self._install(operator, "restore_archive")
            if set(payload) not in [{"record"}, {"record", "takeover"}]:
                raise ValueError(
                    "restore requires record, explicit takeover or archive"
                )
            record = payload["record"]
            if not isinstance(record, dict) or not isinstance(record.get("task"), dict):
                raise ValueError("invalid task checkpoint")  # noqa: TRY004 — stable restore contract
            if not isinstance(record.get("scenario"), dict):
                raise ValueError("task checkpoint scenario is required")  # noqa: TRY004 — stable restore validation
            _interactive_bounds(record["scenario"], record["task"])
            origin = record["task"].get("policy_origin")
            takeover = payload.get("takeover", False)
            if type(takeover) is not bool:
                raise ValueError("takeover must be boolean")
            if origin != "external" and not takeover:
                raise ValueError(
                    "operator restore requires an external-origin task checkpoint or explicit takeover"
                )
            restored = TaskSession.from_checkpoint(record)
            if takeover:
                task = restored._task.model_dump(mode="json", by_alias=True)
                task.update(policy_origin="external", runtime_audit="compact")
                converted = TaskSession(
                    record["scenario"], task, seed=record["episode_seed"]
                )
                for row in restored._transitions:
                    if converted.step(**row["request"]) != row["transition"]:
                        raise ValueError("takeover changed native transition")
                if (
                    converted._runtime.checkpoint()["backend_report"]
                    != restored._runtime.checkpoint()["backend_report"]
                ):
                    raise ValueError("takeover changed native outcome")
                restored = converted
            operator = OperatorSession(restored)
            operator.source_import = {
                "episode_sha256": record["sha256"],
                "policy_origin": origin,
                "revision": restored.revision,
                "takeover": takeover,
                "task": copy.deepcopy(record["task"]),
                "compact": record["schema"] == "space-env.task-episode/2",
            }
            for event in operator.events:
                event["controller"] = (
                    "imported_agent" if takeover else "external_unknown"
                )
            operator.control = {"controller": "operator", "epoch": 1}
            operator.control_events = [
                {
                    "kind": "checkpoint_takeover",
                    "revision": restored.revision,
                    "time_sec": restored._runtime.time_sec,
                    "epoch": 1,
                    "controller": "operator",
                    "reason": "explicit checkpoint import; original provenance retained",
                }
            ]
            return self._install(operator, "restore_task")

    def get(self, id):
        with self.lock:
            if id not in self.sessions:
                saved = (
                    self.repository.load(id) if self.repository is not None else None
                )
                if saved is None:
                    raise OperatorError("not_found", "no such operator session", 404)
                self.sessions[id] = self._hydrate(id, saved)
            current = self.sessions[id]
            if (
                self.repository is not None
                and self.repository.generation(id) != current.generation
            ):
                saved = self.repository.load(id)
                if saved is None:
                    current.closed = True
                    del self.sessions[id]
                    raise OperatorError("not_found", "operator session is closed", 404)
                current = self._hydrate(id, saved)
                self.sessions[id] = current
            return current

    def close(self, id, token):
        with self.lock:
            session = self.get(id)
            if not session.permission(token)[2]:
                raise OperatorError(
                    "forbidden", "admin token required to close session", 403
                )
            with session.lock, session.session._lock:
                session._apply("close", lambda: setattr(session, "closed", True))
                del self.sessions[id]
            return {"ok": True}

    def recover(self, id):
        with self.lock:
            current = self.get(id)
            with current.lock, current.session._lock:
                if len(current.control_events) >= 1024:
                    raise ValueError("control journal budget exhausted")
                old_credentials = current.credentials.copy()

                def rotate():
                    fresh = OperatorSession(current.session)
                    current.tokens = fresh.tokens
                    current.identities = fresh.identities
                    current.credentials = fresh.credentials
                    current.control = {
                        "controller": current.control["controller"],
                        "epoch": current.control["epoch"] + 1,
                    }
                    current.control_events.append(
                        {
                            "kind": "recovery",
                            "revision": current.session.revision,
                            "time_sec": current.session._runtime.time_sec,
                            "controller": current.control["controller"],
                            "epoch": current.control["epoch"],
                            "reason": "local OS-owner credential recovery",
                        }
                    )

                try:
                    current._apply("recover", rotate)
                except Exception:
                    current.credentials = old_credentials
                    raise
                return {
                    "session_id": id,
                    "tokens": current.credentials.copy(),
                    "state": current.state(current.credentials["operator"]),
                }


STORE = OperatorStore()


def dispatch(method, path, payload=None, token=None):
    if method == "GET" and path == "/api/operator/network/preset":
        from space_env.api.network_workspace import default_network_configuration

        return default_network_configuration()
    if method == "POST" and path == "/api/operator/network":
        from space_env.api.network_workspace import create_network

        return create_network(payload, STORE)
    if method == "GET" and path.split("?")[0] == "/api/operator/workspace/physics":
        from space_env.validation.profile_evidence import profile_contracts

        return profile_contracts()
    parts = path.strip("/").split("/")
    if method == "GET" and parts == ["api", "operator", "workspace", "presets"]:
        from space_env.api.workspace import default_configuration

        return {
            "configuration": default_configuration(),
            "limits": {"actors": 4, "requests": 64, "stations": 16, "decisions": 240},
        }
    if method == "POST" and parts == ["api", "operator", "workspace", "configuration"]:
        from space_env.api.workspace import parse_configuration

        return {
            "configuration": parse_configuration(payload).model_dump(
                mode="json", by_alias=True
            )
        }
    if method == "POST" and parts == ["api", "operator", "workspace"]:
        from space_env.api.workspace import create_mission

        return create_mission(payload, STORE)
    if parts == ["api", "operator", "catalogue"] and method == "GET":
        return {
            "durable": STORE.repository is not None,
            "active_sessions": len(STORE.sessions),
            "presets": [
                {
                    "id": "relay-demo",
                    "title": "EO → CPU → ISL → Ground ACK",
                    "scope": "synthetic three-actor coupled mission, 20 SI seconds",
                },
                {
                    "id": "fleet-demo",
                    "title": "4 satellites · 12 downlinks · 10 minutes",
                    "scope": "synthetic separate J2 arcs; fixed ephemeris resources",
                },
            ],
        }
    if method == "POST" and parts == ["api", "operator", "sessions"]:
        return STORE.create(payload)
    if method == "POST" and parts == ["api", "operator", "restore"]:
        return STORE.restore(payload)
    if len(parts) == 5 and parts[:3] == ["api", "operator", "transfers"]:
        from space_env.api.operator_artifacts import ARTIFACTS

        id, action = parts[3:]
        if method == "GET" and action == "status":
            return ARTIFACTS.status(STORE, id, token)
        if method == "GET" and action == "content":
            return ARTIFACTS.download(STORE, id, token)
        if method == "PUT" and action == "chunks":
            return ARTIFACTS.append(STORE, id, token, payload)
        if method == "POST" and action in {"commit", "cancel"}:
            if payload != {}:
                raise ValueError("transfer operation requires an empty JSON object")
            return getattr(ARTIFACTS, action)(STORE, id, token)
        raise OperatorError("not_found", "unknown transfer operation", 404)
    if len(parts) not in {4, 5} or parts[:3] != ["api", "operator", "sessions"]:
        raise OperatorError("not_found", "unknown operator route", 404)
    session = STORE.get(parts[3])
    session.permission(token)
    suffix = "" if len(parts) == 4 else parts[4]
    if method == "POST" and suffix == "replan":
        from space_env.api.workspace_replan import replan_workspace

        return replan_workspace(STORE, session.id, token, payload)
    if method == "POST" and suffix == "evolve":
        from space_env.api.workspace_dynamic import evolve_workspace

        if session.session._scenario.get("metadata", {}).get("network_workspace"):
            from space_env.api.network_dynamic import evolve_network

            evolve_workspace = evolve_network

        return evolve_workspace(STORE, session.id, token, payload)
    if method == "POST" and suffix == "round":
        from space_env.api.fleet_round import dispatch_round

        return dispatch_round(session, token, payload)
    if method == "POST" and suffix in {"verify", "export", "close"} and payload != {}:
        raise ValueError("this operation requires an empty JSON object")
    if method == "GET" and suffix == "":
        return session.state(token)
    if method == "GET" and suffix == "suggest":
        return session.suggest(token)
    if method == "GET" and suffix == "archive":
        return session.archive(token)
    if method == "POST" and suffix == "handover":
        return session.handover(token, payload)
    if method == "GET" and suffix == "checkpoint":
        return session.checkpoint(token)
    if method == "POST" and suffix == "artifacts":
        from space_env.api.operator_artifacts import ARTIFACTS

        return ARTIFACTS.create(STORE, session.id, token, payload)
    if method == "POST" and suffix == "events":
        return session.read_events(token, payload)
    if method == "POST" and suffix == "step":
        return session.step(token, payload)
    if method == "POST" and suffix == "verify":
        return session.verify(token)
    if method == "POST" and suffix == "export":
        return session.export(token)
    if method == "POST" and suffix == "close":
        return STORE.close(parts[3], token)
    raise OperatorError("not_found", "unknown operator operation", 404)
