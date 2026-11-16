"""JSON decisions and source-bound replay shared by three distinct executors.

Records/checkpoints contain privileged setup; only observe() is agent-facing.
These records are deliberately not passed off as native LLMEpisode records.
"""

from __future__ import annotations

import copy
import platform
from importlib.metadata import PackageNotFoundError, version
from threading import RLock

from pydantic import Field

from space_env.llm.episode import sources
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest


def environment_identity():
    packages = {}
    for name in ("space-env", "numpy", "pydantic", "scipy", "gymnasium"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


class Decision(StrictModel):
    expected_revision: int = Field(ge=0, strict=True)
    idempotency_key: str = Field(min_length=1, max_length=128)
    action: dict


class DecisionSession:
    kind = "abstract"

    def _start(self, configuration):
        self._lock = RLock()
        self._configuration = copy.deepcopy(configuration)
        self._sources = sources()
        self._environment = environment_identity()
        self.revision = 0
        self._done = False
        self._history, self._cache = [], {}

    def step(self, action, *, expected_revision, idempotency_key):
        request = Decision(
            action=action,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
        ).model_dump(mode="json")
        with self._lock:
            body_hash = digest(request["action"])
            if idempotency_key in self._cache:
                previous = self._cache[idempotency_key]
                if previous[0] != body_hash:
                    raise ValueError("idempotency conflict")
                return copy.deepcopy(previous[1])
            if expected_revision != self.revision:
                raise ValueError("stale revision")
            if self._done:
                raise ValueError("episode done")
            # Subclasses validate the entire action before mutating their state.
            result = self._apply(copy.deepcopy(action))
            self.revision += 1
            response = {
                "schema": "space-env.industrial-transition/1",
                "revision": self.revision,
                **result,
                "observation": self.observe(),
            }
            self._history.append(
                {"request": request, "response_sha256": digest(response)}
            )
            self._cache[idempotency_key] = (body_hash, copy.deepcopy(response))
            return response

    def checkpoint(self):
        with self._lock:
            body = {
                "schema": "space-env.industrial-checkpoint/1",
                "kind": self.kind,
                "configuration": copy.deepcopy(self._configuration),
                "sources": copy.deepcopy(self._sources),
                "environment": copy.deepcopy(self._environment),
                "history": copy.deepcopy(self._history),
                "state_sha256": digest(self._witness()),
                "revision": self.revision,
            }
            return {**body, "sha256": digest(body)}

    @classmethod
    def from_checkpoint(cls, value):
        if not isinstance(value, dict):
            raise TypeError("checkpoint object required")
        body = {k: v for k, v in value.items() if k != "sha256"}
        if (
            set(body)
            != {
                "schema",
                "kind",
                "configuration",
                "sources",
                "environment",
                "history",
                "state_sha256",
                "revision",
            }
            or body["schema"] != "space-env.industrial-checkpoint/1"
            or body["kind"] != cls.kind
            or value.get("sha256") != digest(body)
            or body["sources"] != sources()
            or body["environment"] != environment_identity()
            or type(body["revision"]) is not int
            or not 0 <= body["revision"] <= 512
            or not isinstance(body["history"], list)
            or len(body["history"]) != body["revision"]
        ):
            raise ValueError("checkpoint schema, source or integrity mismatch")
        restored = cls(body["configuration"])
        for row in body["history"]:
            if set(row) != {"request", "response_sha256"}:
                raise ValueError("invalid history row")
            request = Decision.model_validate(row["request"]).model_dump(mode="json")
            if request["expected_revision"] != restored.revision:
                raise ValueError("nonsequential checkpoint history")
            response = restored.step(**request)
            if digest(response) != row["response_sha256"]:
                raise ValueError("decision replay mismatch")
        if digest(restored._witness()) != body["state_sha256"]:
            raise ValueError("state replay mismatch")
        return restored

    def metrics(self):
        raise NotImplementedError

    def _witness(self):
        raise NotImplementedError


def restore(value):
    from space_env.industrial.campaign import AgentCampaignSession
    from space_env.industrial.native_proximity import NativeProximitySession
    from space_env.industrial.native_service import NativeServiceSession
    from space_env.industrial.operations import OperationsSession
    from space_env.industrial.orbits import OrbitSession
    from space_env.industrial.proximity import ProximitySession
    from space_env.industrial.service import ServiceSession

    types = {
        c.kind: c
        for c in (
            AgentCampaignSession,
            OperationsSession,
            ProximitySession,
            ServiceSession,
            NativeServiceSession,
            NativeProximitySession,
            OrbitSession,
        )
    }
    if not isinstance(value, dict) or value.get("kind") not in types:
        raise ValueError("unknown industrial session kind")
    return types[value["kind"]].from_checkpoint(value)
