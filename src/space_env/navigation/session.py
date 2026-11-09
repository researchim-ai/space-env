"""Causal synthetic navigation over native declared-force orbital trajectories.

The public view is deliberately separate from TaskSession's ideal telemetry.
Checkpoint/configuration and diagnostics are trainer-only artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import math
from threading import RLock

import numpy as np

from space_env.navigation.controls import (
    own_impulses,
    predict_controlled,
    sensor_power_scenario,
)
from space_env.navigation.dynamics import DeclaredDynamics
from space_env.navigation.models import SensorSpec, measurement_model, update
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import digest


class NavigationSession:
    def __init__(self, scenario, sensors):
        runtime = MissionRuntime(
            sensor_power_scenario(scenario, sensors),
            access=AccessPolicy(truth=True, forecast=False),
        )
        self._initialize(scenario, sensors, runtime)

    @classmethod
    def _attach(cls, scenario, sensors, runtime):
        """Trusted task integration: share the already constructed native executor."""
        result = cls.__new__(cls)
        result._initialize(scenario, sensors, runtime)
        return result

    def _initialize(self, scenario, sensors, runtime):
        self._lock = RLock()
        self._scenario = copy.deepcopy(scenario)
        if not isinstance(sensors, dict) or not 1 <= len(sensors) <= 8:
            raise ValueError("one to eight actor sensors required")
        self._specs = {a: SensorSpec.model_validate(v) for a, v in sensors.items()}
        self._runtime = runtime
        config = getattr(self._runtime._backend, "config", None)
        if (
            scenario["backend"] != "reference-dynamic"
            or config is None
            or self._runtime.horizon_sec > 3600
        ):
            raise ValueError("navigation/1 requires reference-dynamic horizon <=3600s")
        actors = set(config.orbits)
        if set(sensors) - actors:
            raise ValueError("unknown sensor actor")
        self._environments = {
            actor: DeclaredDynamics(config, actor) for actor in self._specs
        }
        for actor, spec in self._specs.items():
            if (
                self._environments[actor].perturbed
                and spec.force_profile != "matched_native"
            ):
                raise ValueError(
                    "perturbed truth requires explicit matched_native estimator force profile"
                )
            if spec.reference_actor is not None and (
                spec.reference_actor not in actors or spec.reference_actor == actor
            ):
                raise ValueError("range beacon must be a distinct known actor")
            count = math.floor(self._runtime.horizon_sec / spec.cadence_sec) + 1
            if count > spec.max_samples:
                raise ValueError("sensor sample budget does not cover native horizon")
        self._filters = {
            a: {
                "x": np.array(s.initial_estimate_si),
                "p": np.array(s.initial_covariance_si),
                "epoch": 0.0,
                "index": 0,
                "history": [],
                "pending": [],
                "controls": [],
                "burns": {},
            }
            for a, s in self._specs.items()
        }
        self.revision = 0
        self._calls = []
        self._cache = {}
        self._consume(self._runtime, self._filters)

    def _state(self, runtime, actor, clock):
        # Only acquired past/current samples. Never query the future provider arc.
        if clock > runtime.time_sec + 1e-10:
            raise ValueError("future navigation measurement forbidden")
        return runtime._backend.mission.ephemerides[actor].state(clock)

    def _sync_burns(self, runtime, filters):
        for row in runtime.command_status():
            command = row["command"]
            actor = command["actor_id"]
            if actor not in filters or command["kind"] != "finite_burn":
                continue
            if self._specs[actor].control_profile != "inertial_finite_burn":
                raise ValueError(
                    "finite burn requires explicit estimator control profile"
                )
            if row["status"] not in {"accepted", "running", "completed", "cancelled"}:
                raise ValueError(
                    "unknown finite-burn execution cannot be inferred by estimator"
                )
            entry = filters[actor]["burns"].setdefault(
                command["command_id"],
                {
                    "start": row["start_sec"],
                    "end": row["until_sec"],
                    "thrust_n": command["burn"]["thrust_n"],
                    "direction_gcrs": list(command["burn"]["direction_gcrs"]),
                },
            )
            if row["status"] == "cancelled":
                entry["end"] = min(entry["end"], runtime.time_sec)

    def _consume(self, runtime, filters):
        self._sync_burns(runtime, filters)
        for actor, spec in self._specs.items():
            f = filters[actor]
            # Acquire at the actual boundary, before a subsequent same-epoch command.
            # Pending packets own their noisy value; later maneuvers cannot rewrite it.
            while f["index"] < spec.max_samples:
                index = f["index"]
                acquired = index * spec.cadence_sec
                if acquired > runtime.time_sec + 1e-10:
                    break
                token = f"nav/1/{actor}/{spec.seed}/{index}".encode()
                rng = np.random.default_rng(
                    int.from_bytes(hashlib.sha256(token).digest()[:16], "big")
                )
                dropped = bool(rng.random() < spec.dropout_probability)
                mission = runtime._backend.mission
                if spec.continuous_power_w > 0 or spec.minimum_soc > 0:
                    if abs(acquired - runtime.time_sec) > 1e-9:
                        raise ValueError(
                            "powered measurement acquisition requires a native event boundary"
                        )
                    soc = mission.energy_wh[actor] / mission.vehicles[actor].battery_wh
                    unavailable = soc <= spec.minimum_soc + 1e-12
                else:
                    unavailable = False
                row = {
                    "sample_id": index,
                    "acquired_sec": acquired,
                    "received_sec": acquired + spec.latency_sec,
                    "kind": spec.kind,
                    "frame": "GCRS",
                    "dropped": dropped or unavailable,
                }
                if unavailable:
                    row["unavailable"] = "power"
                row["_control_ids"] = [
                    r["command"]["command_id"] for r in own_impulses(runtime, actor)
                ]
                if not row["dropped"]:
                    beacon = (
                        None
                        if spec.reference_actor is None
                        else self._state(runtime, spec.reference_actor, acquired)
                    )
                    expected, _ = measurement_model(
                        spec.kind, self._state(runtime, actor, acquired), beacon
                    )
                    noise = np.array(spec.standard_deviation)
                    value = expected + rng.normal(size=len(expected)) * noise
                    if spec.true_bias:
                        value += np.array(spec.true_bias)
                    row.update(
                        value=value.tolist(),
                        noise_covariance=np.diag(noise**2).tolist(),
                    )
                    if beacon is not None:
                        row.update(
                            reference_actor=spec.reference_actor,
                            known_reference_state_si=beacon.tolist(),
                        )
                f["pending"].append(row)
                f["index"] += 1
            while (
                f["pending"]
                and f["pending"][0]["received_sec"] <= runtime.time_sec + 1e-10
            ):
                row = f["pending"].pop(0)
                visible = row.pop("_control_ids")
                if not row["dropped"]:
                    x, p, applied = predict_controlled(
                        runtime,
                        actor,
                        spec,
                        f["x"],
                        f["p"],
                        f["epoch"],
                        row["acquired_sec"],
                        f["controls"],
                        visible,
                        environment=self._environments[actor],
                        burns=f["burns"],
                    )
                    x, p, info = update(
                        x,
                        p,
                        row["value"],
                        spec.kind,
                        row["noise_covariance"],
                        row.get("known_reference_state_si"),
                        spec.nis_gate,
                    )
                    f.update(x=x, p=p, epoch=row["acquired_sec"], controls=applied)
                    row["filter"] = info
                f["history"].append(row)

    def _advance_native(self, runtime, filters, duration, advance):
        """Split at acquisition/reception times; preserve native energy history."""
        self._sync_burns(runtime, filters)
        target = runtime.time_sec + duration
        while runtime.time_sec < target - 1e-10:
            now = runtime.time_sec
            candidates = [target]
            for actor, spec in self._specs.items():
                f = filters[actor]
                if f["index"] < spec.max_samples:
                    candidates.append(f["index"] * spec.cadence_sec)
                if f["pending"]:
                    candidates.append(f["pending"][0]["received_sec"])
            boundary = min(t for t in candidates if t > now + 1e-10)
            advance(boundary - now)
            self._consume(runtime, filters)
        if runtime.time_sec < target:
            advance(target - runtime.time_sec)
            self._consume(runtime, filters)

    def _view(self, runtime, filters, actor):
        if actor not in filters:
            raise ValueError("unknown navigation actor")
        f = filters[actor]
        spec = self._specs[actor]
        x, p, _ = predict_controlled(
            runtime,
            actor,
            spec,
            f["x"],
            f["p"],
            f["epoch"],
            runtime.time_sec,
            f["controls"],
            environment=self._environments[actor],
            burns=f["burns"],
        )
        return {
            "schema": "space-env.navigation-observation/1",
            "actor": actor,
            "time_sec": runtime.time_sec,
            "frame": "GCRS",
            "estimate_si": x.tolist(),
            "covariance_si": p.tolist(),
            "filter_epoch_sec": f["epoch"],
            "estimate_prediction_age_sec": runtime.time_sec - f["epoch"],
            "received_samples": copy.deepcopy(f["history"][-1:]),
            "received_sample_count": len(f["history"]),
            "prediction_profile": {
                "force": spec.force_profile,
                "control": spec.control_profile,
                "mass": "declared deterministic propulsion",
            },
            "scope": "synthetic instantaneous measurements; known ideal beacon if present; declared-force EKF with nominal known controls; no own truth state",
        }

    def observe(self, actor):
        with self._lock:
            return self._view(self._runtime, self._filters, actor)

    def measurement_history(self, actor):
        """All received packets, including drops; no truth or injected bias."""
        with self._lock:
            if actor not in self._filters:
                raise ValueError("unknown navigation actor")
            return copy.deepcopy(self._filters[actor]["history"])

    def advance(self, dt_sec, *, expected_revision, idempotency_key):
        with self._lock:
            if (
                isinstance(dt_sec, bool)
                or not isinstance(dt_sec, (int, float))
                or not math.isfinite(dt_sec)
                or dt_sec <= 1e-9
            ):
                raise ValueError("positive finite navigation advance required")
            if (
                not isinstance(idempotency_key, str)
                or not 1 <= len(idempotency_key) <= 128
            ):
                raise ValueError("bounded nonempty idempotency key required")
            if isinstance(expected_revision, bool) or not isinstance(
                expected_revision, int
            ):
                raise TypeError("integer revision required")
            request = {
                "dt_sec": float(dt_sec),
                "expected_revision": expected_revision,
                "idempotency_key": idempotency_key,
            }
            if idempotency_key in self._cache:
                old, response = self._cache[idempotency_key]
                if old != request:
                    raise ValueError("idempotency conflict")
                return copy.deepcopy(response)
            if expected_revision != self.revision or len(self._calls) >= 512:
                raise ValueError(
                    "revision mismatch or navigation decision budget exhausted"
                )
            if self._runtime.time_sec + dt_sec > self._runtime.horizon_sec + 1e-10:
                raise ValueError("advance exceeds native navigation horizon")
            candidate = self._runtime.fork()
            filters = copy.deepcopy(self._filters)

            def advance_native(duration):
                return candidate.advance(
                    duration,
                    expected_revision=candidate.revision,
                    idempotency_key=f"navigation/{idempotency_key}/{candidate.revision}",
                )

            self._advance_native(candidate, filters, dt_sec, advance_native)
            response = {
                "revision": self.revision + 1,
                "time_sec": candidate.time_sec,
                "observations": {
                    a: self._view(candidate, filters, a) for a in sorted(filters)
                },
            }
            digest(response)  # Reject nonfinite serialization before commit.
            self._runtime = candidate
            self._filters = filters
            self.revision += 1
            self._calls.append(request)
            self._cache[idempotency_key] = (request, copy.deepcopy(response))
            return response

    def diagnostics(self, actor):
        """Trainer-only truth errors/NEES. Never part of observe/advance output."""
        with self._lock:
            view = self._view(self._runtime, self._filters, actor)
            error = np.array(view["estimate_si"]) - self._state(
                self._runtime, actor, self._runtime.time_sec
            )
            p = np.array(view["covariance_si"])
            rank = int(np.linalg.matrix_rank(p))
            nees = float(error @ np.linalg.solve(p, error)) if rank == 6 else None
            return {
                "actor": actor,
                "time_sec": self._runtime.time_sec,
                "error_si": error.tolist(),
                "position_error_m": float(np.linalg.norm(error[:3])),
                "velocity_error_m_s": float(np.linalg.norm(error[3:])),
                "nees": nees,
                "covariance_rank": rank,
                "scope": "privileged trainer diagnostics; not agent observations",
            }

    def checkpoint(self):
        from space_env.llm.episode import sources

        with self._lock:
            body = {
                "schema": "space-env.navigation-checkpoint/1",
                "scenario": copy.deepcopy(self._scenario),
                "sensors": {
                    a: s.model_dump(mode="json") for a, s in self._specs.items()
                },
                "calls": copy.deepcopy(self._calls),
                "observations_sha256": digest(
                    {a: self.observe(a) for a in sorted(self._specs)}
                ),
                "model_sources": sources(),
            }
            return body | {"sha256": digest(body)}

    @classmethod
    def from_checkpoint(cls, payload):
        from space_env.llm.episode import sources

        keys = {
            "schema",
            "scenario",
            "sensors",
            "calls",
            "observations_sha256",
            "model_sources",
            "sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != keys
            or payload["schema"] != "space-env.navigation-checkpoint/1"
            or payload["sha256"]
            != digest({k: v for k, v in payload.items() if k != "sha256"})
            or payload["model_sources"] != sources()
        ):
            raise ValueError("navigation checkpoint schema/source/seal mismatch")
        if not isinstance(payload["calls"], list) or len(payload["calls"]) > 512:
            raise ValueError("navigation checkpoint replay budget exceeded")
        result = cls(payload["scenario"], payload["sensors"])
        for call in payload["calls"]:
            result.advance(**call)
        if result.checkpoint() != payload:
            raise ValueError("navigation cold replay mismatch")
        return result
