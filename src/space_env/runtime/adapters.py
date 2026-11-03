"""Backend adapters preserve each executor's declared frame and resource semantics."""

from __future__ import annotations

import copy
from typing import Protocol, runtime_checkable

from space_env.runtime.schema import RuntimeCommand, ScenarioSpec
from space_env.simulation.commands import MissionCommand
from space_env.simulation.snapshot import encode, world_from_payload, world_payload


@runtime_checkable
class Backend(Protocol):
    actors: list[str]
    capabilities: frozenset[str]

    def now(self) -> float: ...
    def horizon(self) -> float: ...
    def validate_command(self, command: RuntimeCommand) -> None: ...
    def advance(self, dt: float, active: list[dict]) -> dict: ...
    def observe(self, actor: str) -> dict: ...
    def report(self) -> dict: ...
    def truth(self) -> dict: ...
    def capture(self): ...
    def restore(self, state) -> None: ...


class ReferenceBackend:
    capabilities = frozenset(
        {
            "commands.wait",
            "commands.downlink",
            "commands.processing",
            "events.inject",
            "observe.agent",
            "observe.truth",
            "forecast.geometry",
            "forecast.eclipse",
            "checkpoint.prefix_replay",
        }
    )

    def __init__(self, spec: ScenarioSpec):
        from space_env.reference.operations import ReferenceDownlinkMission

        definition = copy.deepcopy(spec.definition)
        if set(spec.actor_order) != set(definition.get("ephemerides", {})):
            raise ValueError("actor order does not match reference definition")
        definition["ephemerides"] = {
            a: definition["ephemerides"][a] for a in spec.actor_order
        }
        definition["vehicles"] = {
            a: definition["vehicles"][a] for a in spec.actor_order
        }
        factory = getattr(self, "mission_factory", ReferenceDownlinkMission)
        self.mission = factory(definition)
        self.actors = list(spec.actor_order)
        self.definition = {
            **definition,
            "vehicles": {
                a: v.model_dump(mode="json") for a, v in self.mission.vehicles.items()
            },
            "products": [
                p.model_dump(mode="json") for p in self.mission.products.values()
            ],
            "stations": [
                s.model_dump(mode="json") for s in self.mission.stations.values()
            ],
            "outages": [o.model_dump(mode="json") for o in self.mission.outages],
            "horizon_sec": self.mission.horizon_sec,
            "resource_step_sec": self.mission.resource_step_sec,
        }
        if self.mission.radio_network is not None:
            self.definition["radio_network"] = self.mission.radio_network.config.model_dump(mode="json")
            self.capabilities = self.capabilities | {"physics.sampled_rf", "ground.acquisition", "ground.aggregate_capacity"}
        if self.mission.dtn is not None:
            self.definition["dtn_network"] = self.mission.dtn.config.model_dump(mode="json")
            self.capabilities = self.capabilities | {"commands.relay", "network.custody", "network.concurrent_isl"}
        # Effective and executed configuration are identical, including defaults.
        self.mission = factory(self.definition)

    def now(self):
        return self.mission.clock

    def horizon(self):
        return self.mission.horizon_sec

    def validate_command(self, c):
        if self.mission.dtn is not None and self.mission.dtn.receiving(c.actor_id) and c.kind not in {"relay","wait"}:
            raise ValueError("actor receiver is busy draining an in-flight fragment")
        if c.kind == "relay":
            if self.mission.dtn is None:
                raise ValueError("relay requires DTN configuration")
            self.mission.dtn.validate(c,self.mission)
        elif c.kind == "downlink":
            if c.destination_id not in self.mission.stations:
                raise ValueError("reference downlink requires a known destination")
            if c.product_id:
                p = self.mission.products.get(c.product_id)
                if (
                    p is None
                    or p.actor != c.actor_id
                    or (p.destination and p.destination != c.destination_id)
                ):
                    raise ValueError("downlink product ownership/destination mismatch")
        elif c.kind == "processing" and c.product_id:
            p = self.mission.products.get(c.product_id)
            if p is None or p.actor != c.actor_id:
                raise ValueError("CPU product must belong to actor")

    def admit(self, c):
        if c.kind == "relay":
            expiry = None
            life = getattr(self,"lifecycle",None)
            if life is not None:
                artifact = life.artifacts.get(c.product_id)
                expiry = None if artifact is None else artifact.get("expires_sec")
            self.mission.dtn.admit(c,self.mission,expiry)

    def advance(self, dt, active):
        actions, products, processing, jobs = (
            {},
            {},
            dict.fromkeys(self.actors, False),
            {},
        )
        for row in active:
            c = row["command"]
            a = c["actor_id"]
            if c["kind"] == "downlink":
                actions[a] = c["destination_id"]
                if c["product_id"]:
                    products[a] = c["product_id"]
            elif c["kind"] == "processing":
                processing[a] = True
                if c["product_id"]:
                    jobs[a] = c["product_id"]
        self.mission.step(
            dt, actions, products=products, processing=processing, processing_jobs=jobs,
            relay_controls=[r["command"]["command_id"] for r in active if r["command"]["kind"]=="relay"],
            relay_blocked=[r["command"]["actor_id"] for r in active if r["command"]["kind"] not in {"relay","wait"}],
        )
        return {}

    def inject(self, events):
        self.mission.apply_events(events)

    def cancel(self, command_id):
        if self.mission.dtn is not None:
            self.mission.dtn.cancel(command_id,self.now())

    def observe(self, actor):
        view = self.mission.observation(actor)["actors"][actor]
        return {
            **view,
            "products": [
                {
                    "id": p.id,
                    "remaining_bits": self.mission.remaining[p.id],
                    "ready_sec": p.ready_sec,
                    "deadline_sec": p.deadline_sec,
                    "priority": p.priority,
                    "processing_remaining_sec": self.mission.processing_remaining[p.id],
                    "cancelled": p.id in self.mission.cancelled,
                }
                for p in self.mission.products.values()
                if p.actor == actor
            ],
        }

    def forecast(self, actor, station):
        return self.mission.service_windows(actor, station)

    def geometry_token(self, actor):
        return self.mission.ephemerides[actor].sha256

    def forecast_eclipses(self, actor, **options):
        from space_env.reference.geometry_events import eclipse_windows

        options["end_sec"] = (
            self.horizon() if options.get("end_sec") is None else options["end_sec"]
        )
        result = eclipse_windows(self.mission.ephemerides[actor], **options)
        result["actor_id"] = actor
        return result

    def report(self):
        return self.mission.report()

    def truth(self):
        return {
            "report": self.report(),
            "states": {
                a: {
                    "frame": "GCRS",
                    "units": "m,m/s",
                    "state_si": e.state(self.now()).tolist(),
                    "epoch_utc": e.epoch.shifted(self.now()).iso(),
                }
                for a, e in self.mission.ephemerides.items()
            },
        }

    def capture(self):
        return copy.deepcopy(self.mission)

    def restore(self, state):
        self.mission = state

    def metrics(self):
        return {"on_time_bits": self.mission.report()["on_time_bits"]}


class LegacyBackend:
    capabilities = frozenset(
        {
            "commands.wait",
            "commands.downlink",
            "commands.imaging",
            "observe.agent",
            "observe.truth",
            "checkpoint.prefix_replay",
        }
    )

    def __init__(self, spec: ScenarioSpec):
        self.definition = copy.deepcopy(spec.definition)
        self.world = world_from_payload(self.definition)
        if list(self.world.engines) != list(spec.actor_order):
            raise ValueError("actor order does not match legacy snapshot")
        if any(
            r["status"] in {"accepted", "running"}
            for r in self._native_records().values()
        ):
            raise ValueError(
                "initial legacy definition must not contain active native commands"
            )
        self.actors = list(spec.actor_order)

    def _native_records(self):
        return {k: v["record"] for k, v in self.world.commands.export().items()}

    def now(self):
        return self.world.now_sec()

    def horizon(self):
        return self.world.horizon_sec

    def validate_command(self, c):
        if c.product_id:
            raise ValueError("legacy adapter does not support product-level selection")
        if c.kind == "downlink" and c.destination_id not in {
            s.name for s in self.world.engines[c.actor_id].stations
        }:
            raise ValueError("legacy downlink requires a configured station")
        if c.kind == "imaging" and c.request_id:
            targets = {
                t.name for t in self.world.engines[c.actor_id].scenario.ground_targets
            }
            if (
                c.request_id not in targets
                and c.request_id not in self.world.task_requests
            ):
                raise ValueError("unknown imaging request")
        if any(
            r["command_id"] == "runtime:" + c.command_id
            for r in self._native_records().values()
        ):
            raise ValueError("native command ID already exists")

    def advance(self, dt, active):
        native = self._native_records()
        new = []
        orders = dict.fromkeys(self.actors, "wait")
        kinds = {"downlink": "comms", "wait": "wait", "imaging": "imaging"}
        for row in active:
            c = row["command"]
            cid = "runtime:" + c["command_id"]
            if cid not in native:
                new.append(
                    MissionCommand(
                        cid,
                        c["actor_id"],
                        kinds[c["kind"]],
                        request_id=c["request_id"],
                        destination_id=c["destination_id"],
                        duration_sec=c["duration_sec"],
                    )
                )
            orders[c["actor_id"]] = kinds[c["kind"]]
        if new:
            accepted = self.world.accept(new)
            orders.update(accepted)
            for row in active:
                record = self.world.commands.get(
                    "runtime:" + row["command"]["command_id"]
                )
                if record["status"] in {"rejected", "failed", "cancelled"}:
                    orders[row["command"]["actor_id"]] = "wait"
        self.world.step(dt, orders)
        self.world.settle()
        return {
            r["command"]["command_id"]: self.world.commands.get(
                "runtime:" + r["command"]["command_id"]
            )["status"]
            for r in active
        }

    def cancel(self, command_id):
        if any(
            r["command_id"] == "runtime:" + command_id
            for r in self._native_records().values()
        ):
            self.world.cancel_operation("runtime:" + command_id)

    def observe(self, actor):
        return self.world.observe(actor)

    def report(self):
        return {
            "profile": "legacy-training/1",
            "time_sec": self.now(),
            "actors": {a: self.world.observe(a) for a in self.actors},
            "commands": encode(self.world.commands.export()),
            "receipts": {a: encode(e.receipts) for a, e in self.world.engines.items()},
            "metrics": self.metrics(),
        }

    def capture(self):
        return world_payload(self.world)

    def truth(self):
        return {
            "report": self.report(),
            "states": {
                a: {
                    "native_frame_tag": e.orbit.frame,
                    "physics_profile": "legacy-training/1",
                    "units": "m,m/s",
                    "r_m": e.orbit.r.tolist(),
                    "v_m_s": e.orbit.v.tolist(),
                    "state": e.get_state_snapshot().model_dump(mode="json"),
                }
                for a, e in self.world.engines.items()
            },
        }

    def restore(self, state):
        self.world = world_from_payload(state)

    def metrics(self):
        return {
            "delivered_bits": sum(
                float(e.delivered_bits) for e in self.world.engines.values()
            )
        }


def make_backend(spec: ScenarioSpec) -> Backend:
    if spec.backend in {
        "reference-dynamic",
        "reference-lifecycle",
        "reference-spacecraft",
    }:
        from space_env.reference.dynamic import DynamicReferenceBackend

        return DynamicReferenceBackend(spec)
    return (
        ReferenceBackend(spec)
        if spec.backend == "reference-fixed"
        else LegacyBackend(spec)
    )
