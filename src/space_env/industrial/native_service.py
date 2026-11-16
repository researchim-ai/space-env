"""Streaming products over one coupled native orbit/RF/resource executor.

No parallel service bit/energy ledger: deliveries, custody, power and thermal
accounting come from MissionRuntime. Prefixes keep their absolute mission clock.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

from pydantic import Field, model_validator

from space_env.industrial.session import DecisionSession
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.runtime.schema import RuntimeCommand, StrictModel
from space_env.simulation.snapshot import digest


def _unused(base, occupied):
    """Allocate from committed past names only, including foreign prefix IDs."""
    candidate = base
    counter = 0
    while candidate in occupied:
        counter += 1
        candidate = f"{base}/{counter}"
    return candidate


class Stream(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    source: str
    destination: str
    size_bits: int = Field(ge=1, le=10**9, strict=True)
    chunk_bits: int = Field(default=100000, ge=1, le=10**9, strict=True)
    release_sec: float = Field(default=0.0, ge=0, strict=True)
    interval_sec: float = Field(default=3.0, ge=0.01, le=3600, strict=True)
    deadline_sec: float = Field(gt=0, le=3600, strict=True)
    priority: int = Field(default=1, ge=0, le=1000000, strict=True)

    @model_validator(mode="after")
    def schedule(self):
        count = math.ceil(self.size_bits / self.chunk_bits)
        if (
            count > 256
            or self.release_sec + (count - 1) * self.interval_sec >= self.deadline_sec
        ):
            raise ValueError("stream exceeds chunk budget or release/deadline envelope")
        return self


class NativeServiceEvent(StrictModel):
    at_sec: float = Field(ge=0, le=3600, strict=True)
    type: Literal[
        "spacecraft_fault",
        "spacecraft_fault_cleared",
        "spacecraft_safe_mode",
        "spacecraft_recover",
        "station_outage_added",
        "station_recovered",
        "flow_cancelled",
        "flow_priority_changed",
    ]
    payload: dict

    @model_validator(mode="after")
    def payload_shape(self):
        expected = (
            {"flow_id"}
            if self.type == "flow_cancelled"
            else {"flow_id", "priority"}
            if self.type == "flow_priority_changed"
            else {"actor", "subsystem"}
            if self.type in {"spacecraft_fault", "spacecraft_fault_cleared"}
            else {"actor"}
            if self.type.startswith("spacecraft_")
            else {"station", "start_sec", "end_sec"}
            if self.type == "station_outage_added"
            else {"station"}
        )
        if set(self.payload) != expected:
            raise ValueError("invalid native service event fields")
        if "priority" in self.payload and (
            type(self.payload["priority"]) is not int
            or not 0 <= self.payload["priority"] <= 1000000
        ):
            raise ValueError("integer flow priority required")
        if "subsystem" in self.payload and self.payload["subsystem"] not in {
            "bus",
            "adcs",
            "payload",
            "cpu",
            "radio",
            "propulsion",
        }:
            raise ValueError("unknown fault subsystem")
        return self


class NativeServiceSpec(StrictModel):
    scenario: dict | None = None
    prefix_checkpoint: dict | None = None
    streams: tuple[Stream, ...] = Field(min_length=1, max_length=32)
    events: tuple[NativeServiceEvent, ...] = Field(default=(), max_length=128)
    decision_sec: float = Field(default=2.0, ge=0.01, le=10, strict=True)

    @model_validator(mode="after")
    def contract(self):
        if (self.scenario is None) == (self.prefix_checkpoint is None):
            raise ValueError("provide exactly one scenario or native prefix checkpoint")
        if (
            len({s.id for s in self.streams}) != len(self.streams)
            or sum(math.ceil(s.size_bits / s.chunk_bits) for s in self.streams) > 256
        ):
            raise ValueError("duplicate streams or total chunk budget exceeded")
        return self


class NativeControl(RuntimeCommand):
    command_id: str = "auto"
    forecast: None = None

    @model_validator(mode="after")
    def no_forecast(self):
        if self.forecast is not None:
            raise ValueError("agent controls cannot carry privileged forecast bindings")
        return self


class NativeServiceAction(StrictModel):
    commands: tuple[NativeControl, ...] = Field(default=(), max_length=8)
    cancel_actors: tuple[str, ...] = Field(default=(), max_length=16)


class NativeServiceSession(DecisionSession):
    kind = "native_service"

    def __init__(self, configuration):
        self.spec = NativeServiceSpec.model_validate(copy.deepcopy(configuration))
        self._start(self.spec.model_dump(mode="json"))
        access = AccessPolicy(truth=True, forecast=False)
        self._runtime = (
            MissionRuntime(self.spec.scenario, access=access, audit_mode="compact")
            if self.spec.scenario is not None
            else MissionRuntime.from_checkpoint(
                self.spec.prefix_checkpoint, access=access
            )
        )
        backend, mission = self._runtime._backend, self._runtime._backend.mission
        if (
            self._runtime._spec.backend != "reference-spacecraft"
            or getattr(backend, "bus", None) is None
            or mission.radio_network is None
            or mission.dtn is None
        ):
            raise ValueError("native service requires coupled spacecraft, RF and DTN")
        self._begin = self._runtime.time_sec
        self._actors = tuple(self._runtime.describe_capabilities()["actors"])
        if (
            not 0
            < self._runtime.horizon_sec - self._begin
            <= self.spec.decision_sec * 512
            or len(self._actors) > 16
        ):
            raise ValueError("native service exceeds horizon/actor/decision budget")
        self._flow = {
            s.id: {"priority": s.priority, "cancelled": False}
            for s in self.spec.streams
        }
        self._schedule = []
        for flow in self.spec.streams:
            if (
                flow.source not in self._actors
                or flow.destination not in mission.stations
                or flow.release_sec < self._begin
                or flow.deadline_sec > self._runtime.horizon_sec
            ):
                raise ValueError(
                    "stream endpoints or absolute times outside native mission"
                )
            for index in range(math.ceil(flow.size_bits / flow.chunk_bits)):
                pid = "stream-" + digest({"flow": flow.id, "index": index})[:24]
                if pid in mission.products or pid in getattr(
                    backend.lifecycle, "producers", {}
                ):
                    raise ValueError("stream product collides with inherited inventory")
                self._schedule.append(
                    (
                        flow.release_sec + index * flow.interval_sec,
                        flow.id,
                        pid,
                        min(flow.chunk_bits, flow.size_bits - index * flow.chunk_bits),
                    )
                )
        self._schedule.sort()
        for event in self.spec.events:
            p = event.payload
            if (
                not self._begin <= event.at_sec < self._runtime.horizon_sec
                or (
                    event.type.startswith("flow_")
                    and (
                        p["flow_id"] not in self._flow
                        or event.at_sec
                        < next(
                            f.release_sec
                            for f in self.spec.streams
                            if f.id == p["flow_id"]
                        )
                    )
                )
                or (
                    event.type.startswith("spacecraft_")
                    and p["actor"] not in self._actors
                )
                or (
                    event.type.startswith("station_")
                    and p["station"] not in mission.stations
                )
            ):
                raise ValueError("event target or absolute time outside native mission")
        self._chunks, self._events_done, self._event_log = {}, set(), []
        self._rejections = 0
        self._runtime, self._chunks, self._flow, self._events_done, self._event_log = (
            self._settle(
                self._runtime,
                self._chunks,
                self._flow,
                self._events_done,
                self._event_log,
            )
        )

    def _inject(self, runtime, kind, payload, token):
        mission = runtime._backend.mission
        life = runtime._backend.lifecycle
        occupied = {row["id"] for row in [*mission.events, *life.events]}
        event_id = _unused(f"service/{runtime.revision}/{token}", occupied)
        key = _unused(f"service-event/{runtime.revision}/{token}", runtime._cache)
        return runtime.inject(
            events=[
                {
                    "id": event_id,
                    "type": kind,
                    "payload": payload,
                    "time_sec": runtime.time_sec,
                    "source": "native-service/1",
                    "reason": "causal stream/environment boundary",
                }
            ],
            expected_revision=runtime.revision,
            idempotency_key=key,
        )

    def _settle(self, runtime, chunks, flows, done, log):
        now = runtime.time_sec
        if now >= runtime.horizon_sec - 1e-9:
            return runtime, chunks, flows, done, log
        for index, event in enumerate(self.spec.events):
            if index in done or event.at_sec > now + 1e-9:
                continue
            p = event.payload
            if event.type.startswith("flow_"):
                outcomes = {
                    r["id"]: r for r in runtime._backend.mission.product_outcomes()
                }
                ids = [pid for _, fid, pid, _ in self._schedule if fid == p["flow_id"]]
                if all(
                    pid in outcomes and outcomes[pid]["status"] == "delivered_on_time"
                    for pid in ids
                ):
                    done.add(index)
                    log.append(
                        {
                            "index": index,
                            "at_sec": now,
                            "status": "ignored_completed_flow",
                        }
                    )
                    continue
            if event.type == "flow_cancelled":
                flows[p["flow_id"]]["cancelled"] = True
            elif event.type == "flow_priority_changed":
                flows[p["flow_id"]]["priority"] = p["priority"]
                for pid, row in chunks.items():
                    mission = runtime._backend.mission
                    if (
                        row["flow_id"] == p["flow_id"]
                        and row["admitted"]
                        and mission.remaining[pid] > 0
                        and pid not in mission.cancelled
                    ):
                        self._inject(
                            runtime,
                            "product_priority_changed",
                            {"product_id": pid, "priority": p["priority"]},
                            f"priority/{index}/{pid}",
                        )
            else:
                try:
                    self._inject(runtime, event.type, p, f"environment/{index}")
                except RuntimeContractError as exc:
                    raise RuntimeError("native environment event rejected") from exc
            done.add(index)
            log.append({"index": index, "at_sec": now})
        # Cancellation requests drain in-flight fragments through native cancel;
        # source bytes are never teleported away from a pinned custody transfer.
        for pid, row in chunks.items():
            if not flows[row["flow_id"]]["cancelled"] or not row["admitted"]:
                continue
            for control in runtime.command_status():
                c = control["command"]
                if c["product_id"] == pid and control["status"] in {
                    "accepted",
                    "running",
                }:
                    runtime.cancel(
                        c["command_id"],
                        expected_revision=runtime.revision,
                        idempotency_key=_unused(
                            f"service-cancel/{runtime.revision}/{c['command_id']}",
                            runtime._cache,
                        ),
                    )
            mission = runtime._backend.mission
            if (
                mission.remaining[pid] > 0
                and not mission.dtn.pinned(pid)
                and pid not in mission.cancelled
            ):
                self._inject(
                    runtime, "product_cancelled", {"product_id": pid}, "cancel/" + pid
                )
        by_id = {f.id: f for f in self.spec.streams}
        for at, fid, pid, bits in self._schedule:
            if at > now + 1e-9 or pid in chunks:
                continue
            flow = by_id[fid]
            row = {
                "flow_id": fid,
                "size_bits": bits,
                "admitted": False,
                "status": "cancelled_before_release"
                if flows[fid]["cancelled"]
                else "rejected_capacity",
            }
            if not flows[fid]["cancelled"]:
                trial = runtime.fork()
                try:
                    self._inject(
                        trial,
                        "onboard_product_added",
                        {
                            "id": pid,
                            "actor": flow.source,
                            "size_bits": float(bits),
                            "ready_sec": now,
                            "deadline_sec": flow.deadline_sec,
                            "priority": flows[fid]["priority"],
                            "destination": flow.destination,
                        },
                        "release/" + pid,
                    )
                except RuntimeContractError as exc:
                    # Only a validated native memory admission failure is a
                    # rejected packet; other execution errors are not capacity.
                    if "memory" not in str(exc).lower():
                        raise RuntimeError("native product release failed") from exc
                else:
                    runtime = trial
                    row.update(admitted=True, status="admitted")
            chunks[pid] = row
        return runtime, chunks, flows, done, log

    def observe(self):
        runtime = self._runtime
        mission = runtime._backend.mission
        actors = runtime.observe()["actors"]
        ground = []
        for actor in self._actors:
            for station in mission.stations:
                budget = mission.radio_network.budget(mission, actor, station)
                tx = mission.radio_network.config.transmitters[actor]
                rx = mission.radio_network.config.receivers[station]
                ground.append(
                    {
                        "actor": actor,
                        "station": station,
                        "visible": station in actors[actor]["contacts"],
                        "rate_bps": budget["rate_bps"],
                        "outage": runtime.station_observation(station)["outage"],
                        "body_antenna": tx.beamwidth_deg is not None,
                        "band_compatible": rx.min_frequency_hz
                        <= tx.frequency_hz
                        <= rx.max_frequency_hz,
                    }
                )
        links = []
        for link in mission.dtn.config.links:
            budget = mission.dtn.budget(
                mission, {"source": link.source, "destination": link.destination}
            )
            links.append(
                {
                    "source": link.source,
                    "destination": link.destination,
                    "rate_bps": budget["rate_bps"],
                    "range_m": budget["range_m"],
                    "reason": budget["reason"],
                }
            )
        return {
            "schema": "space-env.native-service-observation/1",
            "revision": self.revision,
            "time_sec": runtime.time_sec,
            "horizon_sec": runtime.horizon_sec,
            "decision_sec": self.spec.decision_sec,
            "objective": {
                "kind": "all_streams_delivered_on_time",
                "completion": "native_ground_acknowledgement",
                "reward": "sum_completed_flow_priorities_minus_0.1_native_rejections",
                "duration": "operate_until_native_horizon",
            },
            "supported_commands": [
                c.removeprefix("commands.")
                for c in runtime.describe_capabilities()["supported"]
                if c.startswith("commands.")
            ],
            "actors": actors,
            "inventory": {a: runtime.inventory(a) for a in self._actors},
            "commands": runtime.command_status(),
            "stations": {s: runtime.station_observation(s) for s in mission.stations},
            "ground_links": ground,
            "isl_links": links,
            "flows": [
                {**f.model_dump(mode="json"), **copy.deepcopy(self._flow[f.id])}
                for f in self.spec.streams
                if f.release_sec <= runtime.time_sec
            ],
            "scope": "central union of ideal current actor telemetry and sampled link budgets; no future forecasts or admission guarantee",
        }

    def _apply(self, value):
        action = NativeServiceAction.model_validate(value)
        if (
            len(set(action.cancel_actors)) != len(action.cancel_actors)
            or set(action.cancel_actors) - set(self._actors)
            or any(c.actor_id not in self._actors for c in action.commands)
        ):
            raise ValueError("unknown actor or duplicate cancellation")
        for command in action.commands:
            if (
                command.product_id in self._chunks
                and self._flow[self._chunks[command.product_id]["flow_id"]]["cancelled"]
            ):
                raise ValueError("cancelled stream cannot receive new commands")
        before = self.metrics()["completed_priority"]
        candidate = self._runtime.fork()
        feedback = {"accepted": True, "code": "executed"}
        try:
            for row in candidate.command_status():
                if row["command"]["actor_id"] in action.cancel_actors and row[
                    "status"
                ] in {"accepted", "running"}:
                    candidate.cancel(
                        row["command"]["command_id"],
                        expected_revision=candidate.revision,
                        idempotency_key=_unused(
                            f"service-action/{self.revision}/{candidate.revision}/cancel",
                            candidate._cache,
                        ),
                    )
            if action.commands:
                occupied = {
                    r["command"]["command_id"] for r in candidate.command_status()
                }
                commands = []
                for i, c in enumerate(action.commands):
                    cid = _unused(f"service-control/{candidate.revision}/{i}", occupied)
                    occupied.add(cid)
                    commands.append(
                        c.model_copy(update={"command_id": cid}).model_dump(mode="json")
                    )
                candidate.submit(
                    commands,
                    expected_revision=candidate.revision,
                    idempotency_key=_unused(
                        f"service-action/{self.revision}/{candidate.revision}/submit",
                        candidate._cache,
                    ),
                )
        except RuntimeContractError:
            candidate = self._runtime.fork()
            feedback = {"accepted": False, "code": "native_rejected"}
        chunks, flows, done, log = (
            copy.deepcopy(self._chunks),
            copy.deepcopy(self._flow),
            self._events_done.copy(),
            copy.deepcopy(self._event_log),
        )
        start = candidate.time_sec
        end = min(candidate.horizon_sec, start + self.spec.decision_sec)
        while candidate.time_sec < end - 1e-9:
            now = candidate.time_sec
            future = [
                at
                for at, _, pid, _ in self._schedule
                if pid not in chunks and at > now + 1e-9
            ] + [
                e.at_sec
                for i, e in enumerate(self.spec.events)
                if i not in done and e.at_sec > now + 1e-9
            ]
            stop = min(end, min(future, default=end))
            candidate.advance(
                stop - now,
                expected_revision=candidate.revision,
                idempotency_key=_unused(
                    f"service-advance/{self.revision}/{candidate.revision}",
                    candidate._cache,
                ),
            )
            candidate, chunks, flows, done, log = self._settle(
                candidate, chunks, flows, done, log
            )
        self._runtime, self._chunks, self._flow, self._events_done, self._event_log = (
            candidate,
            chunks,
            flows,
            done,
            log,
        )
        self._rejections += int(not feedback["accepted"])
        self._done = end >= candidate.horizon_sec - 1e-9
        reward = (
            self.metrics()["completed_priority"]
            - before
            - 0.1 * int(not feedback["accepted"])
        )
        return {
            "reward": float(reward),
            "terminated": self._done,
            "elapsed_sec": candidate.time_sec - start,
            "feedback": feedback,
        }

    def metrics(self):
        report = self._runtime._backend.mission.report()
        outcomes = {p["id"]: p for p in report["products"]}
        completed = []
        for flow in self.spec.streams:
            ids = [pid for _, fid, pid, _ in self._schedule if fid == flow.id]
            if not self._flow[flow.id]["cancelled"] and all(
                pid in outcomes and outcomes[pid]["status"] == "delivered_on_time"
                for pid in ids
            ):
                completed.append(flow.id)
        balances = report["inventory_ledger"]
        dtn = report["dtn"]
        return {
            "success": self._done and len(completed) == len(self.spec.streams),
            "terminal": self._done,
            "completed_flows": len(completed),
            "total_flows": len(self.spec.streams),
            "completed_priority": sum(self._flow[f]["priority"] for f in completed),
            "released_chunks": len(self._chunks),
            "rejected_chunks": sum(
                r["status"] == "rejected_capacity" for r in self._chunks.values()
            ),
            "delivered_bits": sum(
                outcomes[pid]["delivered_bits"]
                for pid in self._chunks
                if pid in outcomes
            ),
            "on_time_chunks": sum(
                outcomes[pid]["status"] == "delivered_on_time"
                for pid in self._chunks
                if pid in outcomes
            ),
            "cancelled_flows": sum(f["cancelled"] for f in self._flow.values()),
            "native_rejections": self._rejections,
            "service_started_sec": self._begin,
            "time_sec": self._runtime.time_sec,
            "maximum_balance_error_bits": max(
                [
                    *(abs(r["balance_error_bits"]) for r in balances.values()),
                    *(abs(v) for v in dtn["receiver_balance_error_bits"].values()),
                    abs(dtn["wire_balance_error_bits"]),
                ],
                default=0.0,
            ),
            "receiver_balance_error_bits": copy.deepcopy(
                dtn["receiver_balance_error_bits"]
            ),
            "wire_balance_error_bits": dtn["wire_balance_error_bits"],
            "native_coupled_executor": True,
        }

    def _witness(self):
        return {
            "native": self._runtime.checkpoint(compact=True),
            "chunks": copy.deepcopy(self._chunks),
            "flows": copy.deepcopy(self._flow),
            "events_done": sorted(self._events_done),
            "events": copy.deepcopy(self._event_log),
            "metrics": self.metrics(),
        }


def native_service_recipe(profile="nominal"):
    from space_env.tasks.heterogeneous_missions import heterogeneous_mission

    if profile not in {
        "nominal",
        "outage",
        "receiver_failure",
        "cancel_inflight",
        "pointing",
        "low_memory",
        "prefix",
    }:
        raise ValueError("unknown native service profile")
    case = heterogeneous_mission()
    scenario = case["scenario"]
    d = scenario["definition"]
    d["lifecycle"].update(
        requests=[], jobs=[], areas_of_interest=[], cloud_masks=[], payloads={}
    )
    ground = d["operations"]["radio_network"]["receivers"]["Ground"]
    ground.update(min_frequency_hz=8100000000.0, max_frequency_hz=8300000000.0)
    for actor, tx in d["operations"]["radio_network"]["transmitters"].items():
        if actor.startswith("relay-"):
            tx["frequency_hz"] = 8200000000.0
    scenario["id"] = "native-stream-" + profile
    scenario["lineage"] = "native-stream-formation/v1"
    cfg = {
        "scenario": scenario,
        "streams": [
            {
                "id": "routine",
                "source": "camera-00",
                "destination": "Ground",
                "size_bits": 400000,
                "chunk_bits": 100000,
                "interval_sec": 3.0,
                "deadline_sec": 40.0,
                "priority": 2,
            },
            {
                "id": "urgent",
                "source": "camera-01",
                "destination": "Ground",
                "size_bits": 300000,
                "chunk_bits": 100000,
                "release_sec": 4.0,
                "interval_sec": 3.0,
                "deadline_sec": 35.0,
                "priority": 9,
            },
        ],
    }
    if profile == "outage":
        cfg["events"] = [
            {
                "at_sec": 4.0,
                "type": "station_outage_added",
                "payload": {"station": "Ground", "start_sec": 4.0, "end_sec": 20.0},
            }
        ]
    elif profile == "receiver_failure":
        cfg["events"] = [
            {
                "at_sec": 4.0,
                "type": "spacecraft_fault",
                "payload": {"actor": "relay-00", "subsystem": "radio"},
            },
            {
                "at_sec": 18.0,
                "type": "spacecraft_fault_cleared",
                "payload": {"actor": "relay-00", "subsystem": "radio"},
            },
        ]
    elif profile == "cancel_inflight":
        d["operations"]["dtn_network"]["ack_processing_sec"] = 1.0
        cfg["events"] = [
            {"at_sec": 0.5, "type": "flow_cancelled", "payload": {"flow_id": "routine"}}
        ]
    elif profile == "pointing":
        for actor in ("relay-00", "relay-01"):
            d["operations"]["radio_network"]["transmitters"][actor]["beamwidth_deg"] = (
                10.0
            )
            d["spacecraft_bus"][actor]["attitude"]["initial_quaternion_wxyz"] = [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
    elif profile == "low_memory":
        d["operations"]["vehicles"]["camera-00"]["memory_bits"] = 150000.0
    elif profile == "prefix":
        from space_env.tasks.modeling_suite import _inputs

        engine = copy.deepcopy(
            _inputs()["chemical"]["definition"]["actuation"]["engines"]["a"]
        )
        d["actuation"] = {"engines": {"camera-00": engine}, "max_segments": 64}
        scenario["fidelity"]["trajectory"] = "coupled-gcrs-thrust"
        runtime = MissionRuntime(
            scenario, access=AccessPolicy(truth=True), audit_mode="compact"
        )
        runtime.submit(
            [
                {
                    "command_id": "prefix-burn",
                    "actor_id": "camera-00",
                    "kind": "body_burn",
                    "duration_sec": 3.0,
                    "body_burn": {
                        "frame": "BODY",
                        "direction_body": [0.0, 0.0, 1.0],
                        "thrust_n": 0.5,
                        "firing_sec": 2.0,
                    },
                }
            ],
            expected_revision=0,
            idempotency_key="prefix-burn",
        )
        runtime.advance(
            4.0, expected_revision=runtime.revision, idempotency_key="prefix-coast"
        )
        cfg.pop("scenario")
        cfg["prefix_checkpoint"] = runtime.checkpoint(compact=True)
        for stream in cfg["streams"]:
            stream["release_sec"] = stream.get("release_sec", 0.0) + 8.0
    return NativeServiceSpec.model_validate(cfg).model_dump(mode="json")


def native_service_policy(observation):
    actors = observation["actors"]
    busy = {
        r["command"]["actor_id"]
        for r in observation["commands"]
        if r["status"] in {"accepted", "running"}
    }
    for value in actors.values():
        busy.update(
            r["destination"]
            for r in value.get("dtn", {}).get("transfers", [])
            if r["status"] not in {"completed", "failed", "cancelled", "expired"}
        )
    used = busy.copy()
    stations = set()
    commands = []

    def healthy(actor):
        bus = actors[actor].get("spacecraft_bus", {})
        return not bus.get("safe") and not {"bus", "radio"} & set(bus.get("faults", []))

    products = sorted(
        (
            (a, p)
            for a, v in observation["inventory"].items()
            for p in v["products"]
            if not p["cancelled"]
            and not p["acknowledged"]
            and not p["custody_pinned"]
            and p["remaining_bits"] > 0
        ),
        key=lambda pair: (-pair[1]["priority"], pair[1]["deadline_sec"], pair[1]["id"]),
    )
    remaining = observation["horizon_sec"] - observation["time_sec"]
    for actor, p in products:
        if actor in used or not healthy(actor):
            continue
        ground = [
            r
            for r in observation["ground_links"]
            if r["actor"] == actor
            and r["station"] == p["destination"]
            and r["visible"]
            and not r["outage"]
            and r["station"] not in stations
        ]
        direct = next((r for r in ground if r["rate_bps"] > 0), None)
        if direct:
            commands.append(
                {
                    "actor_id": actor,
                    "kind": "downlink",
                    "duration_sec": min(2.0, remaining),
                    "destination_id": direct["station"],
                    "product_id": p["id"],
                }
            )
            stations.add(direct["station"])
            used.add(actor)
            continue
        pointing = next(
            (r for r in ground if r["body_antenna"] and r["band_compatible"]), None
        )
        if pointing:
            commands.append(
                {
                    "actor_id": actor,
                    "kind": "slew",
                    "duration_sec": min(10.0, remaining),
                    "slew": {"frame": "GCRS", "station_id": pointing["station"]},
                }
            )
            used.add(actor)
            continue
        link = next(
            (
                l
                for l in observation["isl_links"]
                if l["source"] == actor
                and l["rate_bps"] > 0
                and l["destination"] not in used
                and healthy(l["destination"])
            ),
            None,
        )
        if link:
            commands.append(
                {
                    "actor_id": actor,
                    "kind": "relay",
                    "duration_sec": min(2.0, remaining),
                    "destination_id": link["destination"],
                    "product_id": p["id"],
                    "relay": {"ttl_sec": remaining},
                }
            )
            used.update((actor, link["destination"]))
    return {"commands": commands}
