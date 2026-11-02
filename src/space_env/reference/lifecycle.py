"""Explicit request/acquire/transform/deliver lifecycle over the resource executor.

Products are versioned synthetic resource metadata, not rendered images. Storage
is backed by the existing transport inventory, with a separate provenance ledger.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import math
import re
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from space_env.reference.acquisition import acquisition_geometry
from space_env.reference.actuation import CoupledActuation
from space_env.reference.area_coverage import (
    AreaOfInterest,
    AreaRequirement,
    RectangularFootprint,
    area_report,
    coverage_from_samples,
    coverage_satisfied,
    domain,
    geometry_manifest,
    merge_coverage,
    rectangular_footprint,
)
from space_env.reference.observations import (
    ObservationBudgetError,
    ObservationRequirement,
    ObservationSettings,
    assemble_products,
    capture_observation,
    evaluate_observations,
    merge_observations,
)
from space_env.reference.payload_quality import (
    CloudMaskCell,
    OpticalModel,
    PayloadQuality,
    QualityEvidenceBudgetError,
    QualityRequirement,
    QualitySampling,
    SarModel,
    merge_evidence,
    usable_coverage,
)
from space_env.reference.resource_lanes import ResourceLanes
from space_env.reference.unloading import MomentumUnloader
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest

_TERMINAL = {"completed", "failed", "cancelled", "expired"}


class CustomerRequest(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    target_id: str = Field(min_length=1)
    eligible_actors: tuple[str, ...] = Field(min_length=1, max_length=64)
    release_sec: float = Field(default=0.0, ge=0, strict=True)
    deadline_sec: float = Field(gt=0, strict=True)
    priority: int = Field(default=1, ge=0, strict=True)
    acquisition_sec: float = Field(gt=1e-6, strict=True)
    raw_product_id: str = Field(min_length=1, max_length=128)
    raw_size_bits: int = Field(gt=0, le=2**53, strict=True)
    raw_ttl_sec: float | None = Field(default=None, gt=0, strict=True)
    delivery_product_id: str = Field(min_length=1, max_length=128)
    destination_id: str = Field(min_length=1)
    max_attempts: int = Field(default=1, ge=1, le=100, strict=True)
    coverage: AreaRequirement | None = None
    quality: QualityRequirement | None = None
    observations: ObservationRequirement | None = None

    @model_validator(mode="after")
    def order(self):
        if not self.release_sec < self.deadline_sec or len(
            set(self.eligible_actors)
        ) != len(self.eligible_actors):
            raise ValueError(
                "request requires ordered release/deadline and unique eligible actors"
            )
        if any(
            not v.strip()
            for v in (
                self.id,
                self.target_id,
                self.raw_product_id,
                self.delivery_product_id,
                self.destination_id,
                *self.eligible_actors,
            )
        ):
            raise ValueError("request identifiers must be nonblank")
        return self


class OutputProduct(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    size_bits: int = Field(gt=0, le=2**53, strict=True)
    stage: Literal["intermediate", "final"] = "final"
    ttl_sec: float | None = Field(default=None, gt=0, strict=True)


class TransformationJob(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    inputs: tuple[str, ...] = Field(min_length=1, max_length=128)
    outputs: tuple[OutputProduct, ...] = Field(min_length=1, max_length=128)
    work_sec: float = Field(gt=1e-6, strict=True)
    workspace_bits: int = Field(default=0, ge=0, le=2**53, strict=True)
    power_w: float = Field(default=0.0, ge=0, strict=True)
    min_soc: float = Field(default=0.1, ge=0, lt=1, strict=True)
    switching_sec: float = Field(default=0.0, ge=0, strict=True)
    transformer: str = Field(
        default="metadata-surrogate/1", min_length=1, max_length=128
    )
    consume_inputs: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def unique(self):
        if (
            not self.id.strip()
            or len(set(self.inputs)) != len(self.inputs)
            or any(not p.strip() for p in self.inputs)
        ):
            raise ValueError("job requires nonblank ID and unique input IDs")
        if len({p.id for p in self.outputs}) != len(self.outputs) or any(
            not p.id.strip() for p in self.outputs
        ):
            raise ValueError("job outputs must be unique and nonblank")
        return self


class PayloadSettings(StrictModel):
    footprint: RectangularFootprint | None = None
    imaging_model: OpticalModel | SarModel | None = Field(default=None, discriminator="kind")
    power_w: float = Field(default=20.0, ge=0, strict=True)
    min_soc: float = Field(default=0.1, ge=0, lt=1, strict=True)


class LifecycleSpec(StrictModel):
    observation_settings: ObservationSettings = Field(default_factory=ObservationSettings)
    cloud_masks: tuple[CloudMaskCell, ...] = Field(default=(), max_length=4096)
    quality_sampling: QualitySampling = Field(default_factory=QualitySampling)
    areas_of_interest: tuple[AreaOfInterest, ...] = Field(default=(), max_length=128)
    coverage_geometry_manifest: dict[str, str] = Field(default_factory=dict)
    requests: tuple[CustomerRequest, ...] = Field(default=(), max_length=10000)
    jobs: tuple[TransformationJob, ...] = Field(default=(), max_length=10000)
    payloads: dict[str, PayloadSettings]
    execution_step_sec: float = Field(default=0.5, ge=0.01, le=5, strict=True)
    allow_final_discard: bool = Field(default=False, strict=True)
    transformer_manifests: dict[str, str] = Field(default_factory=dict)
    max_execution_steps: int = Field(default=100000, ge=1, le=1000000, strict=True)


class ProductTransformer(Protocol):
    version: str
    source_sha256: str

    def materialize(self, inputs: list[dict], output: OutputProduct) -> dict: ...


class MetadataTransformer:
    version = "metadata-surrogate/1"

    @property
    def source_sha256(self):
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def materialize(self, inputs, output):
        return {
            "representation": "synthetic-resource-metadata",
            "transformer": self.version,
            "content_sha256": digest(
                {
                    "version": self.version,
                    "inputs": [p["content_sha256"] for p in inputs],
                    "output": output.model_dump(mode="json"),
                }
            ),
            "quality": {
                "model": "unqualified-surrogate",
                "scientific_image_algorithm": False,
            },
        }


class GeometryFeatureTransformer(MetadataTransformer):
    version = "geometry-features/1"

    def materialize(self, inputs, output):
        payload = [
            {
                "product_id": row["id"],
                "acquisition": row.get("acquisition"),
                "parent_payload": row.get("payload"),
            }
            for row in inputs
        ]
        return {
            "representation": "json-geometry-features/1",
            "payload": payload,
            "content_sha256": digest(payload),
            "transformer": self.version,
            "size_model": "declared resource bits, not serialized JSON byte count",
            "quality": {"scientific_image_algorithm": False},
        }


_TRANSFORMER_FACTORIES = {
    "metadata-surrogate/1": MetadataTransformer,
    "geometry-features/1": GeometryFeatureTransformer,
}


def register_product_transformer(factory):
    """Trusted SDK extension. Must be deterministic, JSON-producing and deepcopyable.

    source_sha256 is caller-declared provenance for external code, not attestation.
    Loading a checkpoint requires the same version/fingerprint to be registered.
    """
    plugin = factory()
    if (
        not isinstance(plugin.version, str)
        or not plugin.version.strip()
        or not re.fullmatch(r"[a-f0-9]{64}", plugin.source_sha256)
    ):
        raise ValueError("transformer requires version and SHA256 fingerprint")
    if plugin.version in _TRANSFORMER_FACTORIES:
        raise ValueError("transformer version already registered; use a new version")
    _TRANSFORMER_FACTORIES[plugin.version] = factory


def resolve_transformers(manifests, jobs):
    resolved = dict(manifests)
    for name in ("metadata-surrogate/1", "geometry-features/1"):
        resolved.setdefault(name, _TRANSFORMER_FACTORIES[name]().source_sha256)
    if any(job.transformer not in resolved for job in jobs.values()):
        raise ValueError("external transformer requires explicit scenario fingerprint")
    plugins = {}
    for name, expected in resolved.items():
        if name not in _TRANSFORMER_FACTORIES:
            raise ValueError("required transformer version is not registered")
        plugin = _TRANSFORMER_FACTORIES[name]()
        if (
            not re.fullmatch(r"[a-f0-9]{64}", expected)
            or plugin.version != name
            or plugin.source_sha256 != expected
        ):
            raise ValueError("transformer fingerprint mismatch")
        plugins[name] = plugin
    return resolved, plugins


class MaterializationError(ValueError):
    """Declared transform failure: work/energy spent, but no output is committed."""


def graph(requests, jobs):
    producers = {}
    origins = {}
    for request in requests.values():
        if request.raw_product_id in producers:
            raise ValueError("duplicate product ID")
        producers[request.raw_product_id] = None
        origins[request.raw_product_id] = {request.id}
    for job in jobs.values():
        for output in job.outputs:
            if output.id in producers:
                raise ValueError("duplicate product ID")
            producers[output.id] = job.id
    if len(producers) > 10000:
        raise ValueError("lifecycle product limit exceeded")
    incoming, children = {}, {name: set() for name in jobs}
    for name, job in jobs.items():
        if any(p not in producers for p in job.inputs):
            raise ValueError("unknown input in lifecycle graph")
        parents = {producers[p] for p in job.inputs if producers[p] is not None}
        incoming[name] = len(parents)
        for parent in parents:
            children[parent].add(name)
    ready = [name for name, count in incoming.items() if count == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        name = heapq.heappop(ready)
        visited += 1
        ancestry = set().union(*(origins[p] for p in jobs[name].inputs))
        for output in jobs[name].outputs:
            origins[output.id] = ancestry
        for child in sorted(children[name]):
            incoming[child] -= 1
            if incoming[child] == 0:
                heapq.heappush(ready, child)
    if visited != len(jobs):
        raise ValueError("lifecycle job graph cycle")
    destinations = {}
    for request in requests.values():
        if (
            request.delivery_product_id not in origins
            or request.id not in origins[request.delivery_product_id]
        ):
            raise ValueError("delivery product must descend from its request")
        if (
            request.delivery_product_id in destinations
            and destinations[request.delivery_product_id] != request.destination_id
        ):
            raise ValueError("shared delivery product requires one destination")
        destinations[request.delivery_product_id] = request.destination_id
        if request.observations is not None:
            for pid in request.observations.support_product_ids:
                if pid == request.delivery_product_id or pid not in producers:
                    raise ValueError("support product must be a distinct managed product")
                if pid in destinations and destinations[pid] != request.destination_id:
                    raise ValueError("support product requires one delivery destination")
                destinations[pid] = request.destination_id
    return producers, origins, destinations


class ProductLifecycle:
    def __init__(self, spec: LifecycleSpec, backend):
        self.spec = spec
        self.thermal_network = None
        self.actuation = None if backend.config.actuation is None else CoupledActuation(backend.config.actuation, backend)
        self.unloader = None if backend.config.unloading is None else MomentumUnloader(backend.config.unloading, backend)
        self.scheduler = None if backend.config.execution is None else ResourceLanes(backend.config.execution)
        self.aois = {a.id: a for a in spec.areas_of_interest}
        if len(self.aois) != len(spec.areas_of_interest):
            raise ValueError("duplicate AOI ID")
        self.coverage_geometry_manifest = {}
        if self.aois:
            if backend.config.spacecraft_bus is None:
                raise ValueError("area acquisition requires explicit coupled spacecraft attitude")
            self.coverage_geometry_manifest = geometry_manifest()
            if spec.coverage_geometry_manifest and spec.coverage_geometry_manifest != self.coverage_geometry_manifest:
                raise ValueError("coverage geometry manifest mismatch")
            for aoi in self.aois.values():
                domain(aoi)
        elif spec.coverage_geometry_manifest:
            raise ValueError("geometry manifest requires AOI definitions")
        self.requests = {r.id: r for r in spec.requests}
        self.jobs = {j.id: j for j in spec.jobs}
        if len(self.requests) != len(spec.requests) or len(self.jobs) != len(spec.jobs):
            raise ValueError("duplicate request or job ID")
        if set(spec.payloads) - set(backend.actors):
            raise ValueError("unknown payload actor")
        self.producers, self.origins, self.destinations = graph(
            self.requests, self.jobs
        )
        if set(self.producers) & set(backend.mission.products):
            raise ValueError("managed/preloaded product ID collision")
        for r in self.requests.values():
            self._validate_request(r, backend)
        self.quality = None
        if spec.cloud_masks or any(row.imaging_model is not None for row in spec.payloads.values()):
            if not self.aois:
                raise ValueError("typed imaging models/cloud masks require explicit coupled area scenario")
            self.quality = PayloadQuality(spec.payloads, spec.cloud_masks, spec.quality_sampling, self.aois, backend.horizon())
        self.request_state = {
            r.id: {"status": "pending", "attempts": 0, "actor": None, "reason": None}
            for r in spec.requests
        }
        self.job_state = {
            j.id: {
                "status": "pending",
                "remaining_sec": j.work_sec,
                "work_sec": 0.0,
                "switching_sec": 0.0,
                "setup_remaining_sec": 0.0,
                "reserved": False,
                "actor": None,
                "reason": None,
            }
            for j in spec.jobs
        }
        self.transport = backend.mission.dtn
        self.artifacts = {}
        self.operations = {}
        self.events = []
        self.transitions = []
        self.last_job = {}
        self.consumed_by = {}
        self.removals = []
        self.charges = []
        self.transformer_manifests, self.transformers = resolve_transformers(
            spec.transformer_manifests, self.jobs
        )
        self.transformer = self.transformers["metadata-surrogate/1"]
        self.settle(backend)

    def _validate_request(self, r, backend):
        if r.observations is not None and r.coverage is None:
            raise ValueError("revisit/stereo requires explicit AOI coverage")
        if r.quality is not None and r.coverage is None:
            raise ValueError("quality requirement requires explicit AOI coverage")
        for actor in r.eligible_actors:
            payload = self.spec.payloads.get(actor)
            model = None if payload is None else payload.imaging_model
            if model is not None and r.coverage is None:
                raise ValueError("typed imaging model requires area acquisition")
            if r.quality is not None and (model is None or model.kind != r.quality.kind):
                raise ValueError("quality requirement must match every eligible actor imaging model")
        if r.coverage is not None:
            if r.coverage.aoi_id not in self.aois:
                raise ValueError("unknown coverage AOI")
            if any(a not in self.spec.payloads or self.spec.payloads[a].footprint is None
                   for a in r.eligible_actors):
                raise ValueError("area request requires explicit rectangular footprint for every actor")
        if (
            r.deadline_sec > backend.horizon()
            or r.target_id not in backend.acquisition_targets
        ):
            raise ValueError("unknown request target or deadline outside mission")
        if r.destination_id not in backend.mission.stations:
            raise ValueError("unknown delivery destination")
        if any(
            a not in backend.config.acquisition_sensors or a not in self.spec.payloads
            for a in r.eligible_actors
        ):
            raise ValueError(
                "request actor requires sensor geometry and payload resource model"
            )

    def transition(self, rid, status, now, reason=None):
        state = self.request_state[rid]
        allowed = {
            "pending": {"released", "cancelled", "expired", "failed"},
            "released": {"accepted", "cancelled", "expired", "failed"},
            "accepted": {"scheduled"},
            "scheduled": {"running", "released", "failed", "cancelled", "expired"},
            "running": {"completed", "released", "failed", "cancelled", "expired"},
        }
        if state["status"] == status:
            return
        if status not in allowed.get(state["status"], set()):
            raise ValueError("invalid request state transition")
        self.transitions.append(
            {
                "request_id": rid,
                "from": state["status"],
                "to": status,
                "time_sec": now,
                "reason": reason,
            }
        )
        state.update(status=status, reason=reason)

    def occupied(self, backend, actor):
        return sum(
            backend.mission.remaining[p.id]
            for p in backend.mission.products.values()
            if p.actor == actor
        ) + backend.mission.transport_memory(actor)

    def reservations(self, actor, *, exclude=None):
        jobs = sum(
            self.jobs[j].workspace_bits + sum(p.size_bits for p in self.jobs[j].outputs)
            for j, row in self.job_state.items()
            if row["reserved"] and row["actor"] == actor and j != exclude
        )
        acquisitions = sum(
            self.requests[row["request_id"]].raw_size_bits
            for row in self.operations.values()
            if row["kind"] == "imaging"
            and row["status"] == "running"
            and row["actor"] == actor
        )
        return jobs + acquisitions

    def pinned(self, pid):
        transport = ["@custody"] if self.transport is not None and self.transport.pinned(pid) else []
        return transport + [
            j
            for j, row in self.job_state.items()
            if pid in self.jobs[j].inputs and row["reserved"]
        ]

    def retained(self, pid):
        return [
            j
            for j, row in self.job_state.items()
            if pid in self.jobs[j].inputs and row["status"] not in _TERMINAL
        ]

    def deliverable(self, pid):
        return pid not in self.producers or (
            pid in self.destinations and not self.retained(pid)
        )

    def validate(self, command, backend):
        if command.kind == "imaging":
            r = self.requests.get(command.request_id)
            if r is None or command.actor_id not in r.eligible_actors:
                raise ValueError("unknown request or ineligible actor")
            state = self.request_state[r.id]
            if (
                state["status"] != "released"
                or not r.release_sec <= backend.now() < r.deadline_sec
            ):
                raise ValueError("request is not released/available")
            if command.duration_sec < r.acquisition_sec:
                raise ValueError("imaging window shorter than required acquisition")
        elif command.kind == "transform":
            if (
                command.job_id not in self.jobs
                or self.job_state[command.job_id]["status"] in _TERMINAL
            ):
                raise ValueError("unknown or terminal transform job")
            ancestry = set().union(
                *(self.origins[p] for p in self.jobs[command.job_id].inputs)
            )
            if any(self.requests[r].release_sec > backend.now() for r in ancestry):
                raise ValueError("job demand has not been released")
            if any(
                command.actor_id not in self.requests[r].eligible_actors
                for r in ancestry
            ):
                raise ValueError("transform actor cannot acquire all required inputs")
            if self.job_state[command.job_id]["actor"] not in {None, command.actor_id}:
                raise ValueError("persistent workspace belongs to another actor")
            if any(
                p in self.artifacts and self.artifacts[p]["actor"] != command.actor_id
                for p in self.jobs[command.job_id].inputs
            ):
                raise ValueError("transform inputs must belong to command actor")
        elif command.kind == "relay" and not self.deliverable(command.product_id):
            raise ValueError("managed relay requires an unretained final deliverable")
        elif (
            command.kind == "downlink"
            and command.product_id
            and not self.deliverable(command.product_id)
        ):
            raise ValueError("raw/intermediate or retained input is not deliverable")
        elif command.kind == "processing" and command.product_id in self.producers:
            raise ValueError("managed products require explicit transform jobs")

    def admit(self, command, backend):
        self.validate(command, backend)
        if command.kind == "imaging":
            r = self.requests[command.request_id]
            state = self.request_state[r.id]
            state["attempts"] += 1
            state["actor"] = command.actor_id
            self.transition(r.id, "accepted", backend.now())
            self.transition(r.id, "scheduled", backend.now())
        self.operations[command.command_id] = {
            "kind": command.kind,
            "actor": command.actor_id,
            "request_id": command.request_id,
            "job_id": command.job_id,
            "status": "accepted",
            "progress_sec": 0.0,
            "switch_remaining_sec": 0.0,
            "reason": None,
            "until_sec": backend.now() + command.duration_sec,
        }

    def fail_acquisition(self, operation, backend, reason):
        operation.update(status="failed", reason=reason)
        if getattr(backend, "bus", None) is not None:
            backend.bus.stop_tracking(operation["actor"])
        rid = operation["request_id"]
        state = self.request_state[rid]
        if state["status"] not in _TERMINAL:
            self.transition(
                rid,
                "released"
                if state["attempts"] < self.requests[rid].max_attempts
                else "failed",
                backend.now(),
                reason,
            )

    def cancel(self, cid, backend):
        operation = self.operations.get(cid)
        if operation is None or operation["status"] in _TERMINAL:
            return False
        if operation["kind"] == "imaging":
            self.fail_acquisition(operation, backend, "operator_cancelled")
        else:
            state = self.job_state[operation["job_id"]]
            if state["status"] not in _TERMINAL:
                state.update(
                    status="paused",
                    reason="command_cancelled",
                    setup_remaining_sec=operation["switch_remaining_sec"],
                )
        operation["status"] = "cancelled"
        return True

    def _native_event(self, backend, kind, payload):
        mission = backend.mission
        event = {
            "id": f"@lifecycle/{len(mission.events)}",
            "type": kind,
            "time_sec": backend.now(),
            "source": "reference-lifecycle/1",
            "reason": "managed inventory operation",
            "payload": payload,
        }
        mission.apply_events([event])

    def add_artifacts(self, backend, rows):
        # Peak allocation: materialize all outputs BEFORE consuming input bytes.
        actor = rows[0]["actor"]
        if (
            self.occupied(backend, actor) + sum(p["size_bits"] for p in rows)
            > backend.mission.vehicles[actor].memory_bits
        ):
            raise MaterializationError("output materialization exceeds memory")
        for row in rows:
            if row["id"] in self.artifacts or row["id"] in backend.mission.products:
                raise MaterializationError("duplicate materialization")
        if backend.now() >= backend.horizon():
            raise MaterializationError(
                "materialization at mission horizon is unsupported"
            )
        events = []
        for index, row in enumerate(rows):
            events.append(
                {
                    "id": f"@lifecycle/{len(backend.mission.events) + index}",
                    "type": "onboard_product_added",
                    "time_sec": backend.now(),
                    "source": "reference-lifecycle/1",
                    "reason": "atomic managed materialization",
                    "payload": {
                        "id": row["id"],
                        "actor": row["actor"],
                        "size_bits": row["size_bits"],
                        "ready_sec": backend.now(),
                        "deadline_sec": backend.horizon(),
                        "priority": max(
                            self.requests[r].priority for r in row["request_ids"]
                        ),
                        "destination": self.destinations.get(row["id"], ""),
                    },
                }
            )
        try:
            backend.mission.apply_events(events)
        except ValueError as error:
            raise MaterializationError(str(error)) from error
        self.artifacts.update({row["id"]: row for row in rows})

    def remove_artifact(self, pid, backend, category, *, allow_pinned=False):
        if self.pinned(pid) and not allow_pinned:
            raise ValueError("artifact is pinned by persistent job workspace")
        remaining = backend.mission.remaining[pid]
        if remaining > 0:
            if backend.now() >= backend.horizon():
                # Terminal housekeeping is not an operator event after horizon.
                mission = backend.mission
                mission.discarded[pid] += remaining
                mission.remaining[pid] = 0.0
                mission.cancelled[pid] = backend.now()
                mission.processing_remaining[pid] = 0.0
            else:
                self._native_event(backend, "product_cancelled", {"product_id": pid})
            self.removals.append(
                {
                    "product_id": pid,
                    "actor": self.artifacts[pid]["actor"],
                    "category": category,
                    "bits": remaining,
                    "time_sec": backend.now(),
                }
            )
        self.artifacts[pid]["status"] = category
        if category in {"expired", "discarded"}:
            self.fail_unreachable(
                backend, f"required_artifact_{category}", product_id=pid
            )

    def _needs(self, goal, *, product_id=None, job_id=None):
        # Iterative walk: valid deep DAGs must not hit Python recursion limits.
        pending, seen = [goal], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current == product_id:
                return True
            if current in self.artifacts and self.artifacts[current]["status"] in {
                "resident",
                "delivered",
            }:
                continue
            producer = self.producers[current]
            if producer is None:
                continue
            if producer == job_id:
                return True
            pending.extend(self.jobs[producer].inputs)
        return False

    def fail_unreachable(self, backend, reason, *, product_id=None, job_id=None):
        for rid, request in self.requests.items():
            if self.request_state[rid]["status"] not in _TERMINAL and any(
                self._needs(pid, product_id=product_id, job_id=job_id) for pid in self.delivery_ids(request)
            ):
                self.transition(rid, "failed", backend.now(), reason)

    def _consume_ready(self, backend):
        for pid in sorted(self.consumed_by):
            if not self.retained(pid) and self.artifacts[pid]["status"] == "resident":
                self.remove_artifact(pid, backend, "transformed_in")

    @staticmethod
    def delivery_ids(request):
        return (request.delivery_product_id, *(request.observations.support_product_ids if request.observations else ()))

    def receipt_time(self, request, backend):
        times = []
        for pid in self.delivery_ids(request):
            artifact = self.artifacts.get(pid)
            if artifact is None or pid in backend.mission.cancelled or backend.mission.remaining[pid] != 0:
                return None
            receipts = [row for row in backend.mission.receipts if row["product_id"] == pid and row["station"] == request.destination_id]
            if not math.isclose(math.fsum(row["bits"] for row in receipts), artifact["size_bits"],
                                rel_tol=0, abs_tol=4 * math.ulp(float(artifact["size_bits"]))):
                return None
            ack = backend.mission.acknowledged_time(pid)
            if ack is None:
                return None
            times.append(ack)
        return max(times)

    def request_product(self, request):
        products = [self.artifacts.get(pid) for pid in self.delivery_ids(request)]
        if any(row is None for row in products):
            return None
        if len(products) == 1:
            return products[0]
        try:
            return assemble_products(products, self.spec.observation_settings.max_records_per_product,
                                     self.spec.quality_sampling.max_evidence_records)
        except (ObservationBudgetError, QualityEvidenceBudgetError) as error:
            return {"observation_error": str(error)}

    def evaluate_request(self, request, product):
        if request.observations is None:
            return None
        if product is not None and "observation_error" in product:
            return {"status": "budget_exceeded", "reason": product["observation_error"],
                    "satisfied": False, "joint_area_m2": 0.0, "joint_fraction": 0.0}
        return evaluate_observations(request.observations, product, self.aois[request.coverage.aoi_id],
                                     request.quality, request.coverage.min_fraction, self.spec.observation_settings)

    def settle(self, backend):
        now = backend.now()
        mission = backend.mission
        if self.transport is not None:
            self.transport.settle(mission)
            self.transport.sync_artifacts(mission,self.artifacts)
            for pid, expired in self.transport.expired_products.items():
                if pid in self.artifacts and self.artifacts[pid]["status"] == "resident":
                    self.removals.append({"product_id":pid,"actor":expired["actor"],"category":"expired","bits":expired["bits"],"time_sec":expired["time_sec"]})
                    self.artifacts[pid]["status"] = "expired"
                    self.fail_unreachable(backend,"required_artifact_expired",product_id=pid)
        for rid, r in self.requests.items():
            state = self.request_state[rid]
            if state["status"] == "pending" and now >= r.release_sec:
                self.transition(rid, "released", now)
            if state["status"] not in _TERMINAL:
                completed = self.receipt_time(r, backend)
                if completed is not None and completed <= r.deadline_sec + 4 * math.ulp(r.deadline_sec):
                    product = self.request_product(r)
                    geometry_ok = coverage_satisfied(r.coverage, product or {})
                    quality_ok = r.quality is None or usable_coverage(
                        r.quality, product.get("area_coverage", {}).get(r.coverage.aoi_id),
                        product.get("quality_evidence", []))["usable_fraction"] >= r.coverage.min_fraction
                    observation = self.evaluate_request(r, product)
                    if geometry_ok and quality_ok and (observation is None or observation["satisfied"]):
                        self.transition(rid, "completed", now)
                        state["delivered_sec"] = completed
                    else:
                        reason = ("observation_budget" if observation is not None and observation["status"] == "budget_exceeded"
                                  else "coverage_requirement" if not geometry_ok else "quality_requirement" if not quality_ok
                                  else "observation_requirement")
                        self.transition(rid, "failed", now, reason)
            if state["status"] not in _TERMINAL and now >= r.deadline_sec:
                self.transition(rid, "expired", now, "delivery_deadline")
        for pid, artifact in self.artifacts.items():
            if artifact["status"] != "resident":
                continue
            if mission.acknowledged_time(pid) is not None:
                artifact["status"] = "delivered"
            elif (
                artifact["expires_sec"] is not None
                and now >= artifact["expires_sec"]
                and not self.pinned(pid)
            ):
                self.remove_artifact(pid, backend, "expired")
        self._consume_ready(backend)

    def _artifact(self, pid, size, actor, stage, request_ids, ttl, metadata, now):
        return {
            **metadata,
            "id": pid,
            "size_bits": size,
            "actor": actor,
            "stage": stage,
            "request_ids": sorted(request_ids),
            "created_sec": now,
            "expires_sec": None if ttl is None else now + ttl,
            "status": "resident",
        }

    def _finish_acquisition(self, op, backend):
        r = self.requests[op["request_id"]]
        metadata = {
            "representation": "synthetic-resource-metadata",
            "content_sha256": digest(
                {
                    "request": r.model_dump(mode="json"),
                    "actor": op["actor"],
                    "time_sec": backend.now(),
                    "trajectory": backend.mission.ephemerides[op["actor"]].sha256,
                }
            ),
            "quality": {"model": "point-dwell-surrogate/1", "rendered_image": False},
            "acquisition": {
                "target_id": r.target_id,
                "start_sec": backend.now() - op["progress_sec"],
                "end_sec": backend.now(),
                "validated_dwell_sec": op["progress_sec"],
                "execution_step_sec": self.spec.execution_step_sec,
                "coverage_model": "point-dwell; no AOI area coverage",
                "geometry_end": acquisition_geometry(
                    backend.mission.ephemerides[op["actor"]],
                    backend.acquisition_targets[r.target_id],
                    backend.config.acquisition_sensors[op["actor"]],
                    math.nextafter(backend.now(), 0.0),
                ),
            },
            "source": "acquired",
        }
        if r.coverage is not None:
            try:
                area = coverage_from_samples(self.aois[r.coverage.aoi_id], op.get("coverage_samples", []))
            except ValueError as error:
                raise MaterializationError(str(error)) from error
            area["sources"] = {r.raw_product_id: {"sample_count": area["sample_count"],
                                                   "actor": op["actor"], "end_sec": backend.now()}}
            metadata["area_coverage"] = {area["aoi_id"]: area}
            metadata["acquisition"]["coverage_model"] = area["model"]
            metadata["acquisition"]["footprint_samples"] = copy.deepcopy(op.get("coverage_samples", []))
            metadata["acquisition"]["geometry_manifest"] = dict(self.coverage_geometry_manifest)
            metadata["content_sha256"] = digest({"base": metadata["content_sha256"], "area": area})
        if self.quality is not None and op["actor"] in self.quality.models:
            metadata["quality_evidence"] = copy.deepcopy(op.get("quality_evidence", []))
            metadata["sensor_model"] = self.quality.models[op["actor"]].model_dump(mode="json")
            metadata["content_sha256"] = digest({"base": metadata["content_sha256"], "quality": metadata["quality_evidence"], "model": metadata["sensor_model"]})
        if r.coverage is not None:
            metadata["observations"] = copy.deepcopy(op.get("observations", []))
            metadata["content_sha256"] = digest({"base": metadata["content_sha256"], "observations": metadata["observations"]})
        if getattr(backend, "bus", None) is not None:
            metadata["spacecraft_bus"] = backend.bus.observation(op["actor"])
        row = self._artifact(
            r.raw_product_id,
            r.raw_size_bits,
            op["actor"],
            "raw",
            [r.id],
            r.raw_ttl_sec,
            metadata,
            backend.now(),
        )
        self.add_artifacts(backend, [row])
        op["status"] = "completed"

    def _finish_job(self, op, backend):
        name = op["job_id"]
        job = self.jobs[name]
        state = self.job_state[name]
        inputs = [self.artifacts[p] for p in job.inputs]
        ancestry = set().union(*(p["request_ids"] for p in inputs))
        plugin = (
            self.transformer
            if job.transformer == "metadata-surrogate/1"
            else self.transformers[job.transformer]
        )
        if plugin.version != job.transformer:
            raise MaterializationError("transformer version mismatch")
        area_rows = [row for product in inputs for row in product.get("area_coverage", {}).values()]
        inherited_area = merge_coverage(area_rows) if area_rows else {}
        try:
            inherited_quality = merge_evidence(inputs, self.spec.quality_sampling.max_evidence_records)
            inherited_observations = merge_observations(inputs, self.spec.observation_settings.max_records_per_product)
        except ValueError as error:
            raise MaterializationError(str(error)) from error
        rows = [
            self._artifact(
                p.id,
                p.size_bits,
                op["actor"],
                p.stage,
                ancestry,
                p.ttl_sec,
                {
                    **plugin.materialize(copy.deepcopy(inputs), p),
                    "source": "transformed_out",
                    "job_id": name,
                    **({"area_coverage": copy.deepcopy(inherited_area)} if inherited_area else {}),
                    **({"quality_evidence": copy.deepcopy(inherited_quality)} if inherited_quality else {}),
                    **({"observations": copy.deepcopy(inherited_observations)} if inherited_observations else {}),
                },
                backend.now(),
            )
            for p in job.outputs
        ]
        self.add_artifacts(backend, rows)
        state.update(status="completed", reserved=False, reason=None)
        op["status"] = "completed"
        if job.consume_inputs:
            for pid in job.inputs:
                self.consumed_by.setdefault(pid, []).append(name)
        self._consume_ready(backend)

    def actuation_power(self, actor):
        return sum(m.power(actor) for m in (self.actuation, self.unloader) if m is not None)

    def actuation_floor(self, actor):
        return max([m.floor(actor) for m in (self.actuation, self.unloader) if m is not None]+[self.thermal_heater_floor(actor)])

    def thermal_heater_floor(self, actor):
        return self.thermal_network.heater_floor(actor) if self.thermal_network is not None else 0.

    def imaging_power(self, actor):
        base = self.spec.payloads[actor].power_w
        return base if self.quality is None else self.quality.power(actor, base)

    def advance(self, backend, dt, active, native_advance):
        if self.scheduler is not None:
            def execution_order(row):
                command = row["command"]
                lane = {"imaging": "payload", "transform": "cpu", "downlink": "radio"}.get(command["kind"])
                rank = self.scheduler.config.actors[command["actor_id"]].priority.index(lane) if lane else -1
                return (backend.actors.index(command["actor_id"]), rank, command["command_id"])
            active = sorted(active, key=execution_order)
        end = backend.now() + dt
        bus = getattr(backend, "bus", None)
        count = 0
        while backend.now() < end - 1e-9:
            count += 1
            if count > self.spec.max_execution_steps:
                raise ValueError("lifecycle execution step budget exhausted")
            self.settle(backend)
            now = backend.now()
            mission = backend.mission
            if self.transport is not None:
                self.transport.managed_blocked = {r["command"]["actor_id"] for r in active if r["command"]["kind"] not in {"relay","wait"}}
            if bus is not None:
                bus.prepare(backend, active)
            if self.actuation is not None:
                self.actuation.prepare(backend, active)
            if self.unloader is not None:
                self.unloader.prepare(backend, active)
            right = min(
                end,
                (math.floor((now + 1e-9) / self.spec.execution_step_sec) + 1)
                * self.spec.execution_step_sec,
                (math.floor((now + 1e-9) / mission.resource_step_sec) + 1)
                * mission.resource_step_sec,
            )
            times = [
                t
                for r in self.requests.values()
                for t in (r.release_sec, r.deadline_sec)
                if now + 1e-9 < t < right
            ]
            times += [
                p["expires_sec"]
                for p in self.artifacts.values()
                if p["status"] == "resident"
                and p["expires_sec"] is not None
                and now + 1e-9 < p["expires_sec"] < right
            ]
            if times:
                right = min(times)
            if bus is not None:
                right = bus.boundary(backend, right)
            if self.actuation is not None:
                right = self.actuation.boundary(backend, right)
            if self.unloader is not None:
                right = self.unloader.boundary(backend, right)
            if mission.radio_network is not None:
                right = mission.radio_network.boundary(now, right)
            if self.transport is not None:
                right = self.transport.boundary(mission,right)
            powered = {}
            operation_power = {}
            work_rates = {}
            radio_rates = {}
            progressing = []
            if self.quality is not None:
                right = self.quality.boundary(now, right)
            ordinary = []
            for row in active:
                c = row["command"]
                cid, actor = c["command_id"], c["actor_id"]
                op = self.operations.get(cid)
                if op is None:
                    if bus is not None and not bus.ordinary_allowed(row, backend, span=right-now):
                        continue
                    if c["kind"] == "downlink" and not c["product_id"]:
                        ready = [
                            p
                            for p in mission.products.values()
                            if p.actor == actor
                            and self.deliverable(p.id)
                            and mission.remaining[p.id] > 0
                            and mission.processing_remaining[p.id] == 0
                            and not mission.retained_for(p.id)
                            and p.ready_sec <= now
                            and (
                                not p.destination
                                or p.destination == c["destination_id"]
                            )
                        ]
                        if not ready:
                            continue
                        selected = min(
                            ready, key=lambda p: (-p.priority, p.deadline_sec, p.id)
                        )
                        row = {**row, "command": {**c, "product_id": selected.id}}
                    if (
                        c["kind"] == "downlink"
                        and c["product_id"]
                        and not self.deliverable(c["product_id"])
                    ):
                        continue
                    ordinary.append(row)
                    continue
                if op["status"] in _TERMINAL:
                    continue
                if bus is not None:
                    watts = (
                        (self.spec.payloads[actor].power_w if self.quality is None else self.quality.admission_power(actor, self.spec.payloads[actor].power_w))
                        if op["kind"] == "imaging"
                        else self.jobs[op["job_id"]].power_w
                    )
                    reason = bus.allowed(actor, op["kind"], cid, backend, extra_w=watts)
                    if reason:
                        if op["kind"] == "imaging":
                            self.fail_acquisition(op, backend, reason)
                        else:
                            self.job_state[op["job_id"]].update(
                                status="paused", reason=reason
                            )
                        continue
                if op["kind"] == "imaging":
                    r = self.requests[op["request_id"]]
                    state = self.request_state[r.id]
                    if state["status"] in _TERMINAL:
                        op.update(status="failed", reason="request_terminal")
                        continue
                    if op["status"] == "accepted":
                        if (
                            self.occupied(backend, actor)
                            + self.reservations(actor)
                            + r.raw_size_bits
                            > mission.vehicles[actor].memory_bits
                        ):
                            self.fail_acquisition(op, backend, "memory")
                            continue
                        op["status"] = "running"
                        self.transition(r.id, "running", now)
                    right = min(right, now + r.acquisition_sec - op["progress_sec"])
                    if self.quality is not None:
                        duty_right = self.quality.duty_boundary(actor, now, right)
                        if duty_right - now <= 1e-9:
                            self.fail_acquisition(op, backend, "sar_duty")
                            continue
                        right = duty_right
                    power, floor = (
                        self.imaging_power(actor),
                        self.spec.payloads[actor].min_soc,
                    )
                else:
                    job = self.jobs[op["job_id"]]
                    state = self.job_state[job.id]
                    if any(
                        p not in self.artifacts
                        or self.artifacts[p]["status"] != "resident"
                        or mission.remaining[p] != self.artifacts[p]["size_bits"]
                        for p in job.inputs
                    ):
                        state.update(status="blocked", reason="missing_intact_input")
                        continue
                    if not state["reserved"]:
                        needed = job.workspace_bits + sum(
                            p.size_bits for p in job.outputs
                        )
                        if (
                            self.occupied(backend, actor)
                            + self.reservations(actor)
                            + needed
                            > mission.vehicles[actor].memory_bits
                        ):
                            state.update(status="blocked", reason="memory")
                            continue
                        state.update(reserved=True, actor=actor)
                    if state["actor"] != actor:
                        raise ValueError(
                            "persistent workspace belongs to another actor"
                        )
                    if op["status"] == "accepted":
                        op["switch_remaining_sec"] = (
                            job.switching_sec
                            if self.last_job.get(actor) != job.id
                            else state["setup_remaining_sec"]
                        )
                        self.last_job[actor] = job.id
                        op["status"] = "running"
                    state.update(status="running", reason=None)
                    if self.scheduler is None or op["switch_remaining_sec"] > 0:
                        right = min(right, now + (op["switch_remaining_sec"] or state["remaining_sec"]))
                    power, floor = job.power_w, job.min_soc
                sample = min(
                    (math.floor((now + 1e-9) / mission.resource_step_sec) + 0.5)
                    * mission.resource_step_sec,
                    backend.horizon(),
                )
                base = (
                    mission.vehicles[actor].base_power_w
                    if bus is None
                    else bus.base_power(actor, mission.vehicles[actor].base_power_w)
                    + bus.extra_power(actor)
                    + self.actuation_power(actor)
                )
                floor = max(floor, self.thermal_heater_floor(actor))
                load = base + power
                net = mission._generation(actor, sample) - (
                    load if bus is None else bus.battery_load(actor, load)
                )
                reserve = floor * mission.vehicles[actor].battery_wh
                idle_net = mission._generation(actor, sample) - (
                    base if bus is None else bus.battery_load(actor, base)
                )
                if (
                    mission.energy_wh[actor] < reserve - 1e-12
                    and idle_net > 0
                    and op["kind"] == "transform"
                ):
                    right = min(
                        right,
                        now + (reserve - mission.energy_wh[actor]) * 3600 / idle_net,
                    )
                if mission.energy_wh[actor] < reserve - 1e-12 or (
                    net < 0
                    and (mission.energy_wh[actor] - reserve) * 3600 / -net <= 1e-9
                ):
                    if op["kind"] == "imaging":
                        self.fail_acquisition(op, backend, "power")
                    else:
                        state.update(status="paused", reason="power")
                    continue
                if net < 0:
                    right = min(
                        right, now + (mission.energy_wh[actor] - reserve) * 3600 / -net
                    )
                powered[actor] = powered.get(actor, 0.0) + power
                operation_power[cid] = power
                progressing.append(op)
            if self.scheduler is not None:
                progressing, ordinary, work_rates, radio_rates, right, lane_loads, lane_denied = self.scheduler.allocate(
                    backend, progressing, ordinary, right)
                powered = {}
                for op in progressing:
                    cid = next(key for key, value in self.operations.items() if value is op)
                    powered[op["actor"]] = powered.get(op["actor"], 0.0) + operation_power[cid]
            def drop_power(operation, powered=powered, operation_power=operation_power):
                actor = operation["actor"]
                cid = next(key for key, value in self.operations.items() if value is operation)
                powered[actor] = max(0.0, powered.get(actor, 0.0) - operation_power[cid])
            span = right - now
            if span <= 1e-9:
                raise ValueError("lifecycle resource boundary made no progress")
            valid = []
            area_samples = {}
            quality_samples = {}
            observation_samples = {}
            for op in progressing:
                if op["kind"] == "imaging":
                    r = self.requests[op["request_id"]]
                    sensor = backend.config.acquisition_sensors[op["actor"]]
                    target = backend.acquisition_targets[r.target_id]
                    provider = mission.ephemerides[op["actor"]]
                    if bus is not None:
                        reason = bus.imaging_reason(
                            op["actor"],
                            target,
                            (now, now + span / 2, math.nextafter(right, now)),
                            backend,
                            op["request_id"],
                        )
                        if reason:
                            drop_power(op)
                            self.fail_acquisition(op, backend, reason)
                            continue
                    if not all(
                        acquisition_geometry(provider, target, sensor, t)[
                            "feasible_geometry"
                        ]
                        for t in (now, now + span / 2, math.nextafter(right, now))
                    ):
                        drop_power(op)
                        self.fail_acquisition(op, backend, "geometry")
                        continue
                    if self.quality is not None:
                        reason = self.quality.sar_reason(op["actor"], provider, target, (now, now + span / 2, math.nextafter(right, now)))
                        if reason:
                            drop_power(op)
                            self.fail_acquisition(op, backend, reason)
                            continue
                    if r.coverage is not None:
                        footprint = self.spec.payloads[op["actor"]].footprint
                        samples = [rectangular_footprint(provider,
                                     bus.predicted_quaternion(op["actor"], t - now), footprint, t)
                                   for t in (now, now + span / 2, math.nextafter(right, now))]
                        if any(sample["status"] != "ok" for sample in samples):
                            drop_power(op)
                            self.fail_acquisition(op, backend, "footprint_horizon")
                            continue
                        if len(op.get("coverage_samples", [])) + len(samples) > footprint.max_acquisition_samples:
                            raise ValueError("coverage sample budget exceeded")
                        area_samples[op["request_id"]] = samples
                        if len(op.get("observations", [])) + len(samples) > self.spec.observation_settings.max_records_per_product:
                            raise ValueError("observation record budget exceeded")
                        observation_samples[op["request_id"]] = [capture_observation(
                            provider, bus.predicted_quaternion(op["actor"], sample["time_sec"] - now),
                            self.aois[r.coverage.aoi_id], sample, r.raw_product_id, op["actor"]) for sample in samples]
                        if self.quality is not None:
                            quality_samples[op["request_id"]] = self.quality.evidence(
                                op["actor"], r.coverage.aoi_id, samples, provider, r.raw_product_id,
                                self.spec.quality_sampling.max_evidence_records - len(op.get("quality_evidence", [])))
                            by_time = {row["time_sec"]: row["id"] for row in observation_samples[op["request_id"]]}
                            for evidence in quality_samples[op["request_id"]]:
                                evidence["observation_id"] = by_time[evidence["time_sec"]]
                                evidence["id"] = digest({key: value for key, value in evidence.items() if key != "id"})
                valid.append(op)
            if self.transport is not None:
                mission.dtn_managed_memory = {a:self.reservations(a) for a in backend.actors}
            original = mission.vehicles
            cpu_actors = {
                r["command"]["actor_id"]
                for r in ordinary
                if r["command"]["kind"] == "processing"
            }
            mission.vehicles = {
                a: v.model_copy(
                    update={
                        "base_power_w": (
                            v.base_power_w
                            if bus is None
                            else bus.base_power(a, v.base_power_w) + bus.extra_power(a) + self.actuation_power(a)
                        )
                        + powered.get(a, 0.0),
                        "min_tx_soc": max(v.min_tx_soc, self.thermal_heater_floor(a)),
                        "min_processing_soc": max(v.min_processing_soc, self.thermal_heater_floor(a)),
                        "memory_bits": max(1e-9, v.memory_bits - self.reservations(a))
                        if a in cpu_actors
                        else v.memory_bits,
                    }
                )
                for a, v in original.items()
            }
            sar_active = {op["actor"] for op in valid if op["kind"] == "imaging"}
            managed_heat_power = {a:{"cpu":0.,"payload":0.} for a in backend.actors}
            valid_ids = {id(op) for op in valid}
            for cid, op in self.operations.items():
                if id(op) in valid_ids:
                    managed_heat_power[op["actor"]]["cpu" if op["kind"]=="transform" else "payload"] += operation_power[cid]
            before = {
                a: (
                    mission.generated_wh[a],
                    mission.consumed_wh[a],
                    0.0 if bus is None else bus.extra_power(a),
                    (self.quality.heat_adjustment_w(a, bus.models[a].electrical_heat_fraction)
                     if self.quality is not None and bus is not None and a in sar_active else 0.0)
                    + (self.actuation.heat_adjustment(a, bus.models[a].electrical_heat_fraction)
                       if self.actuation is not None else 0.0)
                    + (self.unloader.heat_adjustment(a, bus.models[a].electrical_heat_fraction) if self.unloader is not None else 0.)
                    + (bus.rigid.heat_adjustment(a, span, bus.models[a].electrical_heat_fraction)
                       if bus is not None else 0.0),
                    None if bus is None else bus.network.capture(backend,a,span,managed_heat_power[a],
                        self.quality.heat_adjustment_w(a,bus.models[a].electrical_heat_fraction) if self.quality is not None and a in sar_active else 0.),
                )
                for a in backend.actors
            }
            original_rates = mission.transmitter_rates
            if self.scheduler is not None:
                mission.transmitter_rates = {a: min(original_rates[a], radio_rates.get(a, original_rates[a])) for a in backend.actors}
                lane_before = {"consumed": dict(mission.consumed_wh), "receipt_index": len(mission.receipts)}
            try:
                if self.actuation is not None:
                    self.actuation.apply(backend, span)
                if self.unloader is not None:
                    self.unloader.apply(backend, span)
                native_advance(span, ordinary)
            finally:
                mission.vehicles = original
                mission.transmitter_rates = original_rates
            if bus is not None:
                bus.integrate(backend, now, span, before)
            if self.actuation is not None:
                self.actuation.finish(backend, now, span)
            if self.unloader is not None:
                self.unloader.finish(backend, now, span)
            if self.scheduler is not None:
                self.scheduler.record(backend, now, backend.now(), valid, powered, work_rates, lane_loads, lane_denied, lane_before)
            for op in valid:
                cid = next(key for key, value in self.operations.items() if value is op)
                power = operation_power[cid]
                self.charges.append(
                    {
                        "actor": op["actor"],
                        "kind": op["kind"],
                        "start_sec": now,
                        "end_sec": backend.now(),
                        "energy_wh": power * span / 3600,
                    }
                )
                if op["kind"] == "imaging":
                    if bus is not None and bus.network.recent.get(op["actor"]):
                        op["progress_sec"] += span
                        self.fail_acquisition(op, backend, "thermal_limit")
                        continue
                    if self.quality is not None:
                        self.quality.record_sar(op["actor"], now, backend.now())
                    if op["request_id"] in observation_samples:
                        op.setdefault("observations", []).extend(observation_samples[op["request_id"]])
                    if op["request_id"] in quality_samples:
                        op.setdefault("quality_evidence", []).extend(quality_samples[op["request_id"]])
                    if op["request_id"] in area_samples:
                        op.setdefault("coverage_samples", []).extend(area_samples[op["request_id"]])
                    op["progress_sec"] += span
                    if op["progress_sec"] >= self.requests[
                        op["request_id"]
                    ].acquisition_sec - 4 * math.ulp(backend.now()):
                        try:
                            self._finish_acquisition(op, backend)
                        except MaterializationError as error:
                            self.fail_acquisition(op, backend, str(error))
                else:
                    state = self.job_state[op["job_id"]]
                    if op["switch_remaining_sec"] > 0:
                        used = min(span, op["switch_remaining_sec"])
                        op["switch_remaining_sec"] -= used
                        state["switching_sec"] += used
                        state["setup_remaining_sec"] = op["switch_remaining_sec"]
                    else:
                        work = span * work_rates.get(cid, 1.0)
                        state["remaining_sec"] = max(0.0, state["remaining_sec"] - work)
                        state["work_sec"] += work
                    if state["remaining_sec"] <= 4 * math.ulp(backend.now()):
                        try:
                            if bus is not None and bus.network.recent.get(op["actor"]):
                                raise MaterializationError("thermal_limit")
                            self._finish_job(op, backend)
                        except MaterializationError as error:
                            state.update(
                                status="failed",
                                reserved=False,
                                reason=str(error),
                                lost_work_sec=state["work_sec"],
                            )
                            op.update(status="failed", reason=str(error))
                            self.fail_unreachable(
                                backend,
                                f"transform_failed:{op['job_id']}",
                                job_id=op["job_id"],
                            )
            if bus is not None:
                bus.apply_protection(backend)
            self.settle(backend)
        for row in active:
            op = self.operations.get(row["command"]["command_id"])
            if (
                op is not None
                and op["status"] not in _TERMINAL
                and backend.now() >= op["until_sec"] - 1e-9
            ):
                if op["kind"] == "imaging":
                    self.fail_acquisition(op, backend, "acquisition_window_ended")
                else:
                    state = self.job_state[op["job_id"]]
                    if state["status"] == "running":
                        state.update(status="paused", reason="command_window_ended")
                    op["status"] = "completed"
        statuses = {cid: op["status"] for cid, op in self.operations.items()}
        if bus is not None:
            statuses.update({cid: op["status"] for cid, op in bus.operations.items()})
        if self.actuation is not None:
            statuses.update({cid: op["status"] for cid, op in self.actuation.operations.items()})
        if self.unloader is not None:
            statuses.update({cid: op["status"] for cid,op in self.unloader.operations.items()})
        return statuses

    def inject(self, backend, events):
        from space_env.simulation.snapshot import canonical_json

        if not isinstance(events, list) or not 1 <= len(events) <= 1000:
            raise ValueError("nonempty lifecycle event batch of at most 1000 required")
        canonical_json(events)
        if len(self.events) + len(events) > 10000 or backend.now() >= backend.horizon():
            raise ValueError("lifecycle event budget/horizon exhausted")
        for event in events:
            if not isinstance(event, dict) or set(event) != {
                "id",
                "type",
                "time_sec",
                "source",
                "reason",
                "payload",
            }:
                raise ValueError("invalid lifecycle event envelope")
            if any(
                not isinstance(event[k], str) or not event[k].strip()
                for k in ("id", "type", "source", "reason")
            ):
                raise ValueError("nonblank event identifiers/source/reason required")
            if (
                isinstance(event["time_sec"], bool)
                or event["time_sec"] != backend.now()
                or not isinstance(event["payload"], dict)
            ):
                raise ValueError("event must apply at now with object payload")
            if event["id"].startswith("@lifecycle/") or event["id"] in {
                e["id"] for e in self.events + backend.mission.events
            }:
                raise ValueError("reserved or duplicate event ID")
            kind, payload = event["type"], event["payload"]
            if kind.startswith("spacecraft_"):
                if getattr(backend, "bus", None) is None:
                    raise ValueError("spacecraft event requires coupled bus profile")
                backend.bus.inject(backend, event)
            elif kind == "request_added":
                if set(payload) != {"request", "jobs"} or not isinstance(
                    payload["jobs"], list
                ):
                    raise ValueError("request_added requires request and jobs")
                request = CustomerRequest.model_validate(payload["request"])
                self._validate_request(request, backend)
                if (
                    request.id in self.requests
                    or request.release_sec < backend.now()
                    or len(self.requests) >= 10000
                ):
                    raise ValueError("duplicate/backdated/excessive demand")
                added = [TransformationJob.model_validate(j) for j in payload["jobs"]]
                if (
                    len({j.id for j in added}) != len(added)
                    or any(j.id in self.jobs for j in added)
                    or len(self.jobs) + len(added) > 10000
                ):
                    raise ValueError("duplicate/excessive jobs")
                requests, jobs = (
                    {**self.requests, request.id: request},
                    {**self.jobs, **{j.id: j for j in added}},
                )
                resolve_transformers(self.transformer_manifests, jobs)
                producers, origins, destinations = graph(requests, jobs)
                if (set(producers) - set(self.producers)) & set(
                    backend.mission.products
                ):
                    raise ValueError("new lifecycle product collides with inventory")
                self.requests, self.jobs = requests, jobs
                self.producers, self.origins, self.destinations = (
                    producers,
                    origins,
                    destinations,
                )
                self.request_state[request.id] = {
                    "status": "pending",
                    "attempts": 0,
                    "actor": None,
                    "reason": None,
                }
                for job in added:
                    self.job_state[job.id] = {
                        "status": "pending",
                        "remaining_sec": job.work_sec,
                        "work_sec": 0.0,
                        "switching_sec": 0.0,
                        "setup_remaining_sec": 0.0,
                        "reserved": False,
                        "actor": None,
                        "reason": None,
                    }
            elif kind in {
                "request_cancelled",
                "request_priority_changed",
                "request_deadline_changed",
            }:
                fields = {"request_id"} | (
                    {"priority"}
                    if kind == "request_priority_changed"
                    else {"deadline_sec"}
                    if kind == "request_deadline_changed"
                    else set()
                )
                if (
                    set(payload) != fields
                    or payload.get("request_id") not in self.requests
                ):
                    raise ValueError("invalid request update")
                rid = payload["request_id"]
                if self.request_state[rid]["status"] in _TERMINAL:
                    raise ValueError("request already terminal")
                if kind == "request_cancelled":
                    self.transition(rid, "cancelled", backend.now(), event["reason"])
                    for operation in self.operations.values():
                        if (
                            operation["request_id"] == rid
                            and operation["status"] not in _TERMINAL
                        ):
                            operation.update(
                                status="cancelled", reason="request_cancelled"
                            )
                    raw_id = self.requests[rid].raw_product_id
                    if raw_id not in self.artifacts:
                        self.fail_unreachable(backend, "required_acquisition_cancelled", product_id=raw_id)
                else:
                    field = (
                        "priority"
                        if kind == "request_priority_changed"
                        else "deadline_sec"
                    )
                    updated = CustomerRequest.model_validate(
                        {
                            **self.requests[rid].model_dump(mode="json"),
                            field: payload[field],
                        }
                    )
                    self._validate_request(updated, backend)
                    if updated.deadline_sec <= backend.now():
                        raise ValueError("deadline must follow now")
                    self.requests[rid] = updated
                    for pid, artifact in self.artifacts.items():
                        if (
                            rid in artifact["request_ids"]
                            and backend.mission.remaining[pid] > 0
                        ):
                            p = backend.mission.products[pid]
                            backend.mission.products[pid] = p.model_copy(
                                update={
                                    "priority": max(
                                        self.requests[r].priority
                                        for r in artifact["request_ids"]
                                    )
                                }
                            )
            elif kind == "job_cancelled":
                if set(payload) != {"job_id"} or payload.get("job_id") not in self.jobs:
                    raise ValueError("known job_id required")
                state = self.job_state[payload["job_id"]]
                if state["status"] in _TERMINAL:
                    raise ValueError("job already terminal")
                state.update(
                    status="cancelled",
                    reserved=False,
                    reason=event["reason"],
                    lost_work_sec=state["work_sec"],
                )
                self.fail_unreachable(
                    backend, "required_job_cancelled", job_id=payload["job_id"]
                )
                for operation in self.operations.values():
                    if (
                        operation["job_id"] == payload["job_id"]
                        and operation["status"] not in _TERMINAL
                    ):
                        operation.update(status="cancelled", reason="job_cancelled")
            elif kind == "artifact_discarded":
                if (
                    set(payload) != {"product_id"}
                    or payload.get("product_id") not in self.artifacts
                ):
                    raise ValueError("known lifecycle product_id required")
                pid = payload["product_id"]
                if self.artifacts[pid]["status"] != "resident":
                    raise ValueError("artifact is not resident")
                if pid in self.destinations and not self.spec.allow_final_discard:
                    raise ValueError(
                        "unconfirmed final output discard requires explicit policy"
                    )
                self.remove_artifact(pid, backend, "discarded")
            else:
                pid = payload.get("product_id", payload.get("id"))
                if kind == "onboard_product_added":
                    if pid in self.producers:
                        raise ValueError("cannot inject a managed lifecycle output")
                    from space_env.reference.operations import OnboardProduct

                    injected_product = OnboardProduct.model_validate(payload)
                    actor = injected_product.actor
                    if (
                        actor not in backend.actors
                        or self.occupied(backend, actor)
                        + self.reservations(actor)
                        + injected_product.size_bits
                        > backend.mission.vehicles[actor].memory_bits
                    ):
                        raise ValueError("injection exceeds reserved memory")
                elif kind.startswith("product_") and pid in self.producers:
                    raise ValueError("managed product updates require lifecycle events")
                backend.mission.apply_events([event])
            self.events.append(copy.deepcopy(event))
            self.settle(backend)
        return self.report(backend)

    def observation(self, actor, now):
        visible_areas = {r.coverage.aoi_id for r in self.requests.values()
                         if r.coverage is not None and actor in r.eligible_actors and r.release_sec <= now}
        return {
            **({"payload_quality": self.quality.observation(actor, now, visible_areas)} if self.quality is not None else {}),
            **({"actuation": self.actuation.observation(actor)} if self.actuation is not None else {}),
            **({"unloading": self.unloader.observation(actor)} if self.unloader is not None else {}),
            **({"resource_lanes": self.scheduler.observation(actor)} if self.scheduler is not None else {}),
            **({"areas_of_interest": {name: self.aois[name].model_dump(mode="json")
                                      for name in sorted(visible_areas)},
                "coverage_geometry_manifest": dict(self.coverage_geometry_manifest),
                "payload_footprint": (self.spec.payloads[actor].footprint.model_dump(mode="json")
                                      if actor in self.spec.payloads and self.spec.payloads[actor].footprint else None)} if self.aois else {}),
            "requests": [
                {**r.model_dump(mode="json"), **self.request_state[r.id]}
                for r in self.requests.values()
                if actor in r.eligible_actors and r.release_sec <= now
            ],
            "jobs": [
                {**self.jobs[j].model_dump(mode="json"), **row}
                for j, row in self.job_state.items()
                if all(
                    self.requests[r].release_sec <= now
                    and actor in self.requests[r].eligible_actors
                    for p in self.jobs[j].inputs
                    for r in self.origins[p]
                )
                and (row["actor"] in {None, actor})
                and all(
                    p not in self.artifacts or self.artifacts[p]["actor"] == actor
                    for p in self.jobs[j].inputs
                )
            ],
            "artifacts": [
                copy.deepcopy(p) for p in self.artifacts.values() if p["actor"] == actor
            ],
        }

    def metrics(self):
        complete = sum(r["status"] == "completed" for r in self.request_state.values())
        return {"fulfilled_requests": complete, "total_requests": len(self.requests)}

    def quality_report(self):
        requests = {}
        for rid, request in self.requests.items():
            if request.quality is None:
                continue
            product = self.request_product(request)
            if product is not None and request.coverage.aoi_id in product.get("area_coverage", {}):
                summary = usable_coverage(request.quality, product["area_coverage"][request.coverage.aoi_id], product.get("quality_evidence", []))
            else:
                summary = {"usable_fraction": 0.0, "usable_area_m2": 0.0, "evidence_records": 0,
                           "criterion": request.quality.model_dump(mode="json")}
            requests[rid] = {**summary, "delivered_and_fulfilled": self.request_state[rid]["status"] == "completed"}
        return {"schema": "space-env.payload-quality/1", "requests": requests,
                "models": {a: m.model_dump(mode="json") for a, m in self.quality.models.items()},
                "sar_intervals": copy.deepcopy(self.quality.sar_intervals),
                "scope": "sampled geometry and categorical cloud masks; no measured radiometry or focused SAR imagery"}

    def report(self, backend):
        ledger = {}
        for actor in backend.actors:
            created = {
                k: sum(
                    p["size_bits"]
                    for p in self.artifacts.values()
                    if p.get("origin_actor",p["actor"]) == actor and p["source"] == k
                )
                for k in ("acquired", "transformed_out")
            }
            removed = {
                k: sum(
                    p["bits"]
                    for p in self.removals
                    if p["actor"] == actor and p["category"] == k
                )
                for k in ("transformed_in", "expired", "discarded")
            }
            initial = backend.mission.initial_bits[actor]
            injected = sum(
                backend.mission.products[e["payload"]["id"]].size_bits
                for e in backend.mission.events
                if e["type"] == "onboard_product_added"
                and e["payload"]["actor"] == actor
                and e["payload"]["id"] not in self.artifacts
            )
            removed["discarded"] += sum(
                bits
                for pid, bits in backend.mission.discarded.items()
                if pid not in self.artifacts
                and backend.mission.products[pid].actor == actor
            )
            transmitted = sum(
                r["bits"] for r in backend.mission.receipts if r["actor"] == actor
            )
            remaining = sum(backend.mission.remaining[p.id] for p in backend.mission.products.values() if p.actor == actor)
            custody_in,custody_out = (0.,0.) if self.transport is None else self.transport.custody(actor)
            balance = (
                initial
                + injected
                + sum(created.values())
                + custody_in - custody_out
                - sum(removed.values())
                - transmitted
                - remaining
            )
            ledger[actor] = {
                "initial": initial,
                "injected": injected,
                **created,
                **removed,
                "transmitted": transmitted,
                "remaining": remaining,
                **({"custody_in":custody_in,"custody_out":custody_out,"transport_reserved":self.transport.reserved(actor)} if self.transport is not None else {}),
                "balance_error_bits": balance,
            }
        linked_receipts = []
        acknowledgements = []
        for receipt in backend.mission.receipts:
            pid = receipt["product_id"]
            if pid not in self.artifacts:
                continue
            obligations = [
                r
                for r in self.requests
                if pid in self.delivery_ids(self.requests[r])
                and self.requests[r].destination_id == receipt["station"]
            ]
            linked_receipts.append({**receipt, "request_ids": obligations})
        for pid, artifact in self.artifacts.items():
            if artifact["status"] != "delivered" or pid not in self.destinations:
                continue
            slips = [r for r in linked_receipts if r["product_id"] == pid]
            acknowledgements.append(
                {
                    "product_id": pid,
                    "actor": artifact["actor"],
                    "destination_id": self.destinations[pid],
                    "bits": artifact["size_bits"],
                    "acknowledged_sec": max(r.get("ack_due_sec", r["end_sec"]) for r in slips),
                    "fulfilled_request_ids": [
                        r
                        for r in self.requests
                        if pid in self.delivery_ids(self.requests[r])
                        and self.request_state[r]["status"] == "completed"
                    ],
                }
            )
        for actor, values in ledger.items():
            values["acked"] = sum(
                r["bits"] for r in acknowledgements if r["actor"] == actor
            )
        return {
            "schema": "space-env.product-lifecycle/1",
            "transformer_manifests": dict(self.transformer_manifests),
            "requests": {
                r: {**self.requests[r].model_dump(mode="json"), **s}
                for r, s in self.request_state.items()
            },
            "jobs": copy.deepcopy(self.job_state),
            "artifacts": copy.deepcopy(self.artifacts),
            "operations": copy.deepcopy(self.operations),
            "transitions": copy.deepcopy(self.transitions),
            "events": copy.deepcopy(self.events),
            "charges": copy.deepcopy(self.charges),
            "ledger": ledger,
            "delivery_receipts": linked_receipts,
            "acknowledgements": acknowledgements,
            "metrics": self.metrics(),
            **({"actuation": self.actuation.report(backend)} if self.actuation is not None else {}),
            **({"unloading": self.unloader.report(backend)} if self.unloader is not None else {}),
            **({"resource_lanes": self.scheduler.report()} if self.scheduler is not None else {}),
            **({"observation_requirements": {rid: {**self.evaluate_request(r, self.request_product(r)),
                  "delivery_product_ids": list(self.delivery_ids(r)), "full_delivery": self.receipt_time(r, backend) is not None,
                  "delivered_and_fulfilled": self.request_state[rid]["status"] == "completed"}
                  for rid, r in self.requests.items() if r.observations is not None}}
               if any(r.observations is not None for r in self.requests.values()) else {}),
            **({"payload_quality": self.quality_report()} if self.quality is not None else {}),
            **({"area_coverage": area_report(self.aois, self.artifacts)} if self.aois else {}),
            "delivery_ack_model": ("ideal-round-trip-plus-processing/1" if backend.mission.radio_network is not None else "instant-on-complete-station-receipt/1"),
            "representation": "synthetic resource metadata; no rendered image or scientific compression",
        }
