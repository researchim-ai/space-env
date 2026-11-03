"""Command-driven GCRS mission adapter; physical qualification remains explicit."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from typing import Literal

from pydantic import Field, model_validator

from space_env.reference.acquisition import (
    AcquisitionSensor,
    AcquisitionTarget,
    acquisition_windows,
)
from space_env.reference.actuation import ActuationSettings
from space_env.reference.covariance import NavigationSettings
from space_env.reference.frames import FrameContext
from space_env.reference.lifecycle import (
    GeometryFeatureTransformer,
    LifecycleSpec,
    MetadataTransformer,
    ProductLifecycle,
)
from space_env.reference.orbit import ForceModel, ReferenceState
from space_env.reference.providers import DynamicStateProvider
from space_env.reference.resource_lanes import ParallelExecution, command_claims
from space_env.reference.spacecraft_bus import (
    CoupledResourceMission,
    SpacecraftBus,
    SpacecraftBusModel,
)
from space_env.reference.thrust import FiniteThrust
from space_env.reference.time import EOPSeries, Epoch
from space_env.reference.unloading import UnloadingSettings
from space_env.runtime.adapters import ReferenceBackend
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.owned import owned_copy


class IntegratorSettings(StrictModel):
    sample_sec: float = Field(default=10.0, gt=0, le=60, strict=True)
    rtol: float = Field(default=1e-11, ge=1e-13, le=1e-5, strict=True)
    position_atol_m: float = Field(default=1e-4, gt=0, strict=True)
    velocity_atol_m_s: float = Field(default=1e-7, gt=0, strict=True)
    max_step_sec: float = Field(default=30.0, gt=0, le=120, strict=True)


class PropulsionSettings(StrictModel):
    dry_mass_kg: float = Field(gt=0, strict=True)
    propellant_kg: float = Field(ge=0, strict=True)
    isp_s: float = Field(gt=0, strict=True)
    max_thrust_n: float = Field(gt=0, strict=True)
    max_impulse_m_s: float = Field(default=100.0, gt=0, strict=True)


class InitialOrbit(StrictModel):
    frame: Literal["GCRS"]
    epoch_utc: str
    r_m: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    v_m_s: tuple[FiniteScalar, FiniteScalar, FiniteScalar]


class OrbitSettings(StrictModel):
    initial: InitialOrbit
    propulsion: PropulsionSettings
    forces: dict = Field(default_factory=dict)
    integration: IntegratorSettings = Field(default_factory=IntegratorSettings)
    navigation: NavigationSettings | None = None

    @model_validator(mode="after")
    def force_contract(self):
        if "mass_kg" in self.forces:
            raise ValueError("orbit mass derives from dry mass plus propellant")
        try:
            self.force_model()
        except TypeError as error:
            raise ValueError("unsupported force fields") from error
        return self

    def force_model(self):
        return ForceModel(
            **self.forces,
            mass_kg=self.propulsion.dry_mass_kg + self.propulsion.propellant_kg,
        )


class EOPInput(StrictModel):
    interpolation: Literal["linear UT1-TAI/xp/yp over TAI seconds"] = (
        "linear UT1-TAI/xp/yp over TAI seconds"
    )
    rows: list[dict]
    source: str = Field(min_length=1)
    allow_predictions: bool = Field(default=False, strict=True)


class WeatherInput(StrictModel):
    source: str = Field(min_length=1)
    valid_from_utc: str
    valid_until_utc: str
    status: Literal["observed", "prediction", "synthetic"]
    allow_predictions: bool = Field(default=False, strict=True)


class DynamicMissionSpec(StrictModel):
    schema_id: Literal["space-env.reference-dynamic-spec/1"] = Field(alias="schema")
    orbits: dict[str, OrbitSettings] = Field(min_length=1, max_length=64)
    eop: EOPInput
    horizon_sec: float = Field(gt=1e-9, le=259200, strict=True)
    operations: dict
    weather: WeatherInput | None = None
    max_maneuvers: int = Field(default=256, ge=1, le=10000, strict=True)
    acquisition_targets: tuple[AcquisitionTarget, ...] = Field(
        default=(), max_length=10000
    )
    acquisition_sensors: dict[str, AcquisitionSensor] = Field(default_factory=dict)
    lifecycle: LifecycleSpec | None = None
    spacecraft_bus: dict[str, SpacecraftBusModel] | None = None
    execution: ParallelExecution | None = None
    actuation: ActuationSettings | None = None
    unloading: UnloadingSettings | None = None


class DynamicReferenceBackend(ReferenceBackend):
    capabilities = ReferenceBackend.capabilities | frozenset(
        {"commands.impulse", "commands.finite_burn"}
    )

    def __init__(self, spec):
        config = DynamicMissionSpec.model_validate(spec.definition)
        if (config.lifecycle is not None) != (
            spec.backend in {"reference-lifecycle", "reference-spacecraft"}
        ):
            raise ValueError(
                "lifecycle definition requires explicit reference-lifecycle backend/profile"
            )
        if (config.spacecraft_bus is not None) != (
            spec.backend == "reference-spacecraft"
        ):
            raise ValueError(
                "spacecraft bus definition requires explicit reference-spacecraft profile"
            )
        if config.spacecraft_bus is not None:
            if set(config.spacecraft_bus) != set(spec.actor_order):
                raise ValueError("coupled bus configuration must cover every actor")
            self.capabilities = self.capabilities - {"commands.impulse", "commands.finite_burn"}
            self.mission_factory = CoupledResourceMission
        if config.actuation is not None:
            if config.spacecraft_bus is None:
                raise ValueError("body actuation requires coupled spacecraft profile")
            self.capabilities = self.capabilities | {"commands.body_burn", "physics.body_thrust"}
            self.forecast_scope = "applied thrust only; future pending engine phases/firing are not predicted"
        if config.unloading is not None:
            if config.spacecraft_bus is None:
                raise ValueError("unloading requires coupled spacecraft profile")
            self.capabilities = self.capabilities | {"commands.momentum_unload", "physics.balanced_rcs_couples"}
            self.forecast_scope = "executed actuation only; future pending unloading/burn commands are not predicted"
        if config.execution is not None:
            if config.spacecraft_bus is None or set(config.execution.actors) != set(spec.actor_order):
                raise ValueError("resource lanes require coupled spacecraft and capacities for every actor")
            self.capabilities = self.capabilities - {"commands.processing"}
            self.capabilities = self.capabilities | {"execution.resource_lanes", "resources.memory_bandwidth"}
            self.concurrency = "resource-lanes/1: one managed CPU, payload, radio and exclusive attitude ownership"
        self.lifecycle = None
        self.bus = None
        self.profile = spec.physics_profile
        if set(config.orbits) != set(spec.actor_order):
            raise ValueError("dynamic actors and actor_order differ")
        if set(config.acquisition_sensors) - set(spec.actor_order):
            raise ValueError("acquisition sensor references unknown actor")
        if len({t.id for t in config.acquisition_targets}) != len(
            config.acquisition_targets
        ):
            raise ValueError("duplicate acquisition target ID")
        if any(
            t.available_from_sec >= config.horizon_sec
            or (
                t.available_until_sec is not None
                and t.available_until_sec > config.horizon_sec
            )
            for t in config.acquisition_targets
        ):
            raise ValueError("acquisition target availability exceeds horizon")
        context = FrameContext(
            EOPSeries(**config.eop.model_dump(exclude={"interpolation"}))
        )
        providers = {}
        for actor in spec.actor_order:
            orbit = config.orbits[actor]
            initial = ReferenceState(
                Epoch.utc(orbit.initial.epoch_utc),
                orbit.initial.r_m,
                orbit.initial.v_m_s,
            )
            model = orbit.force_model()
            if model.drag:
                weather = config.weather
                if weather is None:
                    raise ValueError(
                        "drag requires explicit weather validity and source"
                    )
                if weather.status == "prediction" and not weather.allow_predictions:
                    raise ValueError("predicted weather requires explicit opt-in")
                start, end = (
                    Epoch.utc(weather.valid_from_utc),
                    Epoch.utc(weather.valid_until_utc),
                )
                if (
                    initial.epoch.seconds_since(start) < 0
                    or end.seconds_since(initial.epoch.shifted(config.horizon_sec)) < 0
                ):
                    raise ValueError("mission outside explicit weather validity")
            providers[actor] = DynamicStateProvider(
                initial,
                config.horizon_sec,
                context,
                model=model,
                integration=orbit.integration.model_dump(),
                dry_mass_kg=orbit.propulsion.dry_mass_kg,
                isp_s=orbit.propulsion.isp_s,
                navigation=orbit.navigation,
            )
        if "ephemerides" in config.operations or "horizon_sec" in config.operations:
            raise ValueError(
                "dynamic operations must not override ephemerides or horizon"
            )
        operations = {
            **config.operations,
            "horizon_sec": config.horizon_sec,
            "ephemerides": {
                a: copy.deepcopy(provider.initial_payload)
                for a, provider in providers.items()
            },
        }
        super().__init__(spec.model_copy(update={"definition": operations}))
        effective_operations = copy.deepcopy(self.definition)
        effective_operations.pop("ephemerides")
        effective_operations.pop("horizon_sec")
        normalized = config.model_dump(mode="json", by_alias=True)
        normalized["operations"] = effective_operations
        for actor in spec.actor_order:
            forces = asdict(config.orbits[actor].force_model())
            forces.pop("mass_kg")
            normalized["orbits"][actor]["forces"] = forces
        self.definition = normalized
        self.config = config
        self.acquisition_targets = {t.id: t for t in config.acquisition_targets}
        self.acquisition_actors = [
            a for a in spec.actor_order if a in config.acquisition_sensors
        ]
        if self.acquisition_targets and self.acquisition_actors:
            self.capabilities = self.capabilities | {"forecast.acquisition.geometry"}
        for provider in providers.values():
            del provider.initial_payload
        self.mission.ephemerides = providers
        self.navigation_actors = [
            a for a, p in providers.items() if p.navigation is not None
        ]
        if self.navigation_actors:
            self.capabilities = self.capabilities | {"state.covariance.partial_actors"}
        if len(self.navigation_actors) == len(self.actors):
            self.capabilities = self.capabilities | {
                "state.covariance_propagation",
                "state.stm",
            }
        if self.mission.dtn is not None and config.execution is not None:
            self.mission.dtn_lane_bandwidth = {a:r.memory_bandwidth_bps for a,r in config.execution.actors.items()}
        self.geometry_revision = 0
        self.burn_commands = {}
        self.data_validity = {
            "leap_utc_years": [1972, 2026],
            "eop_coverage_utc": [config.eop.rows[0]["utc"], config.eop.rows[-1]["utc"]],
            "eop_prediction_opt_in": config.eop.allow_predictions,
            "eop_statuses": sorted({r["status"] for r in config.eop.rows}),
            "weather": config.weather.model_dump() if config.weather else None,
            "trajectory_horizon_sec": config.horizon_sec,
            "extrapolation": False,
        }
        if config.lifecycle is not None:
            self.lifecycle = ProductLifecycle(config.lifecycle, self)
            self.definition["lifecycle"]["transformer_manifests"] = dict(
                self.lifecycle.transformer_manifests
            )
            if self.lifecycle.quality is not None:
                self.capabilities = self.capabilities | {"products.payload_quality"}
            if self.lifecycle.aois:
                self.definition["lifecycle"]["coverage_geometry_manifest"] = dict(self.lifecycle.coverage_geometry_manifest)
                self.capabilities = self.capabilities | {"products.area_coverage", "products.observation_geometry", "requests.revisit_stereo", "delivery.observation_bundle"}
            self.capabilities = self.capabilities | {
                "commands.imaging",
                "commands.transform",
                "events.lifecycle",
            }
        if config.spacecraft_bus is not None:
            self.bus = SpacecraftBus(config.spacecraft_bus, self)
            self.mission.bus = self.bus
            if self.mission.radio_network is not None:
                for a,tx in self.mission.radio_network.config.transmitters.items():
                    v = self.mission.vehicles[a]
                    if tx.rf_power_w > v.tx_power_w*(1-self.bus.models[a].electrical_heat_fraction)+1e-12:
                        raise ValueError("RF output plus declared radio heat exceeds DC input")
            self.lifecycle.thermal_network = self.bus.network
            if self.bus.network.models:
                self.capabilities = self.capabilities | {"physics.thermal_network.partial_actors", "thermal.autonomous_heaters"}
            self.capabilities = self.capabilities | {
                "commands.slew",
                "events.spacecraft_bus",
                "observe.spacecraft_bus",
            }

        if self.bus is not None and self.bus.rigid.models:
            self.capabilities = self.capabilities | {"commands.wheel_torque", "physics.rigid_attitude.partial_actors"}
            if config.actuation is not None and any(m.rigid_coupling is not None for m in config.actuation.engines.values()):
                self.capabilities = self.capabilities | {"physics.rotating_body_thrust.partial_actors"}

    def command_claims(self, command):
        if self.config.execution is None:
            return ("actor",)
        return command_claims(command, self.config.execution, self.config.spacecraft_bus)

    def validate_batch(self, commands, active):
        if self.lifecycle is None:
            return
        seen = set()
        for command in [*active, *(c.model_dump(mode="json") for c in commands)]:
            key = (("request", command["request_id"]) if command["kind"] == "imaging"
                   else ("job", command["job_id"]) if command["kind"] == "transform" else None)
            if key is not None:
                if key in seen:
                    raise ValueError("managed request/job already has an active command")
                seen.add(key)

    def validate_command(self, c):
        super().validate_command(c)
        if c.kind == "wheel_torque" and (self.bus is None or c.actor_id not in self.bus.rigid.models):
            raise ValueError("wheel torque requires actor rigid attitude model")

        if c.kind == "momentum_unload":
            self.lifecycle.unloader.validate(c, self)
            return
        if c.kind == "body_burn":
            self.lifecycle.actuation.validate(c, self)
            return
        if self.lifecycle is not None:
            self.lifecycle.validate(c, self)
        if c.kind not in {"impulse", "finite_burn"}:
            return
        if self.now() >= self.horizon():
            raise ValueError("maneuver at horizon is outside this profile")
        if (
            sum(
                sum(m["kind"] in {"impulse", "finite_burn"} for m in e.maneuvers)
                for e in self.mission.ephemerides.values()
            )
            >= self.config.max_maneuvers
        ):
            raise ValueError("maneuver history budget exhausted")
        settings = self.config.orbits[c.actor_id].propulsion
        provider = self.mission.ephemerides[c.actor_id]
        mass = provider.mass(self.now())
        if c.kind == "impulse":
            speed = math.hypot(*c.impulse.delta_v_m_s)
            if speed > settings.max_impulse_m_s:
                raise ValueError("impulse exceeds configured delta-v limit")
            from space_env.core.propulsion import G0_M_S2

            if (
                mass * math.exp(-speed / (settings.isp_s * G0_M_S2))
                < settings.dry_mass_kg - 1e-10
            ):
                raise ValueError("insufficient propellant for impulse")
        else:
            if c.burn.thrust_n > settings.max_thrust_n:
                raise ValueError("burn exceeds configured thrust limit")
            thrust = FiniteThrust(
                c.burn.thrust_n,
                settings.isp_s,
                settings.dry_mass_kg,
                c.burn.direction_gcrs,
            )
            if c.duration_sec > thrust.fuel_seconds(mass) + 1e-9:
                raise ValueError("insufficient propellant for finite burn")

    def forecast_acquisition(self, actor, target_id, **options):
        if actor not in self.config.acquisition_sensors:
            raise ValueError("actor has no acquisition sensor geometry configuration")
        if target_id not in self.acquisition_targets:
            raise ValueError("unknown acquisition target")
        options["end_sec"] = (
            self.horizon() if options.get("end_sec") is None else options["end_sec"]
        )
        result = acquisition_windows(
            self.mission.ephemerides[actor],
            self.acquisition_targets[target_id],
            self.config.acquisition_sensors[actor],
            **options,
        )
        result["actor_id"] = actor
        return result

    def admit(self, c):
        if c.kind == "relay":
            return super().admit(c)
        self.validate_command(c)
        if c.kind == "momentum_unload":
            self.lifecycle.unloader.admit(c, self)
            return
        if c.kind == "body_burn":
            self.lifecycle.actuation.admit(c, self)
            return
        if self.bus is not None and c.kind in {"slew", "wheel_torque"}:
            self.bus.admit(c, self)
            return
        if self.lifecycle is not None and c.kind in {"imaging", "transform"}:
            self.lifecycle.admit(c, self)
            return
        provider = self.mission.ephemerides[c.actor_id]
        settings = self.config.orbits[c.actor_id].propulsion
        if c.kind == "impulse":
            provider.impulse(self.now(), c.impulse.delta_v_m_s)
        elif c.kind == "finite_burn":
            provider.burn(
                self.now(),
                c.duration_sec,
                FiniteThrust(
                    c.burn.thrust_n,
                    settings.isp_s,
                    settings.dry_mass_kg,
                    c.burn.direction_gcrs,
                ),
            )
            self.burn_commands[c.command_id] = (c.actor_id, self.now() + c.duration_sec)
        else:
            return
        self.geometry_revision += 1
        self._refresh_geometry(c.actor_id)

    def _refresh_geometry(self, actor):
        provider = self.mission.ephemerides[actor]
        boundaries = set(self.mission._boundaries)
        for name, station in self.mission.stations.items():
            rows = provider.access(station)["windows"]
            self.mission.windows[actor, name] = [
                (r["aos_sec"], r["los_sec"]) for r in rows
            ]
            boundaries.update(t for r in rows for t in (r["aos_sec"], r["los_sec"]))
        # Old boundaries remain harmless integration splits; old windows are replaced.
        self.mission._boundaries = sorted(boundaries)

    def cancel(self, command_id):
        if self.lifecycle is not None and self.lifecycle.unloader is not None and self.lifecycle.unloader.cancel(command_id):
            return True
        if self.lifecycle is not None and self.lifecycle.actuation is not None and self.lifecycle.actuation.cancel(command_id):
            return
        if self.bus is not None and self.bus.cancel(command_id):
            return
        if self.lifecycle is not None and self.lifecycle.cancel(command_id, self):
            return
        entry = self.burn_commands.get(command_id)
        if entry is not None and self.now() < entry[1]:
            self.mission.ephemerides[entry[0]].stop_burn(self.now())
            self.geometry_revision += 1
            self._refresh_geometry(entry[0])
        self.burn_commands.pop(command_id, None)
        super().cancel(command_id)

    def observe(self, actor):
        return self._observation(actor, owned=True)

    def _observe_borrowed(self, actor):
        return self._observation(actor, owned=False)

    def _observation(self, actor, *, owned):
        provider = self.mission.ephemerides[actor]
        mass = provider.mass(self.now())
        observation = {
            **super().observe(actor),
            "orbit": {
                "frame": "GCRS",
                "state_si": provider.state(self.now()).tolist(),
                "epoch_utc": provider.epoch.shifted(self.now()).iso(),
                "mass_kg": mass,
                "propellant_kg": max(0.0, mass - provider.dry_mass_kg),
                "geometry_revision": self.geometry_revision,
            },
        }
        if provider.navigation is not None:
            observation["orbit"]["covariance_si"] = provider.covariance(
                self.now()
            ).tolist()
        if self.lifecycle is not None:
            observation["products"] = [
                p
                for p in observation["products"]
                if p["id"] not in self.lifecycle.producers
            ]
            observation["lifecycle"] = self.lifecycle._observation(actor, self.now(), owned=owned)
        if self.bus is not None:
            observation["spacecraft_bus"] = self.bus.observation(actor)
        return observation

    def advance(self, dt, active):
        if self.lifecycle is None:
            return super().advance(dt, active)
        return self.lifecycle.advance(self, dt, active, super().advance)

    def inject(self, events):
        if self.lifecycle is None:
            return super().inject(events)
        return self.lifecycle.inject(self, events)

    def task_score(self):
        if self.lifecycle is None:
            return None
        mission = self.mission
        life = self.lifecycle
        outcomes = mission.product_outcomes()
        return {
            "completion": life.metrics()["fulfilled_requests"],
            "priority": sum(
                life.requests[rid].priority
                for rid, row in life.request_state.items()
                if row["status"] == "completed"
            ),
            "mbits": sum(
                p["delivered_bits"]
                for p in outcomes
                if p["status"] in {"delivered_on_time", "delivered_late"}
            )
            / 1e6,
            "energy_j": math.fsum(mission.consumed_wh[a] for a in mission.actors) * 3600,
            "discard_mbits": sum(
                sum(
                    mission.discarded[p.id]
                    for p in mission.products.values()
                    if p.actor == a
                )
                for a in mission.actors
            )
            / 1e6,
            "safe": {}
            if self.bus is None
            else {a: bool(row["safe"]) for a, row in self.bus.states.items()},
        }

    def metrics(self):
        return (
            self.lifecycle.metrics()
            if self.lifecycle is not None
            else super().metrics()
        )

    def truth(self):
        result = super().truth()
        for actor in self.navigation_actors:
            provider = self.mission.ephemerides[actor]
            result["states"][actor]["covariance_si"] = provider.covariance(
                self.now()
            ).tolist()
            result["states"][actor]["stm"] = provider.stm(self.now()).tolist()
        return result

    def report(self):
        report = {
            **super().report(),
            "profile": self.profile,
            "geometry_revision": self.geometry_revision,
            "navigation_actors": list(self.navigation_actors),
            "data_validity": copy.deepcopy(self.data_validity),
            "mass_kg": {
                a: p.mass(self.now()) for a, p in self.mission.ephemerides.items()
            },
            "trajectory_revision": {
                a: p.revision for a, p in self.mission.ephemerides.items()
            },
        }
        if self.lifecycle is not None:
            report["lifecycle"] = self.lifecycle.report(self)
        if self.bus is not None:
            report["spacecraft_bus"] = self.bus.report(self)
        return report

    def capture(self):
        # Clone the large scalar/container lifecycle graph once into the shared
        # deepcopy memo. Every mutable node is still owned, never shared with
        # the live backend; transport/bus/mission aliases remain one graph.
        state = (
            self.mission,
            self.geometry_revision,
            self.burn_commands,
            self.lifecycle,
            self.bus,
        )
        memo = {}
        life = state[3]
        if type(life) is ProductLifecycle and all(
            type(plugin) in {MetadataTransformer, GeometryFeatureTransformer}
            for plugin in life.transformers.values()
        ):
            cloned = object.__new__(ProductLifecycle)
            memo[id(life)] = cloned
            cloned.__dict__ = owned_copy(life.__dict__, memo)
        return copy.deepcopy(state, memo)

    def restore(self, state):
        (
            self.mission,
            self.geometry_revision,
            self.burn_commands,
            self.lifecycle,
            self.bus,
        ) = state
