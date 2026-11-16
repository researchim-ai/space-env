"""Agent bridge for sequential goals on one continuous native task runtime.

Phase goals and residuals are evaluator state. The agent supplies catalogue
choices, never environment injections, goal edits, forecasts or runtime resets.
"""

from __future__ import annotations

import copy

from pydantic import Field, model_validator

from space_env.industrial.detour import detour_observation
from space_env.industrial.network_accounting import network_accounting
from space_env.industrial.route_telemetry import RouteTelemetry, RouteTelemetrySpec
from space_env.industrial.routes import relay_route_observation
from space_env.industrial.session import DecisionSession
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.tasks.campaign import CampaignSession, CampaignSpec
from space_env.tasks.schema import TaskSpec


class CampaignConfiguration(StrictModel):
    scenario: dict
    task: TaskSpec
    campaign: CampaignSpec
    seed: int | None = Field(default=None, ge=0, strict=True)
    route_telemetry: RouteTelemetrySpec | None = None
    publish_route_topology: bool = Field(default=False, strict=True)
    publish_detour_graph: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def control_contract(self):
        if self.publish_detour_graph and not self.publish_route_topology:
            raise ValueError("detour graph requires public route topology")
        if self.task.goal is not None:
            raise ValueError("campaign task must omit goal")
        if self.task.max_decisions > 512:
            raise ValueError("agent campaign supports at most 512 decisions")
        if any(action.kind == "inject" for action in self.task.actions):
            raise ValueError(
                "agent campaign catalogue cannot inject environment events"
            )
        return self


class CampaignAction(StrictModel):
    choices: dict[str, tuple[str, ...]] = Field(min_length=1, max_length=65)
    advance_sec: FiniteScalar = Field(gt=1e-9, le=3600)


class AgentCampaignSession(DecisionSession):
    kind = "native_campaign"

    def __init__(self, configuration):
        self.spec = CampaignConfiguration.model_validate(copy.deepcopy(configuration))
        configuration = self.spec.model_dump(mode="json", by_alias=True)
        native_configuration = {
            k: v
            for k, v in configuration.items()
            if k
            not in {"route_telemetry", "publish_route_topology", "publish_detour_graph"}
        }
        self._native = CampaignSession(**native_configuration)
        self._start(configuration)
        self._native_return = 0.0
        self._route_telemetry = (
            RouteTelemetry(self.spec.route_telemetry.model_dump())
            if self.spec.route_telemetry is not None
            else None
        )
        self._capture_routes()

    def _capture_routes(self):
        if self._route_telemetry is None:
            return
        view = self._native.observe()
        requests = {e["id"] for e in view["entities"] if e["kind"] == "request"}
        catalogue = [
            r
            for r in view["catalogue"]
            if not r.get("request_ref") or r["request_ref"] in requests
        ]
        runtime = self._native._session._runtime
        self._route_telemetry.capture(
            runtime.time_sec,
            relay_route_observation(
                runtime, catalogue, topology_catalogue=self._route_catalogue(catalogue)
            ),
        )

    def _route_catalogue(self, catalogue):
        visible = {r["id"] for r in catalogue}
        return [
            a.model_dump(mode="json") for a in self.spec.task.actions if a.id in visible
        ]

    def observe(self):
        # The existing projection enforces navigation visibility and request release.
        # Do not merge campaign.status(), phase goals or truth into this view.
        with self._lock:
            view = self._native.observe()
            view.pop("binding", None)
            # Templates for unreleased demand must not reveal future request IDs.
            requests = {e["id"] for e in view["entities"] if e["kind"] == "request"}
            keep = [
                i
                for i, row in enumerate(view["catalogue"])
                if not row.get("request_ref") or row["request_ref"] in requests
            ]
            view["catalogue"] = [view["catalogue"][i] for i in keep]
            view["action_mask"] = [view["action_mask"][i] for i in keep]
            result = {
                "schema": "space-env.agent-campaign-observation/1",
                "revision": self.revision,
                "native": view,
                "environment_events": self._native._session.environment_status(),
                "control": {
                    "max_decision_sec": self.spec.task.max_decision_sec,
                    "default_decision_sec": self.spec.task.decision_sec,
                    "durations_sec": list(self.spec.task.durations_sec),
                },
            }
            if self._route_telemetry is not None:
                routes, metadata = self._route_telemetry.observation(
                    view["time_sec"], {r["id"] for r in view["catalogue"]}
                )
                result["route_telemetry"] = metadata
            else:
                routes = relay_route_observation(
                    self._native._session._runtime,
                    view["catalogue"],
                    topology_catalogue=self._route_catalogue(view["catalogue"]),
                )
            if routes is not None:
                result["relay_routes"] = routes
                if self.spec.publish_detour_graph:
                    result["detour_graph"] = detour_observation(
                        self._native._session._runtime,
                        self._route_catalogue(view["catalogue"]),
                    )
                if self.spec.publish_route_topology:
                    result["relay_topology"] = [
                        {
                            "action_id": a["id"],
                            **{
                                k: c[k]
                                for k in (
                                    "kind",
                                    "actor_id",
                                    "destination_id",
                                    "product_id",
                                )
                            },
                        }
                        for a in self._route_catalogue(view["catalogue"])
                        for c in a.get("commands", [])
                        if c["kind"] in {"relay", "downlink"}
                    ]
            return result

    def _apply(self, action):
        request = CampaignAction.model_validate(action)
        visible = {row["id"] for row in self.observe()["native"]["catalogue"]}
        if any(
            name not in visible for names in request.choices.values() for name in names
        ):
            raise ValueError("action is not in the current public catalogue")
        before = self._native.status()
        output = self._native.step(
            request.choices,
            advance_sec=request.advance_sec,
            expected_revision=self._native.revision,
            idempotency_key=f"agent-campaign/{self.revision}",
        )
        transition, status = output["transition"], output["campaign"]
        self._native_return += transition["reward"]
        self._done = status["terminal"]
        self._capture_routes()
        completion = len(status["completed_phases"]) - len(before["completed_phases"])
        return {
            "reward": float(transition["reward"] + completion - int(status["failed"])),
            "elapsed_sec": transition["elapsed_sec"],
            "feedback": copy.deepcopy(transition["feedback"]),
        }

    def metrics(self):
        with self._lock:
            status = self._native.status()
            actors = self._native._session._runtime.observe()["actors"]
            health = {
                actor: not (
                    view["spacecraft_bus"]["safe"] or view["spacecraft_bus"]["faults"]
                )
                for actor, view in actors.items()
                if "spacecraft_bus" in view
            }
            report = self._native._session._runtime._backend.report()
            result = {
                "success": status["success"],
                "terminal": status["terminal"],
                "failed": status["failed"],
                "completed_phases": list(status["completed_phases"]),
                "total_phases": len(self.spec.campaign.phases),
                "time_sec": status["time_sec"],
                "native_return": self._native_return,
                "environment_events_attempted": len(
                    self._native._session.environment_status()
                ),
                "environment_events_rejected": sum(
                    not e["accepted"]
                    for e in self._native._session.environment_status()
                ),
                "resource_reset_between_phases": False,
                "flight_calibrated": False,
                "current_bus_health": health,
                "resource_accounting": {
                    "energy": copy.deepcopy(report["energy_ledger"]),
                    "bus": copy.deepcopy(
                        report.get("spacecraft_bus", {}).get("ledgers", {})
                    ),
                },
            }
            network = network_accounting(
                report, self._native._session._runtime._backend.mission
            )
            if network is not None:
                result["network_accounting"] = network
            return result

    def _witness(self):
        result = {"native": self._native.checkpoint(), "metrics": self.metrics()}
        if self._route_telemetry is not None:
            result["route_samples"] = copy.deepcopy(self._route_telemetry.samples)
        return result
