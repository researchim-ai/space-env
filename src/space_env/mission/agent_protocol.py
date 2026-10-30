"""Versioned, validated JSON tool invocations over the existing mission executor."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from space_env.mission.tools import MissionTools, RevisionError, ToolBudgetError


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ObserveArgs(_Args):
    actor: str | None = None


class ForecastArgs(_Args):
    actor: str = Field(min_length=1)
    dt_sec: float = Field(gt=0)


class CommandArgs(_Args):
    command_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    kind: Literal["wait", "imaging", "comms"]
    request_id: str = ""
    destination_id: str = ""
    duration_sec: float = Field(default=0, ge=0)


class SubmitArgs(_Args):
    commands: list[CommandArgs]


class CancelArgs(_Args):
    command_id: str = Field(min_length=1)


class AdvanceArgs(_Args):
    dt_sec: float = Field(gt=0)
    expected_revision: int = Field(ge=0)


class RelayArgs(AdvanceArgs):
    source: str = Field(min_length=1)
    destination: str = Field(min_length=1)


_MODELS = {"observe": ObserveArgs, "forecast": ForecastArgs, "submit": SubmitArgs,
           "cancel": CancelArgs, "advance": AdvanceArgs, "relay": RelayArgs}
_DESCRIPTIONS = {
    "observe": "Read declared requests, masks and sensor telemetry. Consumes a tool call.",
    "forecast": "Forecast nominal orbital geometry only, without future faults or subsystem truth. Does not advance time.",
    "submit": "Queue at most one command per actor; advance executes the pending batch.",
    "cancel": "Drop a pending command or stop a world operation. Delivered data remains delivered.",
    "advance": "Execute pending commands and advance all actors, clipping to the horizon. Requires current revision.",
    "relay": "Advance all actors for one serial sampled relay; received data is available at the end. Requires idle participants and no pending commands.",
}


def tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {"name": name, "description": _DESCRIPTIONS[name],
              "parameters": model.model_json_schema()}} for name, model in _MODELS.items()]


def invoke_tool(tools: MissionTools, name: str, arguments: dict) -> dict:
    """Domain/validation failures are observations, not successful actions.

    Schema failures do not invoke a method or spend the call budget. Once a tool
    method is invoked, its established accounting applies, including failed calls.
    Unexpected programming/I/O errors propagate and must not become training labels.
    """
    try:
        if name not in _MODELS:
            raise ValueError(f"unknown mission tool: {name}")
        kwargs = _MODELS[name].model_validate(arguments).model_dump()
    except ValidationError as exc:
        return {"ok": False, "error": {"type": "ValidationError", "details": [
            {"location": list(error["loc"]), "type": error["type"]} for error in exc.errors()
        ]}}
    except ValueError as exc:
        return {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    try:
        result = getattr(tools, "forecast_geometry" if name == "forecast" else name)(**kwargs)
        if name == "advance":
            # UI reports contain truth trajectories/SOC. The agent protocol exposes
            # only the advance acknowledgement; telemetry must use observe.
            result = {key: result[key] for key in ("revision", "now_sec", "horizon_sec", "statuses")}
        return {"ok": True, "result": result}
    except (ValueError, KeyError, RevisionError, ToolBudgetError) as exc:
        return {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
