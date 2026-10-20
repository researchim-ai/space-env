"""
События: планируемые (маневр, съёмка, сессия связи) и неплановые (аномалия, конъюнкция).
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from space_env.core.time import Time


class EventKind(str, Enum):
    # Планируемые
    MANEUVER = "maneuver"
    IMAGING = "imaging"
    COMMS_PASS = "comms_pass"
    # Неплановые
    ANOMALY = "anomaly"
    CONJUNCTION_WARNING = "conjunction_warning"
    ECLIPSE_START = "eclipse_start"
    ECLIPSE_END = "eclipse_end"
    AOS = "aos"
    LOS = "los"
    CONSTRAINT_VIOLATION = "constraint_violation"
    RECOVERY = "recovery"


class MissionEvent(BaseModel):
    """Одно событие в логе симуляции."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: EventKind
    time: Time | None = None  # заполняется движком при возникновении
    time_sec: float | None = Field(default=None, description="Секунды от начала симуляции")
    payload: dict[str, Any] = Field(default_factory=dict)


class PlannedEvent(BaseModel):
    """Планируемое событие (в плане миссии)."""
    kind: EventKind
    scheduled_at_sec: float = Field(ge=0)
    duration_sec: float = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
