"""Bounded causal dispatch updates; no RF, engine fault or orbit mutation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from space_env.runtime.schema import StrictModel


class DispatchEvent(StrictModel):
    at_sec: float = Field(ge=0, le=95, strict=True)
    kind: Literal["close_windows", "capacity"]
    actor: str | None = None
    start_sec: float | None = Field(default=None, ge=0, le=100, strict=True)
    end_sec: float | None = Field(default=None, gt=0, le=100, strict=True)
    capacity: int | None = Field(default=None, ge=0, le=4, strict=True)

    @model_validator(mode="after")
    def payload(self):
        if self.at_sec % 5:
            raise ValueError("dispatch events use five-second decision boundaries")
        if self.kind == "capacity":
            if self.capacity is None or any(
                v is not None for v in (self.actor, self.start_sec, self.end_sec)
            ):
                raise ValueError("capacity event only accepts capacity")
        elif (
            not self.actor
            or self.start_sec is None
            or self.end_sec is None
            or self.capacity is not None
            or not self.at_sec <= self.start_sec < self.end_sec
            or self.start_sec % 5
            or self.end_sec % 5
        ):
            raise ValueError(
                "closure needs actor and nonretroactive five-second interval"
            )
        return self


def subtract_interval(windows, start, end):
    result = []
    for left, right in windows:
        if right <= start or left >= end:
            result.append([left, right])
        else:
            if left < start:
                result.append([left, start])
            if right > end:
                result.append([end, right])
    return result
