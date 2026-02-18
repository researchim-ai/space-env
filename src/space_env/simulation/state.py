"""
Снимок состояния симуляции в один момент времени.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from space_env.core.orbit import Orbit
    from space_env.core.spacecraft import SpacecraftState
    from space_env.core.time import Time


class SimulationState(BaseModel):
    """Состояние симуляции: время, орбита (r, v, altitude), состояние КА, счётчики миссии."""
    time_sec: float = Field(ge=0)
    altitude_km: float = 0.0
    in_eclipse: bool = False
    battery_soc: float = Field(ge=0, le=1)
    delta_v_remaining_m_s: float = Field(ge=0)
    temp_C: float = 20.0
    # Счётчики (для целей)
    imaging_count: int = Field(default=0, ge=0)
    comms_sessions: int = Field(default=0, ge=0)
    data_downlink_Mb: float = Field(default=0.0, ge=0)
    delta_v_used_m_s: float = Field(default=0.0, ge=0)
