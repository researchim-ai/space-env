"""
Снимок состояния симуляции в один момент времени.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SimulationState(BaseModel):
    """Состояние симуляции: время, орбита, КА, связь, счётчики миссии и сближение."""
    time_sec: float = Field(ge=0)
    altitude_km: float = 0.0
    in_eclipse: bool = False
    shadow: str = "sun"
    battery_soc: float = Field(ge=0, le=1)
    delta_v_remaining_m_s: float = Field(ge=0)
    temp_C: float = 20.0
    heater_on: bool = False
    imaging_count: int = Field(default=0, ge=0)
    comms_sessions: int = Field(default=0, ge=0)
    data_downlink_Mb: float = Field(default=0.0, ge=0)
    stored_bits: float = Field(default=0.0, ge=0)
    generated_bits: float = Field(default=0.0, ge=0)
    delivered_bits: float = Field(default=0.0, ge=0)
    lost_bits: float = Field(default=0.0, ge=0)
    produced_bits: float = Field(default=0.0, ge=0)
    initial_bits: float = Field(default=0.0, ge=0)
    received_bits: float = Field(default=0.0, ge=0)
    relayed_out_bits: float = Field(default=0.0, ge=0)
    delta_v_used_m_s: float = Field(default=0.0, ge=0)
    in_contact: bool = False
    elevation_deg: float = -90.0
    contact_station: str = ""
    payload_mode: str = "idle"
    pointing: str = "nadir"
    conjunction_miss_km: float | None = None
    conjunction_probability: float | None = None
    conjunction_tca_passed: bool = False
    link_ok: bool = True
    bus_voltage_V: float = 28.0
    soc_rate_per_s: float = 0.0
    miss_radial_m: float | None = None
    miss_cross_m: float | None = None
    link_margin_db: float = -99.0
    slew_remaining_deg: float = 0.0
    seconds_to_aos: float | None = None
