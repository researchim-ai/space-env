"""Analytic one-node transmitter thermal surrogate; explicit user parameters."""
from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TransmitterThermal(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    ambient_k: float = Field(default=280., gt=0)
    initial_k: float = Field(default=280., gt=0)
    capacity_j_per_k: float = Field(gt=0)
    conductance_w_per_k: float = Field(gt=0)
    heat_tx_w: float = Field(ge=0)
    trip_k: float = Field(gt=0)
    resume_k: float = Field(gt=0)

    @model_validator(mode="after")
    def thresholds(self):
        if not self.ambient_k < self.resume_k < self.trip_k:
            raise ValueError("thermal thresholds require ambient < resume < trip")
        if not math.isfinite(self.capacity_j_per_k/self.conductance_w_per_k) or not math.isfinite(
                self.ambient_k+self.heat_tx_w/self.conductance_w_per_k):
            raise ValueError("derived thermal constants must be finite")
        return self

    def advance(self, temperature: float, duration: float, transmitting: bool) -> float:
        equilibrium = self.ambient_k+(self.heat_tx_w/self.conductance_w_per_k if transmitting else 0.)
        fraction = -math.expm1(-duration*self.conductance_w_per_k/self.capacity_j_per_k)
        return temperature+(equilibrium-temperature)*fraction

    def crossing(self, temperature: float, target: float, transmitting: bool) -> float:
        equilibrium = self.ambient_k+(self.heat_tx_w/self.conductance_w_per_k if transmitting else 0.)
        if not min(temperature, equilibrium) < target < max(temperature, equilibrium):
            return math.inf
        return -self.capacity_j_per_k/self.conductance_w_per_k*math.log((target-equilibrium)/(temperature-equilibrium))
