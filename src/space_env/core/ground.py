"""Наземная станция: координаты и минимальный угол места."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GroundStation(BaseModel):
    """Станция приёма. Видимость считается в движке по углу места."""

    name: str = "GS"
    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=180)
    alt_m: float = Field(default=0.0, ge=-500, le=9000)
    min_elevation_deg: float = Field(default=5.0, ge=0, le=90)


DEFAULT_GROUND_STATIONS: tuple[GroundStation, ...] = (
    GroundStation(name="Moscow", lat_deg=55.752, lon_deg=37.618, alt_m=150.0, min_elevation_deg=5.0),
)
