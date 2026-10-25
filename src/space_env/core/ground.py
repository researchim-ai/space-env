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


def _dms(degrees: float, minutes: float, seconds: float) -> float:
    sign = -1.0 if degrees < 0 else 1.0
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


DEFAULT_GROUND_STATIONS: tuple[GroundStation, ...] = (
    GroundStation(name="Moscow", lat_deg=55.752, lon_deg=37.618, alt_m=150.0, min_elevation_deg=5.0),
    GroundStation(name="Svalbard", lat_deg=78.229, lon_deg=15.407, alt_m=500.0, min_elevation_deg=5.0),
    GroundStation(name="Inuvik", lat_deg=68.319, lon_deg=-133.549, alt_m=100.0, min_elevation_deg=5.0),
)

# KIR-1, ESA Kiruna: 67°51'25.66" N, 20°57'51.57" E, высота 402.2 м над эллипсоидом WGS-84.
KIRUNA = GroundStation(
    name="Kiruna",
    lat_deg=_dms(67, 51, 25.66),
    lon_deg=_dms(20, 57, 51.57),
    alt_m=402.2,
    min_elevation_deg=5.0,
)
# USGS Landsat: Matera POINT (40.65 16.7). Высота антенны в карточке станции не указана.
MATERA = GroundStation(name="Matera", lat_deg=40.65, lon_deg=16.70, alt_m=0.0, min_elevation_deg=5.0)
# USGS EROS / LGS: POINT (43.7361 -96.6225). Высота в карточке не указана.
SIOUX_FALLS = GroundStation(
    name="Sioux Falls", lat_deg=43.7361, lon_deg=-96.6225, alt_m=0.0, min_elevation_deg=5.0
)
# FCC SES E960131, Tempe: 33°20'32.2" N, 111°53'48.5" W. Высота в заявке не указана.
TEMPE = GroundStation(
    name="Tempe",
    lat_deg=_dms(33, 20, 32.2),
    lon_deg=-_dms(111, 53, 48.5),
    alt_m=0.0,
    min_elevation_deg=5.0,
)
# FCC SES E060300, Fairbanks: 64°49'03.2" N, 147°43'29.8" W.
FAIRBANKS = GroundStation(
    name="Fairbanks",
    lat_deg=_dms(64, 49, 3.2),
    lon_deg=-_dms(147, 43, 29.8),
    alt_m=0.0,
    min_elevation_deg=5.0,
)

_BY_NAME = {
    station.name: station
    for station in (
        *DEFAULT_GROUND_STATIONS,
        KIRUNA,
        MATERA,
        SIOUX_FALLS,
        TEMPE,
        FAIRBANKS,
    )
}


def station_by_name(name: str) -> GroundStation:
    """Станция каталога по имени. Неизвестное имя не подменяется координатами."""
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"станция не из каталога: {name}") from exc
