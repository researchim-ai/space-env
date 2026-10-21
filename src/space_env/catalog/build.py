"""Сборка орбиты и движка из каталога. TLE — снимок Celestrak, средние элементы — карточка миссии."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from space_env.catalog.geometry import mean_semimajor_axis_m, raan_rad_for_ltdn
from space_env.catalog.references import LANDSAT_STATIONS, SENTINEL_STATIONS, reference
from space_env.core.ground import GroundStation
from space_env.core.orbit import Orbit
from space_env.core.space_weather import indices_for
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.fleet import Fleet


_TLE_DIR = Path(__file__).resolve().parent / "tle"


def read_tle(key: str) -> tuple[str, str]:
    path = _TLE_DIR / f"{key}.tle"
    if not path.is_file():
        raise KeyError(f"нет снимка TLE для {key}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    line1 = next(line for line in lines if line.startswith("1 "))
    line2 = next(line for line in lines if line.startswith("2 "))
    return line1, line2


def spacecraft_config(key: str) -> SpacecraftConfig:
    return reference(key).spacecraft_config()


def mean_orbit(key: str, epoch: str = "2026-01-01T00:00:00Z", nu_deg: float = 0.0) -> Orbit:
    """Круговая орбита карточки. Узел выставлен по местному времени нисходящего узла, если оно опубликовано."""
    spec = reference(key)
    moment = Time(epoch)
    raan = 0.0 if spec.ltdn_hours is None else raan_rad_for_ltdn(moment, spec.ltdn_hours)
    return Orbit.from_keplerian(
        mean_semimajor_axis_m(spec.altitude_m),
        0.0,
        np.deg2rad(spec.inclination_deg),
        raan,
        0.0,
        np.deg2rad(nu_deg),
        moment,
    )


def tle_orbit(key: str) -> Orbit:
    """Состояние SGP4 на эпохе снимка. Нужен пакет sgp4."""
    from space_env.integrations.sgp4_prop import orbit_from_tle

    line1, line2 = read_tle(key)
    return orbit_from_tle(line1, line2)


def _stations(key: str) -> list[GroundStation]:
    if key.startswith("sentinel-2"):
        return list(SENTINEL_STATIONS)
    if key == "landsat-9":
        return list(LANDSAT_STATIONS)
    if key == "cubesat-6u":
        return [next(station for station in SENTINEL_STATIONS if station.name == "Svalbard")]
    return []


def _scenario(key: str, moment: Time, targets: list | None, with_stations: bool) -> ScenarioConfig:
    weather = indices_for(moment)
    flux = 150.0 if weather is None else weather.f107
    return ScenarioConfig(
        ground_stations=_stations(key) if with_stations else [],
        ground_targets=list(targets or []),
        f107=flux,
        f107a=150.0 if weather is None else weather.f107a,
        ap=4.0 if weather is None else weather.ap,
    )


def make_engine(
    key: str,
    *,
    use_tle: bool = False,
    epoch: str = "2026-01-01T00:00:00Z",
    nu_deg: float = 0.0,
    with_stations: bool = True,
    ground_targets: list | None = None,
) -> SimulationEngine:
    """Один аппарат. Со станциями поиск окон идёт на 6 ч, как у обычного движка."""
    orbit = tle_orbit(key) if use_tle else mean_orbit(key, epoch, nu_deg=nu_deg)
    craft = Spacecraft(config=spacecraft_config(key))
    craft.reset_state()
    return SimulationEngine(
        orbit,
        craft,
        scenario=_scenario(key, orbit.epoch, ground_targets, with_stations),
    )


def sentinel2_design_fleet(
    epoch: str = "2026-01-01T00:00:00Z",
    *,
    with_stations: bool = False,
) -> Fleet:
    """
    Пара из карточки: одна плоскость, 180° по аномалии.
    Живой TLE на дату снимка может стоять иначе — его даёт tle_orbit.
    """
    from space_env.simulation.fleet import _make_vehicle

    moment = Time(epoch)
    spec = reference("sentinel-2a")
    raan_deg = float(np.rad2deg(raan_rad_for_ltdn(moment, spec.ltdn_hours or 10.5)))
    scenario = ScenarioConfig(ground_stations=_stations("sentinel-2a") if with_stations else [])
    vehicles = {}
    for key, nu in (("sentinel-2a", 0.0), ("sentinel-2b", 180.0)):
        spec_i = reference(key)
        vehicles[spec_i.title] = _make_vehicle(
            spec_i.title,
            mean_semimajor_axis_m(spec.altitude_m),
            spec.inclination_deg,
            raan_deg,
            nu,
            moment,
            scenario,
            spec_i.spacecraft_config(spec_i.title),
        )
    return Fleet(vehicles)


def iridium_next_fleet() -> Fleet:
    """Полная звезда 66/6/2. Станций нет: абонентская линия не замкнута на зеркало."""
    spec = reference("iridium-next")
    return Fleet.walker_star(
        66,
        6,
        2,
        a_m=mean_semimajor_axis_m(spec.altitude_m),
        inc_deg=spec.inclination_deg,
        scenario=ScenarioConfig(ground_stations=[]),
        spacecraft=spec.spacecraft_config("Iridium-NEXT"),
    )
