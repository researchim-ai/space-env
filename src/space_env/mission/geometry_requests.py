"""
Геометрия заявки: точка, полоса, полигон, повтор и стереопара.
Зависимая заявка не открывается, пока предшественник не сдан.
"""
from __future__ import annotations

import math

from space_env.core.constants import R_EARTH_EQUATOR


def ground_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Дуга по сфере экваториального радиуса, м."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    chord = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_EQUATOR * math.asin(min(1.0, math.sqrt(chord)))


def strip_fits(length_m: float, footprint_diameter_m: float) -> bool:
    return float(length_m) <= float(footprint_diameter_m) + 1e-6


def polygon_fits(lat_deg: float, lon_deg: float, vertices: list[tuple[float, float]], footprint_diameter_m: float) -> bool:
    """Все вершины лежат в круге кадра вокруг центра заявки."""
    radius = float(footprint_diameter_m) / 2.0
    return all(ground_distance_m(lat_deg, lon_deg, lat, lon) <= radius + 1.0 for lat, lon in vertices)


def revisit_possible(hit_times_sec: list[float], looks: int, gap_sec: float) -> bool:
    """Хватает ли моментов видимости с заданным промежутком."""
    if looks < 2 or gap_sec <= 0.0:
        return False
    chosen: list[float] = []
    for moment in sorted(float(item) for item in hit_times_sec):
        if not chosen or moment + 1e-9 >= chosen[-1] + gap_sec:
            chosen.append(moment)
        if len(chosen) >= looks:
            return True
    return False


def stereo_possible(off_nadir_deg: list[float], min_separation_deg: float, max_separation_deg: float) -> bool:
    """Два взгляда, разность углов от надира лежит в заданном окне."""
    angles = sorted(float(item) for item in off_nadir_deg)
    for left in range(len(angles)):
        for right in range(left + 1, len(angles)):
            gap = angles[right] - angles[left]
            if min_separation_deg - 1e-9 <= gap <= max_separation_deg + 1e-9:
                return True
    return False


def dependency_open(depends_on: str, done_ids: set[str]) -> bool:
    if not depends_on:
        return True
    return depends_on in done_ids


def classify_request(spec: dict, *, footprint_diameter_m: float, hit_times_sec: list[float], off_nadir_deg: list[float], done_ids: set[str]) -> dict:
    """Один вердикт: достижима заявка или почему нет. Сдача предшественника проверяется отдельно."""
    geometry = str(spec.get("geometry", "point"))
    if not dependency_open(str(spec.get("depends_on", "")), done_ids):
        return {"reachable": False, "reason": "ждёт предшественника"}
    if geometry == "strip" and not strip_fits(float(spec.get("length_km", 0.0)) * 1000.0, footprint_diameter_m):
        return {"reachable": False, "reason": "полоса длиннее кадра"}
    if geometry == "polygon":
        vertices = [(float(item["lat_deg"]), float(item["lon_deg"])) for item in spec.get("vertices", [])]
        if not polygon_fits(float(spec["lat_deg"]), float(spec["lon_deg"]), vertices, footprint_diameter_m):
            return {"reachable": False, "reason": "полигон шире кадра"}
    if geometry == "revisit" and not revisit_possible(hit_times_sec, int(spec.get("looks", 1)), float(spec.get("gap_sec", 0.0))):
        return {"reachable": False, "reason": "мало повторных проходов"}
    if geometry == "stereo" and not stereo_possible(
        off_nadir_deg, float(spec.get("min_stereo_deg", 0.0)), float(spec.get("max_stereo_deg", 0.0))
    ):
        return {"reachable": False, "reason": "нет стереопары"}
    if geometry in {"point", "strip", "polygon"} and not hit_times_sec and geometry != "strip":
        return {"reachable": False, "reason": "вне трассы"}
    if geometry == "point" and not hit_times_sec:
        return {"reachable": False, "reason": "вне трассы"}
    return {"reachable": True, "reason": ""}
