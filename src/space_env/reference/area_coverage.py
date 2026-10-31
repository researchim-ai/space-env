"""Bounded WGS84 polygon/rectangular-camera coverage, with explicit sampling.

Shapely/PROJ are optional and loaded only for area scenarios. This is geometry,
not a radiometric image, continuous swath, terrain or stereo reconstruction.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from itertools import pairwise
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.constants import R_EARTH_EQUATOR, R_EARTH_POLAR
from space_env.core.frames import ecef_to_geodetic
from space_env.reference.spacecraft_bus import rotate_body
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.snapshot import digest

Coordinate = tuple[FiniteScalar, FiniteScalar]  # longitude, latitude, degrees


class AreaOfInterest(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    anchor_lon_deg: float = Field(ge=-180, le=180, strict=True)
    anchor_lat_deg: float = Field(ge=-90, le=90, strict=True)
    shell: tuple[Coordinate, ...] = Field(min_length=4, max_length=2048)
    holes: tuple[tuple[Coordinate, ...], ...] = Field(default=(), max_length=32)
    max_segment_m: float = Field(default=1000.0, ge=10, le=10000, strict=True)
    source_status: Literal["synthetic", "published", "observed"] = "synthetic"
    source: str = Field(default="caller-declared synthetic polygon", min_length=1)

    @model_validator(mode="after")
    def rings(self):
        if not self.id.strip():
            raise ValueError("AOI ID must be nonblank")
        if sum(map(len, (self.shell, *self.holes))) > 10000:
            raise ValueError("AOI vertex budget exceeded")
        for ring in (self.shell, *self.holes):
            if not 4 <= len(ring) <= 2048 or ring[0] != ring[-1]:
                raise ValueError("AOI requires explicitly closed bounded rings")
            if any(not -180 <= lon <= 180 or not -90 <= lat <= 90 for lon, lat in ring):
                raise ValueError("AOI coordinate outside longitude/latitude domain")
        return self


class AreaRequirement(StrictModel):
    aoi_id: str = Field(min_length=1)
    min_fraction: float = Field(default=0.95, gt=0, le=1, strict=True)


class RectangularFootprint(StrictModel):
    half_angle_x_deg: float = Field(gt=0, le=45, strict=True)
    half_angle_y_deg: float = Field(gt=0, le=45, strict=True)
    edge_samples: int = Field(default=8, ge=1, le=64, strict=True)
    max_acquisition_samples: int = Field(default=4096, ge=3, le=20000, strict=True)


def _libraries():
    try:
        import pyproj
        import shapely
    except ImportError as error:
        raise ValueError("area coverage requires space-env[coverage]") from error
    return pyproj, shapely


def geometry_manifest():
    pyproj, shapely = _libraries()
    return {
        "pyproj": pyproj.__version__,
        "proj": pyproj.proj_version_str,
        "shapely": shapely.__version__,
        "geos": shapely.geos_version_string,
    }


@lru_cache(maxsize=128)
def _domain(serialized):
    pyproj, shapely = _libraries()
    aoi = AreaOfInterest.model_validate_json(serialized)
    geod = pyproj.Geod(ellps="WGS84")
    projection = pyproj.CRS.from_proj4(
        f"+proj=laea +lat_0={aoi.anchor_lat_deg} +lon_0={aoi.anchor_lon_deg} +datum=WGS84 +units=m"
    )
    transform = pyproj.Transformer.from_crs("EPSG:4326", projection, always_xy=True)

    vertex_count = 0

    def project(ring):
        nonlocal vertex_count
        dense = []
        for start, end in pairwise(ring):
            _, _, distance = geod.inv(*start, *end)
            if distance < 1e-6:
                raise ValueError("AOI has duplicate consecutive vertices")
            count = max(0, math.ceil(distance / aoi.max_segment_m) - 1)
            if vertex_count + len(dense) + count + 2 > 20000:
                raise ValueError("AOI densification budget exceeded")
            dense.append(start)
            if count:
                dense.extend(geod.npts(*start, *end, count))
        dense.append(ring[-1])
        vertex_count += len(dense)
        if vertex_count > 20000:
            raise ValueError("AOI densification budget exceeded")
        for lon, lat in dense:
            if geod.inv(aoi.anchor_lon_deg, aoi.anchor_lat_deg, lon, lat)[2] > 1000000:
                raise ValueError("AOI exceeds 1000 km anchor radius")
        lon, lat = np.asarray(dense).T
        x, y = transform.transform(lon, lat, errcheck=True)
        return list(zip(x, y))

    polygon = shapely.Polygon(project(aoi.shell), [project(ring) for ring in aoi.holes])
    if not polygon.is_valid or polygon.is_empty or polygon.area <= 1e-4:
        raise ValueError("AOI polygon is invalid, empty or degenerate")
    return polygon, transform


def domain(aoi):
    return _domain(aoi.model_dump_json())


def ray_ellipsoid(origin, direction):
    """First forward hit in metres; None for misses/behind/tangent/inside Earth."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if (
        origin.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite([origin, direction]).all()
    ):
        raise ValueError("ray requires finite three-vectors")
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("ray direction is zero")
    direction = direction / norm
    axes = np.array([R_EARTH_EQUATOR, R_EARTH_EQUATOR, R_EARTH_POLAR])
    o, d = origin / axes, direction / axes
    c = float(o @ o - 1)
    if c <= 0:
        return None
    a, b = float(d @ d), float(o @ d)
    discriminant = b * b - a * c
    if discriminant <= 0:
        return None
    # Stable quadratic roots: avoid subtracting nearly equal terms near surface.
    q = -b - math.copysign(math.sqrt(discriminant), b)
    roots = sorted(t for t in (q / a, c / q) if t > 0)
    return None if not roots else origin + roots[0] * direction


def rectangular_footprint(provider, quaternion, sensor, time_sec):
    """Sample rectangular detector edges; reject rather than fabricate limb clipping."""
    sensor = RectangularFootprint.model_validate(sensor)
    quaternion = np.asarray(quaternion, dtype=float)
    if (
        quaternion.shape != (4,)
        or not np.isfinite(quaternion).all()
        or not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-8)
    ):
        raise ValueError("camera requires a unit body-to-GCRS quaternion")
    if not math.isfinite(time_sec):
        raise ValueError("camera time must be finite")
    rotation = provider.context.matrix(provider.epoch.shifted(time_sec))
    origin = rotation @ np.asarray(provider.state(time_sec))[:3]
    hx, hy = np.tan(np.deg2rad([sensor.half_angle_x_deg, sensor.half_angle_y_deg]))
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    ring = []
    for index, left in enumerate(corners):
        right = corners[(index + 1) % 4]
        for f in np.arange(sensor.edge_samples) / sensor.edge_samples:
            x, y = np.asarray(left) * (1 - f) + np.asarray(right) * f
            direction = rotation @ rotate_body(quaternion, [x, y, 1.0])
            hit = ray_ellipsoid(origin, direction)
            if hit is None:
                return {
                    "status": "unavailable",
                    "reason": "footprint_horizon",
                    "time_sec": float(time_sec),
                }
            lat, lon, _ = ecef_to_geodetic(hit)
            ring.append([float(lon), float(lat)])
    ring.append(ring[0])
    return {"status": "ok", "time_sec": float(time_sec), "shell": ring}


def _project_footprint(aoi, ring):
    pyproj, shapely = _libraries()
    polygon, transform = domain(aoi)
    lon, lat = np.asarray(ring).T
    geod = pyproj.Geod(ellps="WGS84")
    if any(
        geod.inv(aoi.anchor_lon_deg, aoi.anchor_lat_deg, x, y)[2] > 2000000
        for x, y in ring
    ):
        raise ValueError("footprint exceeds 2000 km AOI projection domain")
    x, y = transform.transform(lon, lat, errcheck=True)
    footprint = shapely.Polygon(zip(x, y))
    if (
        not footprint.is_valid
        or not np.isfinite(np.asarray(footprint.exterior.coords)).all()
    ):
        raise ValueError("invalid projected footprint")
    return shapely.intersection(polygon, footprint)


def _geometry(value):
    _, shapely = _libraries()
    return shapely.from_geojson(json.dumps(value))


def coverage_from_samples(aoi, samples):
    _, shapely = _libraries()
    polygons = [_project_footprint(aoi, sample["shell"]) for sample in samples]
    union = shapely.union_all(polygons)
    polygon, _ = domain(aoi)
    return {
        "aoi_id": aoi.id,
        "aoi_sha256": digest(aoi.model_dump(mode="json")),
        "projection": "WGS84/ellipsoidal-LAEA; metres; declared anchor",
        "model": "sampled-rectangular-frame-union/1",
        "aoi_area_m2": float(polygon.area),
        "covered_area_m2": float(union.area),
        "fraction": min(1.0, float(union.area / polygon.area)),
        "sample_count": len(samples),
        "geometry": json.loads(shapely.to_geojson(union)),
    }


def merge_coverage(rows):
    """Union in each AOI's projection; repeated lineage cannot inflate area."""
    _, shapely = _libraries()
    groups = {}
    for row in rows:
        groups.setdefault(row["aoi_id"], []).append(row)
    result = {}
    for name, group in sorted(groups.items()):
        template = group[0]
        if any(row["aoi_sha256"] != template["aoi_sha256"] for row in group):
            raise ValueError("incompatible AOI coverage lineage")
        union = shapely.union_all([_geometry(row["geometry"]) for row in group])
        sources = {}
        for row in group:
            for key, value in row.get("sources", {}).items():
                if key in sources and sources[key] != value:
                    raise ValueError("incompatible acquisition source lineage")
                sources[key] = value
        result[name] = {
            **template,
            "covered_area_m2": float(union.area),
            "fraction": min(1.0, float(union.area / template["aoi_area_m2"])),
            "sample_count": sum(row["sample_count"] for row in sources.values())
            if sources
            else sum(row["sample_count"] for row in group),
            **({"sources": sources} if sources else {}),
            "geometry": json.loads(shapely.to_geojson(union)),
        }
    return result


def coverage_satisfied(requirement, artifact):
    if requirement is None:
        return True
    row = artifact.get("area_coverage", {}).get(requirement.aoi_id)
    return row is not None and row["fraction"] >= requirement.min_fraction


def area_report(aois, artifacts):
    _, shapely = _libraries()
    raw = [
        p
        for p in artifacts.values()
        if p["source"] == "acquired" and p.get("area_coverage")
    ]
    delivered = [
        p
        for p in artifacts.values()
        if p["status"] == "delivered" and p.get("area_coverage")
    ]
    revisits = []
    checks = 0
    # Bound quadratic diagnostics; product union itself includes every product.
    for index, left in enumerate(raw):
        for right in raw[index + 1 :]:
            checks += 1
            for name in sorted(
                left["area_coverage"].keys() & right["area_coverage"].keys()
            ):
                overlap = shapely.intersection(
                    _geometry(left["area_coverage"][name]["geometry"]),
                    _geometry(right["area_coverage"][name]["geometry"]),
                )
                if overlap.area > 0:
                    revisits.append(
                        {
                            "aoi_id": name,
                            "product_ids": [left["id"], right["id"]],
                            "actors": [left["actor"], right["actor"]],
                            "interval_sec": abs(
                                left["created_sec"] - right["created_sec"]
                            ),
                            "overlap_area_m2": float(overlap.area),
                        }
                    )
                if len(revisits) >= 10000:
                    break
            if checks >= 10000 or len(revisits) >= 10000:
                break
        if checks >= 10000 or len(revisits) >= 10000:
            break
    return {
        "schema": "space-env.area-coverage/1",
        "areas": {
            name: {
                "definition": aoi.model_dump(mode="json"),
                "area_m2": float(domain(aoi)[0].area),
            }
            for name, aoi in aois.items()
        },
        "acquired": merge_coverage(
            [r for p in raw for r in p["area_coverage"].values()]
        ),
        "delivered": merge_coverage(
            [r for p in delivered for r in p["area_coverage"].values()]
        ),
        "revisits": revisits,
        "revisit_limit": 10000,
        "revisit_pair_checks": checks,
        "revisits_truncated": len(raw) * (len(raw) - 1) // 2 > checks,
        "scope": "sampled frames, geodesically densified AOI/LAEA union; no continuous swath, cloud, terrain or stereo quality",
    }
