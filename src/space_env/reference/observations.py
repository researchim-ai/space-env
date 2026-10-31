"""Sampled revisit/stereo evidence for a static WGS84 surface.

Photogrammetric Earth-fixed baseline, not simultaneous intersatellite range,
SAR interferometry, stereo reconstruction or a continuous visibility proof.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from functools import lru_cache

import numpy as np
from pydantic import Field, model_validator

from space_env.core.frames import geodetic_to_ecef
from space_env.reference.area_coverage import (
    _geometry,
    _libraries,
    _project_footprint,
    domain,
    merge_coverage,
)
from space_env.reference.payload_quality import evidence_qualifies, merge_evidence
from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import digest


class RevisitRequirement(StrictModel):
    min_observations: int = Field(default=2, ge=2, le=8, strict=True)
    min_interval_sec: float = Field(gt=0, strict=True)
    max_interval_sec: float = Field(gt=0, le=259200, strict=True)
    different_actors: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_interval_sec > self.max_interval_sec:
            raise ValueError("revisit requires ordered time interval")
        return self


class StereoRequirement(StrictModel):
    min_convergence_deg: float = Field(gt=0, lt=120, strict=True)
    max_convergence_deg: float = Field(gt=0, le=120, strict=True)
    min_interval_sec: float = Field(default=0.0, ge=0, strict=True)
    max_interval_sec: float = Field(gt=0, le=259200, strict=True)
    min_baseline_m: float = Field(default=0.0, ge=0, strict=True)
    max_baseline_m: float | None = Field(default=None, gt=0, strict=True)
    different_actors: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if (
            self.min_convergence_deg >= self.max_convergence_deg
            or self.min_interval_sec > self.max_interval_sec
        ):
            raise ValueError("stereo requires ordered angle/time limits")
        if (
            self.max_baseline_m is not None
            and self.min_baseline_m > self.max_baseline_m
        ):
            raise ValueError("stereo requires ordered baseline limits")
        return self


class ObservationRequirement(StrictModel):
    revisit: RevisitRequirement | None = None
    stereo: StereoRequirement | None = None
    support_product_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def criteria(self):
        if self.revisit is None and self.stereo is None:
            raise ValueError("observation requirement needs revisit or stereo")
        if len(set(self.support_product_ids)) != len(self.support_product_ids) or any(
            not name.strip() for name in self.support_product_ids
        ):
            raise ValueError("support products must be unique nonblank IDs")
        return self


class ObservationSettings(StrictModel):
    max_records_per_product: int = Field(default=4096, ge=1, le=100000, strict=True)
    max_frames_per_evaluation: int = Field(default=256, ge=2, le=4096, strict=True)
    max_pair_checks: int = Field(default=20000, ge=1, le=1000000, strict=True)
    max_chain_checks: int = Field(default=20000, ge=1, le=1000000, strict=True)
    cell_size_m: float = Field(default=1000.0, ge=10, le=10000, strict=True)
    max_cells_per_pair: int = Field(default=1024, ge=1, le=4096, strict=True)
    max_cell_checks: int = Field(default=100000, ge=1, le=1000000, strict=True)
    max_points_per_region: int = Field(default=256, ge=5, le=4096, strict=True)


class ObservationBudgetError(ValueError):
    pass


def capture_observation(provider, quaternion, aoi, sample, raw_id, actor):
    _, shapely = _libraries()
    time = sample["time_sec"]
    epoch = provider.epoch.shifted(time)
    state = np.asarray(provider.state(time))
    row = {
        "schema": "space-env.observation/1",
        "raw_id": raw_id,
        "actor": actor,
        "aoi_id": aoi.id,
        "aoi_sha256": digest(aoi.model_dump(mode="json")),
        "time_sec": time,
        "epoch_utc": epoch.iso(),
        "epoch_tai_jd": [epoch.tai1, epoch.tai2],
        "trajectory_sha256": provider.sha256,
        "state_gcrs_si": state.tolist(),
        "position_itrs_m": (provider.context.matrix(epoch) @ state[:3]).tolist(),
        "quaternion_body_to_gcrs_wxyz": np.asarray(quaternion).tolist(),
        "geometry": json.loads(
            shapely.to_geojson(_project_footprint(aoi, sample["shell"]))
        ),
    }
    return {"id": digest(row), **row}


def merge_observations(products, limit):
    rows, exposures = {}, {}
    for product in products:
        for row in product.get("observations", []):
            if row["id"] != digest({k: v for k, v in row.items() if k != "id"}):
                raise ValueError("observation content fingerprint mismatch")
            key = (row["raw_id"], row["time_sec"])
            if key in exposures and exposures[key] != row["id"]:
                raise ValueError("conflicting pose for one acquisition sample")
            exposures[key] = row["id"]
            if row["id"] in rows and rows[row["id"]] != row:
                raise ValueError("incompatible observation lineage")
            rows[row["id"]] = row
            if len(rows) > limit:
                raise ObservationBudgetError("observation record budget exceeded")
    return [copy.deepcopy(rows[key]) for key in sorted(rows)]


def assemble_products(products, observation_limit, quality_limit):
    area_rows = [
        r for product in products for r in product.get("area_coverage", {}).values()
    ]
    return {
        "area_coverage": merge_coverage(area_rows) if area_rows else {},
        "quality_evidence": merge_evidence(products, quality_limit),
        "observations": merge_observations(products, observation_limit),
    }


def stereo_angles(left_position, right_position, ground_points):
    """Angles between two surface-to-camera rays, radians converted to degrees."""
    a = np.asarray(left_position) - np.asarray(ground_points)
    b = np.asarray(right_position) - np.asarray(ground_points)
    if np.any(np.linalg.norm(a, axis=-1) == 0) or np.any(
        np.linalg.norm(b, axis=-1) == 0
    ):
        raise ValueError("camera coincides with stereo ground point")
    return np.rad2deg(
        np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))
    )


def _evaluate_observations(requirement, product, aoi, quality, min_fraction, settings):
    """Fail closed on budgets; count distinct acquisitions, never DAG copies."""
    _, shapely = _libraries()
    stats = {"pair_checks": 0, "chain_checks": 0, "cell_checks": 0}
    result = {
        "schema": "space-env.observation-evaluation/1",
        "status": "evaluated",
        "satisfied": False,
        "joint_area_m2": 0.0,
        "joint_fraction": 0.0,
        "distinct_acquisitions": 0,
        "revisit": None,
        "stereo": None,
        "checks": stats,
    }
    if product is None:
        result["status"] = "missing_products"
        return result
    area = domain(aoi)[0]
    aoi_hash = digest(aoi.model_dump(mode="json"))

    def spend(kind, maximum):
        if stats[kind] >= maximum:
            raise ObservationBudgetError(f"observation {kind} budget exceeded")
        stats[kind] += 1

    try:
        frames = [
            row
            for row in merge_observations([product], settings.max_records_per_product)
            if row["aoi_id"] == aoi.id and row["aoi_sha256"] == aoi_hash
        ]
        if len(frames) > settings.max_frames_per_evaluation:
            raise ObservationBudgetError("observation evaluation frame budget exceeded")
        evidence = {}
        for row in product.get("quality_evidence", []):
            if row.get("observation_id") is not None:
                evidence.setdefault(row["observation_id"], []).append(row)
        geometries = {}
        for frame in frames:
            polygon = _geometry(frame["geometry"]).intersection(area)
            if quality is not None:
                allowed = [
                    row
                    for row in evidence.get(frame["id"], [])
                    if row["raw_id"] == frame["raw_id"]
                    and row["actor"] == frame["actor"]
                    and row["time_sec"] == frame["time_sec"]
                    and row["aoi_id"] == aoi.id
                    and row["aoi_sha256"] == aoi_hash
                    and evidence_qualifies(quality, row)
                ]
                polygon = polygon.intersection(
                    shapely.union_all([_geometry(row["geometry"]) for row in allowed])
                )
            if polygon.area > 0:
                geometries[frame["id"]] = polygon
        frames = sorted(
            [row for row in frames if row["id"] in geometries],
            key=lambda row: (row["time_sec"], row["raw_id"], row["id"]),
        )
        result["distinct_acquisitions"] = len({row["raw_id"] for row in frames})
        joint = area
        if requirement.revisit is not None:
            config = requirement.revisit
            accepted, witnesses = [], []

            def visit(chain, begin, intersection):
                if len(chain) == config.min_observations:
                    accepted.append(intersection)
                    if len(witnesses) < 32:
                        witnesses.append(
                            {
                                "raw_ids": [r["raw_id"] for r in chain],
                                "observation_ids": [r["id"] for r in chain],
                                "actors": [r["actor"] for r in chain],
                                "times_sec": [r["time_sec"] for r in chain],
                            }
                        )
                    return
                for index in range(begin, len(frames)):
                    row = frames[index]
                    spend("chain_checks", settings.max_chain_checks)
                    if any(peer["raw_id"] == row["raw_id"] for peer in chain):
                        continue
                    if config.different_actors and any(
                        peer["actor"] == row["actor"] for peer in chain
                    ):
                        continue
                    if (
                        chain
                        and not config.min_interval_sec
                        <= row["time_sec"] - chain[-1]["time_sec"]
                        <= config.max_interval_sec
                    ):
                        continue
                    shared = intersection.intersection(geometries[row["id"]])
                    if shared.area > 0:
                        visit([*chain, row], index + 1, shared)

            if result["distinct_acquisitions"] >= config.min_observations:
                visit([], 0, area)
            qualifying = shapely.union_all(accepted)
            result["revisit"] = {
                "area_m2": float(qualifying.area),
                "fraction": min(1.0, float(qualifying.area / area.area)),
                "witnesses": witnesses,
                "witnesses_truncated": len(accepted) > len(witnesses),
            }
            joint = joint.intersection(qualifying)
        if requirement.stereo is not None:
            config = requirement.stereo
            accepted, witnesses = [], []
            groups = {}
            for frame in frames:
                groups.setdefault(frame["raw_id"], []).append(frame)
            transform = domain(aoi)[1]
            for first, second in itertools.combinations(sorted(groups), 2):
                for left, right in itertools.product(groups[first], groups[second]):
                    spend("pair_checks", settings.max_pair_checks)
                    delta = abs(left["time_sec"] - right["time_sec"])
                    baseline = float(
                        np.linalg.norm(
                            np.asarray(left["position_itrs_m"])
                            - right["position_itrs_m"]
                        )
                    )
                    if (
                        not config.min_interval_sec <= delta <= config.max_interval_sec
                        or baseline < config.min_baseline_m
                    ):
                        continue
                    if (
                        config.max_baseline_m is not None
                        and baseline > config.max_baseline_m
                    ):
                        continue
                    if config.different_actors and left["actor"] == right["actor"]:
                        continue
                    shared = geometries[left["id"]].intersection(
                        geometries[right["id"]]
                    )
                    if shared.area <= 0:
                        continue
                    size = settings.cell_size_m
                    x0, y0, x1, y1 = shared.bounds
                    ix0, iy0, ix1, iy1 = (
                        math.floor(x0 / size),
                        math.floor(y0 / size),
                        math.floor(x1 / size),
                        math.floor(y1 / size),
                    )
                    if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > settings.max_cells_per_pair:
                        raise ObservationBudgetError("stereo cell grid budget exceeded")
                    pair_cells, angles = [], []
                    for ix in range(ix0, ix1 + 1):
                        for iy in range(iy0, iy1 + 1):
                            spend("cell_checks", settings.max_cell_checks)
                            cell = shared.intersection(
                                shapely.box(
                                    ix * size,
                                    iy * size,
                                    (ix + 1) * size,
                                    (iy + 1) * size,
                                )
                            )
                            if cell.area <= 0:
                                continue
                            points = shapely.get_coordinates(cell)
                            if len(points) + 1 > settings.max_points_per_region:
                                raise ObservationBudgetError(
                                    "stereo spatial point budget exceeded"
                                )
                            p = cell.representative_point()
                            points = np.vstack((points, [p.x, p.y]))
                            lon, lat = transform.transform(
                                points[:, 0],
                                points[:, 1],
                                direction="INVERSE",
                                errcheck=True,
                            )
                            ground = [
                                geodetic_to_ecef(y, x, 0.0) for x, y in zip(lon, lat)
                            ]
                            values = stereo_angles(
                                left["position_itrs_m"],
                                right["position_itrs_m"],
                                ground,
                            )
                            if np.all(values >= config.min_convergence_deg) and np.all(
                                values <= config.max_convergence_deg
                            ):
                                pair_cells.append(cell)
                                angles.extend(values.tolist())
                    pair_area = shapely.union_all(pair_cells)
                    if pair_area.area > 0:
                        accepted.append(pair_area)
                        if len(witnesses) < 32:
                            witnesses.append(
                                {
                                    "raw_ids": [first, second],
                                    "observation_ids": [left["id"], right["id"]],
                                    "actors": [left["actor"], right["actor"]],
                                    "times_sec": [left["time_sec"], right["time_sec"]],
                                    "baseline_itrs_m": baseline,
                                    "min_sampled_angle_deg": min(angles),
                                    "max_sampled_angle_deg": max(angles),
                                }
                            )
            qualifying = shapely.union_all(accepted)
            result["stereo"] = {
                "area_m2": float(qualifying.area),
                "fraction": min(1.0, float(qualifying.area / area.area)),
                "witnesses": witnesses,
                "witnesses_truncated": len(accepted) > len(witnesses),
            }
            joint = joint.intersection(qualifying)
        # Every criterion must hold on the same area in the named product/bundle.
        if aoi.id in product.get("area_coverage", {}):
            joint = joint.intersection(
                _geometry(product["area_coverage"][aoi.id]["geometry"])
            )
        else:
            joint = shapely.GeometryCollection()
        result.update(
            joint_area_m2=float(joint.area),
            joint_fraction=min(1.0, float(joint.area / area.area)),
        )
        result["satisfied"] = result["joint_fraction"] >= min_fraction
    except ObservationBudgetError as error:
        result.update(
            status="budget_exceeded",
            reason=str(error),
            satisfied=False,
            joint_area_m2=0.0,
            joint_fraction=0.0,
        )
    return result


@lru_cache(maxsize=32)
def _cached_evaluation(requirement, product, aoi, quality, min_fraction, settings):
    from space_env.reference.area_coverage import AreaOfInterest
    from space_env.reference.payload_quality import QualityRequirement

    return _evaluate_observations(
        ObservationRequirement.model_validate_json(requirement),
        json.loads(product),
        AreaOfInterest.model_validate_json(aoi),
        None if quality is None else QualityRequirement.model_validate_json(quality),
        min_fraction,
        ObservationSettings.model_validate_json(settings),
    )


def evaluate_observations(requirement, product, aoi, quality, min_fraction, settings):
    data = (
        None
        if product is None
        else {
            key: product.get(key, {} if key == "area_coverage" else [])
            for key in ("observations", "quality_evidence", "area_coverage")
        }
    )
    return copy.deepcopy(
        _cached_evaluation(
            requirement.model_dump_json(),
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False),
            aoi.model_dump_json(),
            None if quality is None else quality.model_dump_json(),
            min_fraction,
            settings.model_dump_json(),
        )
    )
