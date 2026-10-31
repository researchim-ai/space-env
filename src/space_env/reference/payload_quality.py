"""Explicit optical masks/geometry and stripmap SAR resource surrogates.

Evidence is spatial, sampled and immutable after acquisition. No radiometry,
cloud classifier, focused SAR image or flight-calibrated resolution is implied.
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.frames import geodetic_to_ecef
from space_env.reference.area_coverage import (
    _geometry,
    _libraries,
    _project_footprint,
    domain,
)
from space_env.reference.orbit import _ephemerides
from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.snapshot import digest


class OpticalModel(StrictModel):
    kind: Literal["optical"] = "optical"
    model_id: str = Field(min_length=1)
    source_status: Literal["synthetic", "published", "calibrated"]
    source: str = Field(min_length=1)
    instantaneous_fov_rad: float = Field(ge=1e-9, le=0.01, strict=True)
    max_incidence_deg: float = Field(default=70.0, gt=0, le=80, strict=True)
    min_sun_elevation_deg: float = Field(default=5.0, ge=0, le=90, strict=True)

    @model_validator(mode="after")
    def attribution(self):
        if not self.model_id.strip() or not self.source.strip():
            raise ValueError("imaging model ID/source must be nonblank")
        return self


class SarModel(StrictModel):
    kind: Literal["sar"] = "sar"
    model_id: str = Field(min_length=1)
    source_status: Literal["synthetic", "published", "calibrated"]
    source: str = Field(min_length=1)
    bandwidth_hz: float = Field(ge=1.0, le=1e10, strict=True)
    antenna_length_m: float = Field(gt=0, le=1000, strict=True)
    min_incidence_deg: float = Field(default=15.0, gt=0, lt=80, strict=True)
    max_incidence_deg: float = Field(default=60.0, gt=0, le=80, strict=True)
    look_side: Literal["left", "right", "either"] = "either"
    peak_rf_power_w: float = Field(gt=0, le=1e7, strict=True)
    pulse_duty_fraction: float = Field(ge=1e-6, le=1, strict=True)
    amplifier_efficiency: float = Field(ge=1e-4, le=1, strict=True)
    rolling_window_sec: float = Field(gt=0, le=259200, strict=True)
    max_on_sec: float = Field(gt=0, le=259200, strict=True)

    @model_validator(mode="after")
    def bounds(self):
        if not self.model_id.strip() or not self.source.strip():
            raise ValueError("imaging model ID/source must be nonblank")
        if (
            self.min_incidence_deg >= self.max_incidence_deg
            or self.max_on_sec > self.rolling_window_sec
        ):
            raise ValueError(
                "SAR requires ordered incidence and on-time <= rolling window"
            )
        return self

    @property
    def mean_rf_w(self):
        return self.peak_rf_power_w * self.pulse_duty_fraction

    @property
    def dc_power_w(self):
        return self.mean_rf_w / self.amplifier_efficiency


class QualityRequirement(StrictModel):
    kind: Literal["optical", "sar"]
    min_sun_elevation_deg: float | None = Field(default=None, ge=0, le=90, strict=True)
    max_gsd_m: float | None = Field(default=None, gt=0, strict=True)
    max_ground_range_resolution_m: float | None = Field(default=None, gt=0, strict=True)
    max_azimuth_resolution_m: float | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def applicable(self):
        optical = (self.min_sun_elevation_deg, self.max_gsd_m)
        sar = (self.max_ground_range_resolution_m, self.max_azimuth_resolution_m)
        required, forbidden = (
            (optical, sar) if self.kind == "optical" else (sar, optical)
        )
        if any(value is None for value in required) or any(
            value is not None for value in forbidden
        ):
            raise ValueError(
                "quality requires explicit thresholds applicable to its sensor kind"
            )
        return self


class CloudMaskCell(StrictModel):
    id: str = Field(min_length=1)
    aoi_id: str = Field(min_length=1)
    aoi_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bounds_m: tuple[FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar]
    state: Literal["clear", "cloud", "unknown"]
    valid_from_sec: float = Field(ge=0, strict=True)
    valid_until_sec: float = Field(gt=0, strict=True)
    available_from_sec: float = Field(default=0.0, ge=0, strict=True)
    source_status: Literal["synthetic", "observed", "forecast"]
    source: str = Field(min_length=1)
    allow_forecast: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def valid(self):
        x0, y0, x1, y1 = self.bounds_m
        if x0 >= x1 or y0 >= y1 or max(map(abs, self.bounds_m)) > 2000000:
            raise ValueError(
                "cloud mask requires ordered bounds in declared AOI LAEA metres"
            )
        if self.valid_from_sec >= self.valid_until_sec:
            raise ValueError("cloud mask requires ordered validity")
        if (
            self.source_status == "observed"
            and self.available_from_sec < self.valid_from_sec
        ):
            raise ValueError(
                "observed cloud mask cannot be available before validity begins"
            )
        if self.source_status == "forecast" and not self.allow_forecast:
            raise ValueError("forecast cloud masks require explicit allow_forecast")
        if not all(s.strip() for s in (self.id, self.aoi_id, self.source)):
            raise ValueError("cloud mask identifiers/source must be nonblank")
        return self


class QualitySampling(StrictModel):
    cell_size_m: float = Field(default=1000.0, ge=10, le=10000, strict=True)
    max_cells_per_sample: int = Field(default=1024, ge=1, le=4096, strict=True)
    max_points_per_region: int = Field(default=256, ge=5, le=4096, strict=True)
    max_evidence_records: int = Field(default=20000, ge=1, le=100000, strict=True)


@lru_cache(maxsize=32)
def _mask_tree(serialized):
    _, shapely = _libraries()
    rows = json.loads(serialized)
    polygons = [shapely.box(*row["bounds_m"]) for row in rows]
    return polygons, shapely.STRtree(polygons)


def ground_metrics(provider, time_sec, lon, lat, *, geometry=None):
    """Sampled ellipsoidal local normal, geometric Sun, and orbital-plane side."""
    if geometry is None:
        rotation = provider.context.matrix(provider.epoch.shifted(time_sec))
        state = np.asarray(provider.state(time_sec))
        sun_gcrs = _ephemerides(provider.epoch.shifted(time_sec))[0]
    else:
        rotation, state, sun_gcrs = geometry
    point = geodetic_to_ecef(lat, lon, 0.0)
    normal = np.array(
        [
            math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
            math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
            math.sin(math.radians(lat)),
        ]
    )
    to_sat = rotation @ state[:3] - point
    distance = float(np.linalg.norm(to_sat))
    incidence = math.degrees(
        math.acos(float(np.clip(normal @ to_sat / distance, -1, 1)))
    )
    sun = rotation @ sun_gcrs - point
    elevation = math.degrees(
        math.asin(float(np.clip(normal @ sun / np.linalg.norm(sun), -1, 1)))
    )
    side = float(
        (point - rotation @ state[:3]) @ (rotation @ np.cross(state[3:], state[:3]))
    )
    return {
        "incidence_deg": incidence,
        "sun_elevation_deg": elevation,
        "slant_range_m": distance,
        "look_side": "right" if side > 0 else "left" if side < 0 else "nadir",
    }


def resolution_metrics(model, metrics):
    angle = math.radians(metrics["incidence_deg"])
    if model.kind == "optical":
        return {
            "gsd_m": metrics["slant_range_m"]
            * model.instantaneous_fov_rad
            / max(math.cos(angle), 1e-12)
        }
    return {
        "ground_range_resolution_m": 299792458.0
        / (2 * model.bandwidth_hz * max(math.sin(angle), 1e-12)),
        "azimuth_resolution_m": model.antenna_length_m / 2,
    }


class PayloadQuality:
    def __init__(self, payloads, masks, sampling, aois, horizon):
        self.models = {
            actor: row.imaging_model
            for actor, row in payloads.items()
            if row.imaging_model is not None
        }
        self.masks = tuple(masks)
        self.sampling = sampling
        self.aois = dict(aois)
        self.sar_intervals = {
            actor: [] for actor, model in self.models.items() if model.kind == "sar"
        }
        if len({row.id for row in masks}) != len(masks):
            raise ValueError("duplicate cloud mask ID")
        for row in masks:
            if row.aoi_id not in aois or row.aoi_sha256 != digest(
                aois[row.aoi_id].model_dump(mode="json")
            ):
                raise ValueError("cloud mask AOI/hash mismatch")
            if row.valid_until_sec > horizon or row.available_from_sec > horizon:
                raise ValueError(
                    "cloud mask validity/availability exceeds mission horizon"
                )
        self.serialized_masks = json.dumps(
            [row.model_dump(mode="json") for row in masks], sort_keys=True
        )
        polygons, tree = _mask_tree(self.serialized_masks)
        for index, row in enumerate(masks):
            for other in sorted(map(int, tree.query(polygons[index]))):
                peer = masks[other]
                if other <= index or row.aoi_id != peer.aoi_id:
                    continue
                if (
                    max(row.valid_from_sec, peer.valid_from_sec)
                    < min(row.valid_until_sec, peer.valid_until_sec)
                    and polygons[index].intersection(polygons[other]).area > 0
                ):
                    raise ValueError(
                        "overlapping cloud masks have ambiguous space/time truth"
                    )

    def power(self, actor, base):
        model = self.models.get(actor)
        return base + (
            model.dc_power_w if model is not None and model.kind == "sar" else 0.0
        )

    def admission_power(self, actor, base):
        model = self.models.get(actor)
        return base + (
            model.peak_rf_power_w / model.amplifier_efficiency
            if model is not None and model.kind == "sar"
            else 0.0
        )

    def heat_adjustment_w(self, actor, fraction):
        model = self.models.get(actor)
        # Replace generic heat fraction for amplifier DC by DC minus radiated RF.
        return (
            (1 - fraction) * model.dc_power_w - model.mean_rf_w
            if model is not None and model.kind == "sar"
            else 0.0
        )

    def boundary(self, now, right):
        boundaries = [
            t
            for row in self.masks
            for t in (row.available_from_sec, row.valid_from_sec, row.valid_until_sec)
            if now < t < right
        ]
        return min(boundaries, default=right)

    def duty_boundary(self, actor, now, right):
        model = self.models.get(actor)
        if model is None or model.kind != "sar":
            return right
        history = self.sar_intervals[actor]

        def used(end):
            left = end - model.rolling_window_sec
            return (
                end
                - max(now, left)
                + math.fsum(
                    max(0.0, min(end, row["end_sec"]) - max(left, row["start_sec"]))
                    for row in history
                )
            )

        if used(right) <= model.max_on_sec + 1e-12:
            return right
        left, end = now, right
        for _ in range(55):
            mid = (left + end) / 2
            if used(mid) <= model.max_on_sec:
                left = mid
            else:
                end = mid
        return left

    def sar_reason(self, actor, provider, target, times):
        model = self.models.get(actor)
        if model is None or model.kind != "sar":
            return None
        for time in times:
            metrics = ground_metrics(provider, time, target.lon_deg, target.lat_deg)
            if (
                not model.min_incidence_deg
                <= metrics["incidence_deg"]
                <= model.max_incidence_deg
            ):
                return "sar_incidence"
            if model.look_side != "either" and metrics["look_side"] != model.look_side:
                return "sar_look_side"
        return None

    def record_sar(self, actor, start, end):
        model = self.models.get(actor)
        if model is None or model.kind != "sar":
            return
        interval = {
            "start_sec": start,
            "end_sec": end,
            "dc_input_j": model.dc_power_w * (end - start),
            "radiated_rf_j": model.mean_rf_w * (end - start),
            "amplifier_heat_j": (model.dc_power_w - model.mean_rf_w) * (end - start),
        }
        rows = self.sar_intervals[actor]
        if rows and abs(rows[-1]["end_sec"] - start) <= 1e-10:
            rows[-1]["end_sec"] = end
            for key in ("dc_input_j", "radiated_rf_j", "amplifier_heat_j"):
                rows[-1][key] += interval[key]
        else:
            rows.append(interval)

    def evidence(self, actor, aoi_id, samples, provider, raw_id, remaining):
        _, shapely = _libraries()
        model = self.models.get(actor)
        if model is None:
            return []
        aoi = self.aois[aoi_id]
        _, transform = domain(aoi)
        polygons, tree = _mask_tree(self.serialized_masks)
        rows = []
        size = self.sampling.cell_size_m
        for sample in samples:
            time = sample["time_sec"]
            geometry = (
                provider.context.matrix(provider.epoch.shifted(time)),
                np.asarray(provider.state(time)),
                _ephemerides(provider.epoch.shifted(time))[0],
            )
            footprint = _project_footprint(aoi, sample["shell"])
            if footprint.is_empty:
                continue
            x0, y0, x1, y1 = footprint.bounds
            ix0, iy0, ix1, iy1 = (
                math.floor(x0 / size),
                math.floor(y0 / size),
                math.floor(x1 / size),
                math.floor(y1 / size),
            )
            if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > self.sampling.max_cells_per_sample:
                raise ValueError("quality cell budget exceeded")
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    cell = footprint.intersection(
                        shapely.box(
                            ix * size, iy * size, (ix + 1) * size, (iy + 1) * size
                        )
                    )
                    if cell.is_empty or cell.area <= 0:
                        continue
                    regions = []
                    if model.kind == "optical":
                        for index in sorted(map(int, tree.query(cell))):
                            mask = self.masks[index]
                            if (
                                mask.aoi_id == aoi_id
                                and mask.available_from_sec <= sample["time_sec"]
                                and mask.valid_from_sec
                                <= sample["time_sec"]
                                < mask.valid_until_sec
                            ):
                                regions.append(
                                    (
                                        cell.intersection(polygons[index]),
                                        mask.state,
                                        mask.id,
                                    )
                                )
                        known = shapely.union_all([region for region, _, _ in regions])
                        regions.append((cell.difference(known), "unknown", None))
                    else:
                        regions.append((cell, "not_applicable", None))
                    for region, cloud, mask_id in regions:
                        if region.is_empty or region.area <= 0:
                            continue
                        if len(rows) >= remaining:
                            raise ValueError("quality evidence budget exceeded")
                        coordinates = shapely.get_coordinates(region)
                        if len(coordinates) + 1 > self.sampling.max_points_per_region:
                            raise ValueError("quality spatial-point budget exceeded")
                        center = region.representative_point()
                        coordinates = np.vstack((coordinates, [center.x, center.y]))
                        lon, lat = transform.transform(
                            coordinates[:, 0],
                            coordinates[:, 1],
                            direction="INVERSE",
                            errcheck=True,
                        )
                        metrics = [
                            ground_metrics(
                                provider, sample["time_sec"], x, y, geometry=geometry
                            )
                            for x, y in zip(lon, lat)
                        ]
                        resolved = [
                            resolution_metrics(model, point) for point in metrics
                        ]
                        applicable = all(
                            point["incidence_deg"] <= model.max_incidence_deg
                            for point in metrics
                        )
                        if model.kind == "optical":
                            applicable = applicable and all(
                                point["sun_elevation_deg"]
                                >= model.min_sun_elevation_deg
                                for point in metrics
                            )
                        if model.kind == "sar":
                            applicable = applicable and all(
                                point["incidence_deg"] >= model.min_incidence_deg
                                and (
                                    model.look_side == "either"
                                    or point["look_side"] == model.look_side
                                )
                                for point in metrics
                            )
                        evidence = {
                            "aoi_id": aoi_id,
                            "aoi_sha256": digest(aoi.model_dump(mode="json")),
                            "raw_id": raw_id,
                            "actor": actor,
                            "time_sec": sample["time_sec"],
                            "kind": model.kind,
                            "model": model.model_dump(mode="json"),
                            "geometry": json.loads(shapely.to_geojson(region)),
                            "cloud_state": cloud,
                            "mask_id": mask_id,
                            "applicable": applicable,
                            "min_sampled_sun_elevation_deg": min(
                                p["sun_elevation_deg"] for p in metrics
                            ),
                            "min_sampled_incidence_deg": min(
                                p["incidence_deg"] for p in metrics
                            ),
                            "max_sampled_incidence_deg": max(
                                p["incidence_deg"] for p in metrics
                            ),
                            **{
                                key: max(p[key] for p in resolved)
                                for key in resolved[0]
                            },
                        }
                        rows.append({"id": digest(evidence), **evidence})
        return rows

    def observation(self, actor, now, visible_areas):
        model = self.models.get(actor)
        return {
            "model": None if model is None else model.model_dump(mode="json"),
            "sampling": self.sampling.model_dump(mode="json"),
            "cloud_masks": [
                row.model_dump(mode="json")
                for row in self.masks
                if row.aoi_id in visible_areas and row.available_from_sec <= now
            ],
            "sar_intervals": copy.deepcopy(self.sar_intervals.get(actor, [])),
        }


class QualityEvidenceBudgetError(ValueError):
    pass


def merge_evidence(products, limit):
    rows = {}
    for product in products:
        for row in product.get("quality_evidence", []):
            if row["id"] in rows and rows[row["id"]] != row:
                raise ValueError("incompatible quality evidence lineage")
            rows[row["id"]] = row
            if len(rows) > limit:
                raise QualityEvidenceBudgetError("quality evidence budget exceeded")
    return [copy.deepcopy(rows[key]) for key in sorted(rows)]


def usable_coverage(requirement, coverage, evidence):
    if coverage is None:
        return {
            "usable_area_m2": 0.0,
            "usable_fraction": 0.0,
            "evidence_records": 0,
            "criterion": requirement.model_dump(mode="json"),
        }
    _, shapely = _libraries()
    eligible = []
    for row in evidence:
        if (
            row["aoi_id"] != coverage["aoi_id"]
            or row["aoi_sha256"] != coverage["aoi_sha256"]
            or row["kind"] != requirement.kind
            or not row["applicable"]
        ):
            continue
        good = evidence_qualifies(requirement, row)
        if good:
            eligible.append(_geometry(row["geometry"]))
    union = shapely.union_all(eligible).intersection(_geometry(coverage["geometry"]))
    return {
        "usable_area_m2": float(union.area),
        "usable_fraction": min(1.0, float(union.area / coverage["aoi_area_m2"])),
        "evidence_records": len(evidence),
        "criterion": requirement.model_dump(mode="json"),
    }


def evidence_qualifies(requirement, row):
    if row["kind"] != requirement.kind or not row["applicable"]:
        return False
    if requirement.kind == "optical":
        return (row["cloud_state"] == "clear" and row["gsd_m"] <= requirement.max_gsd_m
                and row["min_sampled_sun_elevation_deg"] >= requirement.min_sun_elevation_deg)
    return (row["ground_range_resolution_m"] <= requirement.max_ground_range_resolution_m
            and row["azimuth_resolution_m"] <= requirement.max_azimuth_resolution_m)
