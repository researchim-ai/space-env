"""Counter-keyed assumed sensitivity scenarios; no rejection resampling or execution."""

from __future__ import annotations

import copy
import math
from statistics import NormalDist

from space_env.calibration.schema import EnsembleSpec
from space_env.reference.dynamic import DynamicMissionSpec
from space_env.reference.operations import VehicleResources
from space_env.runtime.schema import ScenarioSpec
from space_env.simulation.snapshot import canonical_json, digest


def _uniform(spec, index, parameter):
    latent = (
        ("group", parameter.latent_group)
        if parameter.latent_group is not None
        else ("parameter", parameter.id)
    )
    bits = (
        int(
            digest(
                {
                    "algorithm": "space-env.counter-uniform/1",
                    "id": spec.id,
                    "seed": spec.seed,
                    "member": index,
                    "latent": latent,
                }
            )[:16],
            16,
        )
        >> 11
    )
    return (bits + 1) / (2**53 + 2)


def _draw(parameter, u):
    if parameter.mode == "uniform":
        value = parameter.low + (parameter.high - parameter.low) * u
    else:
        normal = NormalDist(parameter.mean, parameter.sigma)
        lo, hi = normal.cdf(parameter.low), normal.cdf(parameter.high)
        if hi - lo < 1e-12:
            raise ValueError(
                "truncated normal interval has insufficient binary64 CDF mass"
            )
        probability = lo + (hi - lo) * u
        probability = min(
            math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), probability)
        )
        value = normal.inv_cdf(probability)
    if not math.isfinite(value) or not parameter.low <= value <= parameter.high:
        raise ValueError(
            "draw outside declared finite support; not clipped or resampled"
        )
    return float(value)


def _patch(scenario, parameter, value):
    d = scenario["definition"]
    actor = parameter.actor
    vehicles = d["operations"]["vehicles"]
    if actor not in vehicles or actor not in d["spacecraft_bus"]:
        raise ValueError("uncertainty target must be an existing vehicle/bus actor")
    vehicle, bus = vehicles[actor], d["spacecraft_bus"][actor]
    target = parameter.target
    # A synthetic perturbation cannot inherit the base model's calibration label.
    bus["parameter_status"] = "synthetic"
    marker = "Assumed sensitivity; base calibration not transferred: " + parameter.id
    bus["sources"] = [*bus["sources"], marker]
    if target == "battery_capacity_scale":
        vehicle["battery_wh"] *= value
    elif target == "solar_efficiency_scale":
        vehicle["solar_efficiency"] = vehicle.get("solar_efficiency", 0.3) * value
    elif target == "thermal_capacity_scale":
        if bus.get("thermal_network") is not None:
            raise ValueError(
                "scalar thermal-capacity binding cannot silently replace an active node network"
            )
        bus["thermal"]["capacity_j_per_k"] *= value
    else:
        if bus["attitude"].get("rigid") is None:
            raise ValueError(
                "wheel/inertia sensitivity requires an existing rigid controller"
            )
        rigid = bus["attitude"]["rigid"]["model"]
        rigid["source"] += "; " + marker
        key = {
            "wheel_torque_scale": "max_torque_nm",
            "wheel_momentum_scale": "max_momentum_nms",
            "body_inertia_scale": "inertia_kg_m2",
        }[target]
        if target == "body_inertia_scale":
            rigid[key] = [[float(x * value) for x in row] for row in rigid[key]]
        else:
            rigid[key] = [float(x * value) for x in rigid[key]]


def _structural(scenario):
    envelope = ScenarioSpec.model_validate(scenario)
    if envelope.backend != "reference-spacecraft":
        raise ValueError(
            "sensitivity compiler supports explicit reference-spacecraft scenarios only"
        )
    d = DynamicMissionSpec.model_validate(envelope.definition)
    if d.spacecraft_bus is None or d.lifecycle is None:
        raise ValueError(
            "spacecraft ensemble requires declared lifecycle and bus models"
        )
    actors = set(envelope.actor_order)
    if (
        actors != set(d.orbits)
        or actors != set(d.operations["vehicles"])
        or actors != set(d.spacecraft_bus)
    ):
        raise ValueError("ensemble actor/orbit/vehicle/bus sets must match")
    for vehicle in d.operations["vehicles"].values():
        VehicleResources.model_validate(vehicle)
    return envelope


def sample_parameters(value):
    spec = EnsembleSpec.model_validate(value)
    rows = []
    for index in range(spec.samples):
        draws = {}
        errors = []
        for parameter in sorted(spec.parameters, key=lambda p: p.id):
            try:
                draws[parameter.id] = _draw(parameter, _uniform(spec, index, parameter))
            except ValueError as exc:
                errors.append({"parameter": parameter.id, "message": str(exc)})
        rows.append({"index": index, "draws": draws, "errors": errors})
    result = {
        "schema": "space-env.parameter-draws/1",
        "spec": spec.model_dump(mode="json"),
        "members": rows,
        "attempted": spec.samples,
        "status": "assumed_sensitivity_only",
        "algorithm": "space-env.counter-uniform/1",
        "limitations": "independent latents except explicit common-rank groups; no empirical covariance or calibrated tolerances",
    }
    result["sha256"] = digest(result)
    return result


def compile_ensemble(scenario, value):
    spec = EnsembleSpec.model_validate(value)
    if len(canonical_json(scenario).encode()) > 1_000_000:
        raise ValueError("base scenario exceeds 1MB compilation budget")
    base = _structural(scenario).model_dump(mode="json")
    sampled = sample_parameters(spec)
    rows = []
    used = 0
    parameters = sorted(spec.parameters, key=lambda p: p.id)
    for member in sampled["members"]:
        index, draws = member["index"], member["draws"]
        candidate = copy.deepcopy(base)
        candidate["id"] = f"{base['id']}:sensitivity:{spec.id}:{index}"
        try:
            if member["errors"]:
                raise ValueError("distribution errors: " + str(member["errors"]))
            for parameter in parameters:
                _patch(candidate, parameter, draws[parameter.id])
            candidate = _structural(candidate).model_dump(mode="json")
            row = {
                "index": index,
                "draws": draws,
                "status": "structural_candidate",
                "scenario": candidate,
                "scenario_sha256": digest(candidate),
            }
        except (ValueError, KeyError, TypeError) as exc:
            row = {
                "index": index,
                "draws": draws,
                "status": "rejected",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        used += len(canonical_json(row).encode())
        if used > 16_000_000:
            raise ValueError("ensemble output exceeds 16MB compilation budget")
        rows.append(row)
    result = {
        "schema": "space-env.compiled-parameter-ensemble/1",
        "id": spec.id,
        "seed": spec.seed,
        "spec_sha256": digest(spec.model_dump(mode="json")),
        "base_sha256": digest(base),
        "members": rows,
        "attempted": spec.samples,
        "structural_candidates": sum(
            r["status"] == "structural_candidate" for r in rows
        ),
        "rejected": sum(r["status"] == "rejected" for r in rows),
        "native_execution_validated": False,
        "source_spec": spec.model_dump(mode="json"),
        "status": "assumed_sensitivity_only",
        "industrial_readiness": False,
        "dependency_model": "independent counter latents except explicitly shared common-rank groups; not a fitted covariance",
        "rng_policy": "environment seed unchanged for paired common-random-number runs; no draws inserted into agent metadata",
        "limits": "structural contracts only; no native admission/replay/mission feasibility or empirical uncertainty guarantee",
    }
    result["sha256"] = digest(result)
    return result
