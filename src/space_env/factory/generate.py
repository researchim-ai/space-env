"""Deterministic bounded generation; every draw remains in the catalogue."""

from __future__ import annotations

import copy

import numpy as np

from space_env.eval.statistics import assign_split
from space_env.factory.schema import FactorySpec
from space_env.factory.semantic import mission_fingerprint, split_group
from space_env.planning.benchmark import BenchmarkSpec
from space_env.simulation.snapshot import canonical_json, digest
from space_env.tasks import TaskSession


def _seed(value):
    return int(digest(value)[:16], 16) % (2**63)


def _apply(scenario, task, parameter, target, value):
    d = scenario["definition"]
    ops = d.get("operations", d)
    life = d.get("lifecycle", {})
    changed = 0

    def select(id):
        return target == "*" or target == id

    def field(row, key, factor=True, integer=False):
        nonlocal changed
        result = row[key] * value if factor else value
        row[key] = round(result) if integer else float(result)
        changed += 1

    if parameter in {"memory_scale", "battery_scale", "rate_scale"}:
        key, default = {
            "memory_scale": ("memory_bits", None),
            "battery_scale": ("battery_wh", None),
            "rate_scale": ("rate_bps", 1e6),
        }[parameter]
        for actor, row in ops["vehicles"].items():
            if select(actor):
                if key not in row and default is not None:
                    row[key] = default
                field(row, key)
    elif parameter == "station_terminals":
        for row in ops["stations"]:
            if select(row["name"]):
                row["terminals"] = int(value)
                changed += 1
    elif parameter == "outage_duration_sec":
        for row in ops.get("outages", []):
            if select(row["station"]):
                row["end_sec"] = float(row["start_sec"] + value)
                changed += 1
    elif parameter == "relay_ttl_sec":
        for action in task["actions"]:
            for c in action.get("commands", []):
                if c["kind"] == "relay" and select(c.get("product_id")):
                    c.setdefault("relay", {})["ttl_sec"] = float(value)
                    changed += 1
    else:
        for p in ops.get("products", []):
            if not select(p["id"]):
                continue
            if parameter == "product_bits_scale":
                field(p, "size_bits")
                if p.get("workspace_bits"):
                    field(p, "workspace_bits")
            elif parameter == "deadline_scale":
                p["deadline_sec"] = float(
                    p.get("ready_sec", 0.0)
                    + (p["deadline_sec"] - p.get("ready_sec", 0.0)) * value
                )
                changed += 1
            elif parameter == "processing_scale":
                p.setdefault("processing_sec", 0.0)
                field(p, "processing_sec")
        for r in life.get("requests", []):
            if not (select(r["id"]) or select(r["raw_product_id"])):
                continue
            if parameter == "product_bits_scale":
                field(r, "raw_size_bits", integer=True)
            elif parameter == "acquisition_scale":
                field(r, "acquisition_sec")
            elif parameter == "deadline_scale":
                release = r.get("release_sec", 0.0)
                r["deadline_sec"] = float(
                    release + (r["deadline_sec"] - release) * value
                )
                changed += 1
        for job in life.get("jobs", []):
            if parameter == "processing_scale" and select(job["id"]):
                field(job, "work_sec")
            for p in job["outputs"]:
                if parameter == "product_bits_scale" and select(p["id"]):
                    field(p, "size_bits", integer=True)
    if not changed:
        raise ValueError(
            "distribution has no supported target: " + parameter + "/" + target
        )


def _family_contract(template, scenario, task):
    d = scenario["definition"]
    ops = d.get("operations", d)
    life = d.get("lifecycle", {})
    kinds = {c["kind"] for a in task["actions"] for c in a.get("commands", [])}
    if template.kind == "eo_processing" and not (
        life.get("requests") and life.get("jobs") and {"imaging", "transform"} <= kinds
    ):
        raise ValueError("EO/CPU family requires requests, jobs and declared controls")
    if template.kind == "relay" and not (ops.get("dtn_network") and "relay" in kinds):
        raise ValueError("relay family requires native DTN and a relay template")
    if template.kind == "recovery" and not any(
        a.get("event", {}).get("type")
        in {"spacecraft_recover", "spacecraft_fault_cleared", "station_recovered"}
        for a in task["actions"]
        if a.get("event")
    ):
        raise ValueError("recovery family requires a declared recovery action")
    if template.kind == "shared_terminal":
        users = {}
        for a in task["actions"]:
            for c in a.get("commands", []):
                if c["kind"] == "downlink":
                    users.setdefault(c["destination_id"], set()).add(c["actor_id"])
        if not any(len(actors) >= 2 for actors in users.values()):
            raise ValueError("shared-terminal family requires competing station users")
    if template.kind == "ground" and "downlink" not in kinds:
        raise ValueError("ground family requires a downlink control")


def compile_factory(value):
    spec = FactorySpec.model_validate(copy.deepcopy(value))
    cases, splits, families = [], {}, {}
    total_bytes = 0
    for template in sorted(spec.templates, key=lambda t: t.id):
        group = split_group(template.scenario)
        lineage = "mission-" + group
        split = template.split or assign_split(lineage, "")
        if group in splits and splits[group] != split:
            raise ValueError(
                "same mission geometry cannot cross splits, even after relabelling"
            )
        splits[group] = split
        families.setdefault(template.family, set()).add(split)
        for i in range(template.samples):
            scenario, task = (
                copy.deepcopy(template.scenario),
                copy.deepcopy(template.task),
            )
            scenario.update(id=f"{template.id}-{i:03d}", lineage=lineage)
            task.update(
                id="factory-task",
                seed_policy=spec.seed_policy,
                runtime_audit=spec.runtime_audit,
            )
            parameters, error = [], None
            episode_seed = _seed([spec.seed, group, i, "episode"])
            try:
                for p in sorted(
                    template.distributions, key=lambda p: (p.parameter, p.target)
                ):
                    rng = np.random.default_rng(
                        _seed([spec.seed, group, i, p.parameter, p.target])
                    )
                    draw = (
                        float(p.values[int(rng.integers(len(p.values)))])
                        if p.mode == "choice"
                        else float(rng.uniform(p.low, p.high))
                    )
                    parameters.append(
                        {
                            "parameter": p.parameter,
                            "target": p.target,
                            "value": draw,
                            "unit": p.unit,
                            "source": p.source,
                            "parameter_status": "synthetic",
                        }
                    )
                    _apply(scenario, task, p.parameter, p.target, draw)
                _family_contract(template, scenario, task)
                TaskSession(
                    scenario, task, seed=episode_seed
                )  # Configuration validation; no future rollout.
            except (ValueError, KeyError, TypeError, ImportError) as exc:
                error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            semantic = mission_fingerprint(scenario, task)
            case = {
                "id": scenario["id"],
                "family": template.family,
                "kind": template.kind,
                "source": template.source,
                "parameter_status": "synthetic",
                "split": split,
                "group_sha256": group,
                "semantic_sha256": semantic,
                "episode_seed": episode_seed,
                "parameters": parameters,
                "configuration_error": error,
                "scenario": scenario,
                "task": task,
            }
            total_bytes += len(canonical_json(case).encode())
            if total_bytes > spec.max_generated_bytes:
                raise ValueError("generated catalogue byte budget exceeded")
            cases.append(case)
    if any("ood" in s and len(s) > 1 for s in families.values()):
        raise ValueError("declared OOD family cannot occur in training/holdout")
    # Workload identity collisions are grouped even if template labels differ.
    semantic_splits = {}
    for case in cases:
        h = case["semantic_sha256"]
        if h in semantic_splits and semantic_splits[h] != case["split"]:
            raise ValueError("semantic duplicate crosses splits")
        semantic_splits[h] = case["split"]
    body = {
        "schema": "space-env.factory-catalogue/1",
        "spec": spec.model_dump(mode="json", by_alias=True),
        "cases": cases,
        "coverage": {
            "draws": len(cases),
            "unique_missions": len(semantic_splits),
            "geometry_groups": len(splits),
            "families": {k: sorted(v) for k, v in sorted(families.items())},
            "configuration_errors": sum(
                c["configuration_error"] is not None for c in cases
            ),
        },
        "generated_bytes": total_bytes,
    }
    return {**body, "sha256": digest(body)}


def benchmark_manifest(catalogue):
    spec = FactorySpec.model_validate(catalogue["spec"])
    value = {
        "id": spec.id,
        "cases": [
            {
                "id": c["id"],
                "family": c["family"],
                "scenario": c["scenario"],
                "task": c["task"],
                "split": c["split"],
                "episode_seeds": [c["episode_seed"]],
            }
            for c in catalogue["cases"]
        ],
        "policies": [p.model_dump(mode="json") for p in spec.policies],
        "workers": spec.workers,
        "teacher_gates": None
        if spec.teacher_gates is None
        else spec.teacher_gates.model_dump(mode="json"),
    }
    return BenchmarkSpec.model_validate(value).model_dump(mode="json", by_alias=True)
