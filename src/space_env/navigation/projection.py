"""Allowlisted sensor-task view; no geometric entities or truth-derived masks."""

import copy

from space_env.simulation.snapshot import digest

RESOURCE_FIELDS = {
    "soc",
    "stored_bits",
    "reserved_bits",
    "temperature_k",
    "capacity_bits",
    "battery_wh",
    "tx_seconds",
    "tx_ready",
    "safe",
    "status_code",
    "measured_at",
    "age_sec",
}
PRODUCT_FIELDS = {
    "size_bits",
    "remaining_bits",
    "ready_sec",
    "deadline_sec",
    "priority",
    "remaining_work_sec",
    "created_sec",
    "status_code",
    "measured_at",
    "age_sec",
}
NAVIGATION_FEATURES = tuple(f"p_{i}_{j}" for i in range(6) for j in range(6)) + (
    "filter_epoch_sec",
    "prediction_age_sec",
    "sample_count",
    "last_acquired_sec",
    "last_received_sec",
    "last_dropped",
    "last_accepted",
    "last_nis",
    "last_z_0",
    "last_z_1",
    "last_z_2",
)


def project_navigation(raw, spec, role, navigation):
    actors = set(navigation) if role == "@central" else set(spec.roles[role].actors)
    entities = []
    own_indices = {
        e["index"]
        for e in raw["entities"]
        if e["kind"] == "actor" and e["id"] in actors
    }
    own_products = {
        r["destination"]
        for r in raw["relations"]
        if r["kind"] == "owns" and r["source"] in own_indices
    }
    for entity in raw["entities"]:
        if entity["kind"] == "actor" and entity["id"] in actors:
            row = copy.deepcopy(entity)
            row["values"] = {
                k: v for k, v in row["values"].items() if k in RESOURCE_FIELDS
            }
            row["values"].update(
                zip(
                    ["r_x_m", "r_y_m", "r_z_m", "v_x_m_s", "v_y_m_s", "v_z_m_s"],
                    navigation[entity["id"]]["estimate_si"],
                )
            )
            entities.append(row)
        elif entity["kind"] == "product" and entity["index"] in own_products:
            # Raw per-role projector already scopes inventory ownership; no geometry.
            row = copy.deepcopy(entity)
            row["values"] = {
                k: v for k, v in row["values"].items() if k in PRODUCT_FIELDS
            }
            entities.append(row)
    old_to_new = {row["index"]: i for i, row in enumerate(entities)}
    for i, row in enumerate(entities):
        row["index"] = i
    relations = [
        {
            "kind": r["kind"],
            "source": old_to_new[r["source"]],
            "destination": old_to_new[r["destination"]],
        }
        for r in raw["relations"]
        if r["kind"] == "owns"
        and r["source"] in old_to_new
        and r["destination"] in old_to_new
    ]
    catalogue = [
        {"id": "wait", "kind": "wait", "enabled": True, "reason": "declarative"}
    ]
    for action in spec.actions:
        enabled = role == "@central" or action.role == role
        row = {
            "id": action.id,
            "kind": action.kind,
            "enabled": enabled,
            "reason": "declarative" if enabled else "role_scope",
        }
        if enabled:
            row["actors"] = sorted(
                {c.actor_id for c in action.commands}
                | ({action.actor} if action.actor else set())
            )
            row["command_kinds"] = [c.kind for c in action.commands]
            row["commands"] = [
                {
                    k: v
                    for k, v in c.model_dump(mode="json").items()
                    if k not in {"command_id", "forecast"} and v is not None and v != ""
                }
                for c in action.commands
            ]
        catalogue.append(row)
    result = {
        k: copy.deepcopy(raw[k])
        for k in ["schema", "task_id", "role", "revision", "time_sec", "feature_names"]
    }
    result.update(
        state_source="native-sensor-estimate/1",
        belief={
            "kind": "declared-force-ekf",
            "estimated": True,
            "other_telemetry": "ideal operational resource counters; no exact orbit/geometry",
        },
        entities=entities,
        relations=relations,
        catalogue=catalogue,
        action_mask=[int(r["enabled"]) for r in catalogue],
        mask_scope="declarative role/action mask only; native admission may reject",
        navigation={a: copy.deepcopy(navigation[a]) for a in sorted(actors)},
    )
    result["binding"] = digest(result)
    return result


def numerical_navigation(view):
    """Full estimate covariance, packet values and presence masks for Gym/PZ."""
    values = {
        f"p_{i}_{j}": view["covariance_si"][i][j] for i in range(6) for j in range(6)
    }
    values.update(
        filter_epoch_sec=view["filter_epoch_sec"],
        prediction_age_sec=view["estimate_prediction_age_sec"],
        sample_count=view["received_sample_count"],
    )
    if view["received_samples"]:
        row = view["received_samples"][-1]
        values.update(
            last_acquired_sec=row["acquired_sec"],
            last_received_sec=row["received_sec"],
            last_dropped=float(row["dropped"]),
        )
        if "filter" in row:
            values.update(
                last_accepted=float(row["filter"]["accepted"]),
                last_nis=row["filter"]["nis"],
            )
            values.update({f"last_z_{i}": v for i, v in enumerate(row["value"])})
    return [values.get(k, 0.0) for k in NAVIGATION_FEATURES], [
        int(k in values) for k in NAVIGATION_FEATURES
    ]
