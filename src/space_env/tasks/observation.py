"""Causal projections from public local telemetry; no backend admission oracle."""

from __future__ import annotations

import math

from space_env.simulation.snapshot import digest

FEATURES = (
    "size_bits",
    "remaining_bits",
    "soc",
    "stored_bits",
    "reserved_bits",
    "temperature_k",
    "safe",
    "ready_sec",
    "deadline_sec",
    "priority",
    "progress",
    "remaining_work_sec",
    "created_sec",
    "measured_at",
    "age_sec",
    "status_code",
    "range_m",
    "rate_bps",
    "cn0_db_hz",
    "bandwidth_hz",
    "fade_db",
    "outage",
    "terminals",
    "capacity_bits",
    "battery_wh",
    "blocking_reason",
    "tx_seconds",
    "tx_ready",
    "r_x_m",
    "r_y_m",
    "r_z_m",
    "v_x_m_s",
    "v_y_m_s",
    "v_z_m_s",
    "mass_kg",
    "propellant_kg",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
)
KINDS = {
    "actor": 1,
    "station": 2,
    "request": 3,
    "job": 4,
    "product": 5,
    "transfer": 6,
    "command": 7,
    "link": 8,
    "weather": 9,
}
STATES = {
    s: i + 1
    for i, s in enumerate(
        [
            "pending",
            "released",
            "accepted",
            "scheduled",
            "running",
            "paused",
            "completed",
            "failed",
            "cancelled",
            "expired",
            "resident",
            "delivered",
            "queued",
            "ready",
            "inflight",
            "await_ack",
            "safe",
            "acquiring",
            "locked",
            "capacity_blocked",
            "link_unavailable",
        ]
    )
}
RELATIONS = {
    "owns": 1,
    "contact": 2,
    "requires": 3,
    "produces": 4,
    "source": 5,
    "destination": 6,
    "controls": 7,
    "fulfills": 8,
}


class _ProjectionReads:
    """Owned public reads shared only during one synchronous task refresh.

    No instance survives the refresh or participates in native admission. Each
    projection still selects its own actor scope and owns its entity registry.
    Values are treated as read-only by Projection/admissible, never returned.
    """

    def __init__(self, runtime):
        self._runtime = runtime
        self._reads = {}
        self.time_sec = runtime.time_sec
        self.horizon_sec = runtime.horizon_sec

    def _read(self, method, *args):
        key = (method, *args)
        if key not in self._reads:
            self._reads[key] = getattr(self._runtime, method)(*args)
        return self._reads[key]

    def observe(self, actor):
        return self._read("observe", actor)

    def inventory(self, actor):
        return self._read("inventory", actor)

    def station_observation(self, station):
        return self._read("station_observation", station)

    def pointing_observation(self, actor, *, target_id=None, station_id=None):
        key = ("pointing_observation", actor, target_id, station_id)
        if key not in self._reads:
            self._reads[key] = self._runtime.pointing_observation(
                actor, target_id=target_id, station_id=station_id
            )
        return self._reads[key]

    def command_status(self):
        return self._read("command_status")

    def describe_capabilities(self):
        return self._read("describe_capabilities")

    def command_claims(self, command):
        # Claims depend on the command/configuration, not native admission.
        return self._runtime.command_claims(command)


class Projection:
    def __init__(self, spec):
        self.spec = spec
        self.indices = {}

    def build(self, runtime, role, revision, actions):
        scope = None if role == "@central" else self.spec.roles[role]
        actors = (
            runtime.describe_capabilities()["actors"]
            if scope is None
            else list(scope.actors)
        )
        views = {a: runtime.observe(a)["actors"][a] for a in actors}
        inventory = {
            a: {p["id"]: p for p in runtime.inventory(a)["products"]} for a in actors
        }
        command_rows = [
            r for r in runtime.command_status() if r["command"]["actor_id"] in actors
        ]
        now = runtime.time_sec
        nodes = {}
        edges = []

        def add(kind, id, values=None, state=None):
            key = kind + ":" + id
            fields = {"measured_at": now, "age_sec": 0.0, **(values or {})}
            if state in STATES:
                fields["status_code"] = STATES[state]
            fields = {
                k: float(v)
                for k, v in fields.items()
                if k in FEATURES and v is not None
            }
            if not all(math.isfinite(v) and abs(v) <= 1e300 for v in fields.values()):
                raise ValueError("task entity features exceed finite numeric contract")
            nodes[key] = {"key": key, "kind": kind, "id": id, "values": fields}
            return key

        def edge(kind, left, right):
            edges.append((kind, left, right))

        for a, view in views.items():
            bus = view.get("spacecraft_bus", {})
            local = runtime.inventory(a)
            extra = {}
            orbit = view.get("orbit", {})
            if orbit.get("frame") == "GCRS":
                for key, value in zip(
                    ["r_x_m", "r_y_m", "r_z_m", "v_x_m_s", "v_y_m_s", "v_z_m_s"],
                    orbit["state_si"],
                ):
                    extra[key] = value
                extra.update(
                    mass_kg=orbit.get("mass_kg"),
                    propellant_kg=orbit.get("propellant_kg"),
                )
            if "quaternion_wxyz" in bus:
                extra.update(zip(["q_w", "q_x", "q_y", "q_z"], bus["quaternion_wxyz"]))
            actor = add(
                "actor",
                a,
                {
                    **extra,
                    "soc": view["soc"],
                    "stored_bits": view["stored_bits"],
                    "reserved_bits": view.get("dtn", {}).get("reserved_bits", 0.0),
                    "temperature_k": bus.get(
                        "temperature_k", local["transmitter"]["temperature_k"]
                    ),
                    "capacity_bits": local["memory_capacity_bits"],
                    "battery_wh": local["battery_wh"],
                    "tx_seconds": local["transmitter"]["seconds"],
                    "tx_ready": local["transmitter"]["ready"],
                    "safe": bus.get("safe"),
                },
                "safe" if bus.get("safe") else None,
            )
            for station in view.get("contacts", []):
                key = "station:" + station
                if key not in nodes:
                    add("station", station)
                edge("contact", actor, key)
            radio = view.get("radio_network", {})
            for station, w in radio.get("weather", {}).items():
                sid = "station:" + station
                if sid not in nodes:
                    add("station", station)
                wid = add(
                    "weather",
                    f"{len(a)}:{a}:{station}",
                    {
                        "fade_db": w["fade_db"],
                        "outage": w["outage"],
                        "measured_at": w["time_sec"],
                        "age_sec": max(0.0, now - w["time_sec"]),
                    },
                )
                edge("source", actor, wid)
                edge("destination", wid, sid)
            link = radio.get("link")
            if link is not None:
                station = link["station"]
                sid = "station:" + station
                if sid not in nodes:
                    add("station", station)
                lid = add(
                    "link",
                    f"{len(a)}:{a}:{station}",
                    {
                        "range_m": link.get("range_m"),
                        "rate_bps": link["rate_bps"],
                        "cn0_db_hz": link.get("cn0_db_hz"),
                        "bandwidth_hz": link.get("occupied_bandwidth_hz"),
                        "measured_at": link["sample_sec"],
                        "age_sec": max(0.0, now - link["sample_sec"]),
                    },
                    link["state"] if link["state"] in STATES else None,
                )
                edge("source", actor, lid)
                edge("destination", lid, sid)
            life = view.get("lifecycle", {})
            for r in life.get("requests", []):
                request = add(
                    "request",
                    r["id"],
                    {
                        "ready_sec": r.get("release_sec"),
                        "deadline_sec": r["deadline_sec"],
                        "priority": r.get("priority"),
                    },
                    r["status"],
                )
                edge("controls", actor, request)
            for j in life.get("jobs", []):
                job = add(
                    "job",
                    j["id"],
                    {
                        "remaining_work_sec": j.get("remaining_sec"),
                        "progress": j.get("work_sec"),
                    },
                    j["status"],
                )
                edge("controls", actor, job)
            artifacts = {p["id"]: p for p in life.get("artifacts", [])}
            for pid, p in inventory[a].items():
                artifact = artifacts.get(pid, {})
                state = (
                    "cancelled"
                    if p["cancelled"]
                    else ("delivered" if p["acknowledged"] else "await_ack")
                    if p["remaining_bits"] == 0
                    else artifact.get("status", "resident")
                )
                product = add(
                    "product",
                    pid,
                    {
                        "size_bits": p["size_bits"],
                        "remaining_bits": p["remaining_bits"],
                        "ready_sec": p["ready_sec"],
                        "deadline_sec": p["deadline_sec"],
                        "priority": p["priority"],
                        "remaining_work_sec": p["processing_remaining_sec"],
                        "created_sec": artifact.get("created_sec"),
                    },
                    state,
                )
                edge("owns", actor, product)
                if p.get("destination"):
                    station = p["destination"]
                    if "station:" + station not in nodes:
                        add("station", station)
                    edge("destination", product, "station:" + station)
            for j in life.get("jobs", []):
                for pid in j["inputs"]:
                    edge("requires", "product:" + pid, "job:" + j["id"])
                for p in j["outputs"]:
                    edge("produces", "job:" + j["id"], "product:" + p["id"])
            for p in artifacts.values():
                for rid in p.get("request_ids", []):
                    edge("fulfills", "product:" + p["id"], "request:" + rid)
            for t in view.get("dtn", {}).get("transfers", []):
                transfer = add(
                    "transfer",
                    t["id"],
                    {
                        "size_bits": t["size_bits"],
                        "progress": t["progress_bits"],
                        "remaining_bits": t["size_bits"] - t["offset_bits"],
                        "deadline_sec": t["expires_sec"],
                    },
                    t["status"]
                    if t["status"] in {"completed", "failed", "cancelled", "expired"}
                    else t["state"],
                )
                edge("source", "actor:" + t["source"], transfer)
                edge("destination", transfer, "actor:" + t["destination"])
        station_names = set() if scope is None else set(scope.stations)
        if scope is None:
            station_names = set(getattr(self, "station_ids", ()))
        for station in sorted(station_names):
            v = runtime.station_observation(station)
            add(
                "station",
                station,
                {
                    "safe": not v["outage"],
                    "outage": v["outage"],
                    "terminals": v["terminals"],
                },
                "failed" if v["outage"] else "ready",
            )
        for row in command_rows:
            if row["status"] not in {"accepted", "running"}:
                continue
            c = row["command"]
            key = add(
                "command",
                c["command_id"],
                {"ready_sec": row["start_sec"], "deadline_sec": row["until_sec"]},
                row["status"],
            )
            edge("controls", "actor:" + c["actor_id"], key)
        for key in sorted(nodes):
            if key not in self.indices:
                if len(self.indices) >= self.spec.max_entities:
                    raise ValueError("task entity capacity exceeded")
                self.indices[key] = len(self.indices)
        unique = sorted({(k, a, b) for k, a, b in edges if a in nodes and b in nodes})
        if len(unique) > self.spec.max_relations:
            raise ValueError("task relation capacity exceeded")
        masks = []
        catalogue = []
        for action in [None, *actions]:
            enabled, reason = admissible(
                runtime, self.spec, action, role, views, inventory, command_rows
            )
            masks.append(int(enabled))
            item = {
                "id": "wait" if action is None else action.id,
                "kind": "wait" if action is None else action.kind,
                "enabled": bool(enabled),
                "reason": reason,
            }
            if action is not None and (role == "@central" or action.role == role):
                item["actors"] = sorted(
                    {c.actor_id for c in action.commands}
                    | ({action.actor} if action.actor else set())
                )
                # Static command parameters/config/seed are not part of observations.
                item["command_kinds"] = [c.kind for c in action.commands]
                visible = True
                for c in action.commands:
                    life = views.get(c.actor_id, {}).get("lifecycle", {})
                    if self.spec.pointing_masks and c.kind == "slew" and c.slew.target_id is not None:
                        if not any(r["target_id"] == c.slew.target_id for r in life.get("requests", [])):
                            visible = False
                    if c.product_id and c.product_id not in inventory.get(
                        c.actor_id, {}
                    ):
                        visible = False
                    if c.request_id and not any(
                        r["id"] == c.request_id for r in life.get("requests", [])
                    ):
                        visible = False
                    if c.job_id and not any(
                        j["id"] == c.job_id for j in life.get("jobs", [])
                    ):
                        visible = False
                if visible:
                    item["commands"] = [
                        {
                            k: v
                            for k, v in c.model_dump(mode="json").items()
                            if k not in {"command_id", "forecast"}
                            and v is not None
                            and v != ""
                        }
                        for c in action.commands
                    ]
                if visible and enabled and self.spec.pointing_masks:
                    item["pointing"] = [
                        runtime.pointing_observation(c.actor_id, target_id=c.slew.target_id)
                        for c in action.commands
                        if c.kind == "slew" and c.slew.target_id is not None
                    ]
                if action.event is not None:
                    item["operator_event"] = {
                        "type": action.event["type"],
                        "target_actor": action.event["payload"].get("actor"),
                        "target_station": action.event["payload"].get("station"),
                    }
            catalogue.append(item)
        result = {
            "schema": "space-env.task-observation/1",
            "task_id": self.spec.id,
            "role": role,
            "revision": revision,
            "time_sec": now,
            "state_source": "ideal-current-local-runtime",
            "belief": {"kind": "none", "measurement_age_sec": 0.0, "estimated": False},
            "feature_names": list(FEATURES),
            "entities": [
                dict(nodes[k], index=self.indices[k])
                for k in sorted(nodes, key=self.indices.get)
            ],
            "relations": [
                {"kind": k, "source": self.indices[a], "destination": self.indices[b]}
                for k, a, b in unique
            ],
            "action_mask": masks,
            "catalogue": catalogue,
            "mask_scope": "conservative single-action checks; joint commands rechecked after cancellation; no future success guarantee",
        }
        result["binding"] = digest(result)
        return result


def admissible(runtime, spec, action, role, views=None, inventory=None, commands=None):
    if action is None:
        return True, "wait"
    if role != "@central" and action.role != role:
        return False, "role_scope"
    owner = spec.roles[action.role]
    if views is None:
        views = {a: runtime.observe(a)["actors"][a] for a in owner.actors}
        inventory = {
            a: {p["id"]: p for p in runtime.inventory(a)["products"]}
            for a in owner.actors
        }
        commands = [
            r
            for r in runtime.command_status()
            if r["command"]["actor_id"] in owner.actors
        ]
    active = [r for r in commands if r["status"] in {"accepted", "running"}]
    if action.kind == "cancel_actor":
        return (
            (True, "cancel")
            if any(r["command"]["actor_id"] == action.actor for r in active)
            else (False, "no_active_command")
        )
    if action.kind == "inject":
        event = action.event
        payload = event["payload"]
        if (
            event["type"] == "station_recovered"
            and not runtime.station_observation(payload["station"])["outage"]
        ):
            return False, "station_not_in_outage"
        if event["type"] == "spacecraft_recover" and not views.get(
            payload["actor"], {}
        ).get("spacecraft_bus", {}).get("safe"):
            return False, "actor_not_safe"
        return True, "declared_operator_action"
    supported = runtime.describe_capabilities()["supported"]
    for c in action.commands:
        if "commands." + c.kind not in supported:
            return False, "unsupported_capability"
        if runtime.time_sec + c.duration_sec > runtime.horizon_sec + 1e-9:
            return False, "control_exceeds_horizon"
        claim = set(runtime.command_claims(c)["resources"])
        for row in active:
            if row["command"]["actor_id"] != c.actor_id:
                continue
            other = set(runtime.command_claims(row["command"])["resources"])
            if "actor" in claim | other or claim & other:
                return False, "actor_busy"
        v = views.get(c.actor_id, {})
        bus = v.get("spacecraft_bus", {})
        if bus.get("safe") and c.kind != "wait":
            return False, "safe_mode"
        subsystem = {
            "relay": "radio",
            "downlink": "radio",
            "imaging": "payload",
            "transform": "cpu",
            "processing": "cpu",
            "body_burn": "propulsion",
            "wheel_torque": "adcs",
            "slew": "adcs",
        }.get(c.kind)
        if "bus" in bus.get("faults", []) or subsystem in bus.get("faults", []):
            return False, "subsystem_fault"
        if (
            c.kind in {"downlink", "relay"}
            and not runtime.inventory(c.actor_id)["transmitter"]["ready"]
        ):
            return False, "transmitter_unavailable"
        life = v.get("lifecycle", {})
        if c.kind == "imaging":
            r = next(
                (r for r in life.get("requests", []) if r["id"] == c.request_id), None
            )
            if r is None or r["status"] != "released":
                return False, "request_not_available"
            if spec.pointing_masks:
                pointing = runtime.pointing_observation(c.actor_id, target_id=r["target_id"])
                if pointing["geometry"] is None or not pointing["geometry"]["feasible_geometry"]:
                    return False, "current_geometry_unavailable"
                if (pointing["body_rate_rad_s"] is not None
                    and pointing["body_rate_rad_s"] > pointing["payload_max_body_rate_rad_s"]):
                    return False, "current_body_rate"
                if pointing["settle_remaining_sec"] > 1e-9:
                    return False, "current_settling"
                if pointing["boresight_error_deg"] > pointing["payload_tolerance_deg"]:
                    return False, "current_pointing_error"
        if c.kind == "slew" and spec.pointing_masks and c.slew.target_id is not None:
            requests = [r for r in life.get("requests", [])
                        if r["target_id"] == c.slew.target_id and r["status"] == "released"]
            if not requests:
                return False, "request_not_available"
            pointing = runtime.pointing_observation(c.actor_id, target_id=c.slew.target_id)
            if pointing["geometry"] is None or not pointing["geometry"]["feasible_geometry"]:
                return False, "current_geometry_unavailable"
        if c.kind == "transform":
            j = next((j for j in life.get("jobs", []) if j["id"] == c.job_id), None)
            if j is None or j["status"] in {
                "completed",
                "failed",
                "cancelled",
                "expired",
            }:
                return False, "job_not_available"
            if any(
                p not in inventory[c.actor_id]
                or inventory[c.actor_id][p]["remaining_bits"] <= 0
                for p in j["inputs"]
            ):
                return False, "input_not_local"
        if c.kind in {"relay", "downlink"} and c.product_id:
            p = inventory.get(c.actor_id, {}).get(c.product_id)
            if p is None or p["cancelled"] or p["remaining_bits"] <= 0:
                return False, "product_not_local"
            if p["custody_pinned"] or p["retained"] or not p["deliverable"]:
                return False, "product_pinned"
            if p["processing_remaining_sec"] > 0 or p["ready_sec"] > runtime.time_sec:
                return False, "product_not_ready"
            if c.kind == "relay" and p["remaining_bits"] != p["size_bits"]:
                return False, "product_not_intact"
        if c.kind == "downlink" and c.destination_id not in v.get("contacts", []):
            return False, "no_current_contact"
    return True, "observable_checks_passed"
