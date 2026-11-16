"""Slotted store-and-forward service, shared terminals and reserve commissioning.

Capacities/contact calendars are explicit inputs, not computed RF waveforms or
orbital deployment. Transfers commit at slot end; new custody cannot be reused
inside the same slot. All bit accounting is integer and conserved.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Annotated, Literal

from pydantic import Field, model_validator

from space_env.industrial.session import DecisionSession
from space_env.runtime.schema import StrictModel


class Node(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    kind: Literal["satellite", "ground"] = "satellite"
    terminals: int = Field(default=1, ge=1, le=8, strict=True)
    queue_bits: int = Field(default=10000000, ge=1, le=10**12, strict=True)
    energy_j: float = Field(default=100000.0, ge=0, le=1e12, strict=True)
    idle_power_w: float = Field(default=0.0, ge=0, le=1e6, strict=True)
    terminal_power_w: float = Field(default=10.0, ge=0, le=1e6, strict=True)
    reserve: bool = Field(default=False, strict=True)
    commissioning_steps: int = Field(default=2, ge=1, le=32, strict=True)
    handover_steps: int = Field(default=1, ge=0, le=16, strict=True)


class Link(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    source: str
    destination: str
    bits_per_slot: int = Field(ge=1, le=10**12, strict=True)
    windows: tuple[
        tuple[Annotated[int, Field(strict=True)], Annotated[int, Field(strict=True)]],
        ...,
    ] = ()

    @model_validator(mode="after")
    def intervals(self):
        for pair in self.windows:
            if (
                any(type(v) is not int for v in pair)
                or not 0 <= pair[0] < pair[1] <= 512
            ):
                raise ValueError("contact windows are half-open integer slot intervals")
        return self


class Flow(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    source: str
    destination: str
    size_bits: int = Field(ge=1, le=10**12, strict=True)
    priority: int = Field(default=1, ge=0, le=1000000, strict=True)
    release_step: int = Field(default=0, ge=0, le=511, strict=True)
    deadline_step: int = Field(ge=1, le=512, strict=True)


class ServiceEvent(StrictModel):
    at_step: int = Field(ge=0, le=511, strict=True)
    kind: Literal["node_fault", "node_recovered", "flow_cancelled", "flow_priority"]
    target: str
    priority: int | None = Field(default=None, ge=0, le=1000000, strict=True)

    @model_validator(mode="after")
    def payload(self):
        if (self.kind == "flow_priority") != (self.priority is not None):
            raise ValueError("priority payload only for priority changes")
        return self


class ServiceSpec(StrictModel):
    id: str = Field(default="service-network", min_length=1, max_length=128)
    lineage: str = Field(default="slotted-network/v1", min_length=1, max_length=128)
    nodes: tuple[Node, ...] = Field(min_length=2, max_length=64)
    links: tuple[Link, ...] = Field(min_length=1, max_length=256)
    flows: tuple[Flow, ...] = Field(min_length=1, max_length=512)
    events: tuple[ServiceEvent, ...] = Field(default=(), max_length=512)
    slot_sec: float = Field(default=1.0, ge=0.1, le=60, strict=True)
    horizon_steps: int = Field(default=16, ge=1, le=512, strict=True)

    @model_validator(mode="after")
    def topology(self):
        for rows in (self.nodes, self.links, self.flows):
            if len({r.id for r in rows}) != len(rows):
                raise ValueError("duplicate topology or flow IDs")
        nodes = {n.id: n for n in self.nodes}
        flows = {f.id: f for f in self.flows}
        for link in self.links:
            if (
                link.source not in nodes
                or link.destination not in nodes
                or link.source == link.destination
            ):
                raise ValueError("unknown or self link")
        for flow in self.flows:
            if (
                flow.source not in nodes
                or flow.destination not in nodes
                or flow.source == flow.destination
                or nodes[flow.source].kind != "satellite"
                or nodes[flow.destination].kind != "ground"
                or not flow.release_step < flow.deadline_step <= self.horizon_steps
            ):
                raise ValueError("invalid service flow endpoints or deadline")
        for event in self.events:
            if event.at_step >= self.horizon_steps or event.target not in (
                nodes if event.kind.startswith("node_") else flows
            ):
                raise ValueError("unknown event target or outside horizon")
            if (
                event.kind.startswith("flow_")
                and event.at_step < flows[event.target].release_step
            ):
                raise ValueError("flow change predates release")
        return self


class Transfer(StrictModel):
    link_id: str
    flow_id: str
    bits: int = Field(ge=1, le=10**12, strict=True)


class Handover(StrictModel):
    node_id: str
    destination: str


class ServiceAction(StrictModel):
    transfers: tuple[Transfer, ...] = Field(default=(), max_length=64)
    activate: tuple[str, ...] = Field(default=(), max_length=64)
    handovers: tuple[Handover, ...] = Field(default=(), max_length=64)


class ServiceSession(DecisionSession):
    kind = "service"

    def __init__(self, configuration):
        self.spec = ServiceSpec.model_validate(configuration)
        self._start(self.spec.model_dump(mode="json"))
        self._nodes = {n.id: n for n in self.spec.nodes}
        self._links = {l.id: l for l in self.spec.links}
        self._nodes_state = {
            n.id: {
                "ready_at": None if n.reserve else 0,
                "fault": False,
                "energy_j": n.energy_j,
                "peer": None,
                "handover_until": 0,
            }
            for n in self.spec.nodes
        }
        self._flows = {}
        self._receipts = []
        self._time = 0
        self._boundary()

    def _load(self, node):
        return sum(f["queues"].get(node, 0) for f in self._flows.values())

    def _boundary(self):
        for spec in self.spec.flows:
            if spec.release_step == self._time:
                accepted = (
                    self._load(spec.source) + spec.size_bits
                    <= self._nodes[spec.source].queue_bits
                )
                self._flows[spec.id] = {
                    "id": spec.id,
                    "destination": spec.destination,
                    "size_bits": spec.size_bits,
                    "priority": spec.priority,
                    "deadline_step": spec.deadline_step,
                    "status": "active" if accepted else "rejected_capacity",
                    "queues": {spec.source: spec.size_bits} if accepted else {},
                    "delivered_bits": 0,
                    "discarded_bits": 0 if accepted else spec.size_bits,
                }
        for event in self.spec.events:
            if event.at_step != self._time:
                continue
            if event.kind.startswith("node_"):
                self._nodes_state[event.target]["fault"] = event.kind == "node_fault"
            else:
                flow = self._flows[event.target]
                if flow["status"] != "active":
                    continue
                if event.kind == "flow_priority":
                    flow["priority"] = event.priority
                else:
                    flow["discarded_bits"] += sum(flow["queues"].values())
                    flow["queues"] = {}
                    flow["status"] = "cancelled"
        for flow in self._flows.values():
            if flow["status"] == "active" and flow["deadline_step"] <= self._time:
                flow["discarded_bits"] += sum(flow["queues"].values())
                flow["queues"] = {}
                flow["status"] = "expired"

    def _available(self, name):
        node = self._nodes_state[name]
        return (
            node["ready_at"] is not None
            and node["ready_at"] <= self._time
            and not node["fault"]
            and node["handover_until"] <= self._time
        )

    def observe(self):
        return {
            "schema": "space-env.service-observation/1",
            "revision": self.revision,
            "step": self._time,
            "slot_sec": self.spec.slot_sec,
            "horizon_steps": self.spec.horizon_steps,
            "nodes": [
                {
                    **n.model_dump(mode="json"),
                    **copy.deepcopy(self._nodes_state[n.id]),
                    "queue_used_bits": self._load(n.id),
                    "available": self._available(n.id),
                }
                for n in self.spec.nodes
            ],
            "links": [
                {
                    "id": l.id,
                    "source": l.source,
                    "destination": l.destination,
                    "bits_per_slot": l.bits_per_slot,
                    "in_contact": not l.windows
                    or any(a <= self._time < b for a, b in l.windows),
                }
                for l in self.spec.links
            ],
            "flows": copy.deepcopy(list(self._flows.values())),
        }

    def _apply(self, value):
        action = ServiceAction.model_validate(value)
        if len(set(action.activate)) != len(action.activate) or len(
            {h.node_id for h in action.handovers}
        ) != len(action.handovers):
            raise ValueError("duplicate activation or handover")
        switching = {h.node_id for h in action.handovers}
        for name in action.activate:
            if (
                name not in self._nodes
                or not self._nodes[name].reserve
                or self._nodes_state[name]["ready_at"] is not None
            ):
                raise ValueError("activation requires an unused reserve")
        for h in action.handovers:
            if (
                h.node_id not in self._nodes
                or h.destination not in self._nodes
                or not any(
                    l.source == h.node_id and l.destination == h.destination
                    for l in self.spec.links
                )
                or not self._available(h.node_id)
            ):
                raise ValueError(
                    "handover requires an available source and known outgoing peer"
                )
        terminals, link_use, take, add = Counter(), Counter(), Counter(), Counter()
        selected_peers = {}
        for transfer in action.transfers:
            if (
                transfer.link_id not in self._links
                or transfer.flow_id not in self._flows
            ):
                raise ValueError("unknown link or unreleased flow")
            link, flow = self._links[transfer.link_id], self._flows[transfer.flow_id]
            if (
                flow["status"] != "active"
                or not self._available(link.source)
                or not self._available(link.destination)
                or {link.source, link.destination} & switching
                or (
                    link.windows
                    and not any(a <= self._time < b for a, b in link.windows)
                )
            ):
                raise ValueError("unavailable transfer")
            if (
                self._nodes[link.destination].kind == "ground"
                and link.destination != flow["destination"]
            ):
                raise ValueError("wrong destination ground station")
            peer = self._nodes_state[link.source]["peer"]
            if peer is not None and peer != link.destination:
                raise ValueError("explicit handover required")
            if (
                link.source in selected_peers
                and selected_peers[link.source] != link.destination
            ):
                raise ValueError("one outgoing peer per node per slot")
            selected_peers[link.source] = link.destination
            terminals.update((link.source, link.destination))
            link_use[link.id] += transfer.bits
            take[(transfer.flow_id, link.source)] += transfer.bits
            if link.destination != flow["destination"]:
                add[link.destination] += transfer.bits
        for name, count in terminals.items():
            node = self._nodes[name]
            required = self.spec.slot_sec * (
                node.idle_power_w + count * node.terminal_power_w
            )
            if (
                count > node.terminals
                or self._nodes_state[name]["energy_j"] + 1e-9 < required
            ):
                raise ValueError("shared terminal or energy budget exceeded")
        if any(
            bits > self._links[key].bits_per_slot for key, bits in link_use.items()
        ) or any(
            bits > self._flows[f]["queues"].get(n, 0) for (f, n), bits in take.items()
        ):
            raise ValueError("link capacity or start-of-slot custody exceeded")
        for name in self._nodes:
            outgoing = sum(bits for (f, n), bits in take.items() if n == name)
            if self._load(name) - outgoing + add[name] > self._nodes[name].queue_bits:
                raise ValueError("receiver queue capacity exceeded")
        before = self.metrics()["completed_priority"]
        for name in action.activate:
            self._nodes_state[name]["ready_at"] = (
                self._time + self._nodes[name].commissioning_steps
            )
        for h in action.handovers:
            self._nodes_state[h.node_id].update(
                peer=h.destination,
                handover_until=self._time + self._nodes[h.node_id].handover_steps,
            )
        for transfer in action.transfers:
            link, flow = self._links[transfer.link_id], self._flows[transfer.flow_id]
            flow["queues"][link.source] -= transfer.bits
            # Peer is committed only for admitted transfers; no implicit switch.
            self._nodes_state[link.source]["peer"] = link.destination
            if link.destination == flow["destination"]:
                flow["delivered_bits"] += transfer.bits
                self._receipts.append(
                    {
                        "flow_id": flow["id"],
                        "bits": transfer.bits,
                        "step": self._time + 1,
                        "station": link.destination,
                    }
                )
                if flow["delivered_bits"] == flow["size_bits"]:
                    flow["status"] = "completed"
            else:
                flow["queues"][link.destination] = (
                    flow["queues"].get(link.destination, 0) + transfer.bits
                )
        for name, node in self._nodes.items():
            state = self._nodes_state[name]
            if state["ready_at"] is not None:
                state["energy_j"] = max(
                    0.0,
                    state["energy_j"]
                    - self.spec.slot_sec
                    * (node.idle_power_w + terminals[name] * node.terminal_power_w),
                )
        self._time += 1
        self._boundary()
        self._done = self._time >= self.spec.horizon_steps
        self._conservation()
        return {
            "reward": float(self.metrics()["completed_priority"] - before),
            "terminated": self._done,
            "elapsed_sec": self.spec.slot_sec,
            "feedback": {"accepted": True, "code": "executed"},
        }

    def _conservation(self):
        for flow in self._flows.values():
            if flow["size_bits"] != sum(flow["queues"].values()) + flow[
                "delivered_bits"
            ] + flow["discarded_bits"] or any(v < 0 for v in flow["queues"].values()):
                raise RuntimeError("service bit conservation violated")

    def metrics(self):
        complete = [f for f in self._flows.values() if f["status"] == "completed"]
        return {
            "success": self._done and len(complete) == len(self.spec.flows),
            "terminal": self._done,
            "released_flows": len(self._flows),
            "total_flows": len(self.spec.flows),
            "completed_flows": len(complete),
            "completed_priority": sum(f["priority"] for f in complete),
            "delivered_bits": sum(f["delivered_bits"] for f in self._flows.values()),
            "discarded_bits": sum(f["discarded_bits"] for f in self._flows.values()),
            "queued_bits": sum(sum(f["queues"].values()) for f in self._flows.values()),
            "expired_flows": sum(
                f["status"] == "expired" for f in self._flows.values()
            ),
            "cancelled_flows": sum(
                f["status"] == "cancelled" for f in self._flows.values()
            ),
            "rejected_flows": sum(
                f["status"] == "rejected_capacity" for f in self._flows.values()
            ),
            "physical_orbital_deployment": False,
        }

    def _witness(self):
        return {
            "observation": self.observe(),
            "metrics": self.metrics(),
            "receipts": copy.deepcopy(self._receipts),
        }


def service_policy(observation):
    """Public shortest-path/priority scheduler with explicit terminal reservations."""
    nodes = {n["id"]: n for n in observation["nodes"]}
    links = sorted(observation["links"], key=lambda l: l["id"])
    activate = [
        n["id"] for n in nodes.values() if n["reserve"] and n["ready_at"] is None
    ]
    transfers, handovers, used, link_bits, taken, received = (
        [],
        [],
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    switching = set()

    def route(start, end):
        paths = [(start, [])]
        seen = {start}
        for node, path in paths:
            if node == end:
                return path
            for link in links:
                target = link["destination"]
                if (
                    link["source"] == node
                    and link["in_contact"]
                    and nodes[target]["available"]
                    and target not in seen
                ):
                    seen.add(target)
                    paths.append((target, path + [link]))
        return []

    for flow in sorted(
        observation["flows"],
        key=lambda f: (-f["priority"], f["deadline_step"], f["id"]),
    ):
        if flow["status"] != "active":
            continue
        for owner, amount in sorted(flow["queues"].items()):
            if not amount or not nodes[owner]["available"]:
                continue
            path = route(owner, flow["destination"])
            if not path:
                continue
            link = path[0]
            src, dst = link["source"], link["destination"]
            if (
                src in switching
                or dst in switching
                or any(used[n] >= nodes[n]["terminals"] for n in (src, dst))
            ):
                continue
            if nodes[src]["peer"] not in (None, dst):
                if not used[src]:
                    handovers.append({"node_id": src, "destination": dst})
                    switching.add(src)
                continue
            bits = min(
                amount - taken[(flow["id"], src)],
                link["bits_per_slot"] - link_bits[link["id"]],
                nodes[dst]["queue_bits"] - nodes[dst]["queue_used_bits"] - received[dst]
                if dst != flow["destination"]
                else amount,
            )
            if bits <= 0 or any(
                nodes[n]["energy_j"]
                < observation["slot_sec"]
                * (
                    nodes[n]["idle_power_w"]
                    + (used[n] + 1) * nodes[n]["terminal_power_w"]
                )
                for n in (src, dst)
            ):
                continue
            transfers.append(
                {"link_id": link["id"], "flow_id": flow["id"], "bits": bits}
            )
            used.update((src, dst))
            link_bits[link["id"]] += bits
            taken[(flow["id"], src)] += bits
            if dst != flow["destination"]:
                received[dst] += bits
    return {"transfers": transfers, "activate": activate, "handovers": handovers}
