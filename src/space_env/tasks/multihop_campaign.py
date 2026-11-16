"""Constructive two-ISL-hop campaign; reverse catalogue edge tests loop avoidance."""

import copy

from pydantic import Field

from space_env.tasks.routing_campaign import RoutingCampaignSpec, routing_campaign


class MultihopCampaignSpec(RoutingCampaignSpec):
    forward_link_available: bool = Field(default=True, strict=True)
    sink_memory_bits: int | None = Field(default=None, ge=1, le=2**53, strict=True)


def multihop_campaign(configuration=None):
    spec = MultihopCampaignSpec.model_validate(configuration or {})
    case = routing_campaign(
        {k: getattr(spec, k) for k in RoutingCampaignSpec.model_fields}
    )
    cfg = case["configuration"]
    cfg["publish_route_topology"] = True
    task = cfg["task"]
    ops = cfg["scenario"]["definition"]["operations"]
    network = ops["dtn_network"]
    # r is now an S-band forwarder. Native RF prevents r -> X-band ground delivery.
    ops["radio_network"]["transmitters"]["r"]["frequency_hz"] = 2200000000.0
    network["links"] = [
        e
        for e in network["links"]
        if (e["source"], e["destination"]) in {("a", "r"), ("b", "s")}
    ]
    forward = copy.deepcopy(network["links"][0])
    forward.update(source="r", destination="s")
    if not spec.forward_link_available:
        forward["outages"] = [[0, spec.horizon_sec]]
    backward = copy.deepcopy(forward)
    backward.update(source="s", destination="r", outages=[])
    backward["receiver"].update(
        min_frequency_hz=8100000000.0, max_frequency_hz=8300000000.0
    )
    network["links"].extend([forward, backward])
    task["actions"] = [
        a
        for a in task["actions"]
        if a["id"]
        not in {"late-a-via-s", "late-b-via-r", "late-r-ground", "late-b-ground-r"}
    ]
    template = next(a for a in task["actions"] if a["id"] == "late-a-hop")
    for name, source, destination in [
        ("late-a-forward", "r", "s"),
        ("late-a-return", "s", "r"),
    ]:
        action = copy.deepcopy(template)
        action.update(id=name, role=source)
        action["commands"][0].update(
            command_id=name, actor_id=source, destination_id=destination
        )
        task["actions"].append(action)
    if spec.sink_memory_bits is not None:
        ops["vehicles"]["s"]["memory_bits"] = spec.sink_memory_bits
    case["public_contract"]["require_route_progress"] = True
    case["public_brief"] += (
        " Camera a must use the declared a->r->s custody chain before X-band ground delivery; "
        "camera b uses b->s. The reverse s->r command is present, but select only relay edges "
        "that strictly reduce structural hop distance to a declared ground sender. "
        "Hop distance is a catalogue property, not a forecast of link availability."
    )
    case["setup"]["multihop_settings"] = spec.model_dump(mode="json")
    return case
