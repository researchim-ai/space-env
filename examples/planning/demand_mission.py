"""Fixed incoming-request mechanism cases; no geometry/outcome search."""

import argparse
import copy
from pathlib import Path

from examples.planning.incident_mission import configuration as incident_configuration
from space_env.planning.actor_data import publish_actor_data, verify_actor_data
from space_env.planning.actor_online import ActorCoordinator
from space_env.planning.chain import propose_chain
from space_env.planning.demand_journal import (
    append_demand,
    create_demand_journal,
    verify_demand_journal,
)
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def configuration(mode="batch"):
    if mode not in ("batch", "two_updates", "tight_deadline"):
        raise ValueError("unknown demand case")
    value = incident_configuration()
    d = value["scenario"]["definition"]
    d["horizon_sec"] = 360.0
    keep = {"a-00-t02", "b-00-t06"}
    d["lifecycle"]["requests"] = [
        r for r in d["lifecycle"]["requests"] if r["id"] in keep
    ]
    for r in d["lifecycle"]["requests"]:
        r["deadline_sec"] = 360.0
    raw = {r["raw_product_id"] for r in d["lifecycle"]["requests"]}
    d["lifecycle"]["jobs"] = [
        j for j in d["lifecycle"]["jobs"] if set(j["inputs"]) <= raw
    ]
    value["task"]["actions"] = [
        a
        for a in value["task"]["actions"]
        if any(
            a["id"] == kind + "-" + r
            for kind in ("slew", "imaging", "transform", "downlink")
            for r in keep
        )
    ]
    value["task"].update(max_actions=9, max_decisions=181)
    value["goals"]["coupled"]["native_spec_sha256"] = digest(d)
    additions = []
    for r in d["lifecycle"]["requests"]:
        new = copy.deepcopy(r)
        for k in ("id", "raw_product_id", "delivery_product_id"):
            new[k] += "-incoming"
        new["release_sec"] = 104.0
        job = copy.deepcopy(
            next(
                j
                for j in d["lifecycle"]["jobs"]
                if j["inputs"] == [r["raw_product_id"]]
            )
        )
        job["id"] += "-incoming"
        job["inputs"] = [new["raw_product_id"]]
        job["outputs"][0]["id"] = new["delivery_product_id"]
        additions.append({"request": new, "jobs": [job]})
    if mode == "two_updates":
        additions[1]["request"]["release_sec"] = 118.0
        updates = [
            {"at_sec": 100.0, "batch": {"additions": [additions[0]]}},
            {"at_sec": 114.0, "batch": {"additions": [additions[1]]}},
        ]
    elif mode == "tight_deadline":
        additions[0]["request"]["deadline_sec"] = 106.0
        updates = [{"at_sec": 100.0, "batch": {"additions": [additions[0]]}}]
    else:
        updates = [{"at_sec": 100.0, "batch": {"additions": additions}}]
    return {"mode": mode, "workload": value, "updates": updates}


def policy_step(actor):
    ids = actor._plan["steps"][actor.cursor]["action_ids"]
    owners = {a["id"]: a["role"] for a in actor._plan["task"]["actions"]}
    observations = actor.observe()
    return actor.step(
        {r: [a for a in ids if owners.get(a) == r] or ["wait"] for r in actor.roles},
        expected_bindings={r: o["binding"] for r, o in observations.items()},
        idempotency_key=f"policy/{actor.cursor}",
    )


def run(value, output):
    root = Path(output)
    if root.exists():
        raise ValueError("demand case output must not exist")
    root.mkdir(parents=True)
    atomic_write(root / "input.json", canonical_json(value))
    w = value["workload"]
    proposal = propose_chain(
        MissionRuntime(w["scenario"], access=AccessPolicy(forecast=True)),
        w["scenario"],
        w["task"],
        goals=w["goals"],
    )
    journal = create_demand_journal(proposal)
    actor = ActorCoordinator(journal)
    versions = [actor.catalogue_version]
    atomic_write(root / "initial-reference.json", canonical_json(journal))
    for i, update in enumerate(value["updates"]):
        while actor.cursor * w["task"]["decision_sec"] < update["at_sec"]:
            policy_step(actor)
        atomic_write(root / f"before-{i}.json", canonical_json(actor.checkpoint()))
        journal = append_demand(
            journal, actor.as_execution()["execution"], update["batch"]
        )
        actor = actor.rebind(journal)
        versions.append(actor.catalogue_version)
        atomic_write(root / f"journal-{i}.json", canonical_json(journal))
        atomic_write(root / f"after-{i}.json", canonical_json(actor.checkpoint()))
    atomic_write(root / "journal.json", canonical_json(journal))
    atomic_write(
        root / "journal-audit.json",
        canonical_json(verify_demand_journal(journal, native=True)),
    )
    first = True
    while not actor.done:
        policy_step(actor)
        if first:
            atomic_write(root / "actor-prefix.json", canonical_json(actor.checkpoint()))
            first = False
    record = actor.checkpoint()
    atomic_write(root / "controller.json", canonical_json(record))
    atomic_write(root / "execution.json", canonical_json(actor.as_execution()))
    data = publish_actor_data(record, root / "data")
    audit = verify_actor_data(root / "data", native=True)
    atomic_write(root / "data-audit.json", canonical_json(audit))
    result = {
        "epochs": len(journal["epochs"]),
        "catalogue_versions": versions,
        "fulfilled": data["goal"]["fulfilled"],
        "total": data["goal"]["total"],
        "mission_goal_passed": data["goal"]["passed"],
        "actor_rows": {
            f: sum(
                v["rows"] for v in data["training_files"].values() if v["format"] == f
            )
            for f in ("rl", "sft")
        },
        "expert_teacher_qualified": False,
    }
    atomic_write(root / "result.json", canonical_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode", choices=["batch", "two_updates", "tight_deadline"], default="batch"
    )
    parser.add_argument("--write-spec", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must not exist")
    value = configuration(args.mode)
    if args.write_spec:
        atomic_write(args.output, canonical_json(value))
    else:
        print(canonical_json(run(value, args.output)))
