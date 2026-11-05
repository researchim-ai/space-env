"""Fixed actor/journal mechanism cases. No heldout/teacher superiority claim."""

import argparse
from pathlib import Path

from examples.planning.incident_mission import configuration as incident_configuration
from space_env.planning.actor_data import publish_actor_data, verify_actor_data
from space_env.planning.actor_gym import ActorMissionEnv
from space_env.planning.actor_online import ActorCoordinator
from space_env.planning.chain import propose_chain
from space_env.planning.incident_journal import (
    append_incident,
    create_journal,
    current_plan,
    verify_journal,
)
from space_env.planning.mission_plan import PlanExecution
from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


def configuration(mode="policy_fork"):
    if mode not in ("policy_fork", "eight_epochs", "failed_policy"):
        raise ValueError("unknown actor case")
    epochs = (
        [
            {"at_sec": float(t), "outage_end_sec": float(t + 10)}
            for t in range(100, 157, 8)
        ]
        if mode == "eight_epochs"
        else [{"at_sec": 100.0, "outage_end_sec": 120.0}]
    )
    if mode == "policy_fork":
        epochs.append({"at_sec": 124.0, "outage_end_sec": 140.0})
    return {
        "mode": mode,
        "workload": incident_configuration(),
        "epochs": epochs,
        "deviate_at_sec": 122.0 if mode == "policy_fork" else None,
        "policy": "wait" if mode == "failed_policy" else "follow_plan",
        "run_adapters": mode == "policy_fork",
    }


def decision(controller, wait=False):
    owners = {a["id"]: a["role"] for a in controller._plan["task"]["actions"]}
    ids = (
        ["wait"] if wait else controller._plan["steps"][controller.cursor]["action_ids"]
    )
    return {
        r: [a for a in ids if owners.get(a) == r] or ["wait"] for r in controller.roles
    }


def play_adapters(journal, expected_report_sha):
    reports = {}
    gym = ActorMissionEnv(journal)
    obs, _ = gym.reset(seed=7)
    count = 0
    while not gym.coordinator.done:
        assert gym.observation_space.contains(obs)
        choices = decision(gym.coordinator)
        obs, _, _, _, _ = gym.step(
            {r: gym.action_ids[r].index(choices[r][0]) for r in gym.roles}
        )
        count += 1
    actual = digest(
        gym.coordinator.as_execution()["execution"]["episode"]["runtime"][
            "backend_report"
        ]
    )
    assert actual == expected_report_sha
    reports["gym"] = {
        "steps": count,
        "physical_report_matches_sdk": True,
        "spaces_checked": True,
    }
    try:
        from space_env.planning.actor_parallel import ActorParallelEnv
    except ImportError:
        reports["pettingzoo"] = {"available": False}
        return reports
    parallel = ActorParallelEnv(journal)
    obs, _ = parallel.reset(seed=7)
    count = 0
    while parallel.agents:
        c = parallel.joint.coordinator
        choices = decision(c)
        assert all(
            parallel.observation_space(r).contains(obs[r]) for r in parallel.agents
        )
        obs, _, _, _, _ = parallel.step(
            {
                r: parallel.joint.action_ids[r].index(choices[r][0])
                for r in parallel.agents
            }
        )
        count += 1
    actual = digest(
        parallel.joint.coordinator.as_execution()["execution"]["episode"]["runtime"][
            "backend_report"
        ]
    )
    assert actual == expected_report_sha
    reports["pettingzoo"] = {
        "available": True,
        "steps": count,
        "physical_report_matches_sdk": True,
        "spaces_checked": True,
    }
    return reports


def run(value, output):
    root = Path(output)
    if root.exists():
        raise ValueError("actor case output must not exist")
    root.mkdir(parents=True)
    atomic_write(root / "input.json", canonical_json(value))
    w = value["workload"]
    q = w["task"]["decision_sec"]
    proposal = propose_chain(
        MissionRuntime(w["scenario"], access=AccessPolicy(forecast=True)),
        w["scenario"],
        w["task"],
        goals=w["goals"],
    )
    journal = create_journal(proposal)
    for i, epoch in enumerate(value["epochs"]):
        cursor = round(epoch["at_sec"] / q)
        if i == 1 and value["deviate_at_sec"] is not None:
            c = ActorCoordinator(journal)
            while c.cursor < cursor:
                obs = c.observe()
                choices = decision(c, wait=c.cursor * q == value["deviate_at_sec"])
                c.step(
                    choices,
                    expected_bindings={r: o["binding"] for r, o in obs.items()},
                    idempotency_key=f"policy/{c.cursor}",
                )
            atomic_write(
                root / "previous-actor-prefix.json", canonical_json(c.checkpoint())
            )
            record = c.as_execution()["execution"]
        else:
            execution = PlanExecution(current_plan(journal))
            execution.run(decisions=cursor)
            record = execution.checkpoint()
        journal = append_incident(
            journal,
            record,
            {
                "disruptions": [
                    {"station": s, "end_sec": epoch["outage_end_sec"]}
                    for s in ("g02", "g06")
                ]
            },
        )
        atomic_write(root / "journal.json", canonical_json(journal))
        atomic_write(
            root / "progress.json",
            canonical_json(
                {
                    "expected_epochs": len(value["epochs"]),
                    "completed_epochs": i + 1,
                    "journal_sha256": journal["sha256"],
                }
            ),
        )
    atomic_write(
        root / "journal-audit.json",
        canonical_json(verify_journal(journal, native=True)),
    )
    c = ActorCoordinator(journal)
    first = True
    while not c.done:
        obs = c.observe()
        choices = decision(c, wait=value["policy"] == "wait")
        c.step(
            choices,
            expected_bindings={r: o["binding"] for r, o in obs.items()},
            idempotency_key=f"policy/{c.cursor}",
        )
        if first:
            atomic_write(root / "actor-prefix.json", canonical_json(c.checkpoint()))
            first = False
    checkpoint = c.checkpoint()
    atomic_write(root / "controller.json", canonical_json(checkpoint))
    executed = c.as_execution()
    atomic_write(root / "execution.json", canonical_json(executed))
    data = publish_actor_data(checkpoint, root / "data")
    atomic_write(
        root / "data-audit.json",
        canonical_json(verify_actor_data(root / "data", native=True)),
    )
    native = executed["execution"]["episode"]["runtime"]["backend_report"]
    report_sha = digest(native)
    if value["run_adapters"]:
        atomic_write(
            root / "adapters.json", canonical_json(play_adapters(journal, report_sha))
        )
    counts = {
        f: sum(v["rows"] for v in data["training_files"].values() if v["format"] == f)
        for f in ("rl", "sft")
    }
    result = {
        "epochs": len(journal["epochs"]),
        "journal_bytes": len(canonical_json(journal).encode()),
        "controller_sha256": checkpoint["sha256"],
        "fulfilled": data["goal"]["fulfilled"],
        "total": data["goal"]["total"],
        "mission_goal_passed": data["goal"]["passed"],
        "actor_rows": counts,
        "physical_report_sha256": report_sha,
        "expert_teacher_qualified": False,
    }
    atomic_write(root / "result.json", canonical_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        choices=["policy_fork", "eight_epochs", "failed_policy"],
        default="policy_fork",
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
