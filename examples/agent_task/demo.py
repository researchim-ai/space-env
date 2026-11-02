"""Runtime-backed training: EO → onboard CPU → ISL custody → ground ACK."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from space_env.tasks import MissionTaskEnv, TaskSession, TaskTools, export_dataset
from space_env.tasks.cli import apply_decisions


def build(mode="nominal"):
    path = Path(__file__).resolve().parents[1] / "relay_mission/demo.py"
    spec = importlib.util.spec_from_file_location("custody_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenario, _ = module.build_scenario(mode)
    scenario["lineage"] = "B06-three-actor-training"
    ops = scenario["definition"]["operations"]
    if mode == "reroute":
        ops["outages"] = [{"station": "Ground", "start_sec": 8.2, "end_sec": 8.4}]
    if mode == "loss":
        for seed in range(10000):
            samples = [
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}:@task/6/hop-second-b/0:{f}:{a}:ack".encode()
                    ).digest()[:8],
                    "big",
                )
                / 2**64
                for f, a in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]
            ]
            if samples[0] < 0.4 and all(v >= 0.4 for v in samples[1:]):
                ops["dtn_network"]["seed"] = seed
                break
        else:
            raise RuntimeError("loss witness seed not found")
    actions = []

    def command(id, actor, kind, duration, **kw):
        actions.append(
            {
                "id": id,
                "role": actor,
                "kind": "submit",
                "commands": [
                    {
                        "command_id": id,
                        "actor_id": actor,
                        "kind": kind,
                        "duration_sec": duration,
                        **kw,
                    }
                ],
            }
        )

    command("capture-first", "a", "imaging", 2.0, request_id="customer-a")
    command("compute", "a", "transform", 2.0, job_id="features-a")
    command(
        "hop-first-c",
        "a",
        "relay",
        2.0,
        product_id="final-a",
        destination_id="c",
        relay={"ttl_sec": 15.0},
    )
    command("capture-second", "a", "imaging", 2.0, request_id="after-thermal")
    for id, dest, ttl in [
        ("hop-second-b", "b", 12.0),
        ("hop-second-b-short", "b", 0.4),
        ("hop-second-c", "c", 20.0),
    ]:
        command(
            id,
            "a",
            "relay",
            4.0,
            product_id="raw-after",
            destination_id=dest,
            relay={"ttl_sec": ttl},
        )
    command(
        "ground-first-c",
        "c",
        "downlink",
        2.0,
        product_id="final-a",
        destination_id="Ground",
    )
    command(
        "ground-second-b",
        "b",
        "downlink",
        3.5,
        product_id="raw-after",
        destination_id="Ground",
    )
    command(
        "ground-second-c",
        "c",
        "downlink",
        3.5,
        product_id="raw-after",
        destination_id="Ground",
    )
    actions.append(
        {"id": "cancel-a", "role": "a", "kind": "cancel_actor", "actor": "a"}
    )
    task = {
        "id": "relay-training",
        "roles": {
            "a": {"actors": ["a"], "action_slots": 2},
            "b": {"actors": ["b"]},
            "c": {"actors": ["c"]},
            "operator": {"stations": ["Ground"]},
        },
        "actions": actions,
        "max_decision_sec": 10.0,
        "max_decisions": 32,
        "durations_sec": [0.15, 0.1501, 1.6999, 2.0, 1.0, 4.0, 3.5, 0.25, 10.0],
        "policy_origin": "scripted",
    }
    steps = [
        (["capture-first"], 2.0),
        (["compute"], 2.0),
        (["hop-first-c"], 0.15),
        ([], 0.1501),
        ([], 1.6999),
        (["capture-second"], 2.0),
        (
            [
                "hop-second-b-short" if mode == "ttl" else "hop-second-b",
                "ground-first-c",
            ],
            1.0 if mode == "reroute" else 4.0,
        ),
    ]
    if mode == "reroute":
        steps.append((["cancel-a", "hop-second-c"], 4.0))
    if mode != "ttl":
        steps.append(
            (["ground-second-c" if mode == "reroute" else "ground-second-b"], 3.5)
        )
    steps.extend([([], 0.25), ([], 10.0)])
    plan = {
        "decisions": [
            {
                "choices": {"@central": ids or ["wait"]},
                "advance_sec": dt,
                "expected_revision": i,
                "idempotency_key": f"teacher/{i}",
            }
            for i, (ids, dt) in enumerate(steps)
        ]
    }
    return scenario, task, plan


def verify(session, mode):
    report = session._runtime._backend.report()  # Trusted evaluator only.
    life, net = report["lifecycle"], report["dtn"]
    expected = 1 if mode == "ttl" else 2
    assert life["metrics"] == {"fulfilled_requests": expected, "total_requests": 2}
    assert abs(net["wire_balance_error_bits"]) < 1e-6
    assert all(abs(x) < 1e-6 for x in net["receiver_balance_error_bits"].values())
    for ledger in [report["inventory_ledger"], life["ledger"]]:
        assert all(abs(v["balance_error_bits"]) < 1e-6 for v in ledger.values())
    for ledger in report["spacecraft_bus"]["ledgers"].values():
        assert abs(ledger["battery_balance_error_wh"]) < 1e-9
        assert abs(ledger["thermal_balance_error_j"]) < 1e-6
    if mode == "loss":
        assert net["rx_duplicate_bits"]["b"] > 0
    if mode == "reroute":
        assert net["transfers"]["@task/6/hop-second-b/0"]["status"] == "cancelled"
        assert net["transfers"]["@task/7/hop-second-c/0"]["status"] == "completed"
    if mode == "ttl":
        assert life["artifacts"]["raw-after"]["status"] == "expired"
    return {
        "mode": mode,
        "fulfilled_requests": expected,
        "total_requests": 2,
        "decisions": session.revision,
        "wire_balance_error_bits": net["wire_balance_error_bits"],
    }


def main(directory, mode="nominal", parallel=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scenario, task, plan = build(mode)
    for name, value in [("scenario", scenario), ("task", task), ("plan", plan)]:
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    session = TaskSession(scenario, task)
    for i, decision in enumerate(plan["decisions"]):
        result = session.step(**decision)
        assert result["feedback"]["accepted"], result["feedback"]
        if i in {2, 3}:
            if i == 3:
                assert (
                    session._runtime._backend.mission.dtn.transfers[
                        "@task/2/hop-first-c/0"
                    ]["state"]
                    == "inflight"
                )
            path = directory / ("prefix.json" if i == 2 else "inflight.json")
            session.save(path)
            session = TaskSession.load(path)
    record = session.save(directory / "episode.json")
    assert TaskSession.from_checkpoint(record).checkpoint() == record
    expected = session._runtime.checkpoint()
    cli = TaskSession(scenario, task)
    apply_decisions(cli, plan)
    assert cli._runtime.checkpoint() == expected
    env = MissionTaskEnv(scenario, task, variable_duration=True)
    env.reset()
    tools_session = TaskSession(scenario, task)
    tools = TaskTools(tools_session)
    for decision in plan["decisions"]:
        ids = decision["choices"]["@central"]
        values = [env.action_ids.index(a) for a in ids]
        values += [0] * (task.get("max_joint_actions", 8) - len(values))
        _, _, _, _, info = env.step(
            {
                "choices": np.array(values),
                "duration": env.durations.index(decision["advance_sec"]),
            }
        )
        assert info["feedback"]["accepted"]
        assert tools.joint_step(**decision)["feedback"]["accepted"]
    assert env.session._runtime.checkpoint() == expected
    assert tools_session._runtime.checkpoint() == expected
    if parallel:
        from space_env.tasks.parallel import MissionParallelEnv

        penv = MissionParallelEnv(scenario, task, variable_duration=True)
        penv.reset()
        roles = {a["id"]: a["role"] for a in task["actions"]}
        for decision in plan["decisions"]:
            actions = {}
            for role in penv.agents:
                selected = [
                    penv.action_ids.index(a)
                    for a in decision["choices"]["@central"]
                    if roles.get(a) == role
                ]
                slots = task["roles"][role].get("action_slots", 1)
                selected += [0] * (slots - len(selected))
                actions[role] = {
                    "choices": selected[0] if slots == 1 else np.array(selected),
                    "duration": penv.durations.index(decision["advance_sec"]),
                }
            _, _, _, _, infos = penv.step(actions)
            assert all(v["feedback"]["accepted"] for v in infos.values())
        assert penv.session._runtime.checkpoint() == expected
    for format, role in [("rl", "@central"), ("sft", "@central"), ("rl", "a")]:
        export_dataset(
            record, directory / f"{format}-{role}.jsonl", format=format, role=role
        )
    result = verify(session, mode) | {
        "interface_parity": ["sdk", "cli-plan", "gym", "tools"]
        + (["parallel"] if parallel else []),
        "replay": ["mid-fragment", "inflight", "final"],
        "qualification": "synthetic scripted execution; not expert-policy or flight qualification",
    }
    print(json.dumps(result, allow_nan=False))
    return result


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        sys.argv[2] if len(sys.argv) > 2 else "nominal",
        "--parallel" in sys.argv,
    )
