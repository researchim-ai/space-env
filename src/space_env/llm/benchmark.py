"""Frozen local catalogue-execution benchmark for LLM tool agents.

Scripted baselines are not LLM evaluations. The reference schedule is privileged;
public policies receive only messages/tools and never its timing. External
simulators require matched task/physics contracts and measured runs, not badges.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from space_env.api.network_workspace import create_network
from space_env.api.operator import OperatorStore
from space_env.llm.episode import (
    STEP,
    LLMEpisode,
    export_training,
    seal,
    sources,
    verify_record,
)
from space_env.planning.network_benchmark import NetworkSuite
from space_env.planning.network_demand_benchmark import demand_suite
from space_env.simulation.snapshot import atomic_write, canonical_json, digest

POLICIES = ("wait", "public_memory", "reference_schedule")


def run_agent(episode, policy):
    """Provider-independent callable(messages, tools) -> {name, arguments} or None.

    A provider adapter owns API credentials and latency/token accounting. No
    model/provider is called automatically. Exceptions propagate to the harness.
    """
    while not episode.done:
        call = policy(episode.messages, episode.tool_schemas())
        if call is None:
            break
        episode.invoke(
            call["name"], call["arguments"], call_id=f"call-{len(episode._events)}"
        )
    return episode.record()


def public_policy(method, decision_sec):
    if method not in ("wait", "public_memory"):
        raise ValueError("Unknown public policy")
    used = set()

    def choose(messages, tools):
        last = json.loads(messages[-1]["content"])
        observation = last["observation"]
        choices = {}
        for role, view in observation["roles"].items():
            available = [
                a["id"]
                for a in view["catalogue"]
                if a["id"] != "wait" and a["enabled"] and a["id"] not in used
            ]
            selected = available[:1] if method == "public_memory" else []
            used.update(selected)
            choices[role] = selected or ["wait"]
        return {
            "name": STEP,
            "arguments": canonical_json(
                {
                    "choices": choices,
                    "advance_sec": decision_sec,
                    "expected_revision": observation["revision"],
                    "idempotency_key": f"d/{observation['revision']}",
                }
            ),
        }

    return choose


def reference_policy(steps, action_roles):
    """Privileged demonstration source, explicitly excluded from public agent rank."""
    cursor = 0

    def choose(messages, tools):
        nonlocal cursor
        if cursor >= len(steps):
            return None
        observation = json.loads(messages[-1]["content"])["observation"]
        step = steps[cursor]
        cursor += 1
        choices = {}
        for aid in step["action_ids"]:
            if aid != "wait":
                choices.setdefault(action_roles[aid], []).append(aid)
        if not choices:
            choices = {role: ["wait"] for role in observation["roles"]}
        return {
            "name": STEP,
            "arguments": canonical_json(
                {
                    "choices": choices,
                    "advance_sec": step["advance_sec"],
                    "expected_revision": observation["revision"],
                    "idempotency_key": f"reference/{cursor}",
                }
            ),
        }

    return choose


def summarize(suite, rows):
    spec = NetworkSuite.model_validate(suite)
    expected = {(c.id, p) for c in spec.cases for p in POLICIES}
    if (
        len(rows) != len(expected)
        or {(r["case"], r["policy"]) for r in rows} != expected
    ):
        raise ValueError("Benchmark requires every case/policy, including failures")
    cases = {c.id: c for c in spec.cases}
    for row in rows:
        case = cases[row["case"]]
        if (
            row["split"] != case.split
            or row["group"] != case.group
            or row["priority_total"]
            != sum(r.priority for r in case.configuration.mission.requests)
            or row["status"] not in ("completed", "error")
        ):
            raise ValueError("Benchmark case/split/denominator mismatch")
        if (
            row["status"] == "completed"
            and not 0 <= row["metrics"]["weighted_fulfilled"] <= row["priority_total"]
        ):
            raise ValueError("Invalid benchmark value")
    aggregates = []
    for split in ("train", "development", "test"):
        for policy in POLICIES:
            selected = [
                r for r in rows if r["split"] == split and r["policy"] == policy
            ]
            values = [
                r["metrics"]["weighted_fulfilled"] / max(1, r["priority_total"])
                if r["status"] == "completed"
                else None
                for r in selected
            ]
            aggregates.append(
                {
                    "split": split,
                    "policy": policy,
                    "rows": len(values),
                    "completed": sum(v is not None for v in values),
                    "value_lower": sum(v or 0 for v in values) / len(values),
                    "value_upper": sum(1 if v is None else v for v in values)
                    / len(values),
                    "privileged": policy == "reference_schedule",
                    "external_llm_measured": False,
                }
            )
    return {
        "aggregates": aggregates,
        "competitor_superiority_established": False,
        "llm_training_gain_established": False,
    }


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if (
        report.get("sha256")
        != digest({k: v for k, v in report.items() if k != "sha256"})
        or report["model_sources"] != sources()
    ):
        raise ValueError("Report seal/source mismatch")
    spec = NetworkSuite.model_validate(report["suite"])
    if report["suite"] != demand_suite():
        raise ValueError("Frozen suite mismatch")
    cases = {c.id: c for c in spec.cases}
    for row in report["rows"]:
        case = cases[row["case"]]
        if (
            row["split"] != case.split
            or row["group"] != case.group
            or row["priority_total"]
            != sum(r.priority for r in case.configuration.mission.requests)
        ):
            raise ValueError("Case split/group/denominator mismatch")
        if row["status"] == "completed":
            record = json.loads(
                (directory / (row["case"] + "--" + row["policy"] + ".json")).read_text()
            )
            verify_record(record)
            native_config = record["initial_checkpoint"]["scenario"]["metadata"][
                "network_workspace"
            ]["configuration"]
            if native_config != case.configuration.model_dump(
                mode="json", by_alias=True
            ):
                raise ValueError("Native scenario/configuration mismatch")
            from space_env.simulation.streaming import (
                array_digest,
                file_digest,
                iter_jsonl,
            )

            expected_files = (
                {
                    case.id + "--" + row["policy"] + "--" + fmt + ".jsonl"
                    for fmt in (
                        ("rl", "sft") if record["metrics"]["sft_eligible"] else ("rl",)
                    )
                }
                if case.split == "train"
                else set()
            )
            if set(row.get("training_files", {})) != expected_files:
                raise ValueError("Missing/unexpected training files")
            for name, receipt in row.get("training_files", {}).items():
                if case.split != "train" or name not in {
                    case.id + "--" + row["policy"] + "--" + fmt + ".jsonl"
                    for fmt in ("rl", "sft")
                }:
                    raise ValueError("Invalid training artifact identity")
                from space_env.llm.episode import _training_row

                path = directory / name
                fmt = "sft" if name.endswith("--sft.jsonl") else "rl"
                if list(iter_jsonl(path, max_row_bytes=64000000)) != [
                    _training_row(record, format=fmt)
                ]:
                    raise ValueError("Training rows differ from verified dialogue")
                if (
                    file_digest(path) != receipt["file_sha256"]
                    or path.stat().st_size != receipt["bytes"]
                    or array_digest(iter_jsonl(path, max_row_bytes=64000000))
                    != receipt["rows_sha256"]
                ):
                    raise ValueError("Training artifact checksum mismatch")
            if (
                row["episode_sha256"] != record["sha256"]
                or row["metrics"] != record["metrics"]
                or row["initial_checkpoint_sha256"]
                != record["initial_checkpoint"]["sha256"]
                or record["configuration"]["max_calls"] != 64
                or record["configuration"]["policy_origin"]
                != (
                    "scripted_privileged"
                    if row["policy"] == "reference_schedule"
                    else "scripted_public"
                )
                or record["configuration"]["group"] != case.group
                or record["configuration"]["split"] != case.split
            ):
                raise ValueError("Episode/split/result mismatch")
    for case in spec.cases:
        fingerprints = {
            r["initial_checkpoint_sha256"]
            for r in report["rows"]
            if r["case"] == case.id and r["status"] == "completed"
        }
        if len(fingerprints) > 1:
            raise ValueError("Policies did not receive identical initial native tasks")
    summary = summarize(report["suite"], report["rows"])
    if any(report[k] != v for k, v in summary.items()):
        raise ValueError("Summary mismatch")
    return {
        "valid": True,
        "rows": len(report["rows"]),
        "native_dialogues_replayed": sum(
            r["status"] == "completed" for r in report["rows"]
        ),
        "competitor_superiority_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(canonical_json(audit(args.audit)))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    suite = demand_suite()
    atomic_write(args.output / "suite.json", canonical_json(seal(suite)))
    rows = []
    for case in NetworkSuite.model_validate(suite).cases:
        store = OperatorStore()
        connection = create_network(case.configuration, store)
        operator = store.get(
            connection["session_id"] if "session_id" in connection else connection["id"]
        )
        task = operator.session._task
        meta = operator.session._scenario["metadata"]["network_workspace"]
        initial = operator.session
        for policy in POLICIES:
            row = {
                "case": case.id,
                "group": case.group,
                "split": case.split,
                "policy": policy,
                "priority_total": sum(
                    r.priority for r in case.configuration.mission.requests
                ),
            }
            try:
                episode = LLMEpisode(
                    initial,
                    task_text="Complete the released observation/processing/delivery requests using the declared actor command catalogue. Maximize priority delivered before deadlines.",
                    group=case.group,
                    split=case.split,
                    max_calls=64,
                    policy_origin="scripted_privileged"
                    if policy == "reference_schedule"
                    else "scripted_public",
                )
                choose = (
                    reference_policy(
                        meta["steps"], {a.id: a.role for a in task.actions}
                    )
                    if policy == "reference_schedule"
                    else public_policy(policy, task.decision_sec)
                )
                record = run_agent(episode, choose)
                verify_record(record)
                path = args.output / (case.id + "--" + policy + ".json")
                atomic_write(path, canonical_json(record))
                row.update(
                    status="completed",
                    episode_sha256=record["sha256"],
                    initial_checkpoint_sha256=record["initial_checkpoint"]["sha256"],
                    metrics=record["metrics"],
                )
                if case.split == "train":
                    row["training_files"] = {}
                    for fmt in (
                        ("rl", "sft") if record["metrics"]["sft_eligible"] else ("rl",)
                    ):
                        name = case.id + "--" + policy + "--" + fmt + ".jsonl"
                        row["training_files"][name] = export_training(
                            record, args.output / name, format=fmt
                        )
            except Exception as exc:  # noqa: BLE001 -- retain every failed benchmark cell
                row.update(status="error", error_type=type(exc).__name__, metrics=None)
            rows.append(row)
            atomic_write(
                args.output / "progress.json", canonical_json(seal({"rows": rows}))
            )
            print(case.id, policy, row["status"], row.get("metrics"), flush=True)
        # Local in-memory store; no service or operator messages are sent.
    report = seal(
        {
            "schema": "space-env.llm-catalogue-benchmark/1",
            "suite": suite,
            "rows": rows,
            "model_sources": sources(),
            "external_backends": {
                "bsk_rl": {
                    "installed": importlib.util.find_spec("bsk_rl") is not None,
                    "measured": False,
                    "status": "not_run_no_matched_contract",
                }
            },
            "scope": "operator-compiled catalogue execution; constrained choice of supplied commands, not free-form mission synthesis; scripted baselines not LLM scores; no competitor performance claim",
            **summarize(suite, rows),
        }
    )
    atomic_write(args.output / "report.json", canonical_json(report))
    return 0 if all(r["status"] == "completed" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
