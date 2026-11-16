"""Native finite burn, delivery, then a separate retained-delivery phase.

Small public synthetic continuity example; not full operational mission validation.
"""

import argparse
import copy
from pathlib import Path

from space_env.navigation.finite_suite import compile_suite
from space_env.simulation.snapshot import atomic_write, canonical_json
from space_env.tasks.campaign import CampaignSession


def recipe(kind="finite"):
    if kind == "eo":
        from space_env.tasks.modeling_suite import compile_suite as physical

        cases = {c["id"]: c for c in physical()["cases"]}
        case = copy.deepcopy(cases["delivery"])
        pointing = copy.deepcopy(cases["attitude-nominal"])
        delivery = case["task"].pop("goal")
        delivery["deadline_sec"] = 20.0
        pointing["task"]["goal"]["deadline_sec"] = 30.0
        pointing["task"]["actions"][0]["commands"][0]["duration_sec"] = 20.0
        case["task"]["actions"].extend(pointing["task"]["actions"])
        case["campaign_steps"] = case["reference_steps"] + pointing["reference_steps"]
        return case, {
            "id": "eo-delivery-reorientation",
            "phases": [
                {"id": "acquire-process-deliver", "goal": delivery},
                {"id": "reorient", "goal": pointing["task"]["goal"]},
            ],
        }
    if kind != "finite":
        raise ValueError("unknown campaign recipe")
    case = copy.deepcopy(compile_suite()[0])
    delivery = case["task"].pop("goal")
    delivery["deadline_sec"] = 4.0
    retained = copy.deepcopy(delivery)
    retained["deadline_sec"] = 6.0
    retained["conditions"][0]["not_before_sec"] = 4.0
    campaign = {
        "id": "finite-delivery-continuity",
        "phases": [
            {"id": "deliver", "goal": delivery},
            {"id": "verify-retained-product", "goal": retained},
        ],
    }
    return case, campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mission", choices=["finite", "eo"], default="finite")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case, spec = recipe(args.mission)
    rows = []
    for policy in ["wait", "reference"]:
        session = CampaignSession(case["scenario"], case["task"], spec)
        schedule = case.get(
            "campaign_steps",
            [
                {"action_ids": ["burn"], "advance_sec": 1.0},
                {"action_ids": ["deliver"], "advance_sec": 2.0},
                {"action_ids": ["wait"], "advance_sec": 1.0},
            ],
        )
        while not session.status()["terminal"]:
            index = session.revision
            item = (
                schedule[index]
                if index < len(schedule)
                else {"action_ids": ["wait"], "advance_sec": 1.0}
            )
            actions, duration = item["action_ids"], item["advance_sec"]
            if policy == "wait":
                actions = ["wait"]
            session.step(
                {"@central": actions},
                expected_revision=index,
                idempotency_key=str(index),
                advance_sec=duration,
            )
        checkpoint = session.checkpoint()
        assert CampaignSession.from_checkpoint(checkpoint).checkpoint() == checkpoint
        atomic_write(args.output / (policy + ".json"), canonical_json(checkpoint))
        rows.append(
            {
                "policy": policy,
                "status": session.status(),
                "checkpoint_sha256": checkpoint["sha256"],
                "cold_replay": True,
            }
        )
    atomic_write(
        args.output / "report.json",
        canonical_json(
            {
                "scope": "two fixed scripted synthetic runs, not independent physical validation or agent skill benchmark",
                "rows": rows,
            }
        ),
    )
    print(canonical_json(rows))


if __name__ == "__main__":
    main()
