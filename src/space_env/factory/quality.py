"""Native replay/accounting gates and conservative trajectory duplicate signatures."""

from __future__ import annotations

import math

from space_env.factory.semantic import fingerprint
from space_env.simulation.snapshot import canonical_json, digest

_PRIVATE = {
    "seed",
    "seeds",
    "episode_seed",
    "scenario",
    "truth",
    "runtime",
    "backend_report",
    "checkpoint",
    "rng",
}


def public_view_gate(value):
    if isinstance(value, dict):
        if set(value) & _PRIVATE:
            raise ValueError("privileged field in policy view")
        for child in value.values():
            public_view_gate(child)
    elif isinstance(value, list):
        for child in value:
            public_view_gate(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite policy feature")


def quality_report(run):
    """The caller must cold-replay PlanningRun before evaluating this report."""
    native = run.session._runtime.observe(view="truth")["backend"]["report"]
    checks = []

    def ledger(name, rows, field, tolerance):
        for key, row in rows.items():
            error = row[field]
            if not math.isfinite(error) or abs(error) > tolerance:
                raise ValueError("conservation gate failed: " + name + "/" + key)
        checks.append(name)

    ledger("inventory_bits", native["inventory_ledger"], "balance_error_bits", 1e-6)
    ledger("battery_wh", native["energy_ledger"], "balance_error_wh", 1e-9)
    if "lifecycle" in native:
        ledger(
            "lifecycle_bits", native["lifecycle"]["ledger"], "balance_error_bits", 1e-6
        )
    if "spacecraft_bus" in native:
        ledger(
            "bus_battery_wh",
            native["spacecraft_bus"]["ledgers"],
            "battery_balance_error_wh",
            1e-9,
        )
        ledger(
            "bus_thermal_j",
            native["spacecraft_bus"]["ledgers"],
            "thermal_balance_error_j",
            1e-6,
        )
    if "dtn" in native:
        errors = [
            native["dtn"]["wire_balance_error_bits"],
            *native["dtn"]["receiver_balance_error_bits"].values(),
        ]
        if any(not math.isfinite(x) or abs(x) > 1e-6 for x in errors):
            raise ValueError("DTN wire/receiver conservation gate failed")
        checks.append("dtn_bits")
    acked = (
        math.fsum(
            p["delivered_bits"]
            for p in native["products"]
            if p["status"] in {"delivered_on_time", "delivered_late"}
        )
        / 1e6
    )
    transitions = run.session._transitions
    mbits = math.fsum(
        r["transition"]["reward_components"]["mbits"] for r in transitions
    )
    completion = math.fsum(
        r["transition"]["reward_components"]["completion"] for r in transitions
    )
    expected = (
        native["lifecycle"]["metrics"]["fulfilled_requests"]
        if "lifecycle" in native
        else native["on_time_bits"] / 1e6
    )
    if not math.isclose(mbits, acked, rel_tol=1e-12, abs_tol=1e-9) or not math.isclose(
        completion, expected, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("reward/native ACK gate failed")
    conditional_masks = 0
    actions = {a.id: a for a in run.session._actions}
    for row in transitions:
        t = row["transition"]
        if not t["elapsed_sec"] > 0 or not 0 <= t["discount"] <= 1:
            raise ValueError("physical transition interval/discount gate failed")
        selected = [
            a for ids in row["request"]["choices"].values() for a in ids if a != "wait"
        ]
        enabled = {a["id"] for a in t["observation"]["catalogue"] if a["enabled"]}
        masked = set(selected) - enabled
        if t["feedback"]["accepted"] and masked:
            if not any(actions[a].kind == "cancel_actor" for a in selected):
                raise ValueError("accepted action bypassed causal mask")
            conditional_masks += 1  # B06 explicitly rechecks after cancellation.
        for view in [
            *t["role_observations"].values(),
            *t["next_role_observations"].values(),
        ]:
            public_view_gate(view)
            canonical_json(view)
    return {
        "passed": True,
        "checks": checks
        + [
            "native_ACK_reward",
            "causal_public_views",
            "physical_dt_discount",
            "accepted_masks",
        ],
        "conditional_after_cancel": conditional_masks,
        "transitions": len(transitions),
    }


def _view_signature(view, cache):
    catalogue_key = digest(view["catalogue"])
    if catalogue_key not in cache:
        cache[catalogue_key] = fingerprint({"actions": view["catalogue"]})
    by_index = {
        e["index"]: {"kind": e["kind"], "values": e["values"]} for e in view["entities"]
    }
    return {
        "time_sec": view["time_sec"],
        "catalogue_sha256": cache[catalogue_key],
        "entities": sorted(by_index.values(), key=canonical_json),
        "relations": sorted(
            [
                {
                    "kind": r["kind"],
                    "source_values": by_index[r["source"]],
                    "destination_values": by_index[r["destination"]],
                }
                for r in view["relations"]
            ],
            key=canonical_json,
        ),
    }


def trajectory_fingerprint(run, case, role):
    actions = {a.id: a.model_dump(mode="json") for a in run.session._actions}
    steps, cache = [], {}
    for row in run.session._transitions:
        request, t = row["request"], row["transition"]
        ids = [
            a
            for scope, ids in request["choices"].items()
            for a in ids
            if a != "wait"
            and (
                role == "@central"
                or scope == role
                or (scope == "@central" and actions[a]["role"] == role)
            )
        ]
        steps.append(
            {
                "observation_sha256": digest(
                    _view_signature(t["role_observations"][role], cache)
                ),
                "next_observation_sha256": digest(
                    _view_signature(t["next_role_observations"][role], cache)
                ),
                "actions": [actions[a] for a in ids],
                "elapsed_sec": t["elapsed_sec"],
                "reward": t["reward"],
                "discount": t["discount"],
                "requested_advance_sec": request["advance_sec"],
                "feedback": t["feedback"]
                if role == "@central" or t["feedback"]["role"] in {None, role}
                else {"accepted": False, "code": "joint_rejected"},
                "terminated": t["terminated"],
                "truncated": t["truncated"],
            }
        )
    return fingerprint({"steps": steps})


def sft_reasons(spec, evaluation, row):
    reasons = []
    if row["policy"] not in spec.sft_policies:
        reasons.append("not_selected_sft_policy")
    m = row["metrics"]
    if (
        m.get("completion_fraction") is None
        or m["completion_fraction"] < spec.quality.sft_min_completion
    ):
        reasons.append("incomplete_teacher_episode")
    if m.get("rejections", 0) > spec.quality.sft_max_rejections:
        reasons.append("rejected_teacher_actions")
    if spec.quality.require_teacher_gates and (
        evaluation["teacher"]["status"] != "passed_declared_gates"
        or spec.teacher_gates is None
        or row["policy"] != spec.teacher_gates.candidate
    ):
        reasons.append("teacher_gates_not_passed_for_policy")
    return reasons


def semantic_metrics(metrics):
    return {
        k: v
        for k, v in metrics.items()
        if k not in {"decision_wall_sec", "process_peak_rss_bytes"}
    }


def content_digest(catalogue, data_hashes, audit):
    return digest(
        {
            "catalogue_sha256": catalogue["sha256"],
            "data_sha256": data_hashes,
            "audit": audit,
        }
    )
