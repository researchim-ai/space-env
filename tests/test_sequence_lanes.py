"""Sequence pointing, native lane ownership and full-goal training gates."""

import copy
import json

import pytest

from space_env.llm.benchmark import reference_policy, run_agent
from space_env.llm.episode import LLMEpisode, export_training, verify_record
from space_env.llm.mission_suite import compile_suite
from space_env.planning.sequence_lanes import (
    _Ledger,
    schedule_sequence,
    seal,
    verify_sequence_schedule,
)
from space_env.simulation.snapshot import canonical_json
from space_env.tasks import TaskSession
from space_env.workspace import MissionWorkspace


@pytest.fixture(scope="module")
def native_sequence():
    config = compile_suite()["cases"][0]["configuration"]
    m = MissionWorkspace.network(config, planning_method="sequence_lanes")
    try:
        session = m.operator.session
        meta = copy.deepcopy(session._scenario["metadata"]["network_workspace"])
        agent = LLMEpisode(
            session,
            task_text="Deliver all three products.",
            group="sequence-native",
            split="train",
            policy_origin="scripted_privileged",
        )
        record = run_agent(
            agent,
            reference_policy(
                meta["steps"], {a.id: a.role for a in session._task.actions}
            ),
        )
        verify_record(record)
        return meta, record
    finally:
        m.close()


def test_same_frozen_case_reaches_full_native_goal_and_positive_sft(
    native_sequence, tmp_path
):
    meta, record = native_sequence
    assert (
        meta["schedule"]["predicted_fulfilled"]
        == record["metrics"]["fulfilled"]
        == record["metrics"]["total"]
        == 3
    )
    assert record["metrics"]["success"] and record["metrics"]["sft_eligible"]
    assert not record["metrics"]["failed_calls"]
    assert export_training(record, tmp_path / "sft.jsonl", format="sft")["rows"] == 1
    assert (
        json.loads((tmp_path / "sft.jsonl").read_text())["policy_origin"]
        == "scripted_privileged"
    )
    assert meta["schedule"]["search_nodes"] <= 50000
    assert meta["schedule"]["optimal"] is False
    assert verify_sequence_schedule(
        meta["forecast"], meta["sequence_geometry"], meta["schedule"]
    )["valid"]
    # The fixed scalar wire protocol can issue every planned start.
    actions = record["initial_checkpoint"]["task"]["actions"]
    roles = {a["id"]: a["role"] for a in actions}
    for step in meta["steps"]:
        selected = [roles[a] for a in step["action_ids"] if a != "wait"]
        assert len(selected) == len(set(selected))


def test_pointing_reservation_depends_on_preceding_capture(native_sequence):
    meta, _ = native_sequence
    allocations = meta["schedule"]["allocations"]
    assert allocations[0]["stages"][0]["cells"] == 10
    assert all(a["stages"][0]["cells"] < 10 for a in allocations[1:])
    assert all(
        a["energy_wh"][a["owner"]]
        >= next(
            d["owner_wh"]
            for d in meta["forecast"]["demands"]
            if d["id"] == a["request"]
        )
        for a in allocations
    )


def test_ledger_allows_cpu_and_attitude_but_not_relay_conflicts(native_sequence):
    meta, _ = native_sequence
    v, g = meta["forecast"], meta["sequence_geometry"]
    r = v["demands"][0]
    a, b = list(v["actors"])[:2]
    ledger = _Ledger(v, g)
    assert ledger.reserve(
        r, {"kind": "transform", "actor": a, "cell": 0, "cells": 4}, "Ground"
    )
    assert ledger.reserve(
        r, {"kind": "slew", "actor": a, "cell": 1, "cells": 2}, "Ground"
    )
    assert not ledger.reserve(
        r,
        {"kind": "relay", "actor": a, "destination": b, "cell": 2, "cells": 2},
        "Ground",
    )
    assert not ledger.reserve(
        r,
        {"kind": "relay", "actor": b, "destination": a, "cell": 2, "cells": 2},
        "Ground",
    )
    assert not ledger.reserve(
        r, {"kind": "downlink", "actor": a, "cell": 1, "cells": 1}, "Ground"
    )  # one start per tick


@pytest.mark.parametrize("resource", ["power", "bandwidth"])
def test_shared_resource_limits_reject_overlap_without_mutation(
    native_sequence, resource
):
    meta, _ = native_sequence
    v = meta["forecast"]
    g = copy.deepcopy(meta["sequence_geometry"])
    r = v["demands"][0]
    a = next(iter(v["actors"]))
    if resource == "power":
        g["actors"][a]["max_load_w"] = (
            g["actors"][a]["base_w"]
            + g["io"][r["id"]]["cpu_w"]
            + g["actors"][a]["imaging_w"]
            - 1
        )
    else:
        g["actors"][a]["bandwidth_bps"] = (
            g["io"][r["id"]]["cpu_bps"] + g["io"][r["id"]]["imaging_bps"] - 1
        )
    ledger = _Ledger(v, g)
    assert ledger.reserve(
        r, {"kind": "transform", "actor": a, "cell": 0, "cells": 4}, "Ground"
    )
    before = copy.deepcopy(ledger.occupied)
    assert not ledger.reserve(
        r, {"kind": "imaging", "actor": a, "cell": 1, "cells": 1}, "Ground"
    )
    assert ledger.occupied == before and 1 not in ledger.starts[a]


@pytest.mark.parametrize(
    "fault",
    [
        "width",
        "order",
        "predecessor",
        "cost",
        "optimal",
        "coverage",
        "budget",
        "geometry",
    ],
)
def test_resealed_schedule_cannot_change_physical_witness(native_sequence, fault):
    meta, _ = native_sequence
    v = meta["forecast"]
    g = copy.deepcopy(meta["sequence_geometry"])
    s = copy.deepcopy(meta["schedule"])
    if fault == "width":
        s["allocations"][1]["stages"][0]["cells"] = 0
    elif fault == "order":
        s["allocations"][0]["stages"][2]["cell"] = 0
    elif fault == "predecessor":
        s["allocations"].reverse()
    elif fault == "cost":
        s["allocations"][0]["memory_bits"][s["allocations"][0]["owner"]] = 0
    elif fault == "optimal":
        s["optimal"] = True
    elif fault == "coverage":
        s["predicted_fulfilled"] = 999
    elif fault == "budget":
        s["search_nodes"] = s["max_search_nodes"] + 1
    else:
        g["actors"][next(iter(g["actors"]))]["claims"]["relay"] = ["radio"]
        g = seal({k: x for k, x in g.items() if k != "sha256"})
        s["geometry_sha256"] = g["sha256"]
    s = seal({k: x for k, x in s.items() if k != "sha256"})
    with pytest.raises(ValueError):
        verify_sequence_schedule(v, g, s)


@pytest.mark.parametrize("budget", [1, 20])
def test_search_interruption_keeps_valid_incumbent_and_never_claims_optimum(
    native_sequence, budget
):
    meta, _ = native_sequence
    p = schedule_sequence(
        meta["forecast"], meta["sequence_geometry"], max_nodes=budget, max_orders=2
    )
    assert (
        p["search_nodes"] <= budget
        and p["search_budget_exhausted"]
        and not p["optimal"]
    )
    assert verify_sequence_schedule(meta["forecast"], meta["sequence_geometry"], p)[
        "valid"
    ]


def test_native_cpu_and_slew_windows_really_overlap():
    c = compile_suite()["cases"][1]["configuration"]
    m = MissionWorkspace.network(c, planning_method="sequence_lanes")
    try:
        p = m.operator.session._scenario["metadata"]["network_workspace"]["schedule"]
        stages = [s for a in p["allocations"] for s in a["stages"]]
        assert any(
            a["actor"] == b["actor"]
            and max(a["cell"], b["cell"])
            < min(a["cell"] + a["cells"], b["cell"] + b["cells"])
            for a in stages
            if a["kind"] == "transform"
            for b in stages
            if b["kind"] == "slew"
        )
        state = m.run()["network"]
        assert state["fulfilled"] == p["predicted_fulfilled"]
        from space_env.reference.spacecraft_bus import quaternion_angle

        # Genuine native work, not just overlap of conservative reservations:
        # the attitude changed during a tick fully covered by granted CPU work.
        intervals = m.operator.session._runtime._backend.report()["lifecycle"][
            "resource_lanes"
        ]["intervals"]
        concurrent = False
        for transition in m.operator.session._transitions:
            t = transition["transition"]
            left = t["role_observations"]
            right = t["next_role_observations"]
            for actor in (
                a for a in left if a in m.operator.session._scenario["actor_order"]
            ):

                def quaternion(view, actor=actor):
                    values = next(
                        e["values"]
                        for e in view["entities"]
                        if e["kind"] == "actor" and e["id"] == actor
                    )
                    return [values[k] for k in ("q_w", "q_x", "q_y", "q_z")]

                if (
                    quaternion_angle(quaternion(left[actor]), quaternion(right[actor]))
                    < 1e-5
                ):
                    continue
                start, end = left[actor]["time_sec"], right[actor]["time_sec"]
                work = sum(
                    max(
                        0.0,
                        min(end, interval["end_sec"])
                        - max(start, interval["start_sec"]),
                    )
                    for interval in intervals
                    if any(
                        op["kind"] == "transform"
                        and op["phase"] == "work"
                        and op["work_rate"] > 0
                        for op in interval["actors"][actor]["managed"]
                    )
                )
                concurrent |= work >= end - start - 1e-7
        assert concurrent

        assert not any(
            r["transition"]["feedback"].get("accepted") is False
            for r in m.operator.session._transitions
        )
    finally:
        m.close()


def test_recovery_cannot_keep_sequence_geometry_and_source_stays_unchanged():
    m = MissionWorkspace.network(planning_method="sequence_lanes")
    n = None
    try:
        m.run(decisions=1)
        before = m.operator.session.checkpoint(compact=True)
        n = m.evolve("Conservative recovery.", idempotency_key="recover-sequence")
        assert n.state()["network"]["sequence_lanes"] is None
        assert (
            "sequence_geometry"
            not in n.operator.session._scenario["metadata"]["network_workspace"]
        )
        assert m.operator.session.checkpoint(compact=True) == before
        TaskSession.from_checkpoint(n.operator.session.checkpoint(compact=True))
    finally:
        if n is not None:
            n.close()
        m.close()


def test_operator_geometry_does_not_enter_llm_messages(native_sequence):
    _, record = native_sequence
    text = canonical_json(record["messages"])
    assert (
        "sequence_geometry" not in text
        and "rate_deg_s" not in text
        and "goals" not in text
    )


def test_capture_tracking_uncertainty_is_reserved_before_next_slew(native_sequence):
    from space_env.planning.sequence_lanes import _slew_cells

    meta, _ = native_sequence
    g = meta["sequence_geometry"]
    owner = next(iter(g["actors"]))
    pose = [1.0, 0.0, 0.0, 0.0]
    assert _slew_cells(g, owner, pose, pose, 5.0, 0.0) == 1
    assert _slew_cells(g, owner, pose, pose, 5.0, 10.0) == 2
    # A midpoint pose alone is not the native pose after target tracking.
    assert all(
        a["stages"][0]["cells"] >= 2 for a in meta["schedule"]["allocations"][1:]
    )
