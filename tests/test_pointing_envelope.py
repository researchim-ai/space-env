"""Quaternion cone safety, honest forecast diagnostics and actual native delivery."""

import copy
import math

import numpy as np
import pytest

from space_env.core.attitude import slew_step
from space_env.llm.mission_suite import compile_suite
from space_env.planning.joint_network import schedule_network
from space_env.planning.pointing_envelope import (
    collect_pointing_envelope,
    cone_angle,
    diagnose_network,
    sealed,
)
from space_env.reference.spacecraft_bus import quaternion_angle
from space_env.tasks import TaskSession
from space_env.workspace import MissionWorkspace
from tests.test_joint_network import view


def test_cone_bounds_native_interrupted_slews_and_changes_of_target():
    rng = np.random.default_rng(3)
    q = np.column_stack([np.ones(20), rng.uniform(-0.3, 0.3, (20, 3))])
    q /= np.linalg.norm(q, axis=1)[:, None]
    bound = cone_angle(q)
    assert not bound["fallback"] and 0 < bound["angle_deg"] < 180
    current = q[0]
    for _ in range(300):
        goal = q[rng.integers(len(q))]
        current, _ = slew_step(current, goal, 2.0, float(rng.uniform(0, 50)))
        assert all(
            quaternion_angle(current, target) <= bound["angle_deg"] + 1e-8
            for target in q
        )
    assert cone_angle(-q) == bound


def test_nonacute_cone_falls_back_instead_of_assuming_convex_rotation_ball():
    angle = math.radians(80)
    q = [
        [1, 0, 0, 0],
        [math.cos(angle), math.sin(angle), 0, 0],
        [math.cos(angle), -math.sin(angle), 0, 0],
    ]
    assert cone_angle(q)["fallback"] and cone_angle(q)["angle_deg"] == 180


@pytest.mark.parametrize(
    "q",
    [
        [],
        [[1, 0, 0]],
        [[2, 0, 0, 0]],
        [[float("nan"), 0, 0, 0]],
        [[0, 0, 0, 0]],
        [[1, 0, 0, 0]] * 4097,
    ],
)
def test_invalid_goal_sets_are_rejected(q):
    with pytest.raises(ValueError):
        cone_angle(q)


def test_diagnostics_do_not_turn_singleton_search_failure_into_physical_proof():
    v = view()
    p = schedule_network(v, max_nodes=1)
    d = diagnose_network(v, p)
    assert not d["physical_infeasibility_established"]
    assert d["requests"][0]["isolated_search_exhausted"]
    assert not d["requests"][0]["isolated_witness"]


def test_prefix_capacity_certificate_belongs_to_forecast_not_physics():
    v = view()
    original = v["demands"][0]
    v["demands"] = [
        {**copy.deepcopy(original), "id": f"r{i}", "owners": ["a"]} for i in range(3)
    ]
    v["captures"] = [
        {"request": r["id"], "actor": "a", "cells": [True] * 8} for r in v["demands"]
    ]
    v = sealed({k: x for k, x in v.items() if k != "sha256"})
    p = schedule_network(v)
    d = diagnose_network(v, p)
    assert d["prefix_bounds"][0]["required_cells"] == 9
    assert d["prefix_bounds"][0]["available_cells"] == 8
    assert all(r["isolated_witness"] for r in d["requests"])
    assert not d["physical_infeasibility_established"]


@pytest.mark.parametrize(
    "fault", ["seal", "binding", "unknown-request", "duplicate-request"]
)
def test_invalid_diagnostic_schedule_is_rejected(fault):
    v = view()
    p = schedule_network(v)
    if fault == "seal":
        p["sha256"] = "0" * 64
    else:
        if fault == "binding":
            p["forecast_sha256"] = "0" * 64
        elif fault == "unknown-request":
            p["allocations"][0]["request"] = "invented"
        else:
            p["allocations"].append(copy.deepcopy(p["allocations"][0]))
        p = sealed({k: x for k, x in p.items() if k != "sha256"})
    with pytest.raises(ValueError):
        diagnose_network(v, p)


@pytest.fixture(scope="module")
def native_case():
    config = compile_suite()["cases"][0]["configuration"]
    original = MissionWorkspace.network(config)
    changed = MissionWorkspace.network(config, planning_method="pointing_bounded")
    try:
        old = copy.deepcopy(
            original.operator.session._scenario["metadata"]["network_workspace"]
        )
        new = copy.deepcopy(
            changed.operator.session._scenario["metadata"]["network_workspace"]
        )
        initial = changed.operator.session.checkpoint(compact=True)
        actual = changed.run()["network"]
        final = changed.operator.session.checkpoint(compact=True)
        return config, old, new, initial, actual, final
    finally:
        original.close()
        changed.close()


def test_tracking_guard_preserves_admission_and_energy_reserve(
    native_case,
):
    config, old, new, _initial, actual, final = native_case
    assert old["configuration"] == new["configuration"] == config
    assert old["schedule"]["predicted_fulfilled"] == 1
    assert new["schedule"]["predicted_fulfilled"] == actual["fulfilled"] == 1
    assert [d["slew_cells"] for d in old["forecast"]["demands"]] == [19] * 3
    assert [d["slew_cells"] for d in new["forecast"]["demands"]] == [11] * 3
    assert [d["owner_wh"] for d in old["forecast"]["demands"]] == [
        d["owner_wh"] for d in new["forecast"]["demands"]
    ]
    assert not new["completeness"]["all_requests_catalogued"]
    assert new["schedule"]["optimal"] == (
        new["schedule"]["objective_score"] == new["schedule"]["upper_bound_score"]
    )
    assert new["pointing_envelope"]["actors"]["fleet-00"]["tracking_guard_deg"] == 10.0
    restored = TaskSession.from_checkpoint(final)
    assert restored.checkpoint(compact=True) == final


def test_bound_recollects_geometry_and_rejects_resealed_forecast_substitution(
    native_case,
):
    config, old, *_ = native_case
    v = copy.deepcopy(old["forecast"])
    v["demands"][0]["slew_cells"] = 1
    v = sealed({k: x for k, x in v.items() if k != "sha256"})
    with pytest.raises(ValueError, match="matching initial"):
        collect_pointing_envelope(config, v)


def test_operator_diagnostics_stay_out_of_actor_messages(native_case):
    from space_env.llm.episode import LLMEpisode
    from space_env.simulation.snapshot import canonical_json

    initial = native_case[3]
    agent = LLMEpisode(
        TaskSession.from_checkpoint(initial), task_text="Deliver.", group="scope"
    )
    text = canonical_json(agent.messages)
    assert (
        "pointing_envelope" not in text
        and "prefix_bounds" not in text
        and "base_forecast" not in text
    )


def test_recovery_drops_initial_only_pointing_proofs_without_changing_source():
    mission = MissionWorkspace.network(planning_method="pointing_bounded")
    recovery = None
    try:
        mission.run(decisions=1)
        before = mission.operator.session.checkpoint(compact=True)
        assert mission.state()["network"]["pointing_envelope"] is not None
        recovery = mission.evolve(
            "Replan from real state.", idempotency_key="drop-initial-envelope"
        )
        state = recovery.state()["network"]
        assert state["planning_method"] == "serial_recovery"
        assert state["pointing_envelope"] is None and state["completeness"] is None
        assert mission.operator.session.checkpoint(compact=True) == before
        meta = recovery.operator.session._scenario["metadata"]["network_workspace"]
        assert "base_forecast" not in meta
    finally:
        if recovery is not None:
            recovery.close()
        mission.close()
