"""Distinct costs/geometries, visible SLA semantics and bounded native construction."""

import pytest

from space_env.tasks import TaskSession
from space_env.tasks.heterogeneous_missions import (
    HeterogeneousMissionSpec,
    heterogeneous_mission,
)


@pytest.mark.parametrize("profile", ["balanced", "bulk_tradeoff"])
@pytest.mark.parametrize("shift", [-12, 0, 12])
def test_native_preflight_all_declared_geometries(profile, shift):
    case = heterogeneous_mission({"profile": profile, "longitude_shift_deg": shift})
    session = TaskSession(case["scenario"], case["task"])
    assert session.describe()["objective"] == "sla"
    assert len(case["scenario"]["actor_order"]) == 4
    requests = case["scenario"]["definition"]["lifecycle"]["requests"]
    assert len({r["raw_size_bits"] for r in requests}) >= 3
    assert len({r["acquisition_sec"] for r in requests}) >= 2
    assert (
        len(
            {j["work_sec"] for j in case["scenario"]["definition"]["lifecycle"]["jobs"]}
        )
        >= 2
    )
    assert all(a["id"].startswith("act-") for a in case["task"]["actions"])
    assert len({a["id"] for a in case["task"]["actions"]}) == len(
        case["task"]["actions"]
    )


def test_rotations_change_native_state_and_sites_but_do_not_fake_independent_lineage():
    left = heterogeneous_mission({"longitude_shift_deg": -12})
    right = heterogeneous_mission({"longitude_shift_deg": 12})
    assert (
        left["scenario"]["definition"]["orbits"]
        != right["scenario"]["definition"]["orbits"]
    )
    assert (
        left["scenario"]["definition"]["operations"]["stations"]
        != right["scenario"]["definition"]["operations"]["stations"]
    )
    assert left["scenario"]["lineage"] == right["scenario"]["lineage"]


@pytest.mark.parametrize(
    "bad",
    [
        {"profile": "fake"},
        {"longitude_shift_deg": False},
        {"longitude_shift_deg": 12.0},
        {"longitude_shift_deg": 1},
        {"catalogue_seed": -1},
    ],
)
def test_strict_public_config(bad):
    with pytest.raises(ValueError):
        HeterogeneousMissionSpec.model_validate(bad)


def test_current_commands_budget_tracks_acquisition_cpu_and_product_size():
    case = heterogeneous_mission({"profile": "bulk_tradeoff"})
    d = case["scenario"]["definition"]
    requests = {r["id"]: r for r in d["lifecycle"]["requests"]}
    for action in case["task"]["actions"]:
        for command in action.get("commands", []):
            if command["kind"] == "imaging":
                assert (
                    command["duration_sec"]
                    == requests[command["request_id"]]["acquisition_sec"]
                )
    controls = [
        c
        for a in case["task"]["actions"]
        for c in a.get("commands", [])
        if c["kind"] == "relay"
    ]
    assert len({c["duration_sec"] for c in controls}) >= 2
    assert all(c["relay"]["ttl_sec"] == 60.0 for c in controls)
