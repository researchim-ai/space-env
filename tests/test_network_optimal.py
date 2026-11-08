import copy
import itertools

import pytest

from space_env.planning.joint_network import schedule_network
from space_env.planning.network_optimal import optimize_network, verify_exact_schedule
from space_env.runtime import MissionRuntime
from space_env.validation.network_evidence import seal


def bottleneck_view():
    actors = ("a", "b", "c")
    common = {
        "owners": list(actors),
        "stations": ["g"],
        "release_sec": 0.0,
        "deadline_sec": 5.0,
        "raw_bits": 100,
        "output_bits": 50,
        "slew_cells": 1,
        "capture_cells": 1,
        "cpu_cells": 1,
        "relay_cells": 1,
        "owner_wh": 1.0,
        "relay_wh": 1.0,
        "receiver_wh": 0.5,
        "ground_wh": 1.0,
    }
    demands = [
        {**common, "id": rid, "priority": priority, "ground_cells": width}
        for rid, priority, width in (
            ("bulk", 11, 2),
            ("small-1", 6, 1),
            ("small-2", 6, 1),
        )
    ]
    return seal(
        {
            "schema": "space-env.joint-network-view/1",
            "horizon_sec": 8.0,
            "decision_sec": 1.0,
            "actors": {a: {"memory_bits": 1e6, "usable_wh": 100.0} for a in actors},
            "stations": {"g": 1},
            "demands": demands,
            "edges": [
                {
                    "source": a,
                    "destination": "g",
                    "cells": [False, False, False, True, True, False, False, False],
                }
                for a in actors
            ],
            "captures": [
                {
                    "request": r["id"],
                    "actor": a,
                    "cells": [False, True, False, False, False, False, False, False],
                }
                for r in demands
                for a in actors
            ],
            "geometry_sha256": "0" * 64,
        }
    )


def independent_tiny_bruteforce():
    # Independent enumeration of two ground slots and three one-shot actors;
    # no planner candidate/resource helpers are used to compute the answer.
    choices = [
        [None] + [(actor, slots, value) for actor in range(3) for slots in masks]
        for masks, value in (((3,), 11), ((1, 2), 6), ((1, 2), 6))
    ]
    best = 0
    for combination in itertools.product(*choices):
        used_actors, used_slots, value = set(), 0, 0
        for entry in combination:
            if entry is None:
                continue
            actor, slots, score = entry
            if actor in used_actors or slots & used_slots:
                break
            used_actors.add(actor)
            used_slots |= slots
            value += score
        else:
            best = max(best, value)
    return best


def test_global_reference_beats_priority_on_known_counterexample():
    view = bottleneck_view()
    before = copy.deepcopy(view)
    greedy = schedule_network(view, max_hops=0)
    optimum = optimize_network(view, max_hops=0)
    assert [a["request"] for a in greedy["allocations"]] == ["bulk"]
    assert {a["request"] for a in optimum["allocations"]} == {"small-1", "small-2"}
    assert optimum["objective_score"] == independent_tiny_bruteforce() * 4 + 2 == 50
    assert optimum["optimal"] and optimum["upper_bound_score"] == 50
    assert optimum["enumeration_complete"] and optimum["search_complete"]
    assert verify_exact_schedule(view, optimum)
    assert view == before


def test_shared_capacity_and_memory_change_proven_solution():
    view = bottleneck_view()
    view["stations"]["g"] = 2
    view = seal({k: v for k, v in view.items() if k != "sha256"})
    plan = optimize_network(view, max_hops=0)
    assert plan["optimal"] and plan["predicted_fulfilled"] == 3
    for a in view["actors"].values():
        a["memory_bits"] = 100000.0
    plan = optimize_network(
        seal({k: v for k, v in view.items() if k != "sha256"}), max_hops=0
    )
    assert plan["optimal"] and plan["objective_score"] == 0


def test_budget_and_candidate_cap_keep_honest_upper_bound():
    for kw in ({"max_nodes": 1}, {"max_candidates": 1}):
        plan = optimize_network(bottleneck_view(), max_hops=0, **kw)
        assert not plan["optimal"] and plan["search_budget_exhausted"]
        assert plan["objective_score"] < plan["upper_bound_score"]
        assert plan["total"] == 3
        assert plan["search_nodes"] <= plan["max_search_nodes"]
        assert verify_exact_schedule(bottleneck_view(), plan)


def test_multihop_paths_are_admitted_and_never_use_native_rollout(monkeypatch):
    from tests.test_joint_network import view

    def forbidden(*args, **kwargs):
        raise AssertionError("Native future in exact reference")

    monkeypatch.setattr(MissionRuntime, "submit", forbidden)
    monkeypatch.setattr(MissionRuntime, "advance", forbidden)
    value = view()
    value["demands"][0]["owners"] = ["a"]
    value["captures"] = value["captures"][:1]
    value = seal({k: v for k, v in value.items() if k != "sha256"})
    plan = optimize_network(value)
    assert plan["optimal"] and plan["allocations"][0]["path"] == ["a", "b", "c"]
    assert verify_exact_schedule(value, plan)


@pytest.mark.parametrize(
    "fault", ["binding", "stage", "resource", "denominator", "objective"]
)
def test_resource_witness_admission_rejects_corruption(fault):
    view = bottleneck_view()
    plan = optimize_network(view, max_hops=0)
    if fault == "binding":
        plan["forecast_sha256"] = "f" * 64
    elif fault == "stage":
        plan["allocations"][0]["stages"][-1]["cell"] = 7
    elif fault == "resource":
        plan["allocations"][0]["energy_wh"]["a"] += 1.0
    elif fault == "denominator":
        plan["unserved"] = {}
    else:
        plan["objective_score"] += 1
    plan = seal({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError):
        verify_exact_schedule(view, plan)


def test_small_scope_and_strict_budgets_are_explicit():
    with pytest.raises(ValueError):
        optimize_network(bottleneck_view(), max_nodes=True)
    view = bottleneck_view()
    view["at_sec"] = 1.0
    with pytest.raises(ValueError):
        optimize_network(seal({k: v for k, v in view.items() if k != "sha256"}))


def test_native_exact_small_profile_executes_real_two_hop_delivery():
    from space_env.workspace import MissionWorkspace

    mission = MissionWorkspace.network(planning_method="exact_small")
    state = mission.state()["network"]
    assert state["planning_method"] == "exact_small"
    assert state["optimization"]["optimal_in_forecast_model"]
    result = mission.run()["network"]
    assert result["fulfilled"] == 1
    assert mission.operator.session._runtime._backend.report()["lifecycle"][
        "acknowledgements"
    ]
    mission.close()


def test_variable_ground_setup_width_changes_resources_and_proven_feasibility():
    from space_env.planning.joint_network import ground_width

    view = bottleneck_view()
    for r in view["demands"]:
        r["ground_seconds"] = float(r["ground_cells"])
    for edge in view["edges"]:
        edge["setup_sec"] = [0.0] * 8
        edge["setup_sec"][4] = 1.0
    view = seal({k: v for k, v in view.items() if k != "sha256"})
    assert ground_width(view["demands"][1], view["edges"][0], 4, 1.0) == 2
    plan = optimize_network(view, max_hops=0)
    assert plan["optimal"] and plan["predicted_fulfilled"] == 1
    assert plan["allocations"][0]["request"] == "bulk"
    assert verify_exact_schedule(view, plan)


def test_native_reacquisition_is_budgeted_without_changing_antenna_physics():
    from space_env.planning.network_demand_benchmark import demand_suite
    from space_env.workspace import MissionWorkspace

    case = next(
        c
        for c in demand_suite()["cases"]
        if c["id"] == "demand-equatorial-staggered-cpu"
    )
    mission = MissionWorkspace.network(case["configuration"])
    meta = mission.operator.session._scenario["metadata"]["network_workspace"]
    edge = next(
        e
        for e in meta["forecast"]["edges"]
        if e["source"] == "fleet-00" and e["destination"] == "Ground"
    )
    assert edge["setup_sec"][42] >= 2.7  # actual missed handover in the pre-fix trace
    allocation = next(
        a for a in meta["schedule"]["allocations"] if a["request"] == "request-3"
    )
    assert allocation["stages"][-1]["cells"] > 1
    result = mission.run()["network"]
    native = mission.operator.session._runtime._backend.report()["lifecycle"][
        "requests"
    ]
    assert all(
        native[a["request"]]["status"] == "completed"
        for a in meta["schedule"]["allocations"]
    )
    assert result["fulfilled"] == result["planned"]
    mission.close()


@pytest.mark.parametrize("seed", range(30))
def test_pruning_matches_independent_resource_enumeration(seed):
    import random

    rng = random.Random(seed)
    view = bottleneck_view()
    capacity = rng.choice((1, 2))
    view["stations"]["g"] = capacity
    for resources in view["actors"].values():
        resources["memory_bits"] = rng.choice((100149, 100150, 1e6))
        resources["usable_wh"] = rng.choice((1.999, 2.0, 100.0))
    for request in view["demands"]:
        request["priority"] = rng.randrange(1, 15)
        request["ground_cells"] = rng.choice((1, 2))
        request["owners"] = rng.sample(list(view["actors"]), rng.randrange(1, 4))
    owners = {r["id"]: r["owners"] for r in view["demands"]}
    view["captures"] = [
        c for c in view["captures"] if c["actor"] in owners[c["request"]]
    ]
    choices = []
    for request in view["demands"]:
        width = request["ground_cells"]
        choices.append(
            [None]
            + [
                (actor, start, width, request["priority"] * 4 + 1)
                for actor in request["owners"]
                for start in range(3, 6 - width)
            ]
        )
    expected = 0
    for combination in itertools.product(*choices):
        actors, slots, score = set(), [0, 0], 0
        for item in combination:
            if item is None:
                continue
            actor, start, width, value = item
            if (
                actor in actors
                or view["actors"][actor]["memory_bits"] < 100150
                or view["actors"][actor]["usable_wh"] < 2.0
            ):
                break
            actors.add(actor)
            for cell in range(start - 3, start - 3 + width):
                slots[cell] += 1
            if max(slots) > capacity:
                break
            score += value
        else:
            expected = max(expected, score)
    view = seal({k: v for k, v in view.items() if k != "sha256"})
    for pruning in (False, True):
        result = optimize_network(view, max_hops=0, pruning=pruning)
        assert result["optimal"] and result["objective_score"] == expected
        assert result["cached_states"] <= result["cache_limit"]
        assert result["mandatory_cache_states"] <= result["mandatory_cache_limit"]
        assert verify_exact_schedule(view, result)


def test_actor_symmetry_requires_complete_direct_candidate_universe():
    from tests.test_joint_network import view as relay_view

    symmetric = optimize_network(bottleneck_view(), max_hops=0)
    assert symmetric["equivalent_actor_groups"] == [["a", "b", "c"]]
    partial = optimize_network(bottleneck_view(), max_hops=0, max_candidates=1)
    assert all(len(group) == 1 for group in partial["equivalent_actor_groups"])
    relayed = optimize_network(relay_view())
    assert all(len(group) == 1 for group in relayed["equivalent_actor_groups"])
    with pytest.raises(ValueError):
        optimize_network(bottleneck_view(), pruning=1)
