"""Real owner/path choices and mutual exclusion across otherwise disjoint actors."""

import copy

import pytest

from space_env.planning import BaselinePolicy, PlanningInput
from space_env.simulation.snapshot import canonical_json
from space_env.tasks import TaskSession
from space_env.tasks.choice_missions import ChoiceMissionSpec, choice_mission
from tests.test_planning import public


@pytest.mark.parametrize(
    "kind,field", [("imaging", "request_id"), ("transform", "job_id")]
)
@pytest.mark.parametrize(
    "method", ["priority", "earliest_deadline", "route_aware_rolling"]
)
def test_global_intent_exclusion_without_merging_actor_locks(kind, field, method):
    value = public(
        [
            ("first", "a", kind, "one", 3.0, 20.0, "payload", ""),
            ("second", "b", kind, "two", 2.0, 20.0, "payload", ""),
        ]
    ).view()
    for action in value["observation"]["catalogue"]:
        action["commands"][0][field] = "same-intent"
    decision = BaselinePolicy({"name": method}).decide(
        PlanningInput(canonical_json(value))
    )
    assert len(decision["action_ids"]) == 1
    value["observation"]["catalogue"][1]["commands"][0][field] = "independent-intent"
    independent = BaselinePolicy({"name": method}).decide(
        PlanningInput(canonical_json(value))
    )
    assert set(independent["action_ids"]) == {"first", "second"}


@pytest.fixture(scope="module")
def case():
    return choice_mission({"cameras": 2})


def test_catalogue_contains_all_owners_and_paths_without_stage_names(case):
    requests = case["scenario"]["definition"]["lifecycle"]["requests"]
    cameras = [a for a in case["scenario"]["actor_order"] if a.startswith("camera-")]
    actions = case["task"]["actions"]
    assert len(requests) == 4
    assert all(set(r["eligible_actors"]) == set(cameras) for r in requests)
    for r in requests:
        choices = [a["commands"][0] for a in actions if a.get("request_ref") == r["id"]]
        assert {c["actor_id"] for c in choices if c["kind"] == "imaging"} == set(
            cameras
        )
        for camera in cameras:
            assert (
                len(
                    [
                        c
                        for c in choices
                        if c["kind"] == "relay" and c["actor_id"] == camera
                    ]
                )
                == 2
            )
        assert r["id"].startswith("obj-")
        assert not any(token in r["id"] for token in ["camera", "wave", "capture"])
    assert all(a["id"].startswith("act-") for a in actions)
    assert len({a["id"] for a in actions}) == len(actions)
    assert "steps" not in case and "reference_steps" not in case


def test_each_camera_can_really_acquire_same_request(case):
    request = case["scenario"]["definition"]["lifecycle"]["requests"][0]
    for actor in request["eligible_actors"]:
        session = TaskSession(case["scenario"], case["task"])
        action = next(
            a["id"]
            for a in case["task"]["actions"]
            if a.get("commands")
            and a["commands"][0]["kind"] == "imaging"
            and a["commands"][0]["request_id"] == request["id"]
            and a["role"] == actor
        )
        result = session.step(
            {"@central": [action]},
            expected_revision=0,
            idempotency_key="capture",
            advance_sec=2.0,
        )
        assert result["feedback"]["accepted"]
        row = session._runtime._backend.report()["lifecycle"]["requests"][request["id"]]
        assert row["actor"] == actor
        assert row["status"] not in ("failed", "expired")
        products = session._runtime.inventory(actor)["products"]
        assert request["raw_product_id"] in {p["id"] for p in products}


def test_alias_seed_preserves_geometry_lineage_and_actual_priorities(case):
    other = choice_mission({"cameras": 2, "catalogue_seed": 19})
    assert (
        case["scenario"]["definition"]["orbits"]
        == other["scenario"]["definition"]["orbits"]
    )
    assert case["scenario"]["lineage"] == other["scenario"]["lineage"]
    assert [
        r["priority"] for r in case["scenario"]["definition"]["lifecycle"]["requests"]
    ] == [
        r["priority"] for r in other["scenario"]["definition"]["lifecycle"]["requests"]
    ]
    assert {a["id"] for a in case["task"]["actions"]}.isdisjoint(
        a["id"] for a in other["task"]["actions"]
    )
    TaskSession(other["scenario"], other["task"])


@pytest.mark.parametrize("family", ["memory", "ground_window", "link_blackout"])
def test_resource_variants_construct_and_no_forecast_in_actor_observations(family):
    case = choice_mission({"family": family, "cameras": 2})
    session = TaskSession(case["scenario"], case["task"])
    view = session.observe("camera-00")
    assert "forecast" not in view and "schedule" not in view
    assert not any(a.kind == "inject" for a in session._task.actions)


@pytest.mark.parametrize(
    "bad",
    [
        {"cameras": 1},
        {"cameras": True},
        {"waves": 4},
        {"family": "invented"},
        {"catalogue_seed": -1},
    ],
)
def test_invalid_choice_profiles(bad):
    with pytest.raises(ValueError):
        ChoiceMissionSpec.model_validate(copy.deepcopy(bad))


def test_benchmark_retains_all_cells_and_does_not_call_best_observed_optimal():
    from space_env.validation.choice_benchmark import POLICIES, summarize

    frozen = {"cases": [{"id": "case"}]}
    rows = [{"case": "case", "policy": p, "status": "error"} for p in POLICIES]
    result = summarize(frozen, rows)
    assert all(
        r["attempts"] == 1 and r["errors"] == 1 and r["priority_lower"] == 0
        for r in result["policies"].values()
    )
    assert result["best_observed_priority"]["case"] is None
    with pytest.raises(ValueError, match="matrix"):
        summarize(frozen, rows[:-1])
    rows[0].update(
        status="completed", witness={"fulfilled_priority": 9}, metrics={"fulfilled": 2}
    )
    assert summarize(frozen, rows)["best_observed_priority"]["case"] == 9


def test_non_mapping_choice_input_is_not_silently_defaulted():
    with pytest.raises(ValueError):
        choice_mission(False)


def test_audit_integrity_is_not_a_success_claim(tmp_path, monkeypatch):
    from space_env.llm.episode import seal, sources
    from space_env.simulation.snapshot import digest
    from space_env.validation import choice_benchmark as benchmark

    frozen = {
        "quick": True,
        "workers": 1,
        "cases": [{"id": "case"}],
        "policies": list(benchmark.POLICIES),
    }
    monkeypatch.setattr(benchmark, "manifest", lambda quick, workers: frozen)
    rows = [
        {"case": "case", "policy": p, "status": "error"} for p in benchmark.POLICIES
    ]
    report = seal(
        {
            "schema": "space-env.choice-result/1",
            "sources": sources(),
            "manifest_sha256": digest(frozen),
            "rows": rows,
            "summary": benchmark.summarize(frozen, rows),
            "llm_measured": False,
            "optimality_proven": False,
        }
    )
    (tmp_path / "manifest.json").write_text(canonical_json(seal(frozen)))
    (tmp_path / "report.json").write_text(canonical_json(report))
    result = benchmark.audit(tmp_path)
    assert result["valid"] and result["native_replayed"] == 0 and result["rows"] == 4
    report["optimality_proven"] = True
    report = seal({k: v for k, v in report.items() if k != "sha256"})
    (tmp_path / "report.json").write_text(canonical_json(report))
    with pytest.raises(ValueError, match="claims"):
        benchmark.audit(tmp_path)


def test_benchmark_rejects_unknown_status_and_escaping_artifact(tmp_path):
    from space_env.validation.choice_benchmark import POLICIES, _read, summarize

    rows = [{"case": "case", "policy": p, "status": "skipped"} for p in POLICIES]
    with pytest.raises(ValueError, match="outcome"):
        summarize({"cases": [{"id": "case"}]}, rows)
    with pytest.raises(ValueError, match="escapes"):
        _read(tmp_path, "../outside.json")
