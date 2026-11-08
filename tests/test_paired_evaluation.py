"""Paired fairness, fixed denominators and geometry-cluster uncertainty."""

import copy

import pytest

from space_env.llm.episode import OBSERVE, LLMEpisode
from space_env.llm.mission_suite import TASK, compile_suite
from space_env.llm.paired_evaluation import _interval, evaluate_pairs
from space_env.workspace import MissionWorkspace


def errors(suite, selected):
    return [
        {"case": c, "arm": arm, "status": "error", "error_type": "TimeoutError"}
        for c in selected
        for arm in ("baseline", "candidate")
    ]


def test_missing_results_are_unknown_not_zero_or_dropped():
    suite = compile_suite()
    selected = [c["id"] for c in suite["cases"]]
    report = evaluate_pairs(suite, selected, errors(suite, selected))
    overall = report["aggregates"][0]
    assert overall["expected_pairs"] == overall["unknown_pairs"] == 36
    assert overall["geometry_groups"] == 6
    assert (overall["delta_lower"], overall["delta_upper"]) == (-1, 1)
    assert overall["mean_delta"] is overall["cluster_bootstrap_95"] is None
    assert not report["llm_training_gain_established"]
    assert not report["competitor_superiority_established"]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "metrics", "status", "seed", "samples"]
)
def test_invalid_matrix_and_parameters_fail(fault):
    suite = compile_suite()
    selected = [suite["cases"][0]["id"]]
    rows = errors(suite, selected)
    options = {}
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif fault == "metrics":
        rows[0]["metrics"] = {"success": True}
    elif fault == "status":
        rows[0]["status"] = "pending"
    elif fault == "seed":
        options["seed"] = True
    else:
        options["bootstrap_samples"] = 0
    with pytest.raises(ValueError):
        evaluate_pairs(suite, selected, rows, **options)


def test_bootstrap_uses_whole_geometry_groups_and_local_rng():
    assert _interval([1, 1, -1, -1], ["a", "a", "b", "b"], 1000, 2) == [-1, 1]
    assert _interval([1, -1], ["a", "a"], 1000, 2) is None
    one = _interval([0.1, 0.4, -0.2], ["a", "a", "b"], 1000, 9)
    assert one == _interval([0.1, 0.4, -0.2], ["a", "a", "b"], 1000, 9)


@pytest.fixture(scope="module")
def native():
    suite = compile_suite()
    case = suite["cases"][0]
    workspace = MissionWorkspace.network(
        case["configuration"], planning_method="sequence_lanes"
    )
    try:
        session = workspace.operator.session
        records = []
        for budget in (1, 2):
            episode = LLMEpisode(
                session,
                task_text=TASK,
                group=case["group"],
                split=case["split"],
                max_calls=budget,
            )
            episode.invoke(OBSERVE, "{}", call_id="observe")
            records.append(episode.record())
    finally:
        workspace.close()
    return suite, case, records


def test_native_replay_and_measured_failure_is_known_zero(native):
    suite, case, records = native
    rows = [
        {"case": case["id"], "arm": arm, "status": "completed", "record": records[0]}
        for arm in ("baseline", "candidate")
    ]
    report = evaluate_pairs(suite, [case["id"]], rows)
    overall = report["aggregates"][0]
    assert overall["observed_pairs"] == overall["tied"] == 1
    assert overall["delta_lower"] == overall["delta_upper"] == 0
    assert overall["cluster_bootstrap_95"] is None  # one geometry is not a CI
    rows[1] = {
        "case": case["id"],
        "arm": "candidate",
        "status": "error",
        "error_type": "TimeoutError",
    }
    overall = evaluate_pairs(suite, [case["id"]], rows)["aggregates"][0]
    assert (overall["delta_lower"], overall["delta_upper"]) == (0, 1)


def test_native_matching_rejects_unequal_tool_budgets(native):
    suite, case, records = native
    rows = [
        {"case": case["id"], "arm": arm, "status": "completed", "record": record}
        for arm, record in zip(("baseline", "candidate"), records)
    ]
    with pytest.raises(ValueError, match="Unmatched"):
        evaluate_pairs(suite, [case["id"]], rows)


def test_resealed_fake_metrics_are_not_accepted(native):
    from space_env.llm.episode import seal

    suite, case, records = native
    corrupt = copy.deepcopy(records[0])
    corrupt["metrics"]["weighted_fulfilled"] = 18
    corrupt = seal({k: v for k, v in corrupt.items() if k != "sha256"})
    rows = [
        {"case": case["id"], "arm": arm, "status": "completed", "record": record}
        for arm, record in zip(("baseline", "candidate"), (records[0], corrupt))
    ]
    with pytest.raises(ValueError, match="replay mismatch"):
        evaluate_pairs(suite, [case["id"]], rows)


def test_matching_each_other_does_not_authorize_a_modified_task(native):
    from space_env.tasks import TaskSession

    suite, case, records = native
    initial = records[0]["initial_checkpoint"]
    task = copy.deepcopy(initial["task"])
    task["id"] = "substituted-task"
    session = TaskSession(initial["scenario"], task)
    episode = LLMEpisode(
        session, task_text=TASK, group=case["group"], split=case["split"], max_calls=1
    )
    episode.invoke(OBSERVE, "{}", call_id="observe")
    record = episode.record()
    rows = [
        {"case": case["id"], "arm": arm, "status": "completed", "record": record}
        for arm in ("baseline", "candidate")
    ]
    with pytest.raises(ValueError, match="frozen case compilation|runtime replay mismatch"):
        evaluate_pairs(suite, [case["id"]], rows)
