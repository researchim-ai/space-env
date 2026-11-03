"""Deepcopy equivalence, ownership, native aliases and scalar reward parity."""

import copy

import numpy as np
import pytest

from space_env.planning import PlanningRun
from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.scaling.campaign import campaign_workload
from space_env.simulation.owned import owned_copy
from space_env.tasks.session import score


def test_copy_preserves_cycles_aliases_tuple_children_and_numpy_ownership():
    shared = {"array": np.arange(9).reshape(3, 3), "tuple": (1, [2, 3])}
    value = {"a": shared, "b": shared}
    value["self"] = value
    for clone in [copy.deepcopy(value), owned_copy(value)]:
        assert clone is not value and clone["self"] is clone
        assert clone["a"] is clone["b"] and clone["a"] is not shared
        assert not np.shares_memory(clone["a"]["array"], shared["array"])
        clone["a"]["tuple"][1].append(4)
        assert shared["tuple"][1] == [2, 3]


def test_memo_and_subclasses_retain_standard_hooks():
    class Special(list):
        pass

    original = Special([{"x": []}])
    original.extra = {"y": []}
    clone = owned_copy(original)
    assert type(clone) is Special
    assert clone.extra is not original.extra and clone[0] is not original[0]
    obj = []
    marker = object()
    assert owned_copy(obj, {id(obj): marker}) is marker
    assert owned_copy(3, {id(3): marker}) is copy.deepcopy(3, {id(3): marker})
    assert owned_copy([3], {id(3): marker}) == copy.deepcopy([3], {id(3): marker})


def test_dict_hook_order_matches_deepcopy():
    calls = []

    class Hook:
        def __init__(self, label):
            self.label = label

        def __deepcopy__(self, memo):
            calls.append(self.label)
            return Hook(self.label)

    value = {Hook("key"): Hook("value")}
    copy.deepcopy(value)
    expected = calls.copy()
    calls.clear()
    owned_copy(value)
    assert calls == expected == ["value", "key"]


def test_hook_appending_to_source_list_matches_deepcopy():
    class Appender:
        def __deepcopy__(self, memo):
            original.append(7)
            return "copy"

    for copier in [copy.deepcopy, owned_copy]:
        original = [Appender()]
        assert copier(original) == ["copy", 7]


def workload(mode="nominal"):
    return campaign_workload(
        {
            "cameras": 1,
            "relays": 1,
            "waves": 1,
            "horizon_sec": 20.0,
            "repeats": 1,
            "mode": mode,
        }
    )


@pytest.mark.parametrize("mode", ["nominal", "ack_loss", "ground_outage", "ttl"])
def test_scalar_score_exactly_matches_reference_and_capture_owns_science(
    mode, monkeypatch
):
    w = workload(mode)
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    for _ in range(4):
        backend = run.session._runtime._backend
        fast = score(run.session._runtime)
        with monkeypatch.context() as patch:
            patch.setattr(type(backend), "task_score", lambda self: None)
            assert score(run.session._runtime) == fast
        run.step()
    backend = run.session._runtime._backend
    clone = run.session._runtime.fork()
    assert clone._backend.lifecycle.transport is clone._backend.mission.dtn
    assert clone._backend.lifecycle.transport is not backend.lifecycle.transport
    product = next(iter(backend.lifecycle.artifacts))
    before = copy.deepcopy(backend.lifecycle.artifacts[product])
    clone._backend.lifecycle.artifacts[product]["observations"].clear()
    assert backend.lifecycle.artifacts[product] == before
    # Raw lifecycle/default backend readers keep their historical ownership.
    raw = backend.observe("camera-00")
    raw["lifecycle"]["artifacts"][0]["observations"].clear()
    assert backend.lifecycle.artifacts[product] == before
    public = run.session._runtime.observe("camera-00")
    public["actors"]["camera-00"]["lifecycle"]["artifacts"][0]["observations"].clear()
    assert backend.lifecycle.artifacts[product] == before


def test_scalar_score_does_not_construct_science_report(monkeypatch):
    w = workload()
    run = PlanningRun(w["scenario"], w["task"], w["policy"])

    def forbidden(*args, **kwargs):
        raise AssertionError("full science report in reward path")

    monkeypatch.setattr(run.session._runtime._backend, "report", forbidden)
    assert score(run.session._runtime)["completion"] == 0


def test_scalar_score_is_not_available_without_truth_permission():
    w = workload()
    runtime = MissionRuntime(w["scenario"], access=AccessPolicy())
    with pytest.raises(RuntimeContractError, match="disabled"):
        runtime._task_score()


def test_commit_copy_failure_rolls_back_physics_revision_and_idempotency(monkeypatch):
    from space_env.runtime import mission

    w = workload()
    runtime = MissionRuntime(
        w["scenario"], access=AccessPolicy(truth=True), audit_mode="full"
    )
    before = runtime.checkpoint()
    original = mission.owned_copy

    def fail_commit(value, *args):
        if isinstance(value, dict) and "backend_report" in value:
            raise ValueError("forced evidence copy failure")
        return original(value, *args)

    with monkeypatch.context() as patch:
        patch.setattr(mission, "owned_copy", fail_commit)
        with pytest.raises(RuntimeContractError, match="forced evidence"):
            runtime.advance(1.0, expected_revision=0, idempotency_key="rollback")
    assert runtime.checkpoint() == before
    assert runtime._backend.lifecycle.transport is runtime._backend.mission.dtn
    runtime.advance(1.0, expected_revision=0, idempotency_key="rollback")
    assert runtime.revision == 1


def test_custom_hook_observes_standard_memo_keepalive_order():
    class ViewMemo:
        def __deepcopy__(self, memo):
            return len(memo.get(id(memo), []))

    value = [[], ViewMemo()]
    assert owned_copy(value) == copy.deepcopy(value)


def test_external_transformer_keeps_original_capture_path(monkeypatch):
    from space_env.reference import dynamic

    class External:
        pass

    w = workload()
    run = PlanningRun(w["scenario"], w["task"], w["policy"])
    backend = run.session._runtime._backend
    backend.lifecycle.transformers["external-test"] = External()

    def forbidden(*args):
        raise AssertionError("external transformer must use standard graph copy")

    monkeypatch.setattr(dynamic, "owned_copy", forbidden)
    mission, _, _, life, _ = backend.capture()
    assert life.transport is mission.dtn
    assert life is not backend.lifecycle
    assert type(life.transformers["external-test"]) is External
