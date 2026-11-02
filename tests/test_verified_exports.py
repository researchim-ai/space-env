"""Replay-once export, evidence sharing and public mutation isolation."""

import copy

import pytest

from space_env.planning import PlanningRun
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession, VerifiedEpisode, export_dataset
from tests.test_planning import scenario, task


def run_fixture():
    run = PlanningRun(scenario.__wrapped__(), task())
    run.run(decisions=3)
    return run


@pytest.mark.parametrize("compact", [False, True])
def test_shared_private_evidence_public_ownership(compact):
    run = run_fixture()
    session = run.session
    first, second = [r["transition"] for r in session._transitions[:2]]
    assert first["next_role_observations"] is second["role_observations"]
    assert first["observation"] is first["role_observations"]["@central"]
    req = session._transitions[0]["request"]
    assert session._cache[req["idempotency_key"]]["response"] is first
    before = session.checkpoint(compact=compact)
    result = session.step(**req)
    result["role_observations"].clear()
    session.observe()["catalogue"].clear()
    assert session.checkpoint(compact=compact) == before
    restored = TaskSession.from_checkpoint(before)
    assert restored.checkpoint(compact=compact) == before


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "format,role", [("rl", "@central"), ("rl", "a"), ("sft", "@central")]
)
def test_verified_export_legacy_bytes(compact, format, role, tmp_path):
    record = run_fixture().checkpoint(compact=compact)["episode"]
    legacy = export_dataset(record, tmp_path / "legacy", format=format, role=role)
    verified = VerifiedEpisode(record)
    fresh = verified.export(tmp_path / "new", format=format, role=role)
    assert fresh == legacy
    assert (tmp_path / "legacy").read_bytes() == (tmp_path / "new").read_bytes()


def test_one_native_replay_for_multi_format_bundle(monkeypatch, tmp_path):
    record = run_fixture().checkpoint(compact=True)["episode"]
    original = TaskSession.from_checkpoint.__func__
    calls = []

    def replay(cls, value):
        calls.append(True)
        return original(cls, value)

    monkeypatch.setattr(TaskSession, "from_checkpoint", classmethod(replay))
    verified = VerifiedEpisode(record)
    result = verified.export_bundle(tmp_path / "bundle")
    verified.export(tmp_path / "again")
    assert len(calls) == 1
    assert len(result["files"]) == 2
    assert result["sha256"] == digest(
        {k: v for k, v in result.items() if k != "sha256"}
    )
    for name, meta in result["files"].items():
        assert (
            len((tmp_path / "bundle" / name).read_text().splitlines()) == meta["rows"]
        )


def test_public_iterator_does_not_mutate_future_rows():
    record = run_fixture().checkpoint(compact=True)["episode"]
    verified = VerifiedEpisode(record)
    expected = list(verified.iter_rows())
    iterator = verified.iter_rows()
    first = next(iterator)
    first["next_observation"].clear()
    assert next(iterator) == expected[1]
    assert list(verified.iter_rows()) == expected
    record.clear()
    assert list(verified.iter_rows()) == expected


def test_planning_verification_retains_policy_replay(tmp_path):
    record = run_fixture().checkpoint(compact=True)
    verified = VerifiedEpisode.from_planning_checkpoint(record)
    assert verified.verification["trace_sha256"] == digest(record["trace"])
    assert verified.verification["quality"]["passed"]
    public = verified.verification
    public["quality"]["passed"] = False
    assert verified.verification["quality"]["passed"]
    bad = copy.deepcopy(record)
    bad["trace"][0]["policy_sha256"] = "0" * 64
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError):
        VerifiedEpisode.from_planning_checkpoint(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"formats": ()},
        {"formats": ("rl", "rl")},
        {"roles": ("missing",)},
        {"roles": ("a",)},
        {"max_bytes": 1},
    ],
)
def test_bundle_invalid_or_budget_does_not_publish(kwargs, tmp_path):
    verified = VerifiedEpisode(run_fixture().checkpoint(compact=True)["episode"])
    with pytest.raises(ValueError):
        verified.export_bundle(tmp_path / "out", **kwargs)
    assert not list(tmp_path.iterdir())


def test_bundle_late_write_failure_atomic(monkeypatch, tmp_path):
    import space_env.tasks.dataset as module

    verified = VerifiedEpisode(run_fixture().checkpoint(compact=True)["episode"])
    original = module.atomic_jsonl
    calls = []

    def write(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("disk failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "atomic_jsonl", write)
    with pytest.raises(OSError):
        verified.export_bundle(tmp_path / "out")
    assert not list(tmp_path.iterdir())


def test_existing_bundle_preserved(tmp_path):
    verified = VerifiedEpisode(run_fixture().checkpoint(compact=True)["episode"])
    out = tmp_path / "out"
    out.mkdir()
    (out / "mine").write_text("keep")
    with pytest.raises(ValueError):
        verified.export_bundle(out)
    assert (out / "mine").read_text() == "keep"
