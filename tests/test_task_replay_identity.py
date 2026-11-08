"""Source changes fail before replaying historical controls, with useful cause."""

import pytest

from space_env.runtime import mission
from space_env.tasks import TaskSession
from tests.test_llm_episode import session


def test_changed_runtime_identity_is_not_misreported_as_physical_divergence(monkeypatch):
    record = session().checkpoint(compact=True)
    original = mission.runtime_manifest

    def changed_manifest():
        value = original()
        value["source_sha256"] = "changed-source-fixture"
        return value

    monkeypatch.setattr(mission, "runtime_manifest", changed_manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical controls must not run with changed sources")

    monkeypatch.setattr(TaskSession, "step", forbidden)
    with pytest.raises(ValueError, match="source/data/dependency identity changed"):
        TaskSession.from_checkpoint(record)
