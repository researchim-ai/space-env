"""Блокировка сессии и набор только из сходящегося повтора."""
import json
import threading
from pathlib import Path

import pytest

from space_env.data_generators.mission_dataset import seal_run
from space_env.data_generators.verified_dataset import collect_verified
from space_env.envs.mission_fleet_env import packaged_mission
from space_env.mission.session import MissionSession, RevisionError


def test_two_advances_at_the_same_revision_move_the_clock_once():
    session = MissionSession("family_nominal.yaml")
    results = []
    errors = []

    def go() -> None:
        try:
            results.append(session.advance(30.0, expected_revision=0))
        except RevisionError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=go) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 1
    assert len(errors) == 1
    assert session.revision == 1
    assert session.world.now_sec() == pytest.approx(30.0)


def test_only_a_replay_that_matches_enters_the_dataset(tmp_path: Path):
    sealed = Path(tmp_path / "nominal.json")
    scenario = packaged_mission("family_nominal.yaml")
    seal_run(sealed, scenario, step_sec=300.0)
    copy = Path(tmp_path / "variant.json")
    copy.write_text(sealed.read_text(encoding="utf-8"), encoding="utf-8")
    broken = json.loads(sealed.read_text(encoding="utf-8"))
    broken["on_time"] = ["not-the-pass"]
    bad = Path(tmp_path / "bad.json")
    bad.write_text(json.dumps(broken), encoding="utf-8")
    report = collect_verified(
        "kiruna-family",
        [
            {"variant": "nominal", "path": sealed},
            {"variant": "copy", "path": copy},
            {"variant": "broken", "path": bad},
        ],
    )
    assert [item["variant"] for item in report["accepted"]] == ["nominal", "copy"]
    assert report["one_split"] is True
    assert report["accepted"][0]["split"] == report["accepted"][1]["split"]
    assert report["rejected"][0]["variant"] == "broken"
    assert report["rejected"][0]["accepted"] is False
    assert "split" not in report["rejected"][0]
