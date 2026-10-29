"""Короткий набор миссий: обычная сдача, отказ станции, ранний срок и учебные 12 заявок."""
import json

from space_env.cli import main
from space_env.eval.mission_benchmark import seal_suite


def test_smoke_suite_separates_nominal_outage_and_infeasible(capsys):
    main(["benchmark", "mission_v1_smoke.yaml"])
    report = json.loads(capsys.readouterr().out)
    assert report["suite"] == "mission_v1_smoke"
    assert report["passed"] is True
    by_id = {item["id"]: item for item in report["cases"]}
    assert by_id["nominal"]["on_time"] == ["kiruna-pass"]
    assert by_id["outage"]["on_time"] == []
    assert by_id["outage"]["unmet"][0]["reason"] == "станция недоступна"
    assert by_id["infeasible"]["on_time"] == []
    assert by_id["infeasible"]["unmet"][0]["reason"] == "срок вышел"
    assert len(by_id["curriculum"]["on_time"]) == 12
    assert by_id["curriculum"]["unmet"] == []


def test_smoke_suite_seals_a_training_row_for_every_case(tmp_path):
    bundle = seal_suite("mission_v1_smoke.yaml", tmp_path)
    assert bundle["passed"] is True
    by_id = {item["id"]: item for item in bundle["rows"]}
    assert by_id["nominal"]["example"]["on_time"] == ["kiruna-pass"]
    assert by_id["outage"]["example"]["on_time"] == []
    assert by_id["outage"]["example"]["unmet"] == ["kiruna-pass"]
    assert by_id["infeasible"]["example"]["on_time"] == []
    assert len(by_id["curriculum"]["example"]["on_time"]) == 12
    assert by_id["curriculum"]["example"]["commands"]
