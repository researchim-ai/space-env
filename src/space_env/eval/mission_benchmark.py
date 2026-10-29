"""Короткий набор миссий. Тот же отчёт, что у CLI, без 72-часовых полётов."""
from __future__ import annotations

import time
from pathlib import Path

import yaml

from space_env.mission.session import mission_report


def suite_path(name: str | Path) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    packaged = Path(__file__).resolve().parents[1] / "configs" / "benchmark_suites" / Path(name).name
    checkout = Path.cwd() / "configs" / "benchmark_suites" / Path(name).name
    if checkout.is_file():
        return checkout
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(f"нет набора миссий: {name}")


def _scenario_path(name: str) -> Path:
    from space_env.envs.mission_fleet_env import packaged_mission

    return packaged_mission(name)


def _case_passed(item: dict, report: dict) -> bool:
    if "on_time" in item and list(report["on_time"]) != list(item["on_time"]):
        return False
    if "min_on_time" in item and len(report["on_time"]) < int(item["min_on_time"]):
        return False
    if "unmet_reason" in item:
        if report["on_time"]:
            return False
        if not report["unmet"]:
            return False
        if any(row["reason"] != item["unmet_reason"] for row in report["unmet"]):
            return False
    return True


def run_mission_suite(path: str | Path) -> dict:
    """Исполняет случаи набора. Ожидания в файле — это проверка, не подсказка политике."""
    spec = yaml.safe_load(suite_path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or int(spec.get("version", 0)) != 1:
        raise ValueError("набор миссий имеет version: 1")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("в наборе нет случаев")
    rows = []
    started = time.perf_counter()
    for item in cases:
        if not isinstance(item, dict) or not item.get("id") or not item.get("scenario"):
            raise ValueError("у случая нужны id и scenario")
        began = time.perf_counter()
        report = mission_report(
            _scenario_path(str(item["scenario"])),
            step_sec=float(item.get("step_sec", 60.0)),
            policy=str(item.get("policy", "priority")),
        )
        rows.append(
            {
                "id": str(item["id"]),
                "scenario": Path(str(item["scenario"])).name,
                "on_time": list(report["on_time"]),
                "unmet": report["unmet"],
                "weighted_on_time": report["weighted_on_time"],
                "safety_violations": report["safety_violations"],
                "runtime_sec": time.perf_counter() - began,
                "passed": _case_passed(item, report),
            }
        )
    return {
        "suite": str(spec.get("name", "")),
        "passed": all(row["passed"] for row in rows),
        "cases": rows,
        "runtime_sec": time.perf_counter() - started,
    }


def seal_suite(path: str | Path, directory: str | Path) -> dict:
    """Печатает каждый случай набора и берёт строку обучения только из сходящегося повтора."""
    from space_env.data_generators.mission_dataset import seal_run
    from space_env.data_generators.verified_dataset import sft_example

    spec = yaml.safe_load(suite_path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or int(spec.get("version", 0)) != 1:
        raise ValueError("набор миссий имеет version: 1")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("в наборе нет случаев")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in cases:
        scenario = _scenario_path(str(item["scenario"]))
        target = folder / f"{item['id']}.json"
        record = seal_run(
            target,
            scenario,
            step_sec=float(item.get("step_sec", 60.0)),
            policy=str(item.get("policy", "priority")),
        )
        report = {"on_time": record["on_time"], "unmet": record["unmet"]}
        rows.append(
            {
                "id": str(item["id"]),
                "passed": _case_passed(item, report),
                "example": sft_example(target),
            }
        )
    return {
        "suite": str(spec.get("name", "")),
        "passed": all(row["passed"] and row["example"] is not None for row in rows),
        "rows": rows,
    }
