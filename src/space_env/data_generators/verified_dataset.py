"""Набор только из прогонов, которые повтор подтвердил."""
from __future__ import annotations

import json
from pathlib import Path

from space_env.data_generators.mission_dataset import replay
from space_env.eval.statistics import assign_split


def collect_verified(task_id: str, runs: list[dict]) -> dict:
    """
    runs: variant и path к запечатанной записи.
    Повтор, который не сошёлся, остаётся в rejected и не получает split.
    Варианты одного task_id, которые сошлись, попадают в один кусок.
    """
    accepted = []
    rejected = []
    for item in runs:
        path = Path(item["path"])
        ok = replay(path)
        row = {"task_id": task_id, "variant": str(item["variant"]), "path": str(path), "accepted": bool(ok)}
        if ok:
            row["split"] = assign_split(task_id, row["variant"])
            accepted.append(row)
        else:
            rejected.append(row)
    splits = {row["split"] for row in accepted}
    return {"accepted": accepted, "rejected": rejected, "one_split": len(splits) <= 1}


def sft_example(path: str | Path) -> dict | None:
    """Строка обучения из записи. Повтор, который не сошёлся, строку не даёт."""
    target = Path(path)
    if not replay(target):
        return None
    record = json.loads(target.read_text(encoding="utf-8"))
    commands = []
    for step in record.get("schedule", []):
        for command in step.get("commands", {}).values():
            commands.append(
                {
                    "actor_id": command["actor_id"],
                    "kind": command["kind"],
                    "request_id": command.get("request_id", ""),
                }
            )
    return {
        "commands": commands,
        "on_time": list(record.get("on_time", [])),
        "unmet": [item["request_id"] for item in record.get("unmet", [])],
    }
