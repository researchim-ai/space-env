"""Команда с идентификатором: повтор той же команды ничего не меняет, другая — конфликт."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MissionCommand:
    command_id: str
    actor_id: str
    kind: str
    request_id: str = ""
    destination_id: str = ""
    duration_sec: float = 0.0


def _body(command: MissionCommand) -> tuple:
    return (
        command.actor_id,
        command.kind,
        command.request_id,
        command.destination_id,
        float(command.duration_sec),
    )


class CommandBook:
    """submitted → accepted → completed, failed или cancelled."""

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}
        self._running: dict[str, None] = {}
        self._windows: dict[str, None] = {}

    def submit(self, command: MissionCommand) -> dict:
        self.check(command)
        if command.kind not in {"wait", "imaging", "comms", "cancel"}:
            raise ValueError(f"команда {command.kind} не поддерживается")
        body = _body(command)
        existing = self._records.get(command.command_id)
        if existing is not None:
            if existing["body"] != body:
                raise ValueError(f"команда {command.command_id} уже записана с другим телом")
            return dict(existing["record"])
        record = {
            "command_id": command.command_id,
            "actor_id": command.actor_id,
            "kind": command.kind,
            "request_id": command.request_id,
            "destination_id": command.destination_id,
            "duration_sec": float(command.duration_sec),
            "started_sec": None,
            "until_sec": None,
            "status": "accepted",
        }
        self._records[command.command_id] = {"body": body, "record": record}
        return dict(record)

    def cancel(self, command_id: str) -> dict:
        found = self._records.get(command_id)
        if found is None:
            raise KeyError(command_id)
        if found["record"]["status"] == "completed":
            raise ValueError("завершённую команду нельзя отменить")
        found = self._writable(command_id)
        found["record"]["status"] = "cancelled"
        self._index(command_id)
        return dict(found["record"])

    def complete(self, command_id: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] not in {"accepted", "running"}:
            return
        found = self._writable(command_id)
        found["record"]["status"] = "completed"
        self._index(command_id)

    def mark_running(self, command_id: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] != "accepted":
            return
        found = self._writable(command_id)
        found["record"]["status"] = "running"
        self._index(command_id)

    def arm_window(self, command_id: str, now_sec: float) -> None:
        found = self._records.get(command_id)
        if found is None:
            return
        duration = float(found["record"].get("duration_sec") or 0.0)
        if duration <= 0.0:
            return
        found = self._writable(command_id)
        found["record"]["started_sec"] = float(now_sec)
        found["record"]["until_sec"] = float(now_sec) + duration
        self._index(command_id)

    def running_orders(self) -> dict[str, str]:
        """Команда с ненулевой длительностью, срок которой ещё не вышел."""
        chosen: dict[str, str] = {}
        for command_id in self._running:
            record = self._records[command_id]["record"]
            chosen[record["actor_id"]] = record["kind"]
        return chosen

    def running_ids(self) -> list[str]:
        return list(self._running)

    def running_actors(self) -> set[str]:
        return set(self.running_orders())

    def check(self, command: MissionCommand) -> None:
        """Проверяет команду, не записывая её. Конфликт всплывает до шага мира."""
        if command.kind not in {"wait", "imaging", "comms", "cancel"}:
            raise ValueError(f"команда {command.kind} не поддерживается")
        if not math.isfinite(float(command.duration_sec)) or float(command.duration_sec) < 0:
            raise ValueError("длительность должна быть конечной и неотрицательной")
        body = _body(command)
        existing = self._records.get(command.command_id)
        if existing is not None and existing["body"] != body:
            raise ValueError(f"команда {command.command_id} уже записана с другим телом")

    def fail(self, command_id: str, reason: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] not in {"accepted", "running"}:
            return
        found = self._writable(command_id)
        found["record"]["status"] = "failed"
        found["record"]["reason"] = reason
        self._index(command_id)

    def reject(self, command_id: str, reason: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] not in {"accepted", "running"}:
            return
        found = self._writable(command_id)
        found["record"]["status"] = "rejected"
        found["record"]["reason"] = reason
        self._index(command_id)

    def get(self, command_id: str) -> dict:
        found = self._records.get(command_id)
        if found is None:
            raise KeyError(command_id)
        return dict(found["record"])

    def export(self) -> dict:
        import copy

        return copy.deepcopy(self._records)

    def _writable(self, command_id: str) -> dict:
        """Records are replaced, never mutated across an internal checkpoint."""
        old = self._records[command_id]
        found = {"body": old["body"], "record": dict(old["record"])}
        self._records[command_id] = found
        return found

    def checkpoint(self) -> dict:
        """Internal copy-on-write checkpoint; use export for an external artifact."""
        return dict(self._records)

    def restore_checkpoint(self, records: dict) -> None:
        self._records = dict(records)
        self._reindex()

    def active_windows(self) -> list[dict]:
        return [dict(self._records[command_id]["record"]) for command_id in self._windows]

    def _index(self, command_id: str) -> None:
        record = self._records[command_id]["record"]
        if record["status"] == "running":
            self._running[command_id] = None
        else:
            self._running.pop(command_id, None)
        if record["status"] in {"accepted", "running"} and record.get("until_sec") is not None:
            self._windows[command_id] = None
        else:
            self._windows.pop(command_id, None)

    def _reindex(self) -> None:
        self._running = {}
        self._windows = {}
        for command_id in self._records:
            self._index(command_id)

    def load(self, records: dict) -> None:
        import copy

        self._records = copy.deepcopy(records)
        for found in self._records.values():
            found["body"] = tuple(found["body"])
        self._reindex()

    def orders(self) -> dict[str, str]:
        """По одной принятой команде на аппарат. Отменённые не исполняются."""
        chosen: dict[str, str] = {}
        for found in self._records.values():
            record = found["record"]
            if record["status"] != "accepted":
                continue
            chosen[record["actor_id"]] = record["kind"]
        return chosen
