"""Команда с идентификатором: повтор той же команды ничего не меняет, другая — конфликт."""
from __future__ import annotations

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

    def submit(self, command: MissionCommand) -> dict:
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
        found["record"]["status"] = "cancelled"
        return dict(found["record"])

    def complete(self, command_id: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] not in {"accepted", "running"}:
            return
        found["record"]["status"] = "completed"

    def mark_running(self, command_id: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] != "accepted":
            return
        found["record"]["status"] = "running"

    def arm_window(self, command_id: str, now_sec: float) -> None:
        found = self._records.get(command_id)
        if found is None:
            return
        duration = float(found["record"].get("duration_sec") or 0.0)
        if duration <= 0.0:
            return
        found["record"]["started_sec"] = float(now_sec)
        found["record"]["until_sec"] = float(now_sec) + duration

    def running_orders(self) -> dict[str, str]:
        """Команда с ненулевой длительностью, срок которой ещё не вышел."""
        chosen: dict[str, str] = {}
        for found in self._records.values():
            record = found["record"]
            if record["status"] != "running":
                continue
            chosen[record["actor_id"]] = record["kind"]
        return chosen

    def running_ids(self) -> list[str]:
        return [found["record"]["command_id"] for found in self._records.values() if found["record"]["status"] == "running"]

    def running_actors(self) -> set[str]:
        return set(self.running_orders())

    def check(self, command: MissionCommand) -> None:
        """Проверяет команду, не записывая её. Конфликт всплывает до шага мира."""
        if command.kind not in {"wait", "imaging", "comms", "cancel"}:
            raise ValueError(f"команда {command.kind} не поддерживается")
        body = _body(command)
        existing = self._records.get(command.command_id)
        if existing is not None and existing["body"] != body:
            raise ValueError(f"команда {command.command_id} уже записана с другим телом")

    def fail(self, command_id: str, reason: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] not in {"accepted", "running"}:
            return
        found["record"]["status"] = "failed"
        found["record"]["reason"] = reason

    def reject(self, command_id: str, reason: str) -> None:
        found = self._records.get(command_id)
        if found is None or found["record"]["status"] != "accepted":
            return
        found["record"]["status"] = "rejected"
        found["record"]["reason"] = reason

    def get(self, command_id: str) -> dict:
        found = self._records.get(command_id)
        if found is None:
            raise KeyError(command_id)
        return dict(found["record"])

    def export(self) -> dict:
        import copy

        return copy.deepcopy(self._records)

    def load(self, records: dict) -> None:
        import copy

        self._records = copy.deepcopy(records)

    def orders(self) -> dict[str, str]:
        """По одной принятой команде на аппарат. Отменённые не исполняются."""
        chosen: dict[str, str] = {}
        for found in self._records.values():
            record = found["record"]
            if record["status"] != "accepted":
                continue
            chosen[record["actor_id"]] = record["kind"]
        return chosen
