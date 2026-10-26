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


class CommandBook:
    """submitted → accepted → completed, failed или cancelled."""

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}

    def submit(self, command: MissionCommand) -> dict:
        if command.kind not in {"wait", "imaging", "comms", "cancel"}:
            raise ValueError(f"команда {command.kind} не поддерживается")
        body = (
            command.actor_id,
            command.kind,
            command.request_id,
            command.destination_id,
        )
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
        if found is None or found["record"]["status"] != "accepted":
            return
        found["record"]["status"] = "completed"

    def orders(self) -> dict[str, str]:
        """По одной принятой команде на аппарат. Отменённые не исполняются."""
        chosen: dict[str, str] = {}
        for found in self._records.values():
            record = found["record"]
            if record["status"] != "accepted":
                continue
            chosen[record["actor_id"]] = record["kind"]
        return chosen
