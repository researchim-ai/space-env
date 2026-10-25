"""Исход миссии отдельно от того, что команда была принята."""
from __future__ import annotations


def classify_outcome(
    *,
    command_accepted: bool,
    operation_completed: bool,
    request_delivered: bool,
    safe: bool = True,
) -> dict[str, bool | str]:
    """Четыре уровня: команда, операция, сдача заявки, успех миссии."""
    mission = bool(command_accepted and operation_completed and request_delivered and safe)
    if not command_accepted:
        reason = "команда не принята"
    elif not operation_completed:
        reason = "операция не завершена"
    elif not request_delivered:
        reason = "биты не сданы на станцию"
    elif not safe:
        reason = "нарушено ограничение"
    else:
        reason = ""
    return {
        "command_accepted": bool(command_accepted),
        "operation_completed": bool(operation_completed),
        "request_delivered": bool(request_delivered),
        "mission_succeeded": mission,
        "unmet_reason": reason,
    }
