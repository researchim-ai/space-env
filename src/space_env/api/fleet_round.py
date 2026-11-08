"""Bounded ephemeral proposals; one atomic native clock step per fleet round."""

from __future__ import annotations

import copy
import secrets
import threading
import time

from pydantic import Field

from space_env.runtime.schema import StrictModel


class RoundRequest(StrictModel):
    operation: str
    round_id: str | None = Field(default=None, max_length=64)
    action_ids: tuple[str, ...] = Field(default=(), max_length=64)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class FleetRound:
    """Agents submit role capabilities; operator commits once all roles voted."""

    def __init__(self, operator):
        self.operator = operator
        self.nonce = secrets.token_urlsafe(16)
        self.roles = tuple(
            sorted(
                k
                for k, r in operator.session._task.roles.items()
                if r.actors and not r.demand_actors and not r.stations
            )
        )
        if not self.roles:
            raise ValueError("Нет ролей аппаратов для совместного решения.")
        self.revision = operator.session.revision
        self.epoch = operator.control["epoch"]
        self.votes = {}
        self.receipt = None
        self.commit_key = None
        self.lock = threading.RLock()

    def _guard(self):
        from space_env.api.operator import OperatorError

        self.operator._guard()
        if (
            self.operator.session.revision != self.revision
            or self.operator.control["epoch"] != self.epoch
        ):
            raise OperatorError(
                "stale_round",
                "Миссия или управляющий канал изменились; откройте новый раунд.",
                409,
            )

    def observe(self, token):
        with self.lock, self.operator.lock, self.operator.session._lock:
            self._guard()
            role, _, _ = self.operator.permission(token)
            if role not in self.roles:
                raise ValueError("Нужна capability отдельной роли аппарата.")
            return self.operator.state(token)

    def submit(self, token, action_ids):
        from space_env.api.operator import OperatorError

        with self.lock, self.operator.lock, self.operator.session._lock:
            self._guard()
            role, write, _ = self.operator.permission(token)
            if not write or role not in self.roles:
                raise OperatorError(
                    "forbidden", "Нужна capability участника раунда.", 403
                )
            ids = tuple(action_ids)
            if (
                not ids
                or len(set(ids)) != len(ids)
                or len(ids) > self.operator.session._task.roles[role].action_slots
            ):
                raise ValueError("Неверное число действий роли.")
            for aid in ids:
                action = self.operator.session._by_id.get(aid)
                if aid != "wait" and (
                    action is None or action.role != role or action.kind == "inject"
                ):
                    raise OperatorError(
                        "forbidden", "Действие вне роли участника.", 403
                    )
            if role in self.votes and self.votes[role] != ids:
                raise OperatorError(
                    "vote_conflict",
                    "Решение роли уже зафиксировано в этом раунде.",
                    409,
                )
            self.votes[role] = ids
            return {
                "accepted": True,
                "revision": self.revision,
                "submitted": len(self.votes),
                "required": len(self.roles),
            }

    def commit(self, token, idempotency_key):
        from space_env.api.operator import OperatorError, token_digest

        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 128
        ):
            raise ValueError("Нужен idempotency_key до128 символов.")
        with self.lock, self.operator.lock, self.operator.session._lock:
            role, write, _ = self.operator.permission(token)
            if (
                role != "@central"
                or not write
                or self.operator.identities[token_digest(token)] != "operator"
            ):
                raise OperatorError(
                    "forbidden", "Совместный шаг подтверждает оператор.", 403
                )
            if self.receipt is not None:
                if idempotency_key != self.commit_key:
                    raise OperatorError(
                        "idempotency_conflict",
                        "Раунд уже исполнен с другим ключом.",
                        409,
                    )
                return copy.deepcopy(self.receipt)
            self._guard()
            if set(self.votes) != set(self.roles):
                raise OperatorError(
                    "round_incomplete", "Ожидаются решения остальных аппаратов.", 409
                )
            ids = sorted(
                a for values in self.votes.values() for a in values if a != "wait"
            ) or ["wait"]
            result = self.operator.step(
                token,
                {
                    "action_ids": ids,
                    "expected_revision": self.revision,
                    "control_epoch": self.epoch,
                    "advance_sec": self.operator.session._task.decision_sec,
                    "idempotency_key": "fleet/" + self.nonce + "/" + idempotency_key,
                },
            )
            self.receipt, self.commit_key = copy.deepcopy(result), idempotency_key
            return result


_LOCK = threading.RLock()
_ROUNDS = {}


def dispatch_round(operator, token, payload):
    from space_env.api.operator import OperatorError, token_digest

    request = RoundRequest.model_validate(payload)
    with _LOCK:
        now = time.monotonic()
        for key, value in list(_ROUNDS.items()):
            if now - value[0] > 900:
                del _ROUNDS[key]
        if request.operation == "create":
            role, write, _ = operator.permission(token)
            if (
                role != "@central"
                or not write
                or operator.identities[token_digest(token)] != "operator"
            ):
                raise OperatorError("forbidden", "Раунд открывает оператор.", 403)
            if len(_ROUNDS) >= 32:
                # Receipts are a bounded retry cache, not active proposals.
                # Reclaim an old committed round so sequential decisions do
                # not exhaust the live-round limit after 32 clock steps.
                finished = next(
                    (
                        key
                        for key, (_, old) in _ROUNDS.items()
                        if old.receipt is not None
                    ),
                    None,
                )
                if finished is None:
                    raise OperatorError("capacity", "Лимит открытых раундов.", 429)
                del _ROUNDS[finished]
            rid = secrets.token_urlsafe(18)
            round_ = FleetRound(operator)
            _ROUNDS[rid] = (now, round_)
            return {
                "round_id": rid,
                "roles": list(round_.roles),
                "revision": round_.revision,
                "control_epoch": round_.epoch,
            }
        entry = _ROUNDS.get(request.round_id)
        if entry is None or entry[1].operator is not operator:
            raise OperatorError("round_not_found", "Раунд отсутствует или истёк.", 404)
        round_ = entry[1]
        if request.operation == "submit":
            return round_.submit(token, request.action_ids)
        if request.operation == "observe":
            return round_.observe(token)
        if request.operation == "commit":
            return round_.commit(token, request.idempotency_key)
        raise ValueError("Неизвестная операция раунда.")
