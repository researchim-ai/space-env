"""Stateless capability-bound cursors over role-projected append-only histories."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import hmac
import json
import re
from typing import Literal

from pydantic import Field

from space_env.runtime.schema import StrictModel
from space_env.simulation.snapshot import canonical_json, digest


class EventPageRequest(StrictModel):
    channel: Literal["journal", "navigation"] = "journal"
    cursor: str | None = Field(default=None, min_length=1, max_length=2048)
    limit: int = Field(default=64, ge=1, le=128, strict=True)


class CursorConflict(ValueError):
    """A previously observed public prefix is no longer present."""


def _encode(value):
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid event cursor encoding")
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid event cursor encoding") from exc


def _sign(payload, token):
    return hmac.new(token.encode(), payload, hashlib.sha256).digest()


def page_history(entries, *, session_id, token, request, revision, control_epoch):
    """Page an already authorized projection; never accepts raw native history."""
    request = EventPageRequest.model_validate(request)
    offset = 0
    if request.cursor is not None:
        try:
            encoded, signature = request.cursor.split(".")
        except ValueError as exc:
            raise ValueError("invalid event cursor") from exc
        raw = _decode(encoded)
        if not hmac.compare_digest(_sign(raw, token), _decode(signature)):
            raise ValueError("event cursor does not match this capability")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("invalid event cursor payload") from exc
        if (
            not isinstance(body, dict)
            or set(body)
            != {"schema", "session_id", "channel", "offset", "prefix_sha256"}
            or body["schema"] != "space-env.operator-cursor/1"
            or body["session_id"] != session_id
            or body["channel"] != request.channel
            or type(body["offset"]) is not int
            or body["offset"] < 0
            or not isinstance(body["prefix_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", body["prefix_sha256"])
        ):
            raise ValueError("event cursor scope or payload mismatch")
        offset = body["offset"]
        if offset > len(entries) or digest(entries[:offset]) != body["prefix_sha256"]:
            raise CursorConflict(
                "observed history prefix changed; request a fresh cursor"
            )
    end = min(offset + request.limit, len(entries))
    body = {
        "schema": "space-env.operator-cursor/1",
        "session_id": session_id,
        "channel": request.channel,
        "offset": end,
        "prefix_sha256": digest(entries[:end]),
    }
    raw = canonical_json(body).encode()
    return {
        "schema": "space-env.operator-events/1",
        "session_id": session_id,
        "channel": request.channel,
        "revision": revision,
        "control_epoch": control_epoch,
        "entries": [
            dict(sequence=i, **copy.deepcopy(entries[i])) for i in range(offset, end)
        ],
        "visible_count": len(entries),
        "has_more": end < len(entries),
        "cursor": _encode(raw) + "." + _encode(_sign(raw, token)),
    }
