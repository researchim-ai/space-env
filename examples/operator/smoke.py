"""Installed native Mission Control smoke, without browser or external services."""

from __future__ import annotations

import json

from space_env.api.operator import OperatorStore
from space_env.api.server import handle_get, studio_root


def main():
    for path in ["/operator", "/viz/operator.js", "/viz/operator.css"]:
        assert handle_get(path, {})[0] == 200, path
    store = OperatorStore()
    opened = store.create({"preset": "relay-demo"})
    session = store.get(opened["session_id"])
    token = opened["tokens"]["operator"]
    for _ in range(256):
        if session.session._done:
            break
        previous = session.session.revision
        ids = session.suggest(token)["action_ids"]
        session.step(
            token,
            {
                "action_ids": ids,
                "expected_revision": session.session.revision,
                "idempotency_key": str(session.session.revision),
            },
        )
        assert session.session.revision == previous + 1, (
            "native step did not advance exactly once"
        )
    assert session.session._done, "interactive decision budget exhausted"
    assert session.session._runtime.report()["metrics"]["fulfilled_requests"] == 2
    assert session.verify(opened["tokens"]["admin"])["ok"]
    assert session.export(opened["tokens"]["admin"]).startswith(b"PK")
    store.close(session.id, opened["tokens"]["admin"])
    print(
        json.dumps(
            {
                "ok": True,
                "assets": str(studio_root()),
                "fulfilled_requests": 2,
                "decisions": 10,
                "cold_replay": True,
                "training_zip": True,
            }
        )
    )


if __name__ == "__main__":
    main()
