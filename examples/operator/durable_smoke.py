"""Restart and controller handover acceptance against the native relay mission."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

from space_env.api.operator import OperatorStore


def main():
    with tempfile.TemporaryDirectory(prefix="space-env-durable-") as root:
        path = Path(root) / "operator.sqlite3"
        store = OperatorStore(path)
        opened = store.create({"preset": "relay-demo"})
        op = store.get(opened["session_id"])
        tokens = opened["tokens"]
        for index in range(256):
            if op.session._done:
                break
            if index in (2, 6):
                op.handover(
                    tokens["admin"],
                    {
                        "controller": "agent" if index == 2 else "operator",
                        "expected_revision": op.session.revision,
                        "expected_control_epoch": op.control["epoch"],
                        "idempotency_key": f"handover-{index}",
                        "reason": "Native relay acceptance controller transfer",
                    },
                )
            controller = op.control["controller"]
            op.step(
                tokens[controller],
                {
                    "action_ids": op.suggest(tokens["operator"])["action_ids"],
                    "expected_revision": op.session.revision,
                    "control_epoch": op.control["epoch"],
                    "idempotency_key": f"step-{index}",
                },
            )
            if index == 4:
                before = op.state(tokens["viewer"])
                store = OperatorStore(path)
                op = store.get(opened["session_id"])
                assert op.state(tokens["viewer"]) == before
        assert op.session._done
        assert op.session._runtime.report()["metrics"]["fulfilled_requests"] == 2
        assert op.verify(tokens["admin"])["ok"]
        with zipfile.ZipFile(io.BytesIO(op.export(tokens["admin"]))) as archive:
            provenance = json.loads(archive.read("operator-provenance.json"))
            assert provenance
        copied = store.restore({"archive": op.archive(tokens["admin"])})
        clone = store.get(copied["session_id"])
        assert clone.session.checkpoint() == op.session.checkpoint()
        assert clone.control == op.control
        print(
            json.dumps(
                {
                    "ok": True,
                    "fulfilled_requests": 2,
                    "decisions": op.session.revision,
                    "restart": True,
                    "controller_handover": True,
                    "archive": True,
                    "native_replay": True,
                    "training_zip": True,
                }
            )
        )


if __name__ == "__main__":
    main()
