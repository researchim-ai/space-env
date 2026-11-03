"""Installed B10-C1 acceptance: native step, cursor, online backup, restore and tail."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from space_env.api.operator import OperatorStore
from space_env.api.operator_maintenance import (
    backup_operator_store,
    verify_operator_store,
)
from space_env.scaling.workloads import fleet_workload


def main():
    with tempfile.TemporaryDirectory(prefix="space-env-maintenance-") as directory:
        root = Path(directory)
        w = fleet_workload(
            {"actors": 2, "products": 4, "planes": 2, "repeats": 1, "horizon_sec": 12.0}
        )
        store = OperatorStore(root / "live.sqlite3")
        reply = store.create({"scenario": w["scenario"], "task": w["task"]})
        op = store.get(reply["session_id"])
        viewer, controller = reply["tokens"]["viewer"], reply["tokens"]["operator"]
        cursor = op.read_events(viewer, {})["cursor"]
        request = {
            "action_ids": ["wait"],
            "expected_revision": 0,
            "idempotency_key": "s1",
        }
        op.step(controller, request)
        first = op.read_events(viewer, {"cursor": cursor, "limit": 1})
        assert len(first["entries"]) == 1 and not first["has_more"]
        before = op.checkpoint(reply["tokens"]["admin"])
        receipt = backup_operator_store(root / "live.sqlite3", root / "backup.sqlite3")
        clone = OperatorStore(root / "backup.sqlite3").get(op.id)
        assert clone.checkpoint(reply["tokens"]["admin"]) == before
        assert clone.read_events(viewer, {"cursor": first["cursor"]})["entries"] == []
        clone.step(
            controller, {**request, "expected_revision": 1, "idempotency_key": "s2"}
        )
        assert (
            clone.read_events(viewer, {"cursor": first["cursor"]})["entries"][0][
                "sequence"
            ]
            == 1
        )
        assert op.session.revision == 1 and clone.session.revision == 2
        assert (
            verify_operator_store(root / "backup.sqlite3")["active_native_replayed"]
            == 1
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "source_revision": 1,
                    "restored_revision": 2,
                    "cursor_continuity": True,
                    "backup_sha256": receipt["sha256"],
                    "native_replay": True,
                    "independent_restore": True,
                }
            )
        )


if __name__ == "__main__":
    main()
