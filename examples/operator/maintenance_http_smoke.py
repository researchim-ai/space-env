"""Real loopback HTTP cursor continuation against a restored native SQLite store."""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from space_env.api import operator
from space_env.api.operator_maintenance import backup_operator_store
from space_env.api.server import _Handler
from space_env.scaling.workloads import fleet_workload


def main():
    previous = operator.STORE
    with tempfile.TemporaryDirectory(prefix="operator-http-maintenance-") as directory:
        root = Path(directory)
        store = operator.OperatorStore(root / "live.sqlite3")
        workload = fleet_workload(
            {"actors": 2, "products": 4, "planes": 2, "repeats": 1, "horizon_sec": 12.0}
        )
        opened = store.create(
            {"scenario": workload["scenario"], "task": workload["task"]}
        )
        session = store.get(opened["session_id"])
        token = opened["tokens"]["viewer"]
        operator.STORE = store
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        route = f"http://127.0.0.1:{server.server_port}/api/operator/sessions/{session.id}/events"

        def read(payload, bearer):
            request = Request(
                route,
                data=json.dumps(payload).encode(),
                method="POST",
                headers={
                    "Authorization": f"Bearer {bearer}",
                    "Content-Type": "application/json",
                },
            )
            with urlopen(request, timeout=60) as response:
                assert response.headers["Cache-Control"] == "no-store"
                return json.load(response)

        try:
            initial = read({}, token)
            session.step(
                opened["tokens"]["operator"],
                {
                    "action_ids": ["wait"],
                    "expected_revision": 0,
                    "idempotency_key": "first",
                },
            )
            prefix = read({"cursor": initial["cursor"], "limit": 1}, token)
            assert prefix["entries"][0]["sequence"] == 0
            backup_operator_store(root / "live.sqlite3", root / "snapshot.sqlite3")
            operator.STORE = operator.OperatorStore(root / "snapshot.sqlite3")
            assert read({"cursor": prefix["cursor"]}, token)["entries"] == []
            restored = operator.STORE.get(session.id)
            restored.step(
                opened["tokens"]["operator"],
                {
                    "action_ids": ["wait"],
                    "expected_revision": 1,
                    "idempotency_key": "second",
                },
            )
            suffix = read({"cursor": prefix["cursor"]}, token)
            assert suffix["entries"][0]["sequence"] == 1
            assert suffix["revision"] == 2 and session.session.revision == 1
            try:
                read({"cursor": prefix["cursor"]}, opened["tokens"]["operator"])
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("cursor accepted for a different capability")
            print(
                json.dumps(
                    {
                        "ok": True,
                        "http": True,
                        "backup_restore": True,
                        "scoped_cursor_continuity": True,
                        "cross_capability_rejected": True,
                    }
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            operator.STORE = previous


if __name__ == "__main__":
    main()
