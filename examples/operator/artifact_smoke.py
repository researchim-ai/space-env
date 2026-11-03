"""Actual HTTP artifact transport, including a >8 MiB padded JSON wire fixture.

Padding tests HTTP transport limits, not long-mission fidelity or memory qualification.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import tempfile
import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from space_env.api import operator, operator_artifacts
from space_env.api.operator_artifacts import CHUNK_BYTES, ArtifactStore
from space_env.api.server import _Handler
from space_env.scaling.workloads import fleet_workload


def main():
    previous_store, previous_artifacts = operator.STORE, operator_artifacts.ARTIFACTS
    with tempfile.TemporaryDirectory(prefix="operator-artifact-smoke-") as directory:
        root = Path(directory)
        store = operator.OperatorStore(root / "sessions.sqlite3")
        spool = ArtifactStore(parent=root)
        w = fleet_workload(
            {"actors": 2, "products": 4, "planes": 2, "repeats": 1, "horizon_sec": 12.0}
        )
        opened = store.create({"scenario": w["scenario"], "task": w["task"]})
        session = store.get(opened["session_id"])
        token = opened["tokens"]["admin"]
        operator.STORE, operator_artifacts.ARTIFACTS = store, spool
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, payload=None, *, method="GET", raw=None, headers=None):
            values = {"Authorization": f"Bearer {token}"}
            if payload is not None:
                raw = json.dumps(payload).encode()
                values["Content-Type"] = "application/json"
            values.update(headers or {})
            return urlopen(
                Request(base + path, data=raw, method=method, headers=values),
                timeout=60,
            )

        def control(path, payload):
            with request(path, payload, method="POST") as response:
                return json.load(response)

        def create(payload):
            return control(f"/api/operator/sessions/{session.id}/artifacts", payload)

        try:
            original = session.checkpoint(token)
            archive = create({"direction": "download", "format": "archive"})
            with request(
                f"/api/operator/transfers/{archive['transfer_id']}/content"
            ) as response:
                assert response.headers["X-Artifact-SHA256"] == archive["sha256"]
                assert response.headers["Cache-Control"] == "no-store"
                data = response.read()
            assert hashlib.sha256(data).hexdigest() == archive["sha256"]
            control(f"/api/operator/transfers/{archive['transfer_id']}/cancel", {})
            data += b" " * (9 * 1024 * 1024)

            def header_status(method, path, headers):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=15
                )
                try:
                    connection.putrequest(method, path)
                    for key, value in headers.items():
                        connection.putheader(key, str(value))
                    connection.endheaders()
                    response = connection.getresponse()
                    status = response.status
                    response.read()
                    return status
                finally:
                    connection.close()

            assert (
                header_status(
                    "POST",
                    "/api/operator/restore",
                    {
                        "Content-Length": len(data),
                        "Content-Type": "application/json",
                    },
                )
                == 413
            )
            job = create(
                {
                    "direction": "upload",
                    "format": "archive",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            path = f"/api/operator/transfers/{job['transfer_id']}"
            headers = {
                "Content-Length": 1,
                "Content-Type": "application/octet-stream",
                "X-Upload-Offset": "0",
                "Authorization": f"Bearer {token}",
            }
            assert (
                header_status(
                    "PUT", path + "/chunks", {**headers, "Content-Type": "text/plain"}
                )
                == 415
            )
            assert (
                header_status(
                    "PUT",
                    path + "/chunks",
                    {**headers, "Content-Length": CHUNK_BYTES + 1},
                )
                == 413
            )
            assert (
                header_status(
                    "PUT", path + "/chunks", {**headers, "X-Upload-Offset": "-1"}
                )
                == 400
            )
            assert (
                header_status(
                    "PUT", path + "/chunks", {**headers, "Transfer-Encoding": "chunked"}
                )
                == 400
            )
            assert (
                header_status(
                    "PUT",
                    path + "/chunks",
                    {**headers, "Origin": "https://foreign.invalid"},
                )
                == 403
            )
            assert (
                header_status(
                    "PUT",
                    path + "/chunks",
                    {**headers, "Authorization": "Bearer wrong"},
                )
                == 401
            )
            assert (
                header_status(
                    "PUT", path + "/chunks", {**headers, "X-Upload-Offset": "1"}
                )
                == 409
            )
            for offset in range(0, len(data), CHUNK_BYTES):
                block = data[offset : offset + CHUNK_BYTES]
                with request(
                    path + "/chunks",
                    method="PUT",
                    raw=block,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Upload-Offset": str(offset),
                    },
                ) as response:
                    result = json.load(response)
                assert result["received_bytes"] == offset + len(block)
                if offset == 0:
                    with request(
                        path + "/chunks",
                        method="PUT",
                        raw=block,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-Upload-Offset": "0",
                        },
                    ) as response:
                        assert json.load(response)["received_bytes"] == len(block)
            restored = control(path + "/commit", {})
            assert control(path + "/commit", {}) == restored
            assert len(store.sessions) == 2
            assert (
                store.get(restored["session_id"]).checkpoint(
                    restored["tokens"]["admin"]
                )
                == original
            )
            control(path + "/cancel", {})
            training = create({"direction": "download", "format": "training"})
            with request(
                f"/api/operator/transfers/{training['transfer_id']}/content"
            ) as response:
                target = root / "training.zip"
                with target.open("wb") as output:
                    while block := response.read(65536):
                        output.write(block)
            with zipfile.ZipFile(target) as bundle:
                assert "operator-provenance.json" in bundle.namelist()
            control(f"/api/operator/transfers/{training['transfer_id']}/cancel", {})
            assert spool.reserved == 0 and not spool.jobs
            print(
                json.dumps(
                    {
                        "ok": True,
                        "http": True,
                        "wire_bytes": len(data),
                        "wire_fixture": "native archive plus whitespace padding",
                        "legacy_guard": 413,
                        "chunk_retry": True,
                        "native_restore": True,
                        "idempotent_commit": True,
                        "verified_training_zip": True,
                        "released_spool": True,
                    }
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
            spool.close()
            operator.STORE, operator_artifacts.ARTIFACTS = (
                previous_store,
                previous_artifacts,
            )


if __name__ == "__main__":
    main()
