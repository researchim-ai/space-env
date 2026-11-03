"""Streaming artifact isolation, retries, bounded spool and native import semantics."""

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from space_env.api import operator
from space_env.api import operator_artifacts as artifacts
from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.operator_artifacts import (
    CHUNK_BYTES,
    FILE_BYTES,
    ArtifactChunk,
    ArtifactStore,
)
from space_env.api.server import handle_get, handle_post
from space_env.scaling.workloads import fleet_workload
from space_env.simulation.snapshot import canonical_json


@pytest.fixture(scope="module")
def config():
    w = fleet_workload(
        {"actors": 2, "products": 4, "planes": 2, "repeats": 1, "horizon_sec": 12.0}
    )
    return {"scenario": w["scenario"], "task": w["task"]}


@pytest.fixture
def opened(tmp_path, config):
    store = OperatorStore(tmp_path / "live.sqlite3")
    reply = store.create(copy.deepcopy(config))
    spool = ArtifactStore(parent=tmp_path)
    yield store, store.get(reply["session_id"]), reply, spool
    spool.close()


def data_for(op, reply, format="archive"):
    value = (
        op.archive(reply["tokens"]["admin"])
        if format == "archive"
        else op.checkpoint(reply["tokens"]["admin"])
    )
    return canonical_json(value).encode()


def create_upload(store, op, reply, spool, data, format="archive", **kwargs):
    return spool.create(
        store,
        op.id,
        reply["tokens"]["admin"],
        {
            "direction": "upload",
            "format": format,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            **kwargs,
        },
    )["transfer_id"]


def upload(spool, store, id, token, data, offset=0):
    result = None
    for start in range(0, len(data), CHUNK_BYTES):
        block = data[start : start + CHUNK_BYTES]
        result = spool.append(
            store,
            id,
            token,
            ArtifactChunk(offset + start, len(block), io.BytesIO(block)),
        )
    return result


@pytest.mark.parametrize("format", ["archive", "checkpoint", "training"])
def test_download_stream_matches_native_artifact(opened, format):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    before = op.checkpoint(token)
    status = spool.create(
        store, op.id, token, {"direction": "download", "format": format}
    )
    assert status["state"] == "ready" and not status["busy"]
    assert spool.jobs[status["transfer_id"]]["path"].stat().st_mode & 0o777 == 0o600
    with spool.download(store, status["transfer_id"], token) as body:
        chunks = list(body.iter_chunks(1024))
        assert all(len(c) <= 1024 for c in chunks)
        raw = b"".join(chunks)
        assert len(raw) == len(body)
        assert hashlib.sha256(raw).hexdigest() == body.sha256 == status["sha256"]
        if format == "training":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                assert "operator-provenance.json" in archive.namelist()
        else:
            expected = op.archive(token) if format == "archive" else before
            assert json.loads(raw) == expected
    assert not spool.status(store, status["transfer_id"], token)["busy"]
    assert op.checkpoint(token) == before


@pytest.mark.parametrize("format", ["archive", "checkpoint"])
def test_upload_commit_retries_create_exactly_one_native_session(opened, format):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = data_for(op, r, format)
    id = create_upload(store, op, r, spool, data, format)
    result = upload(spool, store, id, token, data)
    assert result["state"] == "ready" and not result["busy"]
    accepted = spool.commit(store, id, token)
    assert spool.commit(store, id, token) == accepted
    assert len(store.sessions) == 2
    clone = store.get(accepted["session_id"])
    assert clone.id != op.id and accepted["tokens"] != r["tokens"]
    assert clone.session.checkpoint(compact=True) == op.session.checkpoint(compact=True)
    assert clone.verify(accepted["tokens"]["admin"])["ok"]
    assert spool.reserved == 0


def test_chunk_retry_is_byte_exact_and_out_of_order_rejected(opened):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = data_for(op, r)
    id = create_upload(store, op, r, spool, data)
    first = data[:100]
    a = upload(spool, store, id, token, first)
    b = upload(spool, store, id, token, first)
    assert a["received_bytes"] == b["received_bytes"] == 100
    for offset, payload in [(0, b"x" * 100), (101, b"x")]:
        with pytest.raises(OperatorError) as error:
            upload(spool, store, id, token, payload, offset)
        assert error.value.status == 409
    assert spool.status(store, id, token)["received_bytes"] == 100
    upload(spool, store, id, token, data[100:], 100)
    assert spool.commit(store, id, token)["session_id"] != op.id


def test_truncated_and_timed_out_chunks_roll_back_for_retry(opened):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = data_for(op, r)
    id = create_upload(store, op, r, spool, data)
    with pytest.raises(OperatorError):
        spool.append(store, id, token, ArtifactChunk(0, 100, io.BytesIO(data[:50])))
    assert spool.jobs[id]["path"].stat().st_size == 0

    class Slow:
        def read(self, size):
            raise TimeoutError("peer stalled")

    with pytest.raises(TimeoutError):
        spool.append(store, id, token, ArtifactChunk(0, 100, Slow()))
    assert spool.status(store, id, token)["received_bytes"] == 0
    upload(spool, store, id, token, data)
    assert spool.status(store, id, token)["state"] == "ready"


def test_sha_mismatch_removes_bytes_and_quota(opened):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    id = spool.create(
        store,
        op.id,
        token,
        {"direction": "upload", "format": "archive", "bytes": 3, "sha256": "0" * 64},
    )["transfer_id"]
    with pytest.raises(OperatorError) as error:
        upload(spool, store, id, token, b"bad")
    assert error.value.status == 422 and spool.reserved == 0 and not spool.jobs
    assert not list(Path(spool.temp.name).iterdir())


@pytest.mark.parametrize("role", ["operator", "agent", "viewer"])
def test_only_admin_can_create_artifacts(opened, role):
    store, op, r, spool = opened
    with pytest.raises(OperatorError) as error:
        spool.create(
            store,
            op.id,
            r["tokens"][role],
            {"direction": "download", "format": "archive"},
        )
    assert error.value.status == 403 and spool.reserved == 0 and spool.temp is None


def test_session_and_token_rotation_revoke_future_operations(opened, config):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = data_for(op, r)
    id = create_upload(store, op, r, spool, data)
    other = store.create(copy.deepcopy(config))
    with pytest.raises(OperatorError):
        spool.status(store, id, other["tokens"]["admin"])
    rotated = store.recover(op.id)
    with pytest.raises(OperatorError):
        spool.status(store, id, token)
    with pytest.raises(OperatorError):
        spool.status(store, id, rotated["tokens"]["admin"])


def test_cancellation_and_busy_download_lease(opened):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    status = spool.create(
        store, op.id, token, {"direction": "download", "format": "archive"}
    )
    id = status["transfer_id"]
    with spool.download(store, id, token) as body:
        next(body.iter_chunks(10))
        with pytest.raises(OperatorError) as error:
            spool.cancel(store, id, token)
        assert error.value.status == 409
        with pytest.raises(OperatorError):
            spool.close()
    assert spool.cancel(store, id, token) == {"ok": True}
    assert not spool.jobs and spool.reserved == 0


def test_expiry_does_not_delete_a_leased_download(opened, monkeypatch):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    id = spool.create(
        store, op.id, token, {"direction": "download", "format": "archive"}
    )["transfer_id"]
    now = artifacts.time.monotonic()
    with spool.download(store, id, token):
        monkeypatch.setattr(artifacts.time, "monotonic", lambda: now + 10000)
        assert spool.cleanup() == 0
    monkeypatch.setattr(artifacts.time, "monotonic", lambda: now + 20000)
    assert spool.cleanup() == 1 and spool.reserved == 0


def test_job_and_byte_reservation_budgets(opened, tmp_path):
    store, op, r, _ = opened
    token = r["tokens"]["admin"]
    spool = ArtifactStore(parent=tmp_path, quota_bytes=20, max_jobs=1)
    try:
        id = create_upload(store, op, r, spool, b"1234567890")
        with pytest.raises(OperatorError) as error:
            create_upload(store, op, r, spool, b"123")
        assert error.value.status == 429
        spool.cancel(store, id, token)
        with pytest.raises(OperatorError):
            create_upload(store, op, r, spool, b"x" * 21)
        assert spool.reserved == 0
    finally:
        spool.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"direction": "upload", "format": "training", "bytes": 10, "sha256": "a" * 64},
        {"direction": "upload", "format": "archive", "bytes": True, "sha256": "a" * 64},
        {
            "direction": "upload",
            "format": "archive",
            "bytes": FILE_BYTES + 1,
            "sha256": "a" * 64,
        },
        {"direction": "upload", "format": "archive", "bytes": 1},
        {"direction": "download", "format": "archive", "takeover": True},
        {
            "direction": "upload",
            "format": "checkpoint",
            "bytes": 1,
            "sha256": "a" * 64,
            "takeover": "true",
        },
        {"direction": "download", "format": "archive", "filename": "../escape"},
    ],
)
def test_strict_transfer_metadata(opened, payload):
    store, op, r, spool = opened
    with pytest.raises(ValueError):
        spool.create(store, op.id, r["tokens"]["admin"], payload)
    assert spool.reserved == 0 and not spool.jobs


def test_invalid_native_json_never_installs_a_session(opened):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = b'{"schema":"not-an-archive"}'
    id = create_upload(store, op, r, spool, data)
    upload(spool, store, id, token, data)
    with pytest.raises(ValueError):
        spool.commit(store, id, token)
    assert len(store.sessions) == 1
    assert spool.status(store, id, token)["state"] == "failed"


def test_export_to_does_not_overwrite_and_cleans_budget_failure(opened, tmp_path):
    _, op, r, _ = opened
    token = r["tokens"]["admin"]
    path = tmp_path / "existing.zip"
    path.write_bytes(b"owner data")
    with pytest.raises(FileExistsError):
        op.export_to(token, path)
    assert path.read_bytes() == b"owner data"
    path = tmp_path / "too-small.zip"
    with pytest.raises(ValueError):
        op.export_to(token, path, max_bytes=5)
    assert not path.exists()


def test_no_global_authentication_bypass_via_direct_routes(opened, monkeypatch):
    store, op, r, spool = opened
    monkeypatch.setattr(operator, "STORE", store)
    monkeypatch.setattr(artifacts, "ARTIFACTS", spool)
    route = f"/api/operator/sessions/{op.id}/artifacts"
    assert handle_post(route, b'{"direction":"download","format":"archive"}')[0] == 401
    status, raw, _ = handle_post(
        route,
        b'{"direction":"download","format":"archive"}',
        token=r["tokens"]["admin"],
    )
    assert status == 200
    id = json.loads(raw)["transfer_id"]
    status, body, kind = handle_get(
        f"/api/operator/transfers/{id}/content", {}, token=r["tokens"]["admin"]
    )
    assert status == 200 and kind.startswith("application/json")
    with body:
        assert (
            json.loads(b"".join(body.iter_chunks()))["schema"]
            == "space-env.operator-archive/1"
        )
    assert (
        handle_post(
            f"/api/operator/transfers/{id}/cancel", b"{}", token=r["tokens"]["viewer"]
        )[0]
        == 401
    )


def test_cleanup_failure_keeps_accounting_for_retry(opened, monkeypatch):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    id = create_upload(store, op, r, spool, b"abc")
    original = artifacts.shutil.rmtree

    def fail(*args, **kwargs):
        raise OSError("filesystem busy")

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.shutil, "rmtree", fail)
        with pytest.raises(OSError):
            spool.cancel(store, id, token)
        assert spool.reserved == 3 and id in spool.jobs
    assert artifacts.shutil.rmtree == original
    spool.cancel(store, id, token)
    assert spool.reserved == 0


def test_failed_create_releases_private_directory(opened, monkeypatch):
    store, op, r, spool = opened
    spool._root()
    original = artifacts.os.open

    def fail(path, *args, **kwargs):
        if Path(path).name == "payload":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, "open", fail)
        with pytest.raises(OSError):
            create_upload(store, op, r, spool, b"abc")
    assert (
        not spool.jobs
        and spool.reserved == 0
        and not list(Path(spool.temp.name).iterdir())
    )


def test_unacknowledged_native_import_is_not_blindly_repeated(opened, monkeypatch):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = data_for(op, r)
    id = create_upload(store, op, r, spool, data)
    upload(spool, store, id, token, data)
    restore = store.restore

    def lost_ack(payload):
        restore(payload)
        raise OSError("native import committed before response failed")

    monkeypatch.setattr(store, "restore", lost_ack)
    with pytest.raises(OSError):
        spool.commit(store, id, token)
    assert len(store.sessions) == 2
    status = spool.status(store, id, token)
    assert status["state"] == "failed" and status["result_session_id"] in store.sessions
    with pytest.raises(OperatorError):
        spool.commit(store, id, token)
    assert len(store.sessions) == 2


def test_post_commit_cleanup_failure_keeps_quota_and_cached_receipt(
    opened, monkeypatch
):
    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = data_for(op, r)
    id = create_upload(store, op, r, spool, data)
    upload(spool, store, id, token, data)
    unlink = Path.unlink

    def fail_payload(path, *args, **kwargs):
        if path == spool.jobs[id]["path"]:
            raise OSError("cleanup busy")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_payload)
        result = spool.commit(store, id, token)
        assert spool.reserved == len(data)
        assert spool.commit(store, id, token) == result
    spool.cancel(store, id, token)
    assert spool.reserved == 0


def test_network_upload_does_not_hold_the_global_mission_lock(opened):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    store, op, r, spool = opened
    token = r["tokens"]["admin"]
    data = data_for(op, r)
    id = create_upload(store, op, r, spool, data)
    entered, release = threading.Event(), threading.Event()

    class SlowStream(io.BytesIO):
        def read(self, size):
            entered.set()
            assert release.wait(5)
            return super().read(size)

    with ThreadPoolExecutor(2) as pool:
        transfer = pool.submit(
            spool.append,
            store,
            id,
            token,
            ArtifactChunk(0, len(data), SlowStream(data)),
        )
        try:
            assert entered.wait(2)
            mission = pool.submit(
                op.step,
                r["tokens"]["operator"],
                {
                    "action_ids": ["wait"],
                    "expected_revision": 0,
                    "idempotency_key": "during-upload",
                },
            )
            assert mission.result(timeout=2)["feedback"]["accepted"]
        finally:
            release.set()
        assert transfer.result(timeout=5)["state"] == "ready"
    assert op.session.revision == 1
