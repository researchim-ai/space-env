"""B10-C1: scoped cursor continuity, coherent online backup and exclusive recovery."""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from space_env.api import operator
from space_env.api import operator_maintenance as maintenance
from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.server import handle_post
from space_env.scaling.workloads import fleet_workload


@pytest.fixture(scope="module")
def config():
    w = fleet_workload(
        {"actors": 2, "products": 4, "planes": 2, "repeats": 1, "horizon_sec": 12.0}
    )
    value = {"scenario": w["scenario"], "task": w["task"]}
    names = value["scenario"]["actor_order"]
    value["task"]["roles"] = {name: {"actors": [name]} for name in names}
    for action in value["task"]["actions"]:
        action["role"] = action["commands"][0]["actor_id"]
    return value


@pytest.fixture
def opened(tmp_path, config):
    path = tmp_path / "live.sqlite3"
    store = OperatorStore(path)
    reply = store.create(copy.deepcopy(config))
    return path, store, store.get(reply["session_id"]), reply


def step(op, reply, *, role="operator"):
    return op.step(
        reply["tokens"][role],
        {
            "action_ids": ["wait"],
            "expected_revision": op.session.revision,
            "control_epoch": op.control["epoch"],
            "idempotency_key": str(op.session.revision),
        },
    )


def handover(op, reply, controller="agent"):
    return op.handover(
        reply["tokens"]["admin"],
        {
            "controller": controller,
            "expected_revision": op.session.revision,
            "expected_control_epoch": op.control["epoch"],
            "idempotency_key": f"h{op.control['epoch']}",
            "reason": "recorded handover",
        },
    )


def test_empty_cursor_tail_and_exactly_once_paging(opened):
    _, _, op, reply = opened
    token = reply["tokens"]["viewer"]
    head = op.read_events(token, {"limit": 1})
    assert head["entries"] == [] and not head["has_more"]
    step(op, reply)
    first = op.read_events(token, {"limit": 1, "cursor": head["cursor"]})
    assert first["entries"][0]["sequence"] == 0
    assert first["entries"][0]["kind"] == "decision"
    assert op.read_events(token, {"limit": 1, "cursor": head["cursor"]}) == first
    handover(op, reply)
    step(op, reply, role="agent")
    second = op.read_events(token, {"limit": 1, "cursor": first["cursor"]})
    assert second["entries"][0]["kind"] == "control" and second["has_more"]
    third = op.read_events(token, {"limit": 1, "cursor": second["cursor"]})
    assert third["entries"][0]["kind"] == "decision"
    assert third["entries"][0]["event"]["controller"] == "agent"
    assert not third["has_more"]
    assert op.read_events(token, {"cursor": third["cursor"]})["entries"] == []


def test_paging_is_read_only_and_returns_owned_copies(opened):
    _, _, op, r = opened
    step(op, r)
    before = op.checkpoint(r["tokens"]["admin"])
    generation = op.generation
    page = op.read_events(r["tokens"]["viewer"], {})
    page["entries"][0]["event"]["action_ids"].append("not-a-command")
    assert op.read_events(r["tokens"]["viewer"], {})["entries"][0]["event"][
        "action_ids"
    ] == ["wait"]
    assert op.generation == generation
    assert op.checkpoint(r["tokens"]["admin"]) == before


def test_cursor_survives_restart_and_more_steps(opened):
    path, _, op, r = opened
    step(op, r)
    page = op.read_events(r["tokens"]["viewer"], {})
    restored = OperatorStore(path).get(op.id)
    step(restored, r)
    continued = restored.read_events(r["tokens"]["viewer"], {"cursor": page["cursor"]})
    assert [e["sequence"] for e in continued["entries"]] == [1]
    assert continued["entries"][0]["event"]["revision"] == 2


@pytest.mark.parametrize("channel", ["journal", "navigation"])
def test_cursor_is_bound_to_capability_session_and_channel(opened, config, channel):
    _, store, op, r = opened
    cursor = op.read_events(r["tokens"]["viewer"], {"channel": channel})["cursor"]
    with pytest.raises(ValueError):
        op.read_events(r["tokens"]["operator"], {"cursor": cursor, "channel": channel})
    with pytest.raises(ValueError):
        op.read_events(
            r["tokens"]["viewer"],
            {
                "cursor": cursor,
                "channel": "navigation" if channel == "journal" else "journal",
            },
        )
    other = store.create(copy.deepcopy(config))
    with pytest.raises(ValueError):
        store.get(other["session_id"]).read_events(
            other["tokens"]["viewer"], {"cursor": cursor, "channel": channel}
        )


def test_local_role_history_and_navigation_never_include_remote_actor(opened):
    _, _, op, r = opened
    names = list(op.session._task.roles)
    step(op, r, role="role:" + names[0])
    step(op, r, role="role:" + names[1])
    for name in names:
        token = r["tokens"]["role:" + name]
        journal = op.read_events(token, {})
        assert len(journal["entries"]) == 1
        assert journal["entries"][0]["event"]["role"] == name
        nav = op.read_events(token, {"channel": "navigation"})
        assert all(set(item["frame"]["actors"]) == {name} for item in nav["entries"])


@pytest.mark.parametrize(
    "payload",
    [
        {"limit": True},
        {"limit": 0},
        {"limit": 129},
        {"channel": "truth"},
        {"unknown": 1},
        {"cursor": ""},
        {"cursor": "a" * 2049},
        {"cursor": "bad"},
        {"cursor": "##.%%"},
        {"cursor": "a.b"},
    ],
)
def test_invalid_page_contract_is_read_only(opened, payload):
    _, _, op, r = opened
    with pytest.raises(ValueError):
        op.read_events(r["tokens"]["viewer"], payload)
    assert op.session.revision == 0 and op.generation == 1


def test_modified_prefix_is_explicit_conflict(opened):
    _, _, op, r = opened
    step(op, r)
    cursor = op.read_events(r["tokens"]["viewer"], {})["cursor"]
    # Trusted implementation mutation: even an unchanged revision cannot hide it.
    op.events[0]["elapsed_sec"] += 1
    with pytest.raises(OperatorError) as error:
        op.read_events(r["tokens"]["viewer"], {"cursor": cursor})
    assert error.value.code == "cursor_conflict" and error.value.status == 409


def test_cursor_revoked_on_recovery(opened):
    _, store, op, r = opened
    token = r["tokens"]["viewer"]
    cursor = op.read_events(token, {})["cursor"]
    recovered = store.recover(op.id)
    with pytest.raises(OperatorError):
        op.read_events(token, {"cursor": cursor})
    with pytest.raises(ValueError):
        op.read_events(recovered["tokens"]["viewer"], {"cursor": cursor})


def test_http_history_auth_and_bounded_json(monkeypatch, opened):
    _, store, op, r = opened
    monkeypatch.setattr(operator, "STORE", store)
    route = f"/api/operator/sessions/{op.id}/events"
    assert handle_post(route, b"{}")[0] == 401
    status, raw, _ = handle_post(
        route, b'{"channel":"navigation","limit":1}', token=r["tokens"]["viewer"]
    )
    assert status == 200
    data = json.loads(raw)
    assert data["channel"] == "navigation" and len(data["entries"]) == 1
    assert handle_post(route, b'{"limit":true}', token=r["tokens"]["viewer"])[0] == 400


def test_online_backup_restore_checkpoint_cursor_and_permissions(opened, tmp_path):
    source, store, op, r = opened
    step(op, r)
    handover(op, r)
    cursor = op.read_events(r["tokens"]["viewer"], {})["cursor"]
    before = op.checkpoint(r["tokens"]["admin"])
    output = tmp_path / "backup.sqlite3"
    receipt = maintenance.backup_operator_store(source, output)
    assert receipt["ok"] and receipt["bytes"] == output.stat().st_size
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o600
    step(op, r, role="agent")
    restored = OperatorStore(output).get(op.id)
    assert restored.checkpoint(r["tokens"]["admin"]) == before
    assert restored.control == {"controller": "agent", "epoch": 1}
    assert (
        restored.read_events(r["tokens"]["viewer"], {"cursor": cursor})["entries"] == []
    )
    step(restored, r, role="agent")
    assert restored.session.revision == op.session.revision == 2
    assert maintenance.verify_operator_store(output)["active_native_replayed"] == 1
    assert store.get(op.id).session.revision == 2


def test_backup_includes_closed_tombstones(opened, tmp_path):
    source, store, op, r = opened
    store.close(op.id, r["tokens"]["admin"])
    receipt = maintenance.backup_operator_store(source, tmp_path / "closed.sqlite")
    assert receipt["verification"]["closed_journals_checked"] == 1
    assert receipt["verification"]["active_native_replayed"] == 0
    with pytest.raises(OperatorError):
        OperatorStore(tmp_path / "closed.sqlite").get(op.id)


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_backup_never_overwrites_existing_target(opened, tmp_path, kind):
    source, _, _, _ = opened
    destination = tmp_path / "existing"
    if kind == "file":
        destination.write_bytes(b"original")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        maintenance.backup_operator_store(source, destination)
    assert destination.is_symlink() if kind == "symlink" else destination.exists()
    if kind == "file":
        assert destination.read_bytes() == b"original"


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), 601])
def test_backup_rejects_invalid_deadline(opened, tmp_path, timeout):
    source, _, _, _ = opened
    with pytest.raises(ValueError):
        maintenance.backup_operator_store(
            source, tmp_path / "invalid", timeout_sec=timeout
        )
    assert not (tmp_path / "invalid").exists()


def test_failed_verification_publishes_nothing(opened, tmp_path, monkeypatch):
    source, _, _, _ = opened

    def fail(path):
        raise ValueError("cold native replay rejected")

    monkeypatch.setattr(maintenance, "_verify_snapshot", fail)
    with pytest.raises(ValueError):
        maintenance.backup_operator_store(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    assert not list(tmp_path.glob(".operator-backup-*"))


def test_racing_publication_does_not_overwrite(opened, tmp_path, monkeypatch):
    source, _, _, _ = opened
    output = tmp_path / "raced"
    link = maintenance.os.link

    def race(src, dst, **kwargs):
        Path(dst).write_bytes(b"competing owner")
        return link(src, dst, **kwargs)

    monkeypatch.setattr(maintenance.os, "link", race)
    with pytest.raises(FileExistsError):
        maintenance.backup_operator_store(source, output)
    assert output.read_bytes() == b"competing owner"
    assert not list(tmp_path.glob(".operator-backup-*"))


def test_corrupt_snapshot_fails_backup_before_publish(opened, tmp_path):
    source, _, _, _ = opened
    with sqlite3.connect(source) as db:
        db.execute("UPDATE sessions SET sha='corrupt'")
    with pytest.raises(ValueError):
        maintenance.backup_operator_store(source, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_sqlite_timeout_cleans_stage_without_publishing(opened, tmp_path, monkeypatch):
    source, _, _, _ = opened
    values = iter([0.0, 100.0])
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: next(values, 100.0))
    with pytest.raises(TimeoutError):
        maintenance.backup_operator_store(
            source, tmp_path / "timed-out", timeout_sec=1.0
        )
    assert not (tmp_path / "timed-out").exists()
    assert not list(tmp_path.glob(".operator-backup-*"))


def test_missing_source_never_created(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        maintenance.backup_operator_store(missing, tmp_path / "output")
    with pytest.raises(FileNotFoundError):
        maintenance.verify_operator_store(missing)
    assert not missing.exists()


def test_maintenance_cli_backup_verify_restore(opened, tmp_path, capsys):
    source, _, op, r = opened
    backup = tmp_path / "cli-backup"
    maintenance.main(["backup", str(source), str(backup)])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["operation"] == "backup"
    assert not any(value in json.dumps(receipt) for value in r["tokens"].values())
    maintenance.main(["verify", str(backup)])
    assert json.loads(capsys.readouterr().out)["ok"]
    restore = tmp_path / "restored"
    maintenance.main(["restore", str(backup), str(restore)])
    assert json.loads(capsys.readouterr().out)["operation"] == "restore"
    assert OperatorStore(restore).get(op.id).session.revision == 0


def test_backup_is_coherent_while_another_connection_commits(
    opened, tmp_path, monkeypatch
):
    source, _, op, reply = opened
    connect = maintenance.sqlite3.connect
    writes = []

    class CopyConnection(sqlite3.Connection):
        def backup(self, target, *, pages, progress, sleep):
            def progress_and_write(status, remaining, total):
                if not writes:
                    writes.append(True)
                    step(op, reply)
                progress(status, remaining, total)

            return super().backup(
                target, pages=1, progress=progress_and_write, sleep=sleep
            )

    def connection(*args, **kwargs):
        kwargs.setdefault("factory", CopyConnection)
        return connect(*args, **kwargs)

    monkeypatch.setattr(maintenance.sqlite3, "connect", connection)
    output = tmp_path / "concurrent.sqlite"
    receipt = maintenance.backup_operator_store(source, output)
    assert writes and op.session.revision == 1
    copy = OperatorStore(output).get(op.id)
    assert copy.session.revision in {0, 1}
    assert copy.generation == copy.session.revision + 1
    assert receipt["verification"]["sessions"][0]["revision"] == copy.session.revision
    assert copy.verify(reply["tokens"]["admin"])["ok"]


def test_verification_uses_frozen_copy_instead_of_live_heads(opened, monkeypatch):
    source, _, op, reply = opened
    verify = maintenance._verify_snapshot
    snapshots = []

    def verify_and_mutate(path):
        snapshots.append(Path(path))
        assert Path(path) != source
        step(op, reply)
        return verify(path)

    monkeypatch.setattr(maintenance, "_verify_snapshot", verify_and_mutate)
    result = maintenance.verify_operator_store(source)
    assert result["sessions"][0]["revision"] == 0
    assert op.session.revision == 1
    assert all(not path.exists() for path in snapshots)
