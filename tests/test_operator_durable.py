"""Durable commit/restart, fencing, provenance, IO failure and native takeover."""

import copy
import io
import json
import sqlite3
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.operator_storage import OperatorRepository
from space_env.planning import PlanningRun
from space_env.scaling.workloads import fleet_workload
from space_env.simulation.snapshot import digest


@pytest.fixture(scope="module")
def configuration():
    w = fleet_workload(
        {"actors": 2, "products": 4, "planes": 2, "repeats": 1, "horizon_sec": 12.0}
    )
    return {"scenario": w["scenario"], "task": w["task"]}


@pytest.fixture
def opened(tmp_path, configuration):
    path = tmp_path / "operator.sqlite"
    store = OperatorStore(path)
    reply = store.create(copy.deepcopy(configuration))
    return path, store, store.get(reply["session_id"]), reply


def step(revision=0, key="s", epoch=None):
    result = {
        "action_ids": ["wait"],
        "expected_revision": revision,
        "idempotency_key": key,
    }
    if epoch is not None:
        result["control_epoch"] = epoch
    return result


def transfer(op, reply, controller, revision, epoch, key="h"):
    return op.handover(
        reply["tokens"]["admin"],
        {
            "controller": controller,
            "expected_revision": revision,
            "expected_control_epoch": epoch,
            "idempotency_key": key,
            "reason": "controlled test handover",
        },
    )


def test_restart_retains_native_frames_journal_and_hashed_capabilities(opened):
    path, _store, op, reply = opened
    op.step(reply["tokens"]["operator"], step())
    before = op.state(reply["tokens"]["viewer"])
    cp = op.checkpoint(reply["tokens"]["admin"])
    restored = OperatorStore(path).get(op.id)
    assert restored.state(reply["tokens"]["viewer"]) == before
    assert restored.checkpoint(reply["tokens"]["admin"]) == cp
    assert restored.credentials == {}
    with sqlite3.connect(path) as db:
        payload = db.execute("SELECT payload FROM sessions").fetchone()[0]
        assert all(token not in payload for token in reply["tokens"].values())
        assert db.execute("SELECT count(*) FROM commits").fetchone()[0] == 2
    assert path.stat().st_mode & 0o777 == 0o600


def test_fenced_handover_no_time_advance_and_exactly_once(opened):
    path, _, op, reply = opened
    cp = op.checkpoint(reply["tokens"]["admin"])
    first = transfer(op, reply, "agent", 0, 0)
    assert transfer(op, reply, "agent", 0, 0) == first
    assert op.checkpoint(reply["tokens"]["admin"]) == cp and op.generation == 2
    with pytest.raises(OperatorError):
        op.step(reply["tokens"]["operator"], step())
    with pytest.raises(OperatorError):
        op.step(reply["tokens"]["agent"], step())
    op.step(reply["tokens"]["agent"], step(epoch=1))
    transfer(op, reply, "operator", 1, 1, "back")
    with pytest.raises(OperatorError):
        op.step(reply["tokens"]["operator"], step(1, "later", 0))
    op.step(reply["tokens"]["operator"], step(1, "s", 2))
    restored = OperatorStore(path).get(op.id)
    assert [e["controller"] for e in restored.events] == ["agent", "operator"]
    assert restored.session.revision == 2 and restored.control["epoch"] == 2


@pytest.mark.parametrize("role", ["operator", "agent", "viewer", "role:control-0"])
def test_only_admin_can_transfer(opened, role):
    _, _, op, reply = opened
    with pytest.raises(OperatorError) as e:
        op.handover(
            reply["tokens"][role],
            {
                "controller": "agent",
                "expected_revision": 0,
                "expected_control_epoch": 0,
                "idempotency_key": "h",
                "reason": "request",
            },
        )
    assert e.value.status == 403 and op.generation == 1


def test_handover_validation_conflicts_do_not_persist(opened):
    _, _, op, reply = opened
    with pytest.raises(ValueError):
        op.handover(
            reply["tokens"]["admin"],
            {
                "controller": "agent",
                "expected_revision": 0,
                "expected_control_epoch": 0,
                "idempotency_key": "h",
                "reason": "   ",
            },
        )
    transfer(op, reply, "agent", 0, 0)
    with pytest.raises(OperatorError):
        transfer(op, reply, "operator", 0, 0, "stale")
    with pytest.raises(OperatorError):
        transfer(op, reply, "operator", 0, 1, "h")
    assert op.generation == 2


def test_precommit_disk_failure_rolls_back_physics_and_cache(opened, monkeypatch):
    path, _, op, reply = opened
    before = op.checkpoint(reply["tokens"]["admin"])
    original = op.repository.save

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(op.repository, "save", fail)
    with pytest.raises(OperatorError) as error:
        op.step(reply["tokens"]["operator"], step())
    assert error.value.status == 503 and op.generation == 1
    assert op.checkpoint(reply["tokens"]["admin"]) == before
    assert OperatorStore(path).get(op.id).session.revision == 0
    monkeypatch.setattr(op.repository, "save", original)
    op.step(reply["tokens"]["operator"], step())
    assert op.session.revision == 1 and len(op.events) == 1


def test_lost_commit_ack_is_resolved_without_duplicate(opened, monkeypatch):
    path, _, op, reply = opened
    original = op.repository.save

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("commit acknowledgement lost")

    monkeypatch.setattr(op.repository, "save", lost)
    first = op.step(reply["tokens"]["operator"], step())
    assert op.step(reply["tokens"]["operator"], step()) == first
    assert op.generation == 2 and OperatorStore(path).get(op.id).session.revision == 1


def test_two_store_writers_cas_and_refresh(opened):
    path, store, op, reply = opened
    other = OperatorStore(path)
    second = other.get(op.id)
    op.step(reply["tokens"]["operator"], step())
    with pytest.raises(OperatorError):
        second.step(reply["tokens"]["operator"], step(key="other"))
    fresh = other.get(op.id)
    assert fresh.session.revision == 1
    fresh.step(reply["tokens"]["operator"], step(1, "next"))
    assert store.get(op.id).session.revision == 2


def test_simultaneous_control_transfer_has_one_winner(opened):
    _, _, op, reply = opened
    barrier = threading.Barrier(2)

    def call(i):
        barrier.wait()
        try:
            return transfer(op, reply, "agent", 0, 0, str(i))
        except OperatorError:
            return None

    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(call, [0, 1]))
    assert sum(r is not None for r in rows) == 1 and len(op.control_events) == 1


def test_close_tombstone_and_restart(opened):
    path, store, op, reply = opened
    store.close(op.id, reply["tokens"]["admin"])
    assert not OperatorStore(path).sessions
    saved = OperatorRepository(path).load(op.id, include_closed=True)
    assert not saved["active"] and saved["state"]["closed"]


def test_recovery_rotates_every_capability_and_fence(opened):
    path, store, op, reply = opened
    recovered = store.recover(op.id)
    assert recovered["state"]["control"]["epoch"] == 1
    assert not set(reply["tokens"].values()) & set(recovered["tokens"].values())
    with pytest.raises(OperatorError):
        op.state(reply["tokens"]["viewer"])
    restarted = OperatorStore(path).get(op.id)
    assert restarted.state(recovered["tokens"]["viewer"])["revision"] == 0


def test_archive_roundtrip_and_provenance_zip(opened, tmp_path):
    _, _, op, reply = opened
    transfer(op, reply, "agent", 0, 0)
    op.step(reply["tokens"]["agent"], step(epoch=1))
    archive = op.archive(reply["tokens"]["admin"])
    assert all(t not in json.dumps(archive) for t in reply["tokens"].values())
    assert "tokens" not in archive and "identities" not in archive
    target = OperatorStore(tmp_path / "other.sqlite")
    r = target.restore({"archive": archive})
    restored = target.get(r["session_id"])
    assert restored.session.checkpoint(compact=True) == op.session.checkpoint(
        compact=True
    )
    assert restored.frames == op.frames and restored.control_events == op.control_events
    with zipfile.ZipFile(io.BytesIO(restored.export(r["tokens"]["admin"]))) as z:
        sidecar = json.loads(z.read("operator-provenance.json"))
        assert sidecar["steps"][0]["controller"] == "agent"
        rows = [json.loads(x) for x in z.read("role-000/rl.jsonl").splitlines()]
        assert rows[0]["policy_origin"] == "external"


@pytest.mark.parametrize("part", ["frames", "events", "control_events"])
def test_rehashed_archive_tampering_fails_native_validation(opened, part):
    _, _, op, reply = opened
    op.step(reply["tokens"]["operator"], step())
    transfer(op, reply, "agent", 1, 0)
    archive = op.archive(reply["tokens"]["admin"])
    if part == "frames":
        archive[part][0]["actors"]["sat-000"]["r_itrs_m"][0] += 1000
    elif part == "events":
        archive[part][0]["elapsed_sec"] = 999.0
    else:
        archive[part][0]["time_sec"] = 10.0
    archive["sha256"] = digest({k: v for k, v in archive.items() if k != "sha256"})
    with pytest.raises(ValueError):
        OperatorStore().restore({"archive": archive})


@pytest.mark.parametrize("what", ["snapshot", "journal", "generation"])
def test_corrupt_store_refuses_start(opened, what):
    path, _, _op, _ = opened
    with sqlite3.connect(path) as db:
        if what == "snapshot":
            db.execute("UPDATE sessions SET payload='{}'")
        elif what == "journal":
            db.execute("UPDATE commits SET sha=?", ("0" * 64,))
        else:
            db.execute("UPDATE sessions SET generation=99")
    with pytest.raises(ValueError):
        OperatorStore(path)


def test_scripted_agent_prefix_takeover_preserves_native_state(configuration, tmp_path):
    run = PlanningRun(
        configuration["scenario"], configuration["task"], {"name": "priority"}
    )
    run.run(decisions=1)
    record = run.checkpoint(compact=True)["episode"]
    assert record["task"]["policy_origin"] == "scripted"
    store = OperatorStore(tmp_path / "db")
    with pytest.raises(ValueError):
        store.restore({"record": record})
    reply = store.restore({"record": record, "takeover": True})
    op = store.get(reply["session_id"])
    assert (
        op.session._runtime.checkpoint()["backend_report"]
        == record["runtime"]["backend_report"]
    )
    assert op.events[0]["controller"] == "imported_agent"
    assert op.session._task.policy_origin == "external"
    op.step(reply["tokens"]["operator"], step(1, "human", 1))
    restored = OperatorStore(tmp_path / "db").get(op.id)
    assert restored.source_import["episode_sha256"] == record["sha256"]
    assert [e["controller"] for e in restored.events] == ["imported_agent", "operator"]


def test_uncoordinated_sdk_write_is_fenced(opened):
    _, _, op, reply = opened
    op.session.step(
        {"@central": ["wait"]}, expected_revision=0, idempotency_key="outside"
    )
    with pytest.raises(OperatorError):
        op.state(reply["tokens"]["viewer"])


def test_foreign_database_is_not_adopted(tmp_path):
    path = tmp_path / "foreign"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated (value TEXT)")
        db.execute("INSERT INTO unrelated VALUES ('keep')")
    with pytest.raises((ValueError, sqlite3.Error)):
        OperatorStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM unrelated").fetchall() == [("keep",)]


@pytest.mark.parametrize("mode", ["before_commit", "after_commit"])
def test_abrupt_process_death_never_creates_partial_or_double_step(
    tmp_path, configuration, mode
):
    import subprocess
    import sys

    db = tmp_path / "crash.sqlite"
    cfg = tmp_path / "input.json"
    caps = tmp_path / "caps.json"
    cfg.write_text(json.dumps(configuration))
    script = """
import json,os,sqlite3,sys
from pathlib import Path
from space_env.api.operator import OperatorStore
path,cfg,caps,mode=sys.argv[1:]
store=OperatorStore(path);reply=store.create(json.loads(Path(cfg).read_text()));op=store.get(reply['session_id'])
Path(caps).write_text(json.dumps(reply['tokens']|{'id':op.id}))
if mode=='before_commit':
 def die(*args,**kwargs):
  db=sqlite3.connect(path);db.execute('BEGIN IMMEDIATE');db.execute("UPDATE sessions SET payload='uncommitted corruption'");os._exit(73)
 op.repository.save=die
op.step(reply['tokens']['operator'],{'action_ids':['wait'],'expected_revision':0,'idempotency_key':'crash'})
os._exit(72)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(db), str(cfg), str(caps), mode],
        check=False,
        timeout=45,
    )
    assert result.returncode == (73 if mode == "before_commit" else 72)
    keys = json.loads(caps.read_text())
    op = OperatorStore(db).get(keys["id"])
    assert op.session.revision == (0 if mode == "before_commit" else 1)
    op.step(keys["operator"], step(key="crash"))
    assert op.session.revision == 1 and len(op.events) == 1 and op.generation == 2


def test_real_generation_cas_rolls_back_losing_writer(opened, monkeypatch):
    path, _, left, reply = opened
    right = OperatorStore(path).get(left.id)
    barrier = threading.Barrier(2)
    for op in [left, right]:
        original = op.repository.save

        def save(*args, _original=original, **kwargs):
            barrier.wait(timeout=10)
            return _original(*args, **kwargs)

        monkeypatch.setattr(op.repository, "save", save)

    def call(pair):
        op, key = pair
        try:
            op.step(reply["tokens"]["operator"], step(key=key))
            return True
        except OperatorError:
            return False

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(call, [(left, "left"), (right, "right")]))
    assert sum(results) == 1
    saved = OperatorStore(path).get(left.id)
    assert saved.session.revision == 1 and saved.generation == 2
    assert sorted([left.session.revision, right.session.revision]) == [0, 1]


def test_uncertain_storage_fences_writes_until_verified_restart(opened, monkeypatch):
    path, _, op, reply = opened

    def fail(*args, **kwargs):
        raise OSError("unavailable device")

    original_load = op.repository.load
    # Guard succeeds first, then the write and reconciliation read fail.
    calls = []

    def load(*args, **kwargs):
        calls.append(True)
        if len(calls) > 1:
            raise OSError("reconciliation unavailable")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(op.repository, "load", load)
    monkeypatch.setattr(op.repository, "save", fail)
    with pytest.raises(OperatorError):
        op.step(reply["tokens"]["operator"], step())
    assert op.poisoned
    with pytest.raises(OperatorError):
        op.step(reply["tokens"]["operator"], step())
    fresh = OperatorStore(path).get(op.id)
    fresh.step(reply["tokens"]["operator"], step())
    assert fresh.session.revision == 1


def test_local_cli_recovers_to_private_file(opened, tmp_path):
    from space_env.api.operator_recovery import main

    path, _, op, reply = opened
    output = tmp_path / "operator-connection.json"
    main([str(path), op.id, str(output)])
    assert output.stat().st_mode & 0o777 == 0o600
    value = json.loads(output.read_text())
    assert value["schema"] == "space-env.operator-connection/1"
    current = OperatorStore(path).get(op.id)
    assert current.state(value["tokens"]["viewer"])["control"]["epoch"] == 1
    with pytest.raises(OperatorError):
        current.state(reply["tokens"]["viewer"])
    with pytest.raises(ValueError):
        main([str(path), op.id, str(output)])


def test_closed_session_does_not_consume_other_process_budget(opened):
    path, store, op, reply = opened
    other = OperatorStore(path)
    store.close(op.id, reply["tokens"]["admin"])
    assert other.repository.active() == []
    with pytest.raises(OperatorError):
        other.get(op.id)


def test_local_controller_idempotency_namespaces_do_not_collide(
    tmp_path, configuration
):
    config = copy.deepcopy(configuration)
    names = config["scenario"]["actor_order"]
    config["task"]["roles"] = {a: {"actors": [a]} for a in names}
    for action in config["task"]["actions"]:
        action["role"] = action["commands"][0]["actor_id"]
    store = OperatorStore(tmp_path / "roles")
    r = store.create(config)
    op = store.get(r["session_id"])
    op.step(r["tokens"]["role:" + names[0]], step())
    op.step(r["tokens"]["role:" + names[1]], step(1))
    assert op.session.revision == 2
    assert OperatorStore(tmp_path / "roles").get(op.id).session.revision == 2


@pytest.mark.parametrize('prefix', ['-', '--'])
def test_recovery_accepts_generated_dash_prefixed_id(tmp_path, configuration, monkeypatch, prefix):
    import secrets
    from space_env.api.operator_recovery import main
    original = secrets.token_urlsafe
    identifier = prefix + 'a' * (24-len(prefix))
    monkeypatch.setattr(secrets, 'token_urlsafe', lambda n: identifier if n == 18 else original(n))
    path = tmp_path / 'dash.sqlite'
    store = OperatorStore(path)
    reply = store.create(copy.deepcopy(configuration))
    assert reply['session_id'] == identifier
    output = tmp_path / 'connection.json'
    main([str(path), identifier, str(output)])
    assert output.stat().st_mode & 0o777 == 0o600
    connection = json.loads(output.read_text())
    assert OperatorStore(path).get(identifier).state(connection['tokens']['viewer'])['control']['epoch'] == 1
