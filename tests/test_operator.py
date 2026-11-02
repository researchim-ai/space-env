"""Native operator parity, capabilities, revision/idempotency, causal navigation."""

import copy
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from space_env.api.operator import OperatorError, OperatorStore
from space_env.api.server import _Handler, handle_get, handle_post
from space_env.reference import EOPSeries
from space_env.runtime import RuntimeContractError
from space_env.scaling.workloads import fleet_workload
from space_env.tasks import TaskSession, TaskTools


@pytest.fixture(scope="module")
def configuration():
    w = fleet_workload({"actors": 10, "products": 10, "planes": 2, "repeats": 1})
    return {"scenario": w["scenario"], "task": w["task"]}


@pytest.fixture
def opened(configuration):
    store = OperatorStore()
    result = store.create(copy.deepcopy(configuration))
    return store, store.get(result["session_id"]), result


def request(ids=("wait",), revision=0, key="step"):
    return {
        "action_ids": list(ids),
        "expected_revision": revision,
        "idempotency_key": key,
    }


def test_native_sdk_exact_parity_and_read_only_state(opened, configuration):
    _, operator, result = opened
    token = result["tokens"]["operator"]
    task = copy.deepcopy(configuration["task"])
    task.update(policy_origin="external", runtime_audit="compact")
    reference = TaskSession(configuration["scenario"], task)
    assert operator.session.checkpoint(compact=True) == reference.checkpoint(
        compact=True
    )
    for _ in range(3):
        operator.state(token)
        operator.suggest(token)
    assert operator.session.checkpoint(compact=True) == reference.checkpoint(
        compact=True
    )
    ids = operator.suggest(token)["action_ids"]
    actual = operator.step(token, request(ids))
    expected = TaskTools(reference).step(**request(ids))
    assert actual == expected
    assert operator.session.checkpoint(compact=True) == reference.checkpoint(
        compact=True
    )


def test_navigation_uses_sealed_native_eop_and_no_future(opened):
    _, operator, result = opened
    view = operator.state(result["tokens"]["operator"])
    frame = view["frames"][-1]
    assert len(view["frames"]) == 1 and frame["time_sec"] == 0
    for actor, nav in frame["actors"].items():
        provider = operator.session._runtime._backend.mission.ephemerides[actor]
        expected = provider.context.matrix(provider.epoch) @ provider.state(0)[:3]
        assert np.array_equal(nav["r_itrs_m"], expected)
    assert all(f["time_sec"] <= view["time_sec"] for f in view["frames"])
    assert (
        not {"ephemerides", "seed", "definition", "backend_report", "runtime"}
        & view.keys()
    )


def test_sealed_frame_context_never_replaced_by_packaged_table(
    configuration, monkeypatch
):
    def fail(cls):
        raise AssertionError("UI substituted packaged EOP")

    monkeypatch.setattr(EOPSeries, "packaged", classmethod(fail))
    assert OperatorStore().create(copy.deepcopy(configuration))["state"]["frames"]


@pytest.mark.parametrize("name", ["viewer", "admin"])
def test_non_operator_tokens_cannot_advance(opened, name):
    _, operator, result = opened
    before = operator.session.checkpoint(compact=True)
    with pytest.raises(OperatorError) as error:
        operator.step(result["tokens"][name], request())
    assert error.value.status == 403
    assert operator.session.checkpoint(compact=True) == before


def test_invalid_cross_session_and_missing_credentials(opened, configuration):
    store, operator, _result = opened
    other = store.create(copy.deepcopy(configuration))
    for token in [None, "bad", other["tokens"]["operator"]]:
        with pytest.raises(OperatorError) as error:
            operator.state(token)
        assert error.value.status == 401


def test_local_role_scope_and_forbidden_action_has_no_effect(opened):
    _, operator, result = opened
    token = result["tokens"]["role:control-1"]
    state = operator.state(token)
    assert set(state["frames"][0]["actors"]) == {"sat-008", "sat-009"}
    assert {
        e["id"] for e in state["observation"]["entities"] if e["kind"] == "actor"
    } == {"sat-008", "sat-009"}
    with pytest.raises(OperatorError) as error:
        operator.step(token, request(["send-product-0000"]))
    assert error.value.status == 403 and operator.session.revision == 0
    with pytest.raises(OperatorError):
        operator.suggest(token)
    for method in [operator.checkpoint, operator.verify, operator.export]:
        with pytest.raises(OperatorError):
            method(token)


def test_native_rejection_consumes_physical_time(opened):
    _, operator, result = opened
    token = result["tokens"]["operator"]
    ids = operator.suggest(token)["action_ids"]
    operator.step(token, request(ids))
    rejected = operator.step(token, request([ids[0]], 1, "rejected"))
    assert not rejected["feedback"]["accepted"]
    assert rejected["elapsed_sec"] == 2.0
    assert operator.state(token)["time_sec"] == 4.0
    assert len(operator.events) == 2


def test_idempotent_retry_is_exact_and_journal_not_duplicated(opened):
    _, operator, result = opened
    token = result["tokens"]["operator"]
    first = operator.step(token, request())
    first["observation"].clear()
    second = operator.step(token, request())
    assert second["observation"] and operator.session.revision == 1
    assert len(operator.events) == 1 and len(operator.frames) == 2
    with pytest.raises(RuntimeContractError) as e:
        operator.step(token, request(["send-product-0000"]))
    assert e.value.code == "idempotency_conflict"


@pytest.mark.parametrize("same", [False, True])
def test_concurrent_revision_admission_and_retry(opened, same):
    _, operator, result = opened
    token = result["tokens"]["operator"]
    barrier = threading.Barrier(2)

    def call(i):
        barrier.wait()
        try:
            return operator.step(token, request(key="same" if same else str(i)))
        except RuntimeContractError as e:
            return e.code

    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(call, range(2)))
    assert operator.session.revision == 1 and len(operator.events) == 1
    assert sum(isinstance(r, dict) for r in rows) == (2 if same else 1)


def test_restore_rotates_tokens_and_restores_native_journal(opened):
    store, operator, result = opened
    operator.step(result["tokens"]["operator"], request())
    record = operator.checkpoint(result["tokens"]["admin"])
    fresh = store.restore({"record": record})
    restored = store.get(fresh["session_id"])
    assert restored.session.checkpoint(compact=True) == record
    assert (
        len(fresh["state"]["frames"]) == 1
        and fresh["state"]["frames"][0]["time_sec"] == 2.0
    )
    assert len(fresh["state"]["events"]) == 1
    with pytest.raises(OperatorError):
        restored.state(result["tokens"]["operator"])
    assert restored.verify(fresh["tokens"]["admin"])["ok"]


def test_admin_export_is_replay_verified_external_and_deterministic(opened):
    _, operator, result = opened
    operator.step(result["tokens"]["operator"], request())
    before = operator.checkpoint(result["tokens"]["admin"])
    data = operator.export(result["tokens"]["admin"])
    assert data == operator.export(result["tokens"]["admin"])
    assert before == operator.checkpoint(result["tokens"]["admin"])
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == {
            "bundle.json",
            "role-000/rl.jsonl",
            "role-000/sft.jsonl",
            "operator-provenance.json",
        }
        rows = [json.loads(r) for r in archive.read("role-000/rl.jsonl").splitlines()]
        assert rows[0]["policy_origin"] == "external" and rows[0]["replay_verified"]


def test_close_revokes_old_handles_and_keeps_other_session(opened, configuration):
    store, operator, result = opened
    other = store.create(copy.deepcopy(configuration))
    with pytest.raises(OperatorError):
        store.close(operator.id, result["tokens"]["viewer"])
    store.close(operator.id, result["tokens"]["admin"])
    with pytest.raises(OperatorError):
        operator.state(result["tokens"]["operator"])
    assert store.get(other["session_id"])


@pytest.mark.parametrize(
    "change",
    [
        {"expected_revision": True},
        {"advance_sec": float("nan")},
        {"extra": 1},
        {"action_ids": []},
        {"idempotency_key": ""},
    ],
)
def test_strict_step_request_never_advances(opened, change):
    _, operator, result = opened
    with pytest.raises(ValueError):
        operator.step(result["tokens"]["operator"], {**request(), **change})
    assert operator.session.revision == 0


def test_token_label_collisions_cannot_replace_central_capability(configuration):
    custom = copy.deepcopy(configuration)
    old = custom["task"]["roles"].pop("control-0")
    custom["task"]["roles"]["operator"] = old
    for action in custom["task"]["actions"]:
        if action["role"] == "control-0":
            action["role"] = "operator"
    result = OperatorStore().create(custom)
    assert result["tokens"]["operator"] != result["tokens"]["role:operator"]
    assert result["state"]["role"] == "@central"


def test_capacity_and_restore_validation(opened, configuration):
    store, operator, result = opened
    for _ in range(7):
        store.create(copy.deepcopy(configuration))
    with pytest.raises(OperatorError) as e:
        store.create(copy.deepcopy(configuration))
    assert e.value.status == 429
    record = operator.checkpoint(result["tokens"]["admin"])
    bad = copy.deepcopy(record)
    bad["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        OperatorStore().restore({"record": bad})
    for value in [None, {"task": None}, {"task": {"policy_origin": "scripted"}}]:
        with pytest.raises(ValueError):
            OperatorStore().restore({"record": value})


def test_api_and_assets(configuration):
    status, body, _ = handle_post(
        "/api/operator/sessions", json.dumps(configuration).encode()
    )
    assert status == 200
    result = json.loads(body)
    path = "/api/operator/sessions/" + result["session_id"]
    try:
        assert handle_get(path, {})[0] == 401
        assert (
            handle_post(
                path + "/step",
                b'{"action_ids":["wait"],"expected_revision":0,"idempotency_key":"k"}',
                token=result["tokens"]["viewer"],
            )[0]
            == 403
        )
        assert (
            handle_post(
                path + "/step",
                b'{"action_ids":["wait"],"expected_revision":true,"idempotency_key":"k"}',
                token=result["tokens"]["operator"],
            )[0]
            == 400
        )
        for page in ["/operator", "/viz/operator.css", "/viz/operator.js"]:
            assert handle_get(page, {})[0] == 200
        assert handle_get("/viz/../pyproject.toml", {})[0] == 404
        assert (
            handle_post(
                path + "/verify", b'{"ignored":true}', token=result["tokens"]["admin"]
            )[0]
            == 400
        )
    finally:
        handle_post(path + "/close", b"{}", token=result["tokens"]["admin"])


def test_http_origin_content_type_body_limits_and_bearer(configuration):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for headers, expected in [
            (
                {"Origin": "http://evil.example", "Content-Type": "application/json"},
                403,
            ),
            ({"Content-Type": "text/plain"}, 415),
            (
                {
                    "Content-Type": "application/json",
                    "Content-Length": str(8 * 1024 * 1024 + 1),
                },
                413,
            ),
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(
                    Request(
                        base + "/api/operator/sessions", data=b"{}", headers=headers
                    )
                )
            assert error.value.code == expected
        req = Request(
            base + "/api/operator/sessions",
            data=json.dumps(configuration).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req) as response:
            result = json.load(response)
            assert response.headers["Cache-Control"] == "no-store"
        url = base + "/api/operator/sessions/" + result["session_id"]
        with urlopen(
            Request(
                url, headers={"Authorization": "Bearer " + result["tokens"]["viewer"]}
            )
        ) as response:
            assert json.load(response)["permissions"]["step"] is False
        urlopen(
            Request(
                url + "/close",
                data=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + result["tokens"]["admin"],
                },
            )
        ).close()
    finally:
        server.shutdown()
        thread.join(3)
        server.server_close()


def test_coupled_preset_runs_same_native_chain_and_checkpoint():
    store = OperatorStore()
    result = store.create({"preset": "relay-demo"})
    operator = store.get(result["session_id"])
    token = result["tokens"]["operator"]
    reference = TaskSession(
        operator.session._original,
        operator.session._task.model_dump(mode="json", by_alias=True),
    )
    while not operator.session._done:
        decision = operator.suggest(token)
        req = request(
            decision["action_ids"],
            operator.session.revision,
            str(operator.session.revision),
        )
        assert operator.step(token, req) == TaskTools(reference).step(**req)
    assert operator.session.checkpoint(compact=True) == reference.checkpoint(
        compact=True
    )
    assert operator.session._runtime.report()["metrics"]["fulfilled_requests"] == 2
    assert operator.verify(result["tokens"]["admin"])["ok"]
    state = operator.state(token)
    assert state["physics_profile"] == "reference-spacecraft-bus/1"
    assert state["time_sec"] == 20.0 and len(state["events"]) == 10


def test_optional_dependency_failure_is_structured(monkeypatch):
    import space_env.api.operator as module

    def fail(_payload):
        raise ImportError("optional backend missing")

    monkeypatch.setattr(module.STORE, "create", fail)
    status, body, _ = handle_post("/api/operator/sessions", b'{"preset":"fleet-demo"}')
    assert status == 503 and json.loads(body)["error"]["code"] == "missing_dependency"


def test_query_token_is_not_authentication(configuration):
    status, body, _ = handle_post(
        "/api/operator/sessions", json.dumps(configuration).encode()
    )
    assert status == 200
    result = json.loads(body)
    path = "/api/operator/sessions/" + result["session_id"]
    try:
        assert handle_get(path, {"token": [result["tokens"]["operator"]]})[0] == 401
    finally:
        handle_post(path + "/close", b"{}", token=result["tokens"]["admin"])
