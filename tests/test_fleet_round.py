import pytest

from space_env.api.fleet_round import dispatch_round
from space_env.api.operator import OperatorError
from space_env.workspace import MissionWorkspace


def test_role_votes_do_not_advance_clock_and_commit_is_one_atomic_step():
    m = MissionWorkspace()
    r = m.fleet_round()
    for role in r.roles:
        token = m.connection["tokens"]["role:" + role]
        observation = r.observe(token)
        assert observation["role"] == role and "workspace" not in observation
        own = next(
            a["id"]
            for a in observation["observation"]["catalogue"]
            if a["id"] != "wait" and a["enabled"]
        )
        r.submit(token, [own])
        assert m.state()["time_sec"] == 0.0
    result = r.commit(m.connection["tokens"]["operator"], "joint")
    assert result["feedback"]["accepted"]
    assert m.state()["time_sec"] == 5.0 and m.state()["revision"] == 1
    assert r.commit(m.connection["tokens"]["operator"], "joint") == result
    assert m.state()["revision"] == 1


def test_incomplete_foreign_conflicting_and_stale_rounds():
    m = MissionWorkspace()
    r = m.fleet_round()
    with pytest.raises(OperatorError, match="остальных"):
        r.commit(m.connection["tokens"]["operator"], "missing")
    with pytest.raises(OperatorError):
        r.submit(m.connection["tokens"]["role:a-00"], ["slew/survey-east"])
    r.submit(m.connection["tokens"]["role:a-00"], ["wait"])
    with pytest.raises(OperatorError, match="зафиксировано"):
        r.submit(m.connection["tokens"]["role:a-00"], ["slew/survey-west"])
    m.step()
    with pytest.raises(OperatorError, match="изменились"):
        r.submit(m.connection["tokens"]["role:b-00"], ["wait"])


def test_http_round_capabilities_and_joint_wait():
    m = MissionWorkspace()
    op, tokens = m.operator, m.connection["tokens"]
    opened = dispatch_round(op, tokens["operator"], {"operation": "create"})
    for role in opened["roles"]:
        dispatch_round(
            op,
            tokens["role:" + role],
            {
                "operation": "submit",
                "round_id": opened["round_id"],
                "action_ids": ["wait"],
            },
        )
    dispatch_round(
        op,
        tokens["operator"],
        {
            "operation": "commit",
            "round_id": opened["round_id"],
            "idempotency_key": "round",
        },
    )
    assert m.state()["time_sec"] == 5.0


def test_same_client_key_in_two_rounds_does_not_reuse_old_receipt():
    m = MissionWorkspace()
    for _ in range(2):
        round_ = m.fleet_round()
        for role in round_.roles:
            round_.submit(m.connection["tokens"]["role:" + role], ["wait"])
        round_.commit(m.connection["tokens"]["operator"], "same-client-key")
    assert m.state()["time_sec"] == 10.0 and m.state()["revision"] == 2


def test_relay_profile_runs_on_same_sdk_with_real_isl_and_ground_ack():
    m = MissionWorkspace.relay()
    result = m.run()
    report = m.operator.session._runtime._backend.report()
    assert result["done"]
    assert report["lifecycle"]["acknowledgements"]
    assert any(
        row["command"]["kind"] == "relay"
        for row in m.operator.session._runtime.command_status()
    )


def test_http_rounds_continue_beyond_retry_cache_capacity():
    m = MissionWorkspace()
    tokens = m.connection["tokens"]
    for i in range(34):
        opened = dispatch_round(m.operator, tokens["operator"], {"operation": "create"})
        for role in opened["roles"]:
            dispatch_round(
                m.operator,
                tokens["role:" + role],
                {
                    "operation": "submit",
                    "round_id": opened["round_id"],
                    "action_ids": ["wait"],
                },
            )
        dispatch_round(
            m.operator,
            tokens["operator"],
            {
                "operation": "commit",
                "round_id": opened["round_id"],
                "idempotency_key": str(i),
            },
        )
    assert m.state()["revision"] == 34 and m.state()["time_sec"] == 170.0
