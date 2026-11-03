"""Refresh-local reuse must preserve projections and public ownership."""

import copy
from collections import Counter

from space_env.scaling.campaign import campaign_workload
from space_env.tasks import TaskSession
from space_env.tasks.observation import Projection, _ProjectionReads


def make_session():
    w = campaign_workload(
        {"cameras": 1, "relays": 1, "waves": 1, "horizon_sec": 20.0, "repeats": 1}
    )
    return TaskSession(w["scenario"], w["task"])


def test_refresh_matches_uncached_projection_and_keeps_role_registries():
    session = make_session()
    for _ in range(3):
        old = copy.deepcopy(session._projections)
        expected = {
            role: p.build(session._runtime, role, session.revision, session._actions)
            for role, p in old.items()
        }
        actual = session._refresh(
            session._runtime, session._projections, session.revision
        )
        assert actual == expected
        assert len({id(p.indices) for p in session._projections.values()}) == len(old)
        session.step(
            {"@central": ["wait"]},
            expected_revision=session.revision,
            idempotency_key=f"refresh/{session.revision}",
        )


def test_reads_once_per_actor_per_refresh_and_no_cross_refresh_cache(monkeypatch):
    session = make_session()
    calls = Counter()
    runtime = session._runtime
    for name in [
        "observe",
        "inventory",
        "station_observation",
        "command_status",
        "describe_capabilities",
    ]:
        original = getattr(runtime, name)

        def counted(*args, _name=name, _original=original):
            calls[(_name, args)] += 1
            return _original(*args)

        monkeypatch.setattr(runtime, name, counted)
    for _ in range(2):
        session._refresh(runtime, session._projections, session.revision)
    assert calls
    assert all(n == 2 for n in calls.values())
    assert calls[("observe", ("camera-00",))] == 2
    assert calls[("inventory", ("relay-00",))] == 2


def test_scoped_projection_reads_only_its_actors_and_leaves_native_state_unchanged(
    monkeypatch,
):
    session = make_session()
    runtime = session._runtime
    before = runtime._backend.report()
    seen = []
    original = runtime.observe

    def observe(actor):
        seen.append(actor)
        return original(actor)

    monkeypatch.setattr(runtime, "observe", observe)
    role = next(
        r
        for r, scope in session._task.roles.items()
        if list(scope.actors) == ["camera-00"]
    )
    p = Projection(session._task)
    p.station_ids = session._station_ids
    p.build(_ProjectionReads(runtime), role, session.revision, session._actions)
    assert seen == ["camera-00"]
    assert runtime._backend.report() == before


def test_public_views_inventory_and_fork_are_owned():
    session = make_session()
    original = session.observe()
    changed = session.observe()
    changed["entities"][0]["values"].clear()
    changed["catalogue"].clear()
    assert session.observe() == original
    runtime = session._runtime
    inventory = runtime.inventory("camera-00")
    inventory["transmitter"]["ready"] = "corrupted"
    view = runtime.observe("camera-00")
    view["actors"]["camera-00"]["lifecycle"]["requests"][0]["status"] = "corrupted"
    assert runtime.inventory("camera-00")["transmitter"]["ready"] is True
    assert (
        runtime.observe("camera-00")["actors"]["camera-00"]["lifecycle"]["requests"][0][
            "status"
        ]
        == "released"
    )
    fork = runtime.fork()
    fork.advance(1.0, expected_revision=fork.revision, idempotency_key="fork")
    assert runtime.time_sec == 0.0
    assert session.observe() == original


def test_single_actor_native_read_does_not_materialize_other_actors(monkeypatch):
    session = make_session()
    mission = session._runtime._backend.mission
    seen = []
    original = mission.radio_network.observation

    def radio(mission, actor):
        seen.append(actor)
        return original(mission, actor)

    monkeypatch.setattr(mission.radio_network, "observation", radio)
    session._runtime.observe("camera-00")
    assert seen == ["camera-00"]
    seen.clear()
    session._runtime.observe()
    assert seen == list(mission.actors)
