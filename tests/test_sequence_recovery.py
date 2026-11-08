"""Native execution of committed-prefix adaptation, custody and fresh suffixes."""

import copy

import pytest

from space_env.api.network_workspace import default_network_configuration
from space_env.planning.sequence_lanes import verify_sequence_schedule
from space_env.planning.sequence_recovery import collect_recovery_geometry
from space_env.reference.spacecraft_bus import quaternion_angle
from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession
from space_env.workspace import MissionWorkspace


def configuration(*, decision=5.0, multiple=False):
    c = default_network_configuration()
    c["mission"].update(horizon_sec=180.0, decision_sec=decision)
    base = c["mission"]["requests"][0]
    base.update(deadline_sec=175.0, processing_sec=12.0)
    if multiple:
        base["processing_sec"] = 1.0
        for i, lat, lon, priority in [(1, 0.6, -126.5, 9), (2, -0.6, -129.5, 6)]:
            r = copy.deepcopy(base)
            r.update(
                id=f"job-{i}",
                lat_deg=lat,
                lon_deg=lon,
                priority=priority,
                processing_sec=1.0,
            )
            c["mission"]["requests"].append(r)
    return c


def metadata(m):
    return m.operator.session._scenario["metadata"]["network_workspace"]


def verify_suffix(m):
    meta = metadata(m)
    witness = meta["sequence_recovery"]
    assert verify_sequence_schedule(
        witness["forecast"], witness["geometry"], witness["schedule"]
    )["valid"]
    assert (
        witness["geometry"]["committed_physical_sha256"]
        == meta["dynamic"]["committed_physical_sha256"]
    )
    assert (
        witness["geometry"]["committed_time_sec"]
        == m.operator.session._runtime.time_sec
    )
    assert "sequence_geometry" not in meta and "pointing_envelope" not in meta
    assert meta["schedule"]["search_nodes"] <= meta["schedule"]["max_search_nodes"]
    assert not meta["schedule"]["optimal"]
    TaskSession.from_checkpoint(m.operator.session.checkpoint(compact=True))
    return witness


def test_repriority_uses_committed_attitude_and_new_sealed_goals():
    m = MissionWorkspace.network(
        configuration(multiple=True), planning_method="sequence_lanes"
    )
    n = None
    try:
        initial = copy.deepcopy(metadata(m)["sequence_geometry"])
        m.run(decisions=2)
        before = m.operator.session.checkpoint(compact=True)
        n = m.evolve(
            "New priorities",
            idempotency_key="priority",
            events=[
                {
                    "type": "request_priority_changed",
                    "payload": {"request_id": "survey", "priority": 99},
                }
            ],
        )
        witness = verify_suffix(n)
        bus = n.operator.session._runtime._backend.report()["spacecraft_bus"]
        for actor, info in witness["geometry"]["actors"].items():
            assert info["initial_quaternion"] == bus["states"][actor]["quaternion_wxyz"]
        assert (
            quaternion_angle(
                initial["actors"]["fleet-00"]["initial_quaternion"],
                witness["geometry"]["actors"]["fleet-00"]["initial_quaternion"],
            )
            > 1
        )
        assert witness["geometry"]["sha256"] != initial["sha256"]
        assert witness["schedule"]["trials"][0]["order"][0] == "survey"
        assert metadata(n)["forecast"]["demands"][0]["priority"] == 99
        assert n.run()["network"]["fulfilled"] == 3
        assert m.operator.session.checkpoint(compact=True) == before
    finally:
        if n:
            n.close()
        m.close()


@pytest.mark.parametrize("phase,decision", [("raw", 5.0), ("final", 1.0)])
def test_partial_native_work_plus_new_job_after_event(phase, decision):
    c = configuration(decision=decision)
    m = MissionWorkspace.network(c, planning_method="sequence_lanes")
    n = None
    try:
        stages = metadata(m)["schedule"]["allocations"][0]["stages"]
        stage = next(
            s
            for s in stages
            if s["kind"] == ("transform" if phase == "raw" else "downlink")
        )
        m.run(decisions=stage["cell"] + 1)
        before = m.operator.session.checkpoint(compact=True)
        prior = m.operator.session._runtime._backend.report()["lifecycle"]
        addition = copy.deepcopy(c["mission"]["requests"][0])
        addition.update(
            id="new",
            actor_id="fleet-00",
            station="Ground",
            release_sec=m.operator.session._runtime.time_sec + 2 * decision,
            processing_sec=1.0,
        )
        n = m.evolve(
            "Keep custody, add work",
            idempotency_key="mixed",
            additions=[addition],
            events=[
                {
                    "type": "request_priority_changed",
                    "payload": {"request_id": "survey", "priority": 99},
                }
            ],
        )
        witness = verify_suffix(n)
        state = metadata(n)["dynamic"]["states"]["survey"]
        assert state["phase"] == phase
        if phase == "raw":
            assert 0 < state["remaining_cpu_sec"] < 12
            assert (
                state["remaining_cpu_sec"]
                == prior["jobs"]["cpu-survey"]["remaining_sec"]
            )
        else:
            assert 0 < state["remaining_bits"] < 500000 and not state["intact"]
            assert state["owner"] == "fleet-02"
        allocations = {a["request"]: a for a in metadata(n)["schedule"]["allocations"]}
        assert all(
            s["kind"] not in {"slew", "imaging"}
            for s in allocations["survey"]["stages"]
        )
        assert (
            min(s["cell"] * decision for s in allocations["new"]["stages"])
            >= allocations["survey"]["finish_sec"]
        )
        assert (
            witness["forecast"]["actors"]["fleet-02"]["memory_bits"]
            < metadata(n)["forecast"]["actors"]["fleet-02"]["memory_bits"]
        )
        assert n.run()["network"]["fulfilled"] == 2
        report = n.operator.session._runtime._backend.report()["lifecycle"]
        assert len(report["acknowledgements"]) == 2
        assert (
            report["requests"]["survey"]["attempts"]
            == report["requests"]["new"]["attempts"]
            == 1
        )
        assert (
            report["artifacts"]["raw-survey"]["content_sha256"]
            == prior["artifacts"]["raw-survey"]["content_sha256"]
        )
        assert m.operator.session.checkpoint(compact=True) == before
    finally:
        if n:
            n.close()
        m.close()


def test_unsupported_profile_uses_whole_view_serial_fallback():
    m = MissionWorkspace.network(planning_method="sequence_lanes")
    n = None
    try:
        n = m.evolve(
            "Unsupported receiver CPU fault",
            idempotency_key="unsupported",
            events=[
                {
                    "type": "spacecraft_fault",
                    "payload": {"actor": "fleet-01", "subsystem": "cpu"},
                }
            ],
        )
        meta = metadata(n)
        detail = meta["schedule"]["recovery_sequence"]
        assert detail["mode"] == "conservative_fallback"
        assert (
            detail["fallback_reason"] == "unsupported_current_attitude_or_bus_profile"
        )
        assert meta["sequence_recovery"]["geometry"] is None
        assert meta["planning_method"] == "serial_recovery"
        assert "sequence_geometry" not in meta
        assert n.state()["network"]["dynamic"]["sequence_recovery"] == detail
        assert meta["schedule"]["predicted_fulfilled"] == 1
        assert n.run()["network"]["fulfilled"] == 1
    finally:
        if n:
            n.close()
        m.close()


def test_geometry_collection_has_no_native_future_or_physical_mutation(monkeypatch):
    from space_env.api.network_workspace import NetworkConfiguration
    from space_env.runtime import AccessPolicy, MissionRuntime

    m = MissionWorkspace.network()
    n = None
    try:
        n = m.evolve("Fresh geometry", idempotency_key="geometry")
        meta = metadata(n)
        runtime = n.operator.session._runtime
        runtime.access = AccessPolicy(truth=True, forecast=True)
        before = digest(runtime._backend.report())

        def forbidden(*args, **kwargs):
            raise AssertionError("Future execution during geometry collection")

        monkeypatch.setattr(MissionRuntime, "submit", forbidden)
        monkeypatch.setattr(MissionRuntime, "advance", forbidden)
        g = collect_recovery_geometry(
            NetworkConfiguration.model_validate(meta["configuration"]),
            meta["sequence_recovery"]["forecast"],
            runtime,
            n.operator.session._scenario,
        )
        assert digest(runtime._backend.report()) == before
        assert g == meta["sequence_recovery"]["geometry"]
    finally:
        if n:
            n.close()
        m.close()
