"""Fixed-start terminal envelopes; synthetic arithmetic and native integration."""

import copy
import math

import pytest

from space_env.planning.network_optimal import optimize_network
from space_env.planning.terminal_sequence import (
    compact_terminal_schedule,
    sealed,
    verify_terminal_schedule,
)
from tests.test_network_optimal import bottleneck_view


def fixture(capacity=1):
    view = bottleneck_view()
    view["stations"]["g"] = capacity
    view["horizon_sec"] = 10.0
    for request in view["demands"]:
        request["deadline_sec"] = 10.0
    for capture in view["captures"]:
        capture["cells"] = [False, True] + [False] * 8
    for edge in view["edges"]:
        edge["cells"] = [False] * 3 + [True] * 7
        edge["setup_sec"] = [1.1] * 10
    view = sealed({k: v for k, v in view.items() if k != "sha256"})
    geometry = sealed(
        {
            "schema": "space-env.terminal-geometry/1",
            "forecast_sha256": view["sha256"],
            "sample_sec": 0.5,
            "angular_guard_deg": math.degrees(12000 / 300000 * 0.5) * 2,
            "stations": {
                "g": {
                    "acquisition_sec": 0.1,
                    "slew_rate_deg_s": 10.0,
                    "directions": {a: [[1.0, 0.0, 0.0]] * 21 for a in view["actors"]},
                }
            },
        }
    )
    return view, optimize_network(view, max_hops=0), geometry


def test_fixed_starts_and_resources_with_independent_first_acquisition_width():
    view, base, geometry = fixture()
    before = copy.deepcopy((view, base, geometry))
    plan = compact_terminal_schedule(view, base, geometry)
    assert verify_terminal_schedule(view, base, geometry, plan)
    assert (view, base, geometry) == before
    assert plan["objective_score"] == base["objective_score"]
    assert len(plan["allocations"]) == 2
    assert plan["terminal_sequence"]["saved_reservation_sec"] == 2.0
    assert [r["after_cells"] for r in plan["terminal_sequence"]["rows"]] == [3, 2]
    assert plan["terminal_sequence"]["rows"][1]["predecessor"] is not None
    for old, new in zip(base["allocations"], plan["allocations"]):
        assert old["request"] == new["request"]
        assert old["stages"][:-1] == new["stages"][:-1]
        assert old["stages"][-1]["cell"] == new["stages"][-1]["cell"]
        assert new["stages"][-1]["cells"] <= old["stages"][-1]["cells"]
    assert plan["terminal_sequence"]["global_sequence_optimality_established"] is False


def test_multiple_terminals_cannot_be_assigned_by_forecast_compactor():
    view, base, geometry = fixture(2)
    plan = compact_terminal_schedule(view, base, geometry)
    assert plan["allocations"] == base["allocations"]
    assert plan["terminal_sequence"]["saved_reservation_sec"] == 0
    assert all(
        r["terminal"] is None and r["mode"] == "conservative_unpinned"
        for r in plan["terminal_sequence"]["rows"]
    )


@pytest.mark.parametrize("fault", ["binding", "directions", "guard", "capacity"])
def test_geometry_and_sequence_tamper_rejected(fault):
    view, base, geometry = fixture()
    if fault == "binding":
        geometry["forecast_sha256"] = "f" * 64
    elif fault == "directions":
        geometry["stations"]["g"]["directions"]["a"][0] = [2.0, 0.0, 0.0]
    elif fault == "guard":
        geometry["angular_guard_deg"] = 0.0
    else:
        geometry["stations"]["g"]["slew_rate_deg_s"] = 0.0
    geometry = sealed({k: v for k, v in geometry.items() if k != "sha256"})
    with pytest.raises(ValueError):
        compact_terminal_schedule(view, base, geometry)


def test_native_compaction_keeps_ground_ack_and_does_not_claim_optimum():
    from space_env.planning.network_demand_benchmark import demand_suite
    from space_env.workspace import MissionWorkspace

    case = next(
        c
        for c in demand_suite()["cases"]
        if c["id"] == "demand-equatorial-staggered-cpu"
    )
    mission = MissionWorkspace.network(
        case["configuration"], planning_method="sequence_compact"
    )
    try:
        meta = mission.operator.session._scenario["metadata"]["network_workspace"]
        assert verify_terminal_schedule(
            meta["forecast"],
            meta["base_schedule"],
            meta["terminal_geometry"],
            meta["schedule"],
        )
        state = mission.state()["network"]
        assert state["optimization"] is None
        assert (
            state["terminal_sequence"]["global_sequence_optimality_established"]
            is False
        )
        result = mission.run()["network"]
        assert result["fulfilled"] == result["planned"]
        assert result["terminal_sequence"]["saved_reservation_sec"] >= 0
    finally:
        mission.close()


def test_previous_release_is_enclosed_over_whole_reserved_interval():
    view, base, geometry = fixture()
    ordered = sorted(base["allocations"], key=lambda a: a["stages"][-1]["cell"])
    first = ordered[0]
    i = first["stages"][-1]["cell"] * 2
    geometry["stations"]["g"]["directions"][first["path"][-1]][i] = [
        math.cos(math.radians(5)),
        math.sin(math.radians(5)),
        0.0,
    ]
    geometry = sealed({k: v for k, v in geometry.items() if k != "sha256"})
    plan = compact_terminal_schedule(view, base, geometry)
    second = plan["terminal_sequence"]["rows"][1]
    assert second["setup_bound_sec"] == pytest.approx(
        0.1 + (5 + geometry["angular_guard_deg"]) / 10
    )


def test_missing_physical_envelope_and_resealed_output_tamper_are_rejected():
    view, base, geometry = fixture()
    schedule = compact_terminal_schedule(view, base, geometry)
    schedule["terminal_sequence"]["rows"][0]["predecessor"] = "invented"
    schedule = sealed({k: v for k, v in schedule.items() if k != "sha256"})
    with pytest.raises(ValueError):
        verify_terminal_schedule(view, base, geometry, schedule)
    for edge in view["edges"]:
        edge.pop("setup_sec")
    view = sealed({k: v for k, v in view.items() if k != "sha256"})
    base = optimize_network(view, max_hops=0)
    geometry["forecast_sha256"] = view["sha256"]
    geometry = sealed({k: v for k, v in geometry.items() if k != "sha256"})
    with pytest.raises(ValueError):
        compact_terminal_schedule(view, base, geometry)


def test_geometry_collection_uses_no_native_future(monkeypatch):
    from space_env.api.network_workspace import (
        NetworkConfiguration,
        collect_network_forecast,
        default_network_configuration,
    )
    from space_env.planning.terminal_sequence import collect_terminal_geometry
    from space_env.runtime import MissionRuntime

    spec = NetworkConfiguration.model_validate(default_network_configuration())
    view = collect_network_forecast(spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native future used by terminal forecast")

    monkeypatch.setattr(MissionRuntime, "submit", forbidden)
    monkeypatch.setattr(MissionRuntime, "advance", forbidden)
    geometry = collect_terminal_geometry(spec, view)
    base = optimize_network(view)
    plan = compact_terminal_schedule(view, base, geometry)
    assert verify_terminal_schedule(view, base, geometry, plan)


def test_saved_receipt_audit_reproduces_plans_and_rejects_matrix_tamper(tmp_path):
    import json
    from pathlib import Path

    from examples.network.compare_terminal_sequence import audit

    path = (
        Path(__file__).resolve().parents[1]
        / "docs/acceptance/NETWORK_TERMINAL_SEQUENCE_V1.json"
    )
    result = audit(path)
    assert result["valid"] and result["native_rows"] == 24
    assert result["plans_reproduced"] == 2 and result["native_replayed"] is False
    receipt = json.loads(path.read_text())
    receipt["report"]["rows"][0]["method"] = "sequence_compact"
    receipt["report"] = sealed(
        {k: v for k, v in receipt["report"].items() if k != "sha256"}
    )
    receipt = sealed({k: v for k, v in receipt.items() if k != "sha256"})
    changed = tmp_path / "tampered.json"
    changed.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        audit(changed)
