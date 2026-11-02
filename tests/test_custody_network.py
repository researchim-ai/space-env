"""Causal packets/custody/queues, shared endpoint resources and exact product credit."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from space_env.reference.dtn import DTNSettings
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    reference_scenario,
)
from space_env.simulation.snapshot import digest
from tests.test_radio_network import spec as radio_fixture


def rotate_payload(payload, angle):
    p = copy.deepcopy(payload)
    states = np.asarray(p["states_si"])
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    states[:, :3] = states[:, :3] @ matrix.T
    states[:, 3:] = states[:, 3:] @ matrix.T
    p["states_si"] = states.tolist()
    p["sha256"] = digest({k: v for k, v in p.items() if k != "sha256"})
    return p


@pytest.fixture(scope="module")
def spec():
    data = copy.deepcopy(radio_fixture.__wrapped__())
    data["ephemerides"]["b"] = rotate_payload(data["ephemerides"]["b"], 0.01)
    data["dtn_network"] = {
        "source": "synthetic engineering fixture",
        "parameter_status": "synthetic",
        "endpoints": {a: {"power_w": 2.0} for a in data["vehicles"]},
        "links": [
            {
                "source": "a",
                "destination": "b",
                "receiver": data["radio_network"]["receivers"]["Ground"],
            }
        ],
        "fragment_bits": 500000,
        "sample_sec": 0.1,
    }
    return data


def runtime(spec):
    return MissionRuntime(
        reference_scenario(spec, id="custody", lineage="B05-B"),
        access=AccessPolicy(truth=True),
    )


def relay(
    m, *, id="hop", source="a", destination="b", product="a", ttl=20.0, duration=10.0
):
    return m.submit(
        [
            {
                "command_id": id,
                "actor_id": source,
                "kind": "relay",
                "destination_id": destination,
                "product_id": product,
                "duration_sec": duration,
                "relay": {"ttl_sec": ttl},
            }
        ],
        expected_revision=m.revision,
        idempotency_key=id,
    )


def advance(m, dt, key="advance"):
    return m.advance(dt, expected_revision=m.revision, idempotency_key=key)


def network(m):
    return m._backend.mission.dtn


def assert_ledgers(m):
    report = m._backend.report()
    d = report["dtn"]
    assert abs(d["wire_balance_error_bits"]) < 1e-6
    assert max(abs(v) for v in d["receiver_balance_error_bits"].values()) < 1e-6
    assert all(
        abs(v["balance_error_bits"]) < 1e-6 for v in report["inventory_ledger"].values()
    )
    assert all(
        abs(v["balance_error_wh"]) < 1e-11 for v in report["energy_ledger"].values()
    )


def test_custody_after_fragments_arrival_and_ack_not_serialization(spec):
    m = runtime(spec)
    relay(m)
    advance(m, 0.3)
    row = network(m).transfers["hop"]
    native = m._backend.mission
    assert 0 < row["progress_bits"] < 500000
    assert native.products["a"].actor == "a" and native.remaining["a"] == 2e6
    assert network(m).reserved("b") == 2e6 and row["received_bits"] == 0
    assert_ledgers(m)
    advance(m, 4.0, "finish")
    assert (
        native.products["a"].actor == "b"
        and network(m).transfers["hop"]["status"] == "completed"
    )
    assert (
        network(m).reserved("b") == 0
        and not native.receipts
        and native.report()["on_time_bits"] == 0
    )
    assert_ledgers(m)


def test_receiver_does_not_use_inflight_fragment_and_keeps_radio_until_arrival(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["fragment_bits"] = 2000000
    m = runtime(d)
    relay(m)
    advance(m, 2.1)
    row = network(m).transfers["hop"]
    assert row["state"] == "inflight" and row["received_bits"] == 0
    assert network(m).receiving("b")
    assert m._backend.mission.products["a"].actor == "a"
    with pytest.raises(RuntimeContractError, match="receiver is busy"):
        m.submit(
            [
                {
                    "command_id": "ground",
                    "actor_id": "b",
                    "kind": "downlink",
                    "duration_sec": 1.0,
                    "destination_id": "Ground",
                }
            ],
            expected_revision=m.revision,
            idempotency_key="ground",
        )
    advance(m, 0.02, "arrival-ack")
    assert m._backend.mission.products["a"].actor == "b"
    assert_ledgers(m)


def test_fragment_energy_both_endpoints_and_listening_tail(spec):
    m = runtime(spec)
    relay(m)
    advance(m, 4.0)
    native = m._backend.mission
    net = network(m)
    assert native.tx_seconds["a"] == pytest.approx(2.4)
    assert native.rx_seconds["b"] > native.tx_seconds["a"]
    assert native.consumed_wh["a"] * 3600 == pytest.approx(24.0)
    assert native.consumed_wh["b"] * 3600 == pytest.approx(net.rx_energy_j["b"])
    assert net.tx_bits["a"] == pytest.approx(2e6)
    assert_ledgers(m)


def test_receiver_buffer_backpressure_not_silent_drop(spec):
    d = copy.deepcopy(spec)
    d["vehicles"]["b"]["memory_bits"] = 3e6
    m = runtime(d)
    relay(m)
    advance(m, 2.0)
    assert network(m).transfers["hop"]["reason"] == "receiver_memory"
    assert m._backend.mission.tx_seconds["a"] == 0 and network(m).reserved("b") == 0
    assert_ledgers(m)


def test_reservation_prevents_overcommitting_injected_products(spec):
    d = copy.deepcopy(spec)
    d["vehicles"]["b"]["memory_bits"] = 4e6
    m = runtime(d)
    relay(m)
    advance(m, 0.2)
    before = m.checkpoint()
    e = {
        "id": "inject",
        "type": "onboard_product_added",
        "time_sec": m.time_sec,
        "source": "test",
        "reason": "test memory",
        "payload": {
            "id": "extra",
            "actor": "b",
            "size_bits": 1.0,
            "ready_sec": m.time_sec,
            "deadline_sec": 20.0,
        },
    }
    with pytest.raises(RuntimeContractError, match="memory"):
        m.inject([e], expected_revision=m.revision, idempotency_key="inject")
    assert m.checkpoint() == before


def test_loss_retries_pay_resources_and_do_not_create_custody(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["links"][0]["loss_probability"] = 1.0
    d["dtn_network"]["max_retries"] = 1
    m = runtime(d)
    relay(m)
    advance(m, 3.0)
    row = network(m).transfers["hop"]
    assert row["status"] == "failed" and row["reason"] == "retry_budget"
    assert (
        network(m).tx_bits["a"] == pytest.approx(1e6)
        and m._backend.mission.products["a"].actor == "a"
    )
    assert network(m).reserved("b") == 0
    assert_ledgers(m)


def test_ack_loss_deduplicates_receiver_not_reward(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["links"][0]["ack_loss_probability"] = 1.0
    d["dtn_network"]["max_retries"] = 1
    m = runtime(d)
    relay(m)
    advance(m, 3.0)
    assert network(m).rx_unique_bits["b"] == 500000
    assert network(m).rx_duplicate_bits["b"] == 500000
    assert (
        m._backend.mission.products["a"].actor == "a"
        and not m._backend.mission.receipts
    )
    assert network(m).rx_discarded_bits["b"] == 500000
    assert_ledgers(m)


def test_ttl_discards_owned_and_staged_data_not_free_retry_reset(spec):
    m = runtime(spec)
    relay(m, ttl=0.9)
    advance(m, 2.0)
    assert network(m).transfers["hop"]["status"] == "expired"
    assert (
        "a" in m._backend.mission.cancelled and m._backend.mission.remaining["a"] == 0
    )
    assert network(m).reserved("b") == 0
    m.cancel("hop", expected_revision=m.revision, idempotency_key="release-window")
    with pytest.raises(RuntimeContractError, match="intact"):
        relay(m, id="retry-expired")
    assert_ledgers(m)


def test_cancel_partial_and_inflight_drain_preserve_source(spec):
    m = runtime(spec)
    relay(m)
    advance(m, 0.3)
    m.cancel("hop", expected_revision=m.revision, idempotency_key="cancel")
    assert (
        network(m).transfers["hop"]["status"] == "cancelled"
        and network(m).reserved("b") == 0
    )
    assert network(m).wasted_partial_bits["a"] > 0
    assert_ledgers(m)
    d = copy.deepcopy(spec)
    d["dtn_network"]["fragment_bits"] = 2000000
    n = runtime(d)
    relay(n)
    advance(n, 2.1)
    n.cancel("hop", expected_revision=n.revision, idempotency_key="cancel")
    assert network(n).reserved("b") == 2e6
    advance(n, 0.2, "drain")
    assert (
        network(n).reserved("b") == 0 and n._backend.mission.products["a"].actor == "a"
    )
    assert_ledgers(n)


def test_partial_packet_outage_restarts_attempt_and_pays_waste(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["links"][0]["outages"] = [[0.3, 0.6]]
    m = runtime(d)
    relay(m)
    advance(m, 5.0)
    assert network(m).transfers["hop"]["status"] == "completed"
    assert network(m).wasted_partial_bits["a"] > 0 and network(m).tx_bits["a"] > 2e6
    assert_ledgers(m)


def test_earth_occultation_and_range_prevent_radio_cost(spec):
    for angle, limit in [(math.pi, 1e7), (0.01, 10000.0)]:
        d = copy.deepcopy(spec)
        d["ephemerides"]["b"] = rotate_payload(d["ephemerides"]["a"], angle)
        d["dtn_network"]["links"][0]["max_range_m"] = limit
        m = runtime(d)
        relay(m)
        advance(m, 1.0)
        assert (
            network(m).transfers["hop"]["reason"] == "geometry"
            and m._backend.mission.tx_seconds["a"] == 0
        )
        assert_ledgers(m)


def test_receiver_soc_floor_cuts_physical_rx_and_prevents_bad_packet(spec):
    d = copy.deepcopy(spec)
    d["vehicles"]["b"].update(initial_soc=0.5, battery_wh=0.001)
    d["dtn_network"]["endpoints"]["b"].update(power_w=2.0, min_soc=0.25)
    m = runtime(d)
    relay(m)
    advance(m, 2.0)
    assert m._backend.mission.energy_wh["b"] == pytest.approx(0.00025)
    assert m._backend.mission.products["a"].actor == "a"
    assert_ledgers(m)


def test_tx_quota_includes_acquisition_no_overspend(spec):
    d = copy.deepcopy(spec)
    d["vehicles"]["a"]["max_tx_sec"] = 0.25
    m = runtime(d)
    relay(m)
    advance(m, 2.0)
    assert m._backend.mission.tx_seconds["a"] == pytest.approx(0.25)
    assert m._backend.mission.products["a"].actor == "a"
    assert_ledgers(m)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["dtn_network"]["links"].append(d["dtn_network"]["links"][0]),
        lambda d: d["dtn_network"]["links"][0].update(destination="a"),
        lambda d: d["dtn_network"]["links"][0].update(outages=[[2.0, 1.0]]),
        lambda d: d["dtn_network"].update(fragment_bits=True),
        lambda d: d["dtn_network"]["links"][0]["receiver"].update(
            aggregate_rate_bps=1e6
        ),
    ],
)
def test_strict_transport_config(spec, mutation):
    d = copy.deepcopy(spec)
    mutation(d)
    with pytest.raises(ValueError):
        DTNSettings.model_validate(d["dtn_network"])


def test_mid_packet_inflight_ack_and_custody_json_replay(spec, tmp_path):
    m = runtime(spec)
    relay(m)
    for i, dt in enumerate([0.2, 0.4, 0.0001, 0.0105, 3.0]):
        advance(m, dt, f"a{i}")
        assert_ledgers(m)
        path = tmp_path / "state.json"
        m.save(path)
        loaded = MissionRuntime.load(path, access=AccessPolicy(truth=True))
        assert loaded.checkpoint() == m.checkpoint()
        m = loaded
    assert m._backend.mission.products["a"].actor == "b"


def test_atomic_budget_failure_restores_products_buffers_events_and_energy(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["max_events"] = 2
    m = runtime(d)
    relay(m)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError, match="budget"):
        advance(m, 4.0)
    assert m.checkpoint() == before


def test_relay_ground_delivery_only_after_final_ack(spec):
    m = runtime(spec)
    relay(m)
    advance(m, 4.0)
    assert m._backend.metrics()["on_time_bits"] == 0
    m.submit(
        [
            {
                "command_id": "ground",
                "actor_id": "b",
                "kind": "downlink",
                "duration_sec": 3.0,
                "destination_id": "Ground",
                "product_id": "a",
            }
        ],
        expected_revision=m.revision,
        idempotency_key="ground",
    )
    advance(m, 2.0, "serialize-ground")
    assert m._backend.metrics()["on_time_bits"] == 0
    advance(m, 0.1, "ground-ack")
    assert m._backend.metrics()["on_time_bits"] == 2e6
    assert_ledgers(m)


def four_actor(spec):
    d = copy.deepcopy(spec)
    for actor, angle in [("c", 0.02), ("d", 0.03)]:
        d["ephemerides"][actor] = rotate_payload(d["ephemerides"]["a"], angle)
        d["vehicles"][actor] = copy.deepcopy(d["vehicles"]["a"])
        d["radio_network"]["transmitters"][actor] = copy.deepcopy(
            d["radio_network"]["transmitters"]["a"]
        )
        d["dtn_network"]["endpoints"][actor] = {"power_w": 2.0}
    d["products"][1]["actor"] = "c"
    d["dtn_network"]["links"].append(
        {
            "source": "c",
            "destination": "d",
            "receiver": copy.deepcopy(d["radio_network"]["receivers"]["Ground"]),
        }
    )
    return d


def batch_relays(m, rows):
    return m.submit(
        [
            {
                "command_id": id,
                "actor_id": src,
                "kind": "relay",
                "destination_id": dst,
                "product_id": pid,
                "duration_sec": 10.0,
                "relay": {"ttl_sec": 20.0},
            }
            for id, src, dst, pid in rows
        ],
        expected_revision=m.revision,
        idempotency_key="batch",
    )


def test_disjoint_isl_pairs_execute_concurrently_not_serial_blocking(spec):
    m = runtime(four_actor(spec))
    batch_relays(m, [("one", "a", "b", "a"), ("two", "c", "d", "b")])
    advance(m, 4.0)
    assert all(row["status"] == "completed" for row in network(m).transfers.values())
    assert any(
        len(row["transmissions"]) == 2
        and all(p["rate_bps"] > 0 for p in row["transmissions"].values())
        for row in network(m).intervals
    )
    assert (
        network(m).custody_history[0]["time_sec"] < 3.0
        and network(m).custody_history[1]["time_sec"] < 3.0
    )
    assert_ledgers(m)


def test_shared_receiver_half_duplex_and_queued_backpressure(spec):
    d = four_actor(spec)
    d["dtn_network"]["links"][1]["destination"] = "b"
    m = runtime(d)
    batch_relays(m, [("one", "a", "b", "a"), ("two", "c", "b", "b")])
    advance(m, 7.0)
    assert all(row["status"] == "completed" for row in network(m).transfers.values())
    for row in network(m).intervals:
        used = []
        for tid in row["transmissions"]:
            t = network(m).transfers[tid]
            used.extend([t["source"], t["destination"]])
        assert len(used) == len(set(used))
    assert_ledgers(m)


def test_direct_downlink_and_disjoint_isl_share_one_wall_clock(spec):
    d = four_actor(spec)
    d["stations"][0]["terminals"] = 2
    m = runtime(d)
    m.submit(
        [
            {
                "command_id": "ground",
                "actor_id": "c",
                "kind": "downlink",
                "destination_id": "Ground",
                "product_id": "b",
                "duration_sec": 5.0,
            },
            {
                "command_id": "hop",
                "actor_id": "a",
                "kind": "relay",
                "destination_id": "b",
                "product_id": "a",
                "duration_sec": 5.0,
                "relay": {"ttl_sec": 20.0},
            },
        ],
        expected_revision=0,
        idempotency_key="mix",
    )
    advance(m, 4.0)
    assert network(m).transfers["hop"]["status"] == "completed"
    assert m._backend.mission.report()["on_time_bits"] == 2e6
    assert any(
        r["actor"] == "c" and r["start_sec"] < 0.5 for r in m._backend.mission.receipts
    )
    assert_ledgers(m)


def test_active_ground_control_at_receiver_blocks_isl_without_double_power(spec):
    m = runtime(spec)
    m.submit(
        [
            {
                "command_id": "ground",
                "actor_id": "b",
                "kind": "downlink",
                "destination_id": "Ground",
                "product_id": "b",
                "duration_sec": 3.0,
            },
            {
                "command_id": "hop",
                "actor_id": "a",
                "kind": "relay",
                "destination_id": "b",
                "product_id": "a",
                "duration_sec": 8.0,
                "relay": {"ttl_sec": 20.0},
            },
        ],
        expected_revision=0,
        idempotency_key="mix",
    )
    advance(m, 3.0)
    assert network(m).tx_bits["a"] == 0
    advance(m, 4.0, "after-ground")
    assert network(m).transfers["hop"]["status"] == "completed"
    assert_ledgers(m)


def test_multihop_preserves_lifetime_and_origin_not_resetting_at_new_owner(spec):
    d = four_actor(spec)
    d["dtn_network"]["links"].append(
        {
            "source": "b",
            "destination": "d",
            "receiver": copy.deepcopy(d["radio_network"]["receivers"]["Ground"]),
        }
    )
    m = runtime(d)
    relay(m, ttl=5.0)
    advance(m, 3.0)
    relay(m, id="second-hop", source="b", destination="d", ttl=20.0)
    assert network(m).transfers["second-hop"]["expires_sec"] == 5.0
    advance(m, 3.0, "expires-in-second-hop")
    assert (
        m._backend.mission.remaining["a"] == 0
        and network(m).transfers["second-hop"]["status"] == "expired"
    )
    assert_ledgers(m)


def test_seeded_ack_loss_recovers_with_duplicate_and_exact_custody(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"].update(fragment_bits=2000000, max_retries=5, seed=5)
    d["dtn_network"]["links"][0]["ack_loss_probability"] = 0.5
    # The seed is part of the declared scenario, not a mocked success/failure path.
    m = runtime(d)
    relay(m)
    advance(m, 9.0)
    assert network(m).transfers["hop"]["status"] == "completed"
    assert network(m).rx_duplicate_bits["b"] > 0
    assert len(network(m).custody_history) == 1
    assert_ledgers(m)


def test_receiver_memory_queue_retains_all_data(spec):
    d = four_actor(spec)
    d["vehicles"]["b"]["memory_bits"] = 2e6
    d["products"] = []
    d["products"] = [
        {"id": pid, "actor": actor, "size_bits": 2e6, "deadline_sec": 20.0}
        for pid, actor in [("a", "a"), ("b", "c")]
    ]
    d["dtn_network"]["links"][1]["destination"] = "b"
    m = runtime(d)
    batch_relays(m, [("one", "a", "b", "a"), ("two", "c", "b", "b")])
    advance(m, 5.0)
    assert network(m).transfers["one"]["status"] == "completed"
    assert network(m).transfers["two"]["reason"] == "receiver_memory"
    assert (
        m._backend.mission.remaining["b"] == 2e6
        and m._backend.mission.products["b"].actor == "c"
    )
    assert_ledgers(m)


def test_source_cannot_discard_pinned_product(spec):
    m = runtime(spec)
    relay(m)
    before = m.checkpoint()
    e = {
        "id": "cancel-data",
        "type": "product_cancelled",
        "time_sec": 0.0,
        "source": "test",
        "reason": "test pin",
        "payload": {"product_id": "a"},
    }
    with pytest.raises(RuntimeContractError, match="pinned"):
        m.inject([e], expected_revision=m.revision, idempotency_key="delete-data")
    assert m.checkpoint() == before


def test_expiry_while_inflight_closes_wire_and_receiver_ledgers(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["fragment_bits"] = 2000000
    m = runtime(d)
    relay(m, ttl=2.1001)
    advance(m, 3.0)
    assert network(m).abandoned_inflight_bits["a"] == 2e6
    assert network(m).reserved("b") == 0
    assert_ledgers(m)


def test_successful_ground_ack_survives_later_bundle_expiry(spec):
    m = runtime(spec)
    relay(m, ttl=8.0)
    advance(m, 3.0)
    m.submit(
        [
            {
                "command_id": "ground",
                "actor_id": "b",
                "kind": "downlink",
                "destination_id": "Ground",
                "product_id": "a",
                "duration_sec": 3.0,
            }
        ],
        expected_revision=m.revision,
        idempotency_key="ground",
    )
    advance(m, 6.0, "past-expiry")
    assert (
        m._backend.mission.report()["on_time_bits"] == 2e6
        and "a" not in m._backend.mission.cancelled
    )
    assert_ledgers(m)


def test_dt_partitioning_preserves_results_bits_energy_and_custody_time(spec):
    a, b = runtime(spec), runtime(spec)
    for m in [a, b]:
        relay(m)
    advance(a, 4.0)
    for i, dt in enumerate([0.11, 0.37, 0.83, 1.19, 1.5]):
        advance(b, dt, f"a{i}")
    assert a._backend.mission.energy_wh == pytest.approx(
        b._backend.mission.energy_wh, abs=1e-11
    )
    assert network(a).custody_history[0]["time_sec"] == pytest.approx(
        network(b).custody_history[0]["time_sec"], abs=1e-12
    )
    assert network(a).tx_bits == pytest.approx(network(b).tx_bits, abs=1e-6)
    assert_ledgers(a)
    assert_ledgers(b)


def coupled_scenario(spec, thermal=False):
    from tests.test_spacecraft_bus import scenario as bus_fixture

    d = copy.deepcopy(bus_fixture.__wrapped__())
    definition = d["definition"]
    original = definition["orbits"]["b"]["initial"]
    angle = 0.01
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    original["r_m"] = (rot @ original["r_m"]).tolist()
    original["v_m_s"] = (rot @ original["v_m_s"]).tolist()
    definition["operations"]["radio_network"] = copy.deepcopy(spec["radio_network"])
    for v in definition["operations"]["vehicles"].values():
        v["tx_power_w"] = 10.0
    definition["operations"]["dtn_network"] = copy.deepcopy(spec["dtn_network"])
    if thermal:
        from examples.thermal_network.demo import thermal_model

        for a in definition["spacecraft_bus"]:
            definition["spacecraft_bus"][a]["thermal_network"] = thermal_model()
    return d


def test_managed_product_keeps_origin_through_thermal_custody_and_ground_ack(
    spec, tmp_path
):
    from tests import test_reference_lifecycle as life

    m = MissionRuntime(
        coupled_scenario(spec, thermal=True), access=AccessPolicy(truth=True)
    )
    life.acquire(m)
    life.transform(m)
    relay(m, product="final", duration=5.0, ttl=12.0)
    advance(m, 2.0, "custody")
    r = m._backend.report()
    a = r["lifecycle"]["artifacts"]["final"]
    assert a["actor"] == "b" and a["origin_actor"] == "a"
    assert r["lifecycle"]["metrics"]["fulfilled_requests"] == 0
    assert m._backend.lifecycle.transport is m._backend.mission.dtn
    for values in r["lifecycle"]["ledger"].values():
        assert abs(values["balance_error_bits"]) < 1e-6
    m.submit(
        [
            {
                "command_id": "ground",
                "actor_id": "b",
                "kind": "downlink",
                "destination_id": "Ground",
                "product_id": "final",
                "duration_sec": 2.0,
            }
        ],
        expected_revision=m.revision,
        idempotency_key="ground",
    )
    advance(m, 1.0, "ground-ack")
    r = m._backend.report()
    assert r["lifecycle"]["metrics"]["fulfilled_requests"] == 1
    assert m.observe("b")["actors"]["b"]["lifecycle"]["artifacts"][0]["id"] == "final"
    assert (
        r["spacecraft_bus"]["states"]["b"]["thermal_network"]["source_heat_j"]["radio"]
        > 0
    )
    for values in r["spacecraft_bus"]["ledgers"].values():
        assert (
            abs(values["battery_balance_error_wh"]) < 1e-10
            and abs(values["thermal_balance_error_j"]) < 1e-6
        )
    for values in r["lifecycle"]["ledger"].values():
        assert abs(values["balance_error_bits"]) < 1e-6
    path = tmp_path / "thermal-custody.json"
    m.save(path)
    assert (
        MissionRuntime.load(path, access=AccessPolicy(truth=True)).checkpoint()
        == m.checkpoint()
    )
    assert_ledgers(m)


def test_receiver_thermal_protection_rejects_work_not_free_custody(spec):
    from tests import test_reference_lifecycle as life

    d = coupled_scenario(spec, thermal=True)
    node = d["definition"]["spacecraft_bus"]["b"]["thermal_network"]["nodes"]["cpu"]
    node.update(capacity_j_per_k=0.01, trip_k=285.01, resume_k=285.005)
    m = MissionRuntime(d, access=AccessPolicy(truth=True))
    life.acquire(m)
    life.transform(m)
    relay(m, product="final")
    advance(m, 2.0, "thermal-trip")
    assert m._backend.mission.products["final"].actor == "a"
    assert m._backend.report()["spacecraft_bus"]["states"]["b"]["safe"]
    assert m._backend.report()["lifecycle"]["metrics"]["fulfilled_requests"] == 0
    assert_ledgers(m)


def test_coupled_peak_power_is_shared_with_receive_load(spec):
    from tests import test_reference_lifecycle as life

    d = coupled_scenario(spec)
    d["definition"]["spacecraft_bus"]["b"]["power"]["max_load_w"] = 1.0
    m = MissionRuntime(d, access=AccessPolicy(truth=True))
    life.acquire(m)
    life.transform(m)
    relay(m, product="final")
    advance(m, 1.0, "peak-denied")
    assert network(m).transfers["hop"]["reason"] == "endpoint_power_or_health"
    assert m._backend.mission.rx_seconds["b"] == 0
    assert_ledgers(m)


def test_actor_radio_fault_interrupts_relay_with_intact_source(spec):
    from tests import test_reference_lifecycle as life

    m = MissionRuntime(coupled_scenario(spec), access=AccessPolicy(truth=True))
    life.acquire(m)
    life.transform(m)
    relay(m, product="final")
    advance(m, 0.2, "partial")
    m.inject(
        [life.event(m, "spacecraft_fault", {"actor": "b", "subsystem": "radio"})],
        expected_revision=m.revision,
        idempotency_key="fault",
    )
    advance(m, 1.0, "faulted")
    assert m._backend.mission.products["final"].actor == "a"
    assert network(m).wasted_partial_bits["a"] > 0
    assert_ledgers(m)


def test_managed_ttl_expiry_closes_removal_and_custody_ledger(spec):
    from tests import test_reference_lifecycle as life

    m = MissionRuntime(coupled_scenario(spec), access=AccessPolicy(truth=True))
    life.acquire(m)
    life.transform(m)
    relay(m, product="final", ttl=0.2)
    advance(m, 1.0, "expired")
    r = m._backend.report()["lifecycle"]
    assert r["artifacts"]["final"]["status"] == "expired"
    assert all(abs(row["balance_error_bits"]) < 1e-6 for row in r["ledger"].values())
    assert r["requests"]["request"]["status"] == "failed"
    assert_ledgers(m)


def test_coupled_rollback_preserves_transport_and_bus_aliases(spec):
    from tests import test_reference_lifecycle as life

    d = coupled_scenario(spec)
    d["definition"]["operations"]["dtn_network"]["max_events"] = 2
    m = MissionRuntime(d, access=AccessPolicy(truth=True))
    life.acquire(m)
    life.transform(m)
    relay(m, product="final")
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError, match="budget"):
        advance(m, 3.0, "fail-budget")
    assert m.checkpoint() == before
    assert m._backend.lifecycle.transport is m._backend.mission.dtn
    assert m._backend.bus is m._backend.mission.bus


def test_noninteger_packet_inventory_rejected(spec):
    d = copy.deepcopy(spec)
    d["products"][0]["size_bits"] = 1.5
    m = runtime(d)
    with pytest.raises(RuntimeContractError, match="integer bit"):
        relay(m)


def test_native_transaction_rolls_back_expiry_discard_and_owner_changes(spec):
    d = four_actor(spec)
    d["dtn_network"]["max_events"] = 12
    m = runtime(d)
    relay(m, id="one", ttl=0.3)
    relay(m, id="two", source="c", destination="d", product="b")
    native = m._backend.mission
    before = native.report()
    with pytest.raises(ValueError, match="budget"):
        native.step(
            4.0,
            {},
            processing=dict.fromkeys(native.actors, False),
            relay_controls=["one", "two"],
        )
    assert native.report() == before
    assert not native.cancelled and native.remaining["a"] == 2e6


def test_unknown_relay_controls_are_not_silently_ignored(spec):
    m = runtime(spec)
    with pytest.raises(ValueError, match="admitted"):
        m._backend.mission.step(1.0, {}, relay_controls=["not-admitted"])


def test_outage_during_propagation_tail_loses_packet_then_retries(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["links"][0]["outages"] = [[0.6001, 0.601]]
    m = runtime(d)
    relay(m)
    advance(m, 5.0)
    assert network(m).transfers["hop"]["status"] == "completed"
    assert network(m).rx_lost_bits["b"] == 500000
    assert network(m).tx_bits["a"] > 2e6
    assert_ledgers(m)


def test_isl_receiver_pattern_uses_actual_body_attitude(spec):
    from tests import test_reference_lifecycle as life

    d = coupled_scenario(spec)
    d["definition"]["operations"]["dtn_network"]["links"][0]["rx_beamwidth_deg"] = 5.0
    m = MissionRuntime(d, access=AccessPolicy(truth=True))
    life.acquire(m)
    life.transform(m)
    relay(m, product="final")
    advance(m, 1.0, "bad-rx-pointing")
    assert network(m).transfers["hop"]["reason"] == "rf_margin"
    assert m._backend.mission.rx_seconds["b"] == 0
    assert_ledgers(m)


def test_fragment_budget_prevents_unbounded_queue(spec):
    d = copy.deepcopy(spec)
    d["dtn_network"]["fragment_bits"] = 1
    m = runtime(d)
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError, match="fragment budget"):
        relay(m)
    assert m.checkpoint() == before
