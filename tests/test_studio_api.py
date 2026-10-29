"""Снимок группировки и локальный интерфейс студии."""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from space_env.api.server import _Handler, handle_get
from space_env.simulation.fleet_view import describe_fleet


def test_star_shell_does_not_close_the_one_watt_link():
    shell = describe_fleet(66, "star", planes=6, phasing=2)
    assert shell["count"] == 66
    assert shell["closed_links"] == 0
    assert shell["closed_neighbor_links"] == 0
    assert shell["neighbor_km"] == pytest.approx(4033.4, rel=0.02)
    assert shell["nearest_km"] == pytest.approx(797.0, abs=15.0)
    assert shell["nearest_margin_db"] < 0.0


def test_chain_of_six_reports_neighbors():
    chain = describe_fleet(6, "chain")
    assert chain["count"] == 6
    assert chain["planes"] == 1
    assert chain["neighbor_km"] > 0.0


def test_api_serves_scenario_fleet_and_studio():
    status, body, kind = handle_get("/api/health", {})
    assert status == 200 and json.loads(body)["status"] == "ok"
    _status, body, _kind = handle_get("/api/scenario", {})
    scenario = json.loads(body)
    assert scenario["keys"][0] == "sentinel-2a"
    status, body, _kind = handle_get("/api/tasking", {})
    tasking = json.loads(body)
    assert status == 200
    assert tasking["requests"][:2] == ["kiruna-day", "svalbard-radar"]
    assert len(tasking["requests"]) == 16
    assert tasking["requests"][2].startswith("fleet-")
    assert tasking["stations"] == ["Kiruna", "Svalbard", "Matera"]
    _status, page, kind = handle_get("/studio", {})
    assert b"tasking-requests" in page
    assert kind.startswith("text/html")
    assert scenario["until"] == "full"
    status, body, _kind = handle_get("/api/fleet", {"count": ["6"], "layout": ["chain"]})
    assert status == 200 and json.loads(body)["count"] == 6
    status, body, _kind = handle_get("/api/fleet", {"count": ["80"], "layout": ["chain"]})
    assert status == 400 and "66" in json.loads(body)["error"]
    status, body, kind = handle_get("/studio", {})
    page = body.decode()
    assert status == 200 and "text/html" in kind and "Студия смены" in page and 'id="stage"' in page
    assert "Пульт смены" in page and "Открыть смену" in page
    assert 'id="groundmap"' in page and "Аппараты в этом кадре" in page
    assert "Журнал дуги" in page and "Сеансы" in page and "Все дела" in page
    assert "Сближения" in page and "дуга с импульсом из плана" in page
    assert "Журнал дуги" in page and "Журнал смены" in page and "пропуск" in page
    assert "Полёт смены" in page and "расчётная дуга" in page
    assert "Сейчас, км" in page and "Окрестность ±250 км" in page
    assert "/viz/studio.js" in page
    status, body, kind = handle_get("/viz/studio.js", {})
    assert status == 200 and "Трасса текущего витка" in body.decode() and "setArc" in body.decode()
    assert "alongKm" in body.decode() and "setFlight" in body.decode()
    status, body, kind = handle_get("/api/episode", {"layout": ["chain"]})
    episode = json.loads(body)
    assert status == 200 and len(episode["frames"]) >= 2
    assert len(episode["frames"][0]["vehicles"]) == 6
    assert episode["frames"][0]["stations"]
    status, body, _kind = handle_get("/api/episode", {"layout": ["star"]})
    star = json.loads(body)
    assert star["summary"]["count"] == 66 and star["summary"]["closed_links"] == 0
    assert len(star["frames"][0]["vehicles"]) == 66
    status, body, _kind = handle_get("/api/episode", {"layout": ["shift"]})
    shift = json.loads(body)
    assert status == 200
    assert set(shift["frames"][0]["vehicles"]) == {"sentinel-2a", "landsat-9", "sar-small", "cubesat-6u"}
    assert shift["summary"]["minutes"] == 1440
    assert shift["frames"][-1]["t"] == pytest.approx(86_400.0)
    assert "Svalbard" in shift["summary"]["stations"] and "Moscow" in shift["summary"]["stations"]
    assert "Шпицберген" in shift["summary"]["brief"]
    assert "Импульс" in shift["summary"]["brief"]
    assert "Pc" in shift["summary"]["brief"]
    screened = shift["summary"]["conjunctions"]
    hot = next(item for item in screened if item["name"] == "HOT")
    warm = next(item for item in screened if item["name"] == "WARM")
    debris = next(item for item in screened if item["name"] == "debris-fragment")
    assert hot["required"] is True and warm["required"] is False and debris["required"] is False
    assert 10 * 3600 < hot["t_sec"] < warm["t_sec"] < 86_400
    assert hot["coast_pc"] > 1.0e-4 > hot["flown_pc"]
    assert warm["coast_pc"] > 1.0e-4 > warm["flown_pc"]
    assert debris["flown_pc"] < 1.0e-4
    assert set(shift["frames"][0]["companions"]) == {"HOT", "WARM"}
    assert set(shift["frames"][0]["vehicles"]) == {"sentinel-2a", "landsat-9", "sar-small", "cubesat-6u"}

    def separation_m(frame, name):
        own = frame["vehicles"]["sentinel-2a"]["r"]
        other = frame["companions"][name]["r"]
        return sum((own[index] - other[index]) ** 2 for index in range(3)) ** 0.5

    opened = separation_m(shift["frames"][0], "HOT")
    meet = min(shift["frames"], key=lambda frame: abs(frame["t"] - hot["t_sec"]))
    assert opened < 30_000.0
    assert separation_m(meet, "HOT") > opened
    coast_start = shift["frames"][0]["vehicles"]["sentinel-2a"]["r_coast"]
    flown_start = shift["frames"][0]["vehicles"]["sentinel-2a"]["r"]
    assert sum((coast_start[index] - flown_start[index]) ** 2 for index in range(3)) ** 0.5 < 1.0

    def coast_separation_m(frame, name):
        own = frame["vehicles"]["sentinel-2a"]["r_coast"]
        other = frame["companions"][name]["r"]
        return sum((own[index] - other[index]) ** 2 for index in range(3)) ** 0.5

    coast_meet = coast_separation_m(meet, "HOT")
    assert coast_meet < 1_000.0
    assert separation_m(meet, "HOT") > 100_000.0
    events = shift["summary"]["events"]
    burn = next(item for item in events if item["kind"] == "maneuver" and item["required"])
    assert burn["start_sec"] == pytest.approx(0.0)
    assert burn["vehicle"] == "sentinel-2a"
    assert shift["summary"]["burn_spent_m_s"] == pytest.approx(burn["dv_m_s"])
    assert any(item["kind"] == "imaging" and item["required"] for item in events)
    moscow = next(
        item
        for item in events
        if item["vehicle"] == "cubesat-6u" and item["kind"] == "comms" and item["required"] and item["name"] == "Moscow"
    )
    assert moscow["start_sec"] > 3600
    denied = [
        item
        for item in events
        if item["kind"] == "comms" and not item["required"] and "занято" in item["note"]
    ]
    assert denied
    for item in denied:
        holder = item["note"].split("аппаратом ", 1)[1].rstrip(".")
        assert holder != item["vehicle"]
        assert holder in ("sentinel-2a", "landsat-9", "sar-small")
    moscow_seen = [
        frame["t"]
        for frame in shift["frames"]
        if frame["vehicles"]["cubesat-6u"]["contact"] and frame["vehicles"]["cubesat-6u"]["station"] == "Moscow"
    ]
    assert moscow_seen and min(moscow_seen) == pytest.approx(4020.0, abs=120.0)
    svalbard_seen = [
        frame["t"]
        for frame in shift["frames"]
        for name in ("sentinel-2a", "landsat-9", "sar-small")
        if frame["vehicles"][name]["contact"] and frame["vehicles"][name]["station"] == "Svalbard"
    ]
    assert svalbard_seen and min(svalbard_seen) > 3600
    svalbard_dumps = [
        item
        for item in events
        if item["kind"] == "comms" and item["required"] and item["name"] == "Svalbard" and item["start_sec"] > 5400
    ]
    assert {"sentinel-2a", "landsat-9"} <= {item["vehicle"] for item in svalbard_dumps}
    status, _body, _kind = handle_get("/viz/../pyproject.toml", {})
    assert status == 404


def test_studio_root_finds_the_page_and_the_packaged_suite():
    import space_env
    from pathlib import Path

    from space_env.api.server import studio_root

    root = studio_root()
    assert (root / "studio.html").is_file()
    assert (root / "studio.js").is_file()
    suite = Path(space_env.__file__).resolve().parent / "configs" / "benchmark_suites" / "mission_v1_smoke.yaml"
    assert suite.is_file()
    import json

    from space_env.api.server import _SESSIONS, handle_get, handle_post

    _SESSIONS.clear()
    status, body, _kind = handle_post("/api/sessions", b'{"scenario": "family_nominal.yaml"}')
    first = json.loads(body)
    assert status == 200
    status, body, _kind = handle_post("/api/sessions", b'{"scenario": "family_nominal.yaml"}')
    second = json.loads(body)
    assert status == 200 and second["session_id"] != first["session_id"]
    status, moved, _kind = handle_post(
        f"/api/sessions/{first['session_id']}/advance",
        b'{"expected_revision": 0, "dt_sec": 30}',
    )
    moved_body = json.loads(moved)
    assert status == 200 and moved_body["now_sec"] == 30.0
    assert [row["t"] for row in moved_body["track"]] == [0.0, 30.0]
    assert [frame["t"] for frame in moved_body["episode"]["frames"]] == [0.0, 30.0]
    assert moved_body["episode"]["summary"]["layout"] == "session"
    assert len(moved_body["episode"]["frames"][1]["vehicles"]["sentinel-a"]["r"]) == 3
    assert moved_body["episode"]["earth_radius_m"] > 6.0e6
    assert moved_body["vehicles"][0]["name"] == "sentinel-a"
    opened = moved_body["track"][0]
    stepped = moved_body["track"][1]
    assert abs(stepped["lat_deg"] - opened["lat_deg"]) + abs(stepped["lon_deg"] - opened["lon_deg"]) > 0.05
    status, still, _kind = handle_get(f"/api/sessions/{second['session_id']}", {})
    still_body = json.loads(still)
    assert status == 200 and still_body["now_sec"] == 0.0
    assert [row["t"] for row in still_body["track"]] == [0.0]
    again = _SESSIONS[second["session_id"]].report()
    assert len(again["track"]) == 1
    status, _body, _kind = handle_post(
        f"/api/sessions/{first['session_id']}/advance",
        b'{"expected_revision": 0, "dt_sec": 30}',
    )
    assert status == 409
    _SESSIONS.clear()


def test_http_server_answers_health():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/health") as response:
            payload = json.loads(response.read().decode())
        assert payload["status"] == "ok"
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
