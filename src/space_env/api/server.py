"""Студия, снимок группировки и пульт живой смены."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from space_env import __version__
from space_env.api.console import open_shift, step_shift
from space_env.mission.scenario_v2 import scenario_manifest
from space_env.mission.session import RevisionError, MissionSession, mission_report
from space_env.simulation.commands import MissionCommand
from space_env.mission.shift_scenario import load_shift_scenario
from space_env.simulation.fleet_view import describe_fleet, fleet_episode


def studio_root() -> Path:
    """Каталог студии: корень репозитория, затем копия внутри пакета, затем текущий каталог."""
    packaged = Path(__file__).resolve().parents[1] / "studio_assets"
    checkout = Path(__file__).resolve().parents[3] / "viz"
    local = Path.cwd() / "viz"
    for candidate in (checkout, packaged, local):
        if (candidate / "studio.html").is_file():
            return candidate
    return checkout


_SESSIONS: dict[str, object] = {}
_SESSION_FILES = {
    "mission_short.yaml",
    "family_nominal.yaml",
    "family_outage.yaml",
    "family_infeasible.yaml",
}
_MISSION: dict | None = None


def _mission_file() -> Path:
    root = Path(__file__).resolve().parents[3] / "configs" / "scenarios" / "mission_short.yaml"
    if root.is_file():
        return root
    return Path(__file__).resolve().parents[1] / "configs" / "mission_short.yaml"


def _mission_outcome() -> dict:
    global _MISSION
    if _MISSION is None:
        _MISSION = mission_report(_mission_file(), step_sec=60.0, policy="priority")
    return _MISSION


def _json_bytes(payload: dict, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


def handle_get(path: str, query: dict[str, list[str]]) -> tuple[int, bytes, str]:
    """Ответ на GET. Путь без хоста."""
    if path == "/api/health":
        return _json_bytes({"status": "ok", "version": __version__})
    if path == "/api/scenario":
        return _json_bytes(load_shift_scenario())
    if path == "/api/tasking":
        root = Path(__file__).resolve().parents[3] / "configs" / "scenarios" / "eo_72h_16.yaml"
        if not root.is_file():
            root = Path(__file__).resolve().parents[1] / "configs" / "eo_72h_16.yaml"
        return _json_bytes(scenario_manifest(root))
    if path == "/api/mission":
        return _json_bytes(_mission_outcome())
    if path.startswith("/api/sessions/"):
        session_id = path.removeprefix("/api/sessions/").strip("/")
        session = _SESSIONS.get(session_id)
        if session is None:
            return _json_bytes({"error": "нет такой сессии"}, 404)
        return _json_bytes(session.report())
    if path == "/api/fleet":
        try:
            count = int(query.get("count", ["6"])[0])
            layout = query.get("layout", ["chain"])[0]
            planes = int(query.get("planes", ["6"])[0])
            phasing = int(query.get("phasing", ["2"])[0])
            payload = describe_fleet(count, layout, planes=planes, phasing=phasing)
        except ValueError as exc:
            return _json_bytes({"error": str(exc)}, 400)
        return _json_bytes(payload)
    if path == "/api/episode":
        layout = query.get("layout", ["chain"])[0]
        try:
            payload = fleet_episode(layout)
        except ValueError as exc:
            return _json_bytes({"error": str(exc)}, 400)
        return _json_bytes(payload)
    if path in {"/", "/studio"}:
        return _file(studio_root() / "studio.html")
    if path.startswith("/viz/"):
        return _file(studio_root() / path.removeprefix("/viz/"))
    return _json_bytes({"error": "нет такого пути"}, 404)


def handle_post(path: str, body: bytes) -> tuple[int, bytes, str]:
    """Команды пульта и сессии миссии."""
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return _json_bytes({"error": "нужен словарь JSON"}, 400)
    if not isinstance(payload, dict):
        return _json_bytes({"error": "нужен словарь JSON"}, 400)
    if path == "/api/sessions":
        name = str(payload.get("scenario", "mission_short.yaml"))
        if Path(name).name != name or name not in _SESSION_FILES:
            return _json_bytes({"error": "сценарий сессии не из списка"}, 400)
        if len(_SESSIONS) >= 8:
            return _json_bytes({"error": "слишком много сессий"}, 429)
        session = MissionSession(name)
        _SESSIONS[session.session_id] = session
        return _json_bytes(session.report())
    if path.startswith("/api/sessions/") and path.endswith("/advance"):
        session_id = path.removeprefix("/api/sessions/").removesuffix("/advance").strip("/")
        session = _SESSIONS.get(session_id)
        if session is None:
            return _json_bytes({"error": "нет такой сессии"}, 404)
        try:
            commands = [
                MissionCommand(
                    command_id=str(item["command_id"]),
                    actor_id=str(item["actor_id"]),
                    kind=str(item["kind"]),
                    request_id=str(item.get("request_id", "")),
                    destination_id=str(item.get("destination_id", "")),
                )
                for item in payload.get("commands", [])
            ]
            report = session.advance(
                float(payload.get("dt_sec", 60.0)),
                commands,
                expected_revision=int(payload["expected_revision"]),
            )
        except KeyError:
            return _json_bytes({"error": "нужна ревизия и поля команды"}, 400)
        except RevisionError as exc:
            return _json_bytes({"error": str(exc)}, 409)
        except ValueError as exc:
            return _json_bytes({"error": str(exc)}, 400)
        return _json_bytes(report)
    if path not in {"/api/shift/reset", "/api/shift/step"}:
        return _json_bytes({"error": "нет такого пути"}, 404)
    try:
        if path == "/api/shift/reset":
            return _json_bytes(open_shift(int(payload.get("seed", 0))))
        return _json_bytes(step_shift(int(payload["action"])))
    except KeyError:
        return _json_bytes({"error": "нужно поле action"}, 400)
    except RuntimeError as exc:
        return _json_bytes({"error": str(exc)}, 409)
    except ValueError as exc:
        return _json_bytes({"error": str(exc)}, 400)


def _file(path: Path) -> tuple[int, bytes, str]:
    root = studio_root().resolve()
    try:
        target = path.resolve()
        target.relative_to(root)
    except (OSError, ValueError):
        return _json_bytes({"error": "нет такого файла"}, 404)
    if not target.is_file():
        return _json_bytes({"error": "нет такого файла"}, 404)
    kind = "text/html; charset=utf-8"
    if target.suffix == ".js":
        kind = "text/javascript; charset=utf-8"
    elif target.suffix == ".json":
        kind = "application/json; charset=utf-8"
    elif target.suffix == ".css":
        kind = "text/css; charset=utf-8"
    elif target.suffix in {".jpg", ".jpeg"}:
        kind = "image/jpeg"
    elif target.suffix == ".png":
        kind = "image/png"
    return 200, target.read_bytes(), kind


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        status, body, kind = handle_get(parsed.path, parse_qs(parsed.query))
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        status, payload, kind = handle_post(parsed.path, body)
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8766) -> None:
    """Слушает студию, пока процесс не остановят."""
    server = ThreadingHTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
