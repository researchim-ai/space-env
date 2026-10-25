"""Студия, снимок группировки и пульт живой смены."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from space_env import __version__
from space_env.api.console import open_shift, step_shift
from space_env.mission.scenario_v2 import scenario_manifest
from space_env.mission.shift_scenario import load_shift_scenario
from space_env.simulation.fleet_view import describe_fleet, fleet_episode

_VIZ = Path(__file__).resolve().parents[3] / "viz"
if not _VIZ.is_dir():
    _VIZ = Path.cwd() / "viz"


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
        root = Path(__file__).resolve().parents[3] / "configs" / "scenarios" / "eo_72h.yaml"
        if not root.is_file():
            root = Path(__file__).resolve().parents[1] / "configs" / "eo_72h.yaml"
        return _json_bytes(scenario_manifest(root))
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
        return _file(_VIZ / "studio.html")
    if path.startswith("/viz/"):
        return _file(_VIZ / path.removeprefix("/viz/"))
    return _json_bytes({"error": "нет такого пути"}, 404)


def handle_post(path: str, body: bytes) -> tuple[int, bytes, str]:
    """Команды пульта. GET по-прежнему только читает снимки."""
    if path not in {"/api/shift/reset", "/api/shift/step"}:
        return _json_bytes({"error": "нет такого пути"}, 404)
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return _json_bytes({"error": "нужен словарь JSON"}, 400)
    if not isinstance(payload, dict):
        return _json_bytes({"error": "нужен словарь JSON"}, 400)
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
    root = _VIZ.resolve()
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
