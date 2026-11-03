"""Студия, снимок группировки и пульт живой смены."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from space_env import __version__
from space_env.api.console import open_shift, step_shift
from space_env.mission.scenario_v2 import scenario_manifest
from space_env.mission.session import MissionSession, RevisionError, mission_report
from space_env.mission.shift_scenario import load_shift_scenario
from space_env.simulation.commands import MissionCommand
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
    root = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "scenarios"
        / "mission_short.yaml"
    )
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


def handle_get(
    path: str, query: dict[str, list[str]], *, token=None
) -> tuple[int, bytes, str]:
    """Ответ на GET. Путь без хоста."""
    if path.startswith("/api/operator/"):
        return _operator_response("GET", path, token=token)
    if path == "/operator":
        return _file(studio_root() / "operator.html")
    if path == "/api/health":
        return _json_bytes({"status": "ok", "version": __version__})
    if path == "/api/scenario":
        return _json_bytes(load_shift_scenario())
    if path == "/api/tasking":
        root = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "scenarios"
            / "eo_72h_16.yaml"
        )
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


def handle_post(path: str, body: bytes, *, token=None) -> tuple[int, bytes, str]:
    """Команды пульта и сессии миссии."""
    if path.startswith("/api/operator/") and len(body) > 8 * 1024 * 1024:
        return _json_bytes(
            {"error": {"code": "body_limit", "message": "request exceeds 8 MiB"}}, 413
        )
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except (ValueError, UnicodeError, RecursionError):
        return _json_bytes({"error": "нужен словарь JSON"}, 400)
    if not isinstance(payload, dict):
        return _json_bytes({"error": "нужен словарь JSON"}, 400)
    if path.startswith("/api/operator/"):
        return _operator_response("POST", path, payload=payload, token=token)
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
        session_id = (
            path.removeprefix("/api/sessions/").removesuffix("/advance").strip("/")
        )
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
                    duration_sec=float(item.get("duration_sec", 0.0)),
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


def _operator_response(method, path, *, payload=None, token=None):
    try:
        from space_env.api.operator import OperatorError, dispatch
    except ImportError:
        return _json_bytes(
            {
                "ok": False,
                "error": {
                    "code": "missing_dependency",
                    "message": "operator mode requires space-env[reference]",
                },
            },
            503,
        )
    from space_env.api.operator_artifacts import ArtifactChunk, FileBody
    from space_env.runtime import RuntimeContractError
    from space_env.simulation.snapshot import canonical_json

    try:
        if payload is not None and not isinstance(payload, ArtifactChunk):
            canonical_json(payload)
        result = dispatch(method, path, payload, token)
        if isinstance(result, FileBody):
            return 200, result, result.content_type
        if isinstance(result, bytes):
            return 200, result, "application/zip"
        return _json_bytes(result)
    except OperatorError as exc:
        return _json_bytes(
            {"ok": False, "error": {"code": exc.code, "message": str(exc)}}, exc.status
        )
    except RuntimeContractError as exc:
        status = (
            409
            if exc.code
            in {
                "stale_revision",
                "idempotency_conflict",
                "episode_done",
                "external_mutation",
            }
            else 400
        )
        return _json_bytes(exc.as_dict(), status)
    except ImportError:
        return _json_bytes(
            {
                "ok": False,
                "error": {
                    "code": "missing_dependency",
                    "message": "operator mode requires space-env[reference]",
                },
            },
            503,
        )
    except sqlite3.Error:
        return _json_bytes(
            {
                "ok": False,
                "error": {
                    "code": "storage_unavailable",
                    "message": "durable repository unavailable",
                },
            },
            503,
        )
    except TimeoutError:
        return _json_bytes(
            {
                "ok": False,
                "error": {
                    "code": "artifact_timeout",
                    "message": "artifact operation timed out",
                },
            },
            408,
        )
    except OSError:
        return _json_bytes(
            {
                "ok": False,
                "error": {
                    "code": "artifact_io",
                    "message": "local native artifact IO failure",
                },
            },
            500,
        )
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        return _json_bytes(
            {
                "ok": False,
                "error": {"code": "invalid_request", "message": str(exc)[:1000]},
            },
            400,
        )


class _Handler(BaseHTTPRequestHandler):
    def _send(self, response):
        status, body, kind = response
        from space_env.api.operator_artifacts import FileBody

        try:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if isinstance(body, FileBody):
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{body.filename}"'
                )
                self.send_header("X-Artifact-SHA256", body.sha256)
            if self.path.startswith(("/api/operator/", "/operator")):
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            if isinstance(body, FileBody):
                self.connection.settimeout(15)
                for chunk in body.iter_chunks():
                    self.wfile.write(chunk)
            else:
                self.wfile.write(body)
        finally:
            if isinstance(body, FileBody):
                body.close()

    def _token(self):
        value = self.headers.get("Authorization", "")
        return value[7:] if value.startswith("Bearer ") else None

    def _operator_origin(self):
        if not self.path.startswith("/api/operator/"):
            return True
        host = self.headers.get("Host", "")
        try:
            hostname = urlparse("http://" + host).hostname
        except ValueError:
            return False
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            return False
        origin = self.headers.get("Origin")
        return (
            not origin
            or urlparse(origin).scheme in {"http", "https"}
            and urlparse(origin).netloc == host
        )

    def do_GET(self):
        if not self._operator_origin():
            self._send(
                _json_bytes(
                    {
                        "error": {
                            "code": "origin",
                            "message": "local same-origin API required",
                        }
                    },
                    403,
                )
            )
            return
        parsed = urlparse(self.path)
        self._send(handle_get(parsed.path, parse_qs(parsed.query), token=self._token()))

    def do_POST(self):
        if not self._operator_origin():
            self.close_connection = True
            self._send(
                _json_bytes(
                    {
                        "error": {
                            "code": "origin",
                            "message": "local same-origin API required",
                        }
                    },
                    403,
                )
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("negative length")
        except ValueError:
            self.close_connection = True
            self._send(_json_bytes({"error": "invalid Content-Length"}, 400))
            return
        if length > 8 * 1024 * 1024:
            self.close_connection = True
            self._send(_json_bytes({"error": "request exceeds 8 MiB"}, 413))
            return
        if (
            self.path.startswith("/api/operator/")
            and self.headers.get_content_type() != "application/json"
        ):
            self.close_connection = True
            self._send(_json_bytes({"error": "application/json required"}, 415))
            return
        self.connection.settimeout(15)
        try:
            body = self.rfile.read(length)
        except TimeoutError:
            self.close_connection = True
            self._send(_json_bytes({"error": "request body timeout"}, 408))
            return
        parsed = urlparse(self.path)
        self._send(handle_post(parsed.path, body, token=self._token()))

    def do_PUT(self):
        from space_env.api.operator_artifacts import CHUNK_BYTES, ArtifactChunk

        self.close_connection = True  # A rejected chunk may leave unread request bytes.
        parsed = urlparse(self.path)
        if not self._operator_origin():
            self._send(_json_bytes({"error": "local same-origin API required"}, 403))
            return
        if not parsed.path.startswith("/api/operator/transfers/"):
            self._send(_json_bytes({"error": "unknown upload route"}, 404))
            return
        if self.headers.get_content_type() != "application/octet-stream":
            self._send(_json_bytes({"error": "application/octet-stream required"}, 415))
            return
        try:
            if self.headers.get("Transfer-Encoding") is not None:
                raise ValueError("chunked transfer encoding is unsupported")
            for header in ["Content-Length", "X-Upload-Offset"]:
                values = self.headers.get_all(header, [])
                if (
                    len(values) != 1
                    or not values[0].isascii()
                    or not values[0].isdigit()
                ):
                    raise ValueError(
                        "one nonnegative decimal length and offset required"
                    )
            length = int(self.headers["Content-Length"])
            offset = int(self.headers["X-Upload-Offset"])
            if not 1 <= length <= CHUNK_BYTES:
                self._send(
                    _json_bytes({"error": "chunk exceeds 1 MiB or is empty"}, 413)
                )
                return
        except ValueError as exc:
            self._send(_json_bytes({"error": str(exc)}, 400))
            return
        self.connection.settimeout(15)
        deadline = time.monotonic() + 15
        handler = self

        class DeadlineReader:
            def read(self, size):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("artifact chunk deadline exceeded")
                handler.connection.settimeout(remaining)
                return handler.rfile.read1(size)

        response = _operator_response(
            "PUT",
            parsed.path,
            payload=ArtifactChunk(offset, length, DeadlineReader()),
            token=self._token(),
        )
        self._send(response)

    def log_message(self, fmt, *args):
        return


def serve(host: str = "127.0.0.1", port: int = 8766, *, storage=None) -> None:
    """Слушает студию, пока процесс не остановят."""
    from space_env.api import operator

    path = (
        storage
        if storage is not None
        else os.environ.get(
            "SPACE_ENV_OPERATOR_DB", str(Path.cwd() / ".space-env" / "operator.sqlite3")
        )
    )
    operator.STORE = operator.OperatorStore(path)
    server = ThreadingHTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        from space_env.api.operator_artifacts import ARTIFACTS

        ARTIFACTS.close()


if __name__ == "__main__":
    serve()
