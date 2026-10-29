"""Один отчёт миссии для CLI, среды и студии и живая сессия с ревизией."""
from __future__ import annotations

import uuid
import math
import threading
from pathlib import Path

from space_env.mission.scenario_v2 import load_scenario_v2
from space_env.mission.tasking import request_lifecycle, requests_from_scenario, run_scenario
from space_env.simulation.commands import MissionCommand


class RevisionError(RuntimeError):
    """Номер ревизии в запросе не совпал с сессией."""


class MissionSession:
    """Один мир, одна книга команд и растущий номер ревизии."""

    def __init__(self, scenario: str | Path) -> None:
        from space_env.envs.mission_fleet_env import packaged_mission
        from space_env.mission.scenario_v2 import world_from_scenario

        path = Path(scenario)
        if not path.is_file():
            path = packaged_mission(str(scenario))
        spec = load_scenario_v2(path)
        self.scenario = str(path)
        self.session_id = uuid.uuid4().hex[:12]
        self.revision = 0
        self.requests = requests_from_scenario(spec)
        self.world = world_from_scenario(path, attach_stations=True, refresh_passes=False)
        self.world.task_requests = {item.request_id: item for item in self.requests}
        self.world.set_outages(
            [(item["station"], item["start_sec"], item["end_sec"]) for item in spec.get("outages", [])]
        )
        self._lock = threading.Lock()
        self._track: list[dict] = []
        self._frames: list[dict] = []
        self._remember()

    def _remember(self) -> None:
        from space_env.core.access import subsatellite_latlon

        now = float(self.world.now_sec())
        for name, engine in self.world.engines.items():
            lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
            self._track.append(
                {
                    "t": now,
                    "name": name,
                    "lat_deg": round(float(lat), 4),
                    "lon_deg": round(float(lon), 4),
                }
            )
        from space_env.viz.export import _capture

        self._frames.append(_capture(self.world.engines))

    def _vehicles(self) -> list[dict]:
        from space_env.core.access import subsatellite_latlon

        rows = []
        for name, engine in self.world.engines.items():
            snap = engine.get_state_snapshot()
            lat, lon = subsatellite_latlon(engine.orbit.r, engine.now())
            rows.append(
                {
                    "name": name,
                    "lat_deg": round(float(lat), 4),
                    "lon_deg": round(float(lon), 4),
                    "altitude_km": round(float(snap.altitude_km), 1),
                    "soc": round(float(snap.battery_soc), 4),
                    "in_contact": bool(snap.in_contact),
                    "station": snap.contact_station or "",
                    "in_eclipse": bool(snap.in_eclipse),
                }
            )
        return rows

    def report(self) -> dict:
        now = self.world.now_sec()
        receipts = [item for engine in self.world.engines.values() for item in engine.receipts]
        return {
            "session_id": self.session_id,
            "revision": self.revision,
            "now_sec": now,
            "horizon_sec": self.world.horizon_sec,
            "statuses": [
                {"request_id": item.request_id, "status": request_lifecycle(item, receipts, now)}
                for item in self.requests
            ],
            "vehicles": self._vehicles(),
            "track": list(self._track),
            "episode": self.episode(),
        }

    def episode(self) -> dict:
        """Кадры уже прожитого мира. Повторный отчёт кадры не добавляет."""
        from space_env.core.constants import R_EARTH_EQUATOR

        now = float(self.world.now_sec())
        vehicles = self._vehicles()
        receipts = [item for engine in self.world.engines.values() for item in engine.receipts]
        statuses = [
            {"request_id": item.request_id, "status": request_lifecycle(item, receipts, now)}
            for item in self.requests
        ]
        actor = vehicles[0]["name"] if vehicles else ""
        return {
            "version": 1,
            "epoch": self.world.engines[actor].orbit.epoch.datetime.isoformat() if actor else "",
            "earth_radius_m": R_EARTH_EQUATOR,
            "dt_sec": 30.0,
            "layout": "session",
            "summary": {
                "layout": "session",
                "count": len(vehicles),
                "minutes": now / 60.0,
                "brief": "Живая сессия. Глобус и журнал показывают этот мир.",
                "events": [
                    {
                        "vehicle": actor,
                        "kind": "imaging",
                        "name": row["request_id"],
                        "start_sec": 0.0,
                        "end_sec": now,
                        "required": True,
                        "note": f"{row['request_id']}: {row['status']}",
                        "status": row["status"],
                    }
                    for row in statuses
                ],
            },
            "frames": list(self._frames),
        }

    def advance(self, dt_sec: float, commands: list[MissionCommand] | None = None, *, expected_revision: int) -> dict:
        with self._lock:
            if int(expected_revision) != self.revision:
                raise RevisionError("устаревшая ревизия")
            if not math.isfinite(float(dt_sec)) or float(dt_sec) <= 0.0:
                raise ValueError("шаг должен быть положительным и конечным")
            remaining = self.world.horizon_sec - self.world.now_sec()
            if remaining <= 1e-9:
                raise ValueError("горизонт уже пройден")
            dt = min(float(dt_sec), remaining)
            book = self.world.commands.export()
            batch = list(self.world._batch)
            try:
                orders = self.world.accept(list(commands or []))
                self.world.step(dt, orders)
                self.world.settle()
            except Exception:
                self.world.commands.load(book)
                self.world._batch = batch
                raise
            self.revision += 1
            self._remember()
            return self.report()


def mission_report(
    scenario: str | Path,
    *,
    policy: str = "priority",
    step_sec: float = 30.0,
    refresh_passes: bool = False,
) -> dict:
    spec = load_scenario_v2(scenario)
    report = run_scenario(
        scenario,
        step_sec=step_sec,
        policy=policy,
        refresh_passes=refresh_passes,
    )
    requests = requests_from_scenario(spec)
    statuses = [
        {
            "request_id": item.request_id,
            "status": request_lifecycle(item, report["receipts"], spec["horizon_sec"]),
            "destination": item.destination,
            "quality": item.quality,
        }
        for item in requests
    ]
    return {
        "scenario": Path(scenario).name,
        "on_time": list(report["score"]["on_time"]),
        "unmet": report["unmet"],
        "statuses": statuses,
        "weighted_on_time": report["score"]["weighted_on_time"],
        "safety_violations": report["kpis"]["safety_violations"],
        "rejected": report["rejected"],
    }
