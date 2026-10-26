"""Короткая и суточная миссии. Семя меняет координаты и сроки, не только имена."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

SHORT_VEHICLES = 2
SHORT_REQUESTS = 12
DAY_REQUESTS = 32
STATIONS = ("Kiruna", "Svalbard", "Matera")


def build_mission(kind: str, seed: int) -> dict:
    if kind not in {"short", "day"}:
        raise ValueError("миссия бывает short или day")
    rng = np.random.default_rng(seed)
    if kind == "short":
        horizon = 3600.0
        count = SHORT_REQUESTS
        instances = [
            {"id": "optic", "model": "sentinel-2a", "nu_deg": 80.0},
            {"id": "radar", "model": "sar-small", "nu_deg": 45.0},
        ]
    else:
        horizon = 86400.0
        count = DAY_REQUESTS
        instances = [
            {"id": "sentinel-a", "model": "sentinel-2a", "nu_deg": 80.0},
            {"id": "sentinel-b", "model": "sentinel-2a", "nu_deg": 180.0},
            {"id": "landsat", "model": "landsat-9", "nu_deg": 90.0},
            {"id": "radar", "model": "sar-small", "nu_deg": 45.0},
        ]
    requests = []
    for index in range(count):
        release = float(rng.integers(0, max(1, int(horizon * 0.2))))
        deadline = release + float(rng.integers(max(2, int(horizon * 0.2)), max(3, int(horizon * 0.5))))
        requests.append(
            {
                "id": f"req-{index:02d}",
                "lat_deg": round(float(rng.uniform(-60.0, 70.0)), 4),
                "lon_deg": round(float(rng.uniform(-179.0, 179.0)), 4),
                "release_sec": release,
                "deadline_sec": deadline,
                "size_bits": 8.0e6,
                "priority": int(rng.integers(1, 5)),
                "payload": "sar" if index % 5 == 0 else "optical",
                "destination": STATIONS[index % len(STATIONS)],
                "quality": "full",
            }
        )
    return {
        "version": 2,
        "epoch": "2026-10-20T10:00:00Z",
        "horizon_sec": horizon,
        "instances": instances,
        "stations": list(STATIONS),
        "requests": requests,
    }


def write_mission(path: str | Path, kind: str, seed: int) -> Path:
    target = Path(path)
    target.write_text(yaml.safe_dump(build_mission(kind, seed), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target
