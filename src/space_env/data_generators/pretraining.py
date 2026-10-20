"""
Генерация данных для претренинга: шаблоны + simulation-driven тексты, вывод в JSONL.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from space_env.core import Orbit, Time
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig, PowerConfig, PropulsionConfig
from space_env.simulation import SimulationEngine


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "data"


def generate_from_templates(
    num_samples: int = 50,
    seed: int | None = None,
    data_dir: Path | None = None,
) -> Iterator[dict]:
    """Генерирует текстовые чанки из шаблонов с подстановкой параметров."""
    data_dir = data_dir or _get_data_dir()
    templates_path = data_dir / "pretraining_templates.yaml"
    data = _load_yaml(templates_path)
    domains = data.get("domains", {})
    rng = random.Random(seed)
    templates_flat: list[tuple[str, str, dict]] = []
    for domain, lines in domains.items():
        for t in lines:
            templates_flat.append((domain, t, {}))
    if not templates_flat:
        return
    for _ in range(num_samples):
        domain, template, _ = rng.choice(templates_flat)
        # Параметры по умолчанию для подстановки
        params: dict[str, Any] = {
            "name": rng.choice(["Sat-1", "EO-Sat", "Missionsat"]),
            "a_km": rng.uniform(6600, 7200),
            "inc_deg": rng.uniform(45, 98),
            "perigee_km": rng.uniform(400, 600),
            "apogee_km": rng.uniform(420, 620),
            "period_min": rng.uniform(92, 98),
            "solar_m2": rng.uniform(1.0, 5.0),
            "battery_Wh": rng.uniform(100, 500),
            "consumption_W": rng.uniform(10, 50),
            "delta_v": rng.uniform(20, 100),
            "soc_percent": rng.uniform(30, 100),
            "time_utc": "2024-06-15T12:00:00Z",
            "alt_km": rng.uniform(400, 800),
            "imaging_count": rng.randint(5, 20),
            "comms_sessions": rng.randint(2, 10),
            "delta_v_budget": rng.uniform(30, 80),
        }
        try:
            text = template.format(**params)
        except KeyError:
            continue
        yield {"text": text, "meta": {"source": "template", "domain": domain}}


def generate_from_simulation(
    num_steps: int = 100,
    step_sec: float = 60.0,
    seed: int | None = None,
) -> Iterator[dict]:
    """Генерирует предложения из лога симуляции (eclipse, SoC, время)."""
    epoch = Time("2024-01-01T00:00:00Z")
    orbit = Orbit.from_keplerian(6800e3, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    sc = Spacecraft(config=SpacecraftConfig(
        power=PowerConfig(solar_panel_area_m2=2.0, battery_capacity_Wh=200.0, consumption_nominal_W=12.0, consumption_peak_W=30.0),
        propulsion=PropulsionConfig(delta_v_total_m_s=50.0, mass_kg=50.0),
    ))
    sc.reset_state()
    engine = SimulationEngine(orbit, sc, seed=seed)
    prev_eclipse = False
    for _ in range(num_steps):
        snap, _ = engine.step(step_sec)
        t_utc = (epoch + snap.time_sec).datetime.strftime("%Y-%m-%d %H:%M UTC")
        if snap.in_eclipse and not prev_eclipse:
            yield {"text": f"В {t_utc} КА вошёл в тень Земли (eclipse).", "meta": {"source": "simulation", "domain": "events"}}
        if not snap.in_eclipse and prev_eclipse:
            yield {"text": f"В {t_utc} КА вышел из тени, солнечные батареи в освещении.", "meta": {"source": "simulation", "domain": "events"}}
        prev_eclipse = snap.in_eclipse
        if snap.time_sec > 0 and int(snap.time_sec) % 600 == 0:  # каждые ~10 мин
            yield {
                "text": (
                    f"Уровень заряда батареи (SoC) на момент {t_utc} — {snap.battery_soc * 100:.0f}%. "
                    f"Высота орбиты {snap.altitude_km:.0f} км, температура {snap.temp_C:.1f} °C, освещение «{snap.shadow}»."
                ),
                "meta": {"source": "simulation", "domain": "spacecraft"},
            }
    yield {"text": f"За {num_steps * step_sec / 3600:.1f} ч симуляции выполнено {num_steps} шагов пропагации орбиты и обновления подсистем.", "meta": {"source": "simulation", "domain": "missions"}}


def generate_glossary_entries(data_dir: Path | None = None) -> Iterator[dict]:
    """Генерирует текстовые чанки из глоссария (термин — определение)."""
    data_dir = data_dir or _get_data_dir()
    path = data_dir / "glossary.yaml"
    data = _load_yaml(path)
    terms = data.get("terms", {})
    for term, definition in terms.items():
        text = f"{term}: {definition}"
        yield {"text": text, "meta": {"source": "glossary", "term": term}}


def generate_pretraining_chunks(
    output_path: str | Path,
    num_template_samples: int = 100,
    num_simulation_steps: int = 150,
    include_glossary: bool = True,
    seed: int | None = 42,
) -> int:
    """
    Пишет в JSONL файл все чанки: шаблоны, simulation-driven, глоссарий.
    Возвращает число записанных строк.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in generate_from_templates(num_template_samples, seed=seed):
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
        for chunk in generate_from_simulation(num_steps=num_simulation_steps, seed=seed):
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
        if include_glossary:
            for chunk in generate_glossary_entries():
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1
    return count
