"""
Генерация данных для SFT: пары instruction/output из YAML и из сценариев симуляции.
Формат: JSONL с полями instruction, output, input (optional), meta.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import yaml

import numpy as np

from space_env.core import Orbit, Time
from space_env.core.spacecraft import PowerConfig, PropulsionConfig, Spacecraft, SpacecraftConfig
from space_env.data_generators.pretraining import _get_data_dir, _load_yaml
from space_env.simulation import SimulationEngine


def generate_sft_from_yaml(data_dir: Path | None = None, seed: int | None = None) -> Iterator[dict]:
    """Читает data/sft_pairs.yaml и выдаёт записи {instruction, output, input?, meta}."""
    data_dir = data_dir or _get_data_dir()
    path = data_dir / "sft_pairs.yaml"
    data = _load_yaml(path)
    rng = random.Random(seed)
    for key in ("qa", "instructions"):
        items = data.get(key, [])
        for item in items:
            if "question" in item and "answer" in item:
                yield {
                    "instruction": item["question"],
                    "output": item["answer"],
                    "input": "",
                    "meta": {"source": "sft_pairs", "type": "qa"},
                }
            elif "instruction" in item and "output" in item:
                yield {
                    "instruction": item["instruction"],
                    "output": item["output"].strip(),
                    "input": "",
                    "meta": {"source": "sft_pairs", "type": "instruction"},
                }


def generate_sft_scenario_pairs(seed: int | None = None) -> Iterator[dict]:
    """Генерирует пары сценарий → решение из заготовленных процедур (аномалия, конъюнкция)."""
    rng = random.Random(seed)
    scenarios = [
        {
            "instruction": "По телеметрии: напряжение шины 28V упало до 25V, температура батареи +15°C. Какие действия рекомендованы?",
            "output": "Рекомендуется: 1) Зафиксировать телеметрию и время события. 2) Снизить нагрузку на шину (отключить некритичные потребители). 3) Перевести КА в safe mode при дальнейшем падении напряжения. 4) Дождаться AOS и передать данные на Землю для анализа.",
        },
        {
            "instruction": "Получено предупреждение о сближении: вероятность столкновения 1e-4, промах 2 км. Доступный Δv 15 м/с. Нужно ли выполнять манёвр уклонения?",
            "output": "При вероятности 1e-4 и промахе 2 км решение принимается по пороговой политике оператора. Обычно при Pc > 1e-5 рассматривают манёвр. Рекомендуется запросить уточнённый прогноз и при подтверждении высокого риска выполнить манёвр уклонения в пределах доступного Δv, согласовав направление с ЦУП.",
        },
        {
            "instruction": "КА в тени (eclipse), SoC 35%. До выхода из тени остаётся ~20 минут. Что делать?",
            "output": "Рекомендуется снизить потребление до минимума (отключить полезную нагрузку, оставить только служебные системы) до выхода из тени. Следить за SoC; при падении ниже порога safe mode (например 20%) перевести КА в safe mode. После выхода на свет — восстановить штатный режим и зарядить батарею.",
        },
    ]
    for s in scenarios:
        yield {
            "instruction": s["instruction"],
            "output": s["output"],
            "input": "",
            "meta": {"source": "scenario", "type": "procedure"},
        }


def generate_sft_from_simulation(seed: int | None = None) -> Iterator[dict]:
    """Одна пара по живому снимку симуляции: телеметрия → режим нагрузки."""
    del seed
    epoch = Time("2024-06-15T12:00:00Z")
    orbit = Orbit.from_keplerian(6800e3, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0, epoch)
    spacecraft = Spacecraft(
        config=SpacecraftConfig(
            power=PowerConfig(
                solar_panel_area_m2=2.0,
                battery_capacity_Wh=200.0,
                consumption_nominal_W=12.0,
                consumption_peak_W=30.0,
            ),
            propulsion=PropulsionConfig(delta_v_total_m_s=50.0, mass_kg=50.0),
        )
    )
    spacecraft.reset_state()
    engine = SimulationEngine(orbit, spacecraft, seed=0)
    snap = engine.get_state_snapshot()
    contact = "есть" if snap.in_contact else "нет"
    instruction = (
        f"Телеметрия КА: высота {snap.altitude_km:.0f} км, SoC {snap.battery_soc * 100:.0f}%, "
        f"температура {snap.temp_C:.0f} °C, освещение «{snap.shadow}», контакт со станцией: {contact}. "
        "Какой режим нагрузки выбрать?"
    )
    if snap.in_eclipse or snap.battery_soc < 0.4:
        output = (
            "Снизить нагрузку до минимума, оставить служебные системы и при необходимости включить нагреватели. "
            "Съёмку не проводить в тени. Связь возможна только при контакте со станцией."
        )
    else:
        output = (
            "Держать номинальную нагрузку. Съёмка допустима, пока КА на свету и в надирной ориентации. "
            "Сеанс связи начинать только при угле места выше порога станции."
        )
    yield {
        "instruction": instruction,
        "output": output,
        "input": "",
        "meta": {"source": "simulation", "type": "procedure", "shadow": snap.shadow},
    }


def generate_sft_records(
    output_path: str | Path,
    include_yaml: bool = True,
    include_scenarios: bool = True,
    include_simulation: bool = True,
    seed: int | None = 42,
) -> int:
    """
    Записывает SFT-записи в JSONL. Возвращает число записей.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        if include_yaml:
            for rec in generate_sft_from_yaml(seed=seed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        if include_scenarios:
            for rec in generate_sft_scenario_pairs(seed=seed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        if include_simulation:
            for rec in generate_sft_from_simulation(seed=seed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count
