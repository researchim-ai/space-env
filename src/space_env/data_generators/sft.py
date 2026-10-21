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
    seen: set[str] = set()
    for _ in range(8):
        snap = engine.get_state_snapshot()
        kind = f"{snap.shadow}:{int(snap.in_contact)}:{int(snap.battery_soc < 0.5)}"
        if kind not in seen:
            seen.add(kind)
            yield _sft_from_snapshot(snap)
        engine.step(120.0)


def _sft_from_snapshot(snap) -> dict:
    contact = "есть" if snap.in_contact else "нет"
    aos = "неизвестно" if snap.seconds_to_aos is None else f"{snap.seconds_to_aos / 60.0:.0f} мин"
    instruction = (
        f"Телеметрия КА: высота {snap.altitude_km:.0f} км, SoC {snap.battery_soc * 100:.0f}%, "
        f"температура {snap.temp_C:.0f} °C, освещение «{snap.shadow}», контакт: {contact}, "
        f"запас линии {snap.link_margin_db:.1f} дБ, до AOS {aos}. Какой режим выбрать?"
    )
    if snap.in_eclipse or snap.battery_soc < 0.4:
        output = (
            "Снизить нагрузку до минимума и не снимать в тени. "
            "Сеанс открывать только при положительном запасе линии."
        )
    elif snap.in_contact and snap.link_margin_db >= 0.0:
        output = (
            "Линия закрыта: открыть сеанс и сбросить накопленное. "
            "Съёмку оставить, если аппарат на свету и в надире."
        )
    else:
        output = (
            "Держать номинальную нагрузку. Съёмка допустима на свету в надире. "
            "Связь ждать до положительного запаса над станцией."
        )
    return {
        "instruction": instruction,
        "output": output,
        "input": "",
        "meta": {"source": "simulation", "type": "procedure", "shadow": snap.shadow},
    }


def generate_sft_from_constellation(seed: int | None = None) -> Iterator[dict]:
    """Один эпизод группировки: где пакет и куда его нести, пока он не окажется над Фэрбанксом."""
    del seed
    import gymnasium as gym

    import space_env.envs  # регистрация среды
    from space_env.rl.expert import expert_action

    env = gym.make("space_env/ConstellationOps-v0", max_episode_steps=8)
    _obs, info = env.reset(seed=0)
    gateway = info["gateway"]
    source = info["source"]
    yield {
        "instruction": (
            f"В цепочке из четырёх аппаратов пакет на {source}. "
            f"Наибольший угол места над Фэрбанксом у {gateway}. Кому отдать пакет на этом шаге?"
        ),
        "output": (
            f"Переслать с {source} соседнему аппарату в сторону {gateway}. "
            "За шаг открывается один луч. Награда есть, когда пакет оказывается на аппарате над Фэрбанксом."
        ),
        "input": "",
        "meta": {"source": "constellation", "type": "relay", "gateway": gateway, "holder": source},
    }
    arrived = source == gateway
    for _ in range(8):
        if arrived:
            break
        action = expert_action("space_env/ConstellationOps-v0", env.unwrapped._obs())
        _obs, _reward, _terminated, _truncated, step = env.step(action)
        arrived = step["success"] or step["buffer_mb"][gateway] >= 8.0 - 1e-6
    env.close()
    yield {
        "instruction": "Пакет дошёл до аппарата над Фэрбанксом. Что это значит для фидера Iridium?",
        "output": (
            "Аппарат выбран по углу места, не по закрытой радиолинии. "
            "ЭИИМ фидера в заявке FCC не опубликован, поэтому сдача на шлюз здесь не считается."
        ),
        "input": "",
        "meta": {"source": "constellation", "type": "gateway", "arrived": arrived},
    }


def generate_sft_from_operations(seed: int | None = None) -> Iterator[dict]:
    """Снимки каталога: сдача Sentinel, сверка B* и число скачков по звезде Iridium."""
    del seed
    from space_env.catalog.build import make_engine
    from space_env.catalog.drag import drag_comparison
    from space_env.catalog.mesh import iridium_shell, route

    slots, radius_m, inclination = iridium_shell()
    start = slots[0]
    goal = next(item for item in slots if item.plane == 0 and item.slot == 5)
    path = route(start, goal, a_m=radius_m, inclination_deg=inclination, total=66, planes=6, phasing=2)
    hops = len(path) if path is not None else -1
    yield {
        "instruction": "Сколько проектных скачков Iridium NEXT отделяют слот P1-S1 от противоположного слота той же плоскости?",
        "output": (
            f"Кратчайший путь по соседям, чей луч не пересекает лимб, занимает {hops} скачков. "
            "ЭИИМ Ka-линии в карточке нет, поэтому в путь входит только геометрия."
        ),
        "input": "",
        "meta": {"source": "catalog", "type": "iridium-route", "hops": hops},
    }
    drag = drag_comparison("sentinel-2a")
    yield {
        "instruction": "Чем геометрический Cd·A/m Sentinel-2A отличается от B* в снимке TLE?",
        "output": (
            f"Грань 3.4×1.8 м при Cd 2.2 даёт {drag.geometric_cd_area_per_kg:.4f} м²/кг. "
            f"B* снимка переводится в {drag.tle_cd_area_per_kg:.6f} м²/кг. "
            "B* — подгонка SGP4 и мидель не заменяет."
        ),
        "input": "",
        "meta": {"source": "catalog", "type": "drag"},
    }
    engine = make_engine("sentinel-2a", nu_deg=180.0, with_stations=True)
    snap = engine.get_state_snapshot()
    aos = "нет окна на 6 ч" if snap.seconds_to_aos is None else f"{snap.seconds_to_aos / 60.0:.0f} мин"
    contact = snap.contact_station or "нет"
    yield {
        "instruction": (
            f"Sentinel-2A: память пустая из 2.4 Тбит, запас X-band {snap.link_margin_db:.1f} дБ, "
            f"станция «{contact}», до AOS {aos}. Когда сдавать данные?"
        ),
        "output": (
            "Сдавать только внутри окна, когда линия закрыта и запас неотрицательный. "
            "Запись 450 Мбит/с в память сама по себе сеансом не является."
        ),
        "input": "",
        "meta": {"source": "catalog", "type": "sentinel-pass", "aos_sec": snap.seconds_to_aos},
    }


def generate_sft_from_workplan(seed: int | None = None) -> Iterator[dict]:
    """Один план Sentinel-2A: список дел, включая окно, которое занимать не нужно."""
    del seed
    from space_env.mission.workplan import build_acquisition_plan, maneuver_item

    burn = maneuver_item(
        300.0,
        0.5,
        "retrograde",
        "300 с: импульс 0.5 м/с ретроградно, если фаза пары уже ушла от антипода.",
    )
    plan = build_acquisition_plan("sentinel-2a", horizon_sec=5400.0, extra_items=[burn])
    yield {
        "instruction": (
            "Составь план работы Sentinel-2A на ближайшие 5400 с: какие цели снять, "
            "куда сдавать память и какой сеанс не занимать."
        ),
        "output": "\n".join(plan.lines()),
        "input": "",
        "meta": {"source": "workplan", "type": "sentinel-2a", "tasks": len(plan.required_items())},
    }


def generate_sft_from_fleet_plan(seed: int | None = None) -> Iterator[dict]:
    """План цепочки на горизонт: только те скачки, которые линия реально пропускает."""
    del seed
    from space_env.mission.fleet_plan import build_relay_schedule

    schedule = build_relay_schedule()
    yield {
        "instruction": (
            f"Составь план работы цепочки из {schedule.count} аппаратов на горизонт шагов по "
            f"{schedule.step_sec:.0f} с. Пакет 8 Мбит стоит на {schedule.source}, "
            f"шлюз по углу места над Фэрбанксом — {schedule.gateway}."
        ),
        "output": "\n".join(schedule.lines()),
        "input": "",
        "meta": {
            "source": "fleet-plan",
            "type": "relay-schedule",
            "arrived": schedule.arrived,
            "tasks": len(schedule.tasks),
        },
    }


def generate_sft_records(
    output_path: str | Path,
    include_yaml: bool = True,
    include_scenarios: bool = True,
    include_simulation: bool = True,
    include_operations: bool = True,
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
        if include_operations:
            for rec in generate_sft_from_operations(seed=seed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            for rec in generate_sft_from_constellation(seed=seed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            for rec in generate_sft_from_workplan(seed=seed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            for rec in generate_sft_from_fleet_plan(seed=seed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count
