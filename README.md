# space-env

Среда для генерации данных и обучения RL/LLM агентов в космической отрасли.

## Назначение

- **Pretraining** — синтетические тексты и структурированные данные: орбиты, миссии спутников, подсистемы КА, протоколы.
- **SFT** — диалоги и инструкции: планирование миссий, диагностика аномалий, принятие решений.
- **RL** — интерактивные среды (Gymnasium): выполнение миссий, управление ресурсами, реакция на аномалии и сближения.

Домены: LEO/GEO/MEO миссии, орбитальная механика, наземный сегмент, аномалии и процедуры, моделирование.

## Структура

```
space-env/
├── docs/                         # План реализации и обзор библиотек
├── src/space_env/                # Ядро, миссии, симуляция, среды, RL, оценка
├── data/                         # Глоссарий, шаблоны, пары Q&A
├── configs/                      # Награды, куррикулум, YAML-сценарии
├── scripts/                      # Генерация данных, бенчмарк, REINFORCE
└── tests/
```

Каталог `datasets/` скрипты создают сами; он в `.gitignore`.

## Что умеет версия 0.3

| Слой | Содержание |
|------|------------|
| Ядро | Кауэлл с J2 и экспоненциальной атмосферой, тень, Солнце/альбедо/ИК Земли, угол места, импульс Δv |
| Миссии | Съёмка по наземным целям, один сеанс на зону станции, сближение Клохесси–Уилтшира |
| Симуляция | Шаг с командой, события eclipse/AOS/LOS, Pc по приближению Чена, отказы только по симптомам |
| RL | `MissionPlanner`, `ResourceManager`, `AnomalyResponse`, `ConjunctionAssessment`; регистрация в Gymnasium |
| Данные | Претренинг, SFT (YAML, процедуры и снимок симуляции), траектории по куррикулуму |
| Оценка | Случайный бенчмарк и рубрика терминов. Обучение — линейный REINFORCE |

Подробно — в [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Генерация данных и прогон

```bash
pip install -e ".[dev]"
pytest -q

# Претренинг → datasets/pretraining.jsonl
python scripts/generate_pretraining_data.py -o datasets/pretraining.jsonl -n 100 -s 150

# SFT → datasets/sft.jsonl
python scripts/generate_sft_data.py -o datasets/sft.jsonl

# Траектории RL → datasets/rl_trajectories.jsonl
python scripts/generate_rl_data.py -o datasets/rl_trajectories.jsonl

# Случайный бенчмарк четырёх сред
python scripts/evaluate.py --episodes 2 --steps 12

# Короткое обучение
python scripts/train_reinforce.py --env space_env/AnomalyResponse-v0 --episodes 40
```

Среды: `space_env/MissionPlanner-v0`, `space_env/ResourceManager-v0`, `space_env/AnomalyResponse-v0`, `space_env/ConjunctionAssessment-v0`.

Исходники текстов лежат в `data/` (глоссарий, шаблоны, пары Q&A). Веса наград — в `configs/rewards.yaml`.

## Тесты

```bash
pytest -q
```

## Зависимости

- Базовые: `boinor` (орбитальная механика), `gymnasium`, `pydantic`, `numpy`, `pandas`, `pyyaml`
- Опционально: `sgp4` (TLE), `spiceypy` (ядра SPICE), `ccsdspy`/`spacepackets` (разбор CCSDS), `orekit-jpype` (нужен JDK 11+ или extra `orekit-jdk`)
- Где какая библиотека вызывается: **[docs/INDUSTRY_LIBRARIES.md](docs/INDUSTRY_LIBRARIES.md)**

## Лицензия

См. [LICENSE](LICENSE).
