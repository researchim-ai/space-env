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
├── docs/IMPLEMENTATION_PLAN.md   # Полный план реализации (все фазы)
├── src/                          # Исходный код
├── data/                         # Исходники для генерации: глоссарий, шаблоны, пары Q&A (YAML)
├── datasets/                     # Сгенерированные датасеты (JSONL) — претренинг, SFT
├── configs/                      # Конфиги сценариев и генерации
└── scripts/                      # Скрипты генерации и обучения
```

## План реализации (кратко)

| Фаза | Содержание |
|------|------------|
| 0 | Инфраструктура: Python, зависимости (boinor, gymnasium, pydantic), структура пакета |
| 1 | **Core:** время, орбиты (Keplerian/SGP4), модель КА (power, thermal, propulsion, comms), среда (eclipse, debris) |
| 2 | **Миссии:** типы (LEO/GEO/MEO), цели, сценарии (номинал, аномалии, конъюнкции) |
| 3 | **Симуляция:** движок шага, логирование, воспроизводимость |
| 4 | **RL-среды:** MissionPlanner, AnomalyResponse, ConjunctionAssessment, ResourceManager (Gymnasium) |
| 5 | **Генерация PT:** текст + структура из шаблонов и симуляции |
| 6 | **Генерация SFT:** инструкции и эталонные ответы (в т.ч. из симуляции) |
| 7 | **RL-данные:** траектории, куррикулум, описание наград |
| 8 | **Оценка:** бенчмарки, метрики, эвалюация LLM |

Подробно — в [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Генерация данных

- **Претренинг:** шаблоны (орбиты, КА, процедуры, миссии), тексты из симуляции (eclipse, SoC), глоссарий. Выход — JSONL (`text`, `meta`).
- **SFT:** пары вопрос–ответ и инструкция–выполнение из `data/sft_pairs.yaml` и сценариев (аномалии, конъюнкции). Выход — JSONL в **datasets/** (`instruction`, `output`, `input`, `meta`).

**Скрипты (датасеты пишутся в `datasets/`):**
```bash
# Претренинг → datasets/pretraining.jsonl
python scripts/generate_pretraining_data.py -o datasets/pretraining.jsonl -n 100 -s 150

# SFT → datasets/sft.jsonl
python scripts/generate_sft_data.py -o datasets/sft.jsonl
```

**Исходники в `data/`** (YAML): глоссарий, шаблоны текстов, пары Q&A — их меняют, чтобы менять содержимое датасетов. **Сгенерированные датасеты** — в **`datasets/`**.

## Зависимости

- Базовые: `boinor` (орбитальная механика), `gymnasium`, `pydantic`, `numpy`, `pandas`
- Опционально (отраслевые): `sgp4` (TLE/NORAD), `spiceypy` (NASA SPICE), `ccsdspy`/`spacepackets` (CCSDS), `orekit-jpype` (OreKit; нужен JDK 11+ или `.[orekit-jdk]`)
- Подробный обзор библиотек, используемых NASA, ESA и в миссиях: **[docs/INDUSTRY_LIBRARIES.md](docs/INDUSTRY_LIBRARIES.md)**

## Лицензия

См. [LICENSE](LICENSE).
