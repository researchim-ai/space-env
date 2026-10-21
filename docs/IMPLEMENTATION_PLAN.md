# План и текущая реализация

Версия пакета 0.6.0. Ниже — что считает код, а не список намерений.

## Слои

| Слой | Модули | Поведение |
|------|--------|-----------|
| Время и орбита | `core/time.py`, `core/orbit.py`, `core/propagate.py`, `core/gravity.py` | Кауэлл: поле + J2 + атмосфера + солнечное давление (пушечное ядро). Импульс `with_delta_v` |
| Солнце и тень | `core/sun.py`, `core/environment.py` | Низкоточные экваториальные координаты Солнца, конус umbra/penumbra, поток с 1/r² |
| Кадры и станции | `core/frames.py`, `core/ground.py`, `core/passes.py`, `core/link.py` | GAST, линк-бюджет, окна AOS/LOS. Сеть: Москва, Шпицберген, Инувик |
| КА | `core/spacecraft.py`, `core/power.py`, `core/thermal.py` | SoC, один тепловой узел (Солнце, альбедо, ИК Земли, излучение), ориентация `nadir`/`sun` |
| Миссия | `mission/` | Цели, ограничения, план, инжекты аномалий и сближений, загрузка `configs/scenarios/*.yaml` |
| Шаг | `simulation/engine.py` | Команда → пропагация → инжекты и план → питание и термика → события |
| RL | `envs/`, `rl/` | Четыре среды Gymnasium, веса наград, куррикулум, JSONL-траектории, линейный REINFORCE |
| Данные | `data_generators/`, `scripts/` | Претренинг, SFT (в том числе снимок симуляции), траектории |
| Оценка | `eval/` | Случайный бенчмарк и покрытие терминов в ответе SFT |
| Пакеты CCSDS | `core/packets.py` | Первичный заголовок Space Packet. Разбор файлов — extra `ccsds` |

## Команды движка

`wait`, `imaging`, `comms`, `maneuver`, `nominal_power`, `reduced_power`, `minimal_power`, `safe_mode`, `point_sun`, `point_nadir`, `reboot`.

Съёмка засчитывается, если за шаг надирный прибор видит ещё не снятую цель, КА не в umbra и разворот на надир закончен. Связь открывается одной командой до LOS, если запас линка неотрицательный; скорость ступенчатая. Манёвр списывает Δv и меняет только свой вектор скорости.

Сближение: второе тело интегрируется Кауэллом. Ковариация едет в RTN. Pc — интеграл по диску твёрдого тела. KVN-сообщение CDM разбирает `core/cdm.py` (промах, скорость, ковариации, AREA_PC). Сообщённая вероятность не подменяет расчёт.

Отказы для обучения (`overload`, `stuck_heater`, `radio_latch`) не попадают в наблюдение. Сброс нагрузки снимает перегруз, safe mode ещё и залипший нагреватель, перезагрузка радио занимает 3 шага и не чинит шину.

## Среды

| Идентификатор | Смысл эпизода |
|---------------|----------------|
| `space_env/MissionPlanner-v0` | Набрать съёмки и сеансы, не выйти за бюджет Δv |
| `space_env/ResourceManager-v0` | Дожить до конца эпизода, не разрядив батарею и не выйдя из диапазона температур |
| `space_env/AnomalyResponse-v0` | Снять отказ по симптомам. Перезагрузка не лечит шину |
| `space_env/ConjunctionAssessment-v0` | Жечь Δv, только если интеграл Pc выше 10⁻⁴ |
| `space_env/Constellation-v0` | Цепочка аппаратов: одно действие выбирает борт и команду, пары скринятся по дистанции |

`info["success"]` на последнем шаге — критерий бенчмарка.

## Что сознательно не вшито в репозиторий

- Ядра SPICE и каталог `orekit-data`. Обёртки есть, без этих файлов вызов завершится ошибкой библиотеки.
- Полный ряд нутации IAU 2000A и бюллетень DUT1. В кадре учтён только главный член уравнения равноденствий, UT1 = UTC.
- NRLMSISE-00 вызывается при `atmosphere: nrlmsise` и установленном пакете `nrlmsise00`. По умолчанию — таблица среднего Солнца и множитель F10.7.
- Ковариация сближения растёт шумом процесса на дуге минут. Это не оценка орбиты по измерениям и не живой CDM.
- Обучение нейросетью: эвристика и линейный REINFORCE. Зарегистрированные среды можно отдать в Stable-Baselines3 отдельно.

## Проверки

```bash
pip install -e ".[dev]"
pytest -q
```

Скрипты пишут в каталог `datasets/` (он в `.gitignore`):

```bash
python scripts/generate_pretraining_data.py -o datasets/pretraining.jsonl -n 100 -s 150
python scripts/generate_sft_data.py -o datasets/sft.jsonl
python scripts/generate_rl_data.py -o datasets/rl_trajectories.jsonl
python scripts/evaluate.py --episodes 2 --steps 12
python scripts/train_reinforce.py --env space_env/AnomalyResponse-v0 --episodes 40
```
