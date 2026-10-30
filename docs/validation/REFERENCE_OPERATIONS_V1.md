# Исполнение downlink на reference-эфемеридах

Дата31 октября2026. Рабочее дерево0.134.0. Новый профиль:
`reference-fixed-downlink/1`. Это отдельный исполнитель ограниченного класса задач,
не полная миграция MissionWorld и не эксплуатационный цифровой двойник.

## Сквозной пример

```bash
python examples/reference_operations/demo.py /tmp/reference-downlink
python -m space_env.cli reference-replay /tmp/reference-downlink/run.json
```

Два аппарата на близких полярных GCRS орбитах передают существующие продукты
через один терминал. Отключение станции — [2,4). Продукт2Мбит завершён на2с,
продукт3Мбит — на7с. В итогах5Мбит своевременной доставки, проверенный replay
и одна RL JSONL-строка. Начальные продукты явно загружены в память; пример не
имитирует получение нового изображения и не засчитывает контакт как доставку.

## SDK и среда для обучения

```python
from space_env.reference.operations import ReferenceDownlinkMission
from space_env.reference.operations_env import ReferenceDownlinkEnv
from space_env.data_generators.reference_operations import ReferenceOperationsRecorder, replay_reference_operations
from space_env.rl.trajectories import rollout_episode

# spec — self-contained JSON configuration, как в demo.py.
mission = ReferenceDownlinkMission(spec)
report = mission.step(10, {'a': 'Ground', 'b': 'Ground'})
env = ReferenceDownlinkEnv(spec, dt_sec=10)
observation, info = env.reset(seed=7)
rows = rollout_episode(env, seed=7)
recorder = ReferenceOperationsRecorder(spec, lineage='mission-family-1')
recorder.step(10, {'a': 'Ground', 'b': 'Ground'})
recorder.save('run.json')
assert replay_reference_operations('run.json')['ok']
```

Это публичный Gymnasium class, пока без автоматической регистрации среди legacy
сред. Совместимость проверена `gymnasium.utils.env_checker.check_env` и существующим
rollout writer. PettingZoo, RLlib и обученный policy baseline этим не квалифицированы.

Joint action — MultiDiscrete: на каждом акторе0 означает ожидание,1..N — выбранную
станцию. Выбор сохраняется на длительность внешнего шага; при входе в контакт
передача может начаться внутри этого шага. Актор без записи в SDK actions ожидает.
Шаги длительностью≤1нс отвергаются вместо молчаливого отсутствия прогресса.
Неверное действие не двигает мир; после done требуется reset. Последний шаг среды
обрезается горизонтом. Seed задаёт RNG среды/action space; физика пока детерминирована.

Observation Dict: clock, SoC, stored_bits, текущая geometric contact matrix и таблица
products. Поля products: remaining_bits, ready_sec, deadline_sec, priority,
actor_index, destination_index; destination0 означает любую станцию. Порядок акторов,
станций, product IDs и названия полей возвращаются reset info. Hardware constants и
полные исходные эфемериды доступны в task spec, не дублируются в каждом tensor.
Это joint-control профиль с объявленной информацией, не distributed partial-observation MARL.

Reward — Mбит **новых полностью завершённых в срок продуктов**. Частичная доставка
и delivered_late дают0. Terminated допускает завершение всех продуктов поздно;
success отдельно требует своевременного завершения каждого. Incomplete на горизонте
даёт truncation. Пустая задача не является содержательным benchmark для качества policy.

## Конфигурация и физический контракт

Schema `space-env.reference-downlink-spec/1` требует:

- ephemerides: actor → sealed reference orbit с GCRS r/v, общей эпохой, EOP и hashes;
- vehicles: actor → memory_bits, battery_wh, initial_soc/min_tx_soc, base/tx power,
  fixed rate_bps; необязательные solar_area_m2/solar_efficiency;
- stations: явные GroundStation fields; shared bandwidth и RF band filtering
  отвергаются как неподдерживаемые, а не молча игнорируются;
- products: уникальные id, actor, size_bits≥1, ready_sec, deadline_sec, priority,
  необязательная destination. Весь inventory существует и занимает память с t=0.

Опциональны horizon_sec, resource_step_sec и station outages. Требуется
ready < deadline <= horizon; horizon покрывается всеми эфемеридами. Ограничения
конфигурации64 актора/128 станций/10000 продуктов — input limits, не доказанная
производительность на максимальных workload. Сетки/покрытие эпhemeris наследуют
[reference access contract](REFERENCE_ACCESS_V1.md).

Конец операции/готовность/окна станции/outage/порог батареи/завершение продукта
разбивают интервал исполнения. Терминал, освободившийся после продукта, может
обслужить следующего актора внутри того же внешнего шага. При конкуренции выбираются
feasible actors по алфавиту; внутри актора продукт выбирается по priority/deadline/id.
Это объявленное arbitration rule, не оптимальный resource-constrained planner.

Скорость канала задана в бит/с и не заменяет link budget. Передача создаёт квитанции
с actor/station/product/start/end/bits, уменьшает память и расходует энергию.
Deadline влияет на оценку, не уничтожает просроченные данные автоматически.
Сохраняются непереданные остатки и completed_sec; on-time score не выдаётся авансом.

## Энергия

Батарея — линейный Wh/SOC reservoir с постоянными base/tx loads и min_tx_soc.
При достижении порога передача останавливается; базовая нагрузка продолжается.
Недоступная базовая энергия считается unserved_base_wh, а не бесплатным питанием.
В ledger: generated, served consumption, curtailment и balance residual.

При solar_area>0 используется ERFA Sun, GCRS position и shadow fraction. Solar mode
явно **ideal_sun_tracking**: независимое идеальное наведение панелей без стоимости
разворота или конфликта с антенной. По умолчанию solar_area=0, а не скрытая зарядка.
Solar sample-and-hold привязан к абсолютной ресурсной сетке, midpoint каждого
кванта. По умолчанию quantum5с, разрешён диапазон.01–30с; это численное приближение,
не независимый error budget энергетики. Разбиение внешнего шага не меняет его сетку.

Нет battery ageing/voltage/current dynamics, temperature coupling, тока пуска,
реального ADCS, RF pointing и гибких панелей. Snapshot орбиты фиксирован: команды
связи не меняют траекторию, манёвры требуют нового расчёта и новой конфигурации.

## Запись, replay и CLI

`space-env.reference-operations-run/1` содержит весь spec со sealed ephemerides,
lineage, software manifest, observations/actions/results и final report. После записи
исходные spec/plan файлы не нужны. Replay сверяет каждый результат и итог; изменённая
software configuration отвергается. Checksum проверяет целостность, не подпись.

```bash
python -m space_env.cli reference-run spec.json plan.json run.json --lineage mission-family-1
python -m space_env.cli reference-replay run.json
```

plan.json: `{"steps":[{"dt_sec":10,"actions":{"a":"Ground","b":"Ground"}}]}`.
Файл сохраняется атомарно после успешного исполнения плана. Ошибка шага откатывает
время, inventory, энергию, ledger, квитанции и interval history. История rollback
использует длины append-only logs, а не полное копирование накопленных списков.

RL JSONL из общего rollout writer сохраняет structured observations/actions;
для self-contained provenance дополнительно хранить task spec/reference run.
SFT tools protocol старой MissionSession не подключён к этому исполнителю автоматически.

## Оставшиеся gates

Это модель ground delivery готового inventory. Acquisition/processing/relay/ACK,
манёвры, thermal/ADCS, полноценный RF, stochastic faults и реальные hardware
паспорта требуют последующих профилей. Архивы/отчёты пока in-memory с накопленными
результатами; throughput/retention/sharding/endurance на больших миссиях не измерены.
Изолированная установка/тесты не означают operational qualification или превосходство
над сторонними симуляторами. Общий MissionWorld остаётся legacy-training/1.
