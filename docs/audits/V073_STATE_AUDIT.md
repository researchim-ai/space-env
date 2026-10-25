# Аудит Space-Env 0.73.0: текущее состояние и ограничения

Дата: 25 октября 2026. HEAD: `fcab948` (полный SHA в `v073_inventory.json`). На начало проверки дерево чистое. Пользователь упомянул 0.72; `pyproject.toml` и импортируемый пакет уже показывают **0.73.0**.

Аудит 0.69 был выборочным, и переносить его выводы на новую версию без перепроверки нельзя. Этот документ заменяет его как описание текущего состояния. План действий: [NEXT_DEVELOPMENT_PLAN_V073.md](../NEXT_DEVELOPMENT_PLAN_V073.md).

## Метод и честные границы покрытия

- Инвентаризированы **104 Python-модуля пакета**, **33 test-модуля**, CLI, конфиги, данные и авторские JS/HTML: **168 файлов**, 15 388 строк Python/config внутри `src/` по правилам manifest.
- [Файловый manifest](v073_inventory.json) содержит пути, размеры, хеши и определения классов/функций. Он позволяет проверить, что в обзор вошли все пакеты, а не только operator console. Автоматическая инвентаризация не равна ручному доказательству корректности каждой строки.
- Предметно разобраны контракты и пути данных core → engine → Fleet → mission → env → reward → export/API; все семейства env, trainers/evaluation/dataset code; новые изменения и тестовые assertions времени, ресурсов, сценариев и validation.
- Runtime-проверки сосредоточены на критичных инвариантах. Они сохранены в [v073_reproduce.py](v073_reproduce.py), результаты — в [v073_probe_results.json](v073_probe_results.json). Скрипт диагностический, ничего не исправляет, private helpers использует только для изоляции семантики.
- Собраны 175 pytest cases, запущен полный прогон. Его результат фиксируется ниже. Не проводились браузерный e2e, независимая квалификация физики, полная проверка каждой ветви ошибки, нагрузочный тест и реальный клиентский пилот.
- Код продукта не менялся, commit/push/PR не выполнялись. Добавлены только материалы аудита и планирования.

## Что изменилось с прежней оценки

**Есть и заслуживает сохранения:**

- В engine появились `stored_bits/generated_bits/delivered_bits/lost_bits`; downlink ограничен `_stored_bits`. Старое утверждение «движок всегда передаёт из пустой памяти» теперь неверно.
- `PayloadConfig` получает memory/store-rate, каталог передаёт параметры регистратора.
- `_body_latlon` и substep-update положения/времени улучшили атмосферный контекст. Старое утверждение «используются только координаты станции на всём шаге» больше не описывает основной путь.
- Orbit имеет `frame`, SGP4 явно возвращает TEME. Это важное начало, хотя полная frame safety не реализована.
- Mission scenario loader отвергает неизвестные поля, дубликаты ключей, отрицательный бюджет и неконечные числа части полей.
- `validation/core_report` добавляет четыре формализованных case: энергия, момент, J2 node, знак drag effect.
- `DemandShift-v0` исполняет одну заявку из YAML с deadline и заданной станцией. Наблюдения строятся по текущей геометрии, успех связан с реально переданным объёмом. Это более подходящая основа для развития request-driven миссий, чем только повторение job.kind.
- Есть две различные заявки Kiruna: выполнимая и с ранним deadline; report честно не ставит `learned`.

**Частично, а не закрыто:** единый resource ledger, корректность времени/кадров, полноценная DSL, наблюдения всего семейства сред, независимая validation, mission benchmark и production API.

## Покрытие подсистем и уровень зрелости

### Физика и подсистемы

`core/`: время, Orbit/boinor, Cowell/RK4, J2/атмосфера, SRP/third body, Earth frames/EOP, Sun/Moon, eclipse, access/passes, link, quaternion slew, power/thermal, relative/CW, Pc/CDM и минимальный CCSDS packet header.

Сильная сторона — физические зависимости уже соединены в одном движке и есть аналитические sanity checks. Граница — упрощённые модели и критичные ошибки базовых контрактов, перечисленные ниже. Полноценные attitude dynamics/actuators, finite-burn propulsion, измерительный навигационный фильтр и calibrated RF/thermal digital twin из этого не следуют.

### Миссионная логика

`mission/`: отдельные workplan/horizon/fleet_plan/campaign реализации. Есть окна, память планировщика, общий ресурс станции, манёвры и расписания. Однако planner и executor считают объём и время разными способами; campaign основан на curated population, screening первого аппарата и эвристиках. Объекты типов `MEO`/`INTERPLANETARY` в enum не подтверждают наличие соответствующих полных миссий.

### Каталог и интеграции

`catalog/`: Sentinel-2A/B, Landsat 9, Iridium NEXT и generic 6U/debris/SAR/GEO; published values и assumptions во многих местах честно разделены. TLE snapshot/live fetch, mean orbit geometry, mesh/routing, traffic helpers. Один model key ещё играет роль instance ID в campaign; нельзя просто создать десятки независимых экземпляров одной модели через текущий shift YAML.

`integrations/`: SGP4, двухтельный Orekit, SPICE Sun и CCSDS parser adapter. Это узкие адаптеры, не общее переключение всего simulator backend. Ядра/данные/версии и независимые эталоны требуют отдельного lifecycle.

### Обучение, данные и оценка

`rl/`: linear/factorized REINFORCE, heuristics, curriculum/rewards, rollout JSONL. `eval/`: smoke benchmark, short tracks, comparison, transfer/scale/operator/demand reports и keyword recall. `data_generators/`: glossary/templates, simulator narratives и SFT procedures/plans.

Этого хватает для обучения и экспериментов на ограниченных упражнениях после исправления дефектов. Пока нет доказательства массовой генерации независимых, физически валидированных миссий, пригодных для заявлений об industrial agent performance.

### Сервис и визуализация

`api/`, `simulation/fleet_view.py`, `viz/export.py`, `viz/`: локальная демонстрация с картой/глобусом, playback, сценарием, screening, manual operator и auto macro. Единая user experience уже есть. Сервис хранит одну process-global смену; UI связан с private env state и заранее рассчитанными дугами. Session isolation, persistence, task-driven UI для DemandShift и единый world stream ещё нужны.

## Все 32 зарегистрированные среды: что именно моделируется

Классификация — описание реализации, не оценка полезности учебных упражнений.

### Последовательное управление одним аппаратом / отдельным процессом

- `MissionPlanner-v0`: imaging/comms/малый импульс, цели на будущей трассе, counters и reward shaping; не общий планировщик заявок.
- `ResourceManager-v0`: режим нагрузки, SoC/температура, выживание на горизонте.
- `AnomalyResponse-v0`: случайный тип одного из трёх отказов и симптомы; есть настоящая вариативность reset, но небольшой класс отказов.
- `ConjunctionAssessment-v0`: случайная encounter geometry, фиксированные варианты манёвра, forecast Pc; optional short tracking arc.
- `SentinelPhasing-v0`: фиксированная пара и ошибка +0.5 м/с, гашение дрейфа; не весь formation maintenance.
- `SentinelAcquisition-v0`, `LandsatAcquisition-v0`, `CubesatAcquisition-v0`: сбор и сдача вокруг заранее выбранного прохода; собственные env buffers расходятся с новым engine recorder.
- `SentinelPassFault-v0`: recovery radio в выбранном проходе с env-local preload; не путать со старым PassFault ниже.
- `PassFault-v0`: короткая учебная recovery sequence; доставка логическая и не подтверждается физическим contact/ledger.
- `DemandShift-v0`: одна заявка, один аппарат, геометрические признаки, deadline и engine bits. Важная новая основа, но success пока «передано >0», не исполнение полного продукта требуемого качества.

### Централизованные задачи группировки

- `Constellation-v0`: один dispatcher выбирает satellite+command, отдельный Fleet buffer; это single-agent MultiDiscrete, не MARL.
- `ConstellationOps-v0`: relay к борту с наибольшим elevation над Fairbanks, без реальной доставки через feeder.
- `FleetShift-v0`: цепочка/relay, формация и штраф за импульс на cold pair; gateway выбран в начале, общая учебная постановка.

### Исполнение заранее построенного плана / распределение окон

- `WorkPlan-v0`: следование списку imaging/comms с закодированным видом дела; отдельный memory accounting.
- `DayShift-v0`: прохождение фиксированного job list смешанного флота.
- `OperatorShift-v0`: сначала awards, затем job list; hints из required plan, success требует совпадения keeper heuristic.
- `DishAward-v0`: последовательный выбор keeper; исполнение сеанса с искусственным preload ожидаемых битов.

### Узкие одношаговые упражнения / выбор стратегии

- `DayMission-v0`: принятие рассчитанного плана.
- `ConjunctionQueue-v0`: выбрать один заданный импульс или wait.
- `DecayWatch-v0`, `DecayWatchHigh-v0`: решение по готовой оценке drag loss; `should_burn` входит в obs.
- `PhaseBurn-v0`: разрешить заранее назначенный второй импульс.
- `Reroute-v0`: выбрать заранее реализованный обход B→C→A.
- `RendezvousHold-v0`: погасить фиксированную ошибку скорости; не docking/servicing simulator.
- `PriorityCoverage-v0`: одна память, приоритетная/запасная цель.
- `DebrisAvoid-v0`: own burn/debris burn/wait на curated geometry.
- `Disposal-v0`: малый retro impulse уменьшает semimajor axis; не доказательство удаления с орбиты или нормативного compliance.
- `StationContention-v0`: выбрать видимый борт с максимальным буфером.
- `OrderBook-v0`: FIFO против priority allocation, не выдача свободного расписания агентом.
- `RadarPass-v0`: включить SAR в eclipse, получить один кадр.
- `LongitudeKeep-v0`: вернуть semimajor axis ближе к geosynchronous; не полный GEO longitude/inclination/eccentricity station keeping.

## Подтверждённые дефекты и незавершённые контракты

### V73-01 — P0: UTC roundtrip теряет 12 часов [runtime + независимый Astropy]

`core/time.py:34`, `jd_to_datetime` берёт целую/дробную часть JD без корректного перехода noon→calendar day. Для `2026-10-20T00:00:00Z`:

- JD = 2461333.5; Astropy возвращает `2026-10-20T00:00:00.000`.
- `Time.datetime` возвращает `2026-10-19T12:00:00+00:00`.
- Roundtrip error = **−43200 секунд**.

Это затрагивает экспорт/нарративы и force context через `.datetime`. Там, где расчёт остаётся в JD, ошибка не обязана проявляться так же; численный ущерб каждому пути ещё надо измерить. `test_time_jd_roundtrip` проверяет только год и тождественность JD самому себе, поэтому дефект не обнаруживает.

### V73-02 — P0: съёмка противоположной стороны Земли [runtime]

`core/access.py:18` корректно считает угол направления, но caller использует только off-nadir/FOV без LOS occultation/положительного elevation цели. Near и antipodal точки дают 0°. Дополнительный integrated probe помещает единственную цель на противоположной стороне: после `engine.step(1, 'imaging')` **count=1, command_ok=True**.

Нужно единое `target_access`: Earth LOS + pointing/FOV + требуемое освещение поверхности/качество. Eclipse спутника не эквивалентен дневному освещению цели.

### V73-03 — P0: frame label есть, безопасность frame отсутствует [runtime + code]

`core/orbit.py:132,144`: `propagate` и `with_delta_v` создают новый Orbit с default GCRF, теряя TEME. Probe с нулевым burn: **TEME → GCRF**, без преобразования векторов. Engine тоже пересобирает Orbit с default frame. В `frames.py` нет общего dispatch по заявленным frame; GAST rotation сама по себе не является полным GCRF→ITRF преобразованием.

Минимум: сохранять frame там, где преобразования нет; запрещать unsupported combinations. Затем validated transforms позиций/скоростей/covariance.

### V73-04 — P0: recorder улучшен, но учёт ещё несогласован [runtime + code]

- `_store_capture` использует численное значение `store_rate_bps` как биты одного кадра, без множителя длительности. Это не интеграл потока за экспозицию.
- `_generated_bits` растёт только на принятые в память биты, `_lost_bits` — отдельно. Для двух кадров 8 bit в память 10 bit получаются generated=10, stored=10, lost=6. Инвариант «generated=stored+delivered+lost» нарушен; при иной семантике счётчик нужно переименовать и добавить total produced.
- Fleet buffer, engine recorder, acquisition/workplan env-local buffers остаются разными источниками истины. Runtime: relay 8 Mbit увеличивает receiver Fleet buffer до 8, но receiver engine stored_bits остаётся 0.
- Fleet relay/downlink принимают dt, но сами не продвигают clock/не резервируют terminal. Повторные вызовы можно делать на том же времени. Это требует явного контракта executor, иначе capacity расходуется многократно.
- `DishAwardEnv` напрямую дописывает `_stored_bits/_generated_bits` из expected ready_bits. Это preload учебного теста, а не исполненная съёмка.

### V73-05 — P0/P1: временная причинность операций и учёт связи [code]

`SimulationEngine.step` сначала пропагирует целый dt, потом запускает planned events и разрешает imaging по samples уже прошедшего интервала. `duration_sec` planned operation не исполняется как lifecycle. Запланированное на t=0 событие не проходит условие `t0 < scheduled <= t1`.

`_accrue_downlink` применяет endpoint contact/rate ко всему dt, не интегрируя реальный доступ внутри шага; возможны завышение/потеря доставки у AOS/LOS. `find_passes` и engine используют не полностью одинаковый набор force/weather/SRP параметров.

Приоритетный тест: partition equivalence по учёту ресурсов и событиям при одной и той же непрерывной команде, с явной numerical tolerance и неизменным control schedule.

### V73-06 — P1: нагреватель учитывается дважды [code]

`engine.py::_update_subsystems`: heater уже входит в `consumption`; затем thermal получает `q_diss_w=consumption+q_fault` и отдельно `q_heater_w=heater_w`. Одна и та же мощность добавляется дважды к узлу. Radio/actuator operation loads при этом не являются полноценно выделенными потребителями. Нужны energy/heat flow contracts и независимый balance test.

### V73-07 — P0 для RL: одношаговый REINFORCE не обучается [алгебра + runtime]

В обоих тренерах `baseline=mean(gains)` текущего эпизода. Если эпизод состоит из одного шага, `gain-baseline=0`: все updates нулевые независимо от награды. Probe на `PriorityCoverage-v0`: lr=0 и lr=10 дают идентичные десять возвратов при одном seed. Это важнее наличия learning script.

Дополнительно linear trainer не возвращает обученные weights/checkpoint; `learned` в report сравнивает training tail с random reward, не held-out mission learning. Нужен train/eval contract и state-independent/running/critic baseline с тестом learning на bandit.

### V73-08 — P1: некоторые success не подтверждаются выполнением миссии [runtime + code]

`PassFault-v0`: последовательность reboot/wait/wait/comms/wait/wait даёт success=True при contact=False и engine delivered_bits=0. В `DayShift/OperatorShift` сеанс теперь принимается по `last_command_ok`, без проверки фактического delivered amount; это не доказательство доставки. В `DemandShift` положительной части объёма достаточно для success, нет required product size/quality.

Учебные критерии допустимы, но их нельзя смешивать с mission SLA. Нужен отдельный outcome evaluator по продуктам, объёму, destination, сроку и ограничениям.

### V73-09 — P1: CDM units и covariance требуют исправления до реальных входов [runtime + code]

`core/cdm.py` удаляет `[units]` и применяет фиксированные множители. Явный input `MISS_DISTANCE=100 [m]` становится 100000 m; `RELATIVE_SPEED=1 [m/s]` — 1000 m/s. Cross covariance `1 [km**2/s]` становится 1000 m²/s вместо 1e6 по размерности.

Нужны явно поддерживаемый профиль стандарта, unit-aware parsing, transforms ковариаций объектов в общую frame/epoch, PSD checks и fixtures из независимого источника. Сейчас blocks просто суммируются; нельзя предполагать одинаковость RTN обоих объектов для произвольной геометрии.

### V73-10 — P1: TCA/Pc и relative/navigation model ограничены [code]

Campaign `_closest` всё ещё sample minimum на 60-секундной сетке и фиксированная covariance. В NumericalEncounter есть 0.5-секундное уточнение, поэтому утверждать «во всех путях нет refinement» неверно. Но prediction context многократно переиспользует исходный `when` между последовательными вызовами; обновление времени внутри отдельного propagate этого не решает полностью.

CW-state velocity и инерциальный relative velocity требуют явного rotating-frame transport contract. Estimation — synthetic range fit, не полноценный operational OD filter. Singular/non-PSD covariance и low-relative-speed encounter требуют диагностики применимости, не молчаливой подстановки определённой Pc.

### V73-11 — P1: составной World по-прежнему не единый [code]

`Fleet.step` проверяет равенство relative clocks после поочерёдных engine steps, но нет общего транзакционного event world/проверки общей абсолютной эпохи при создании. Operator `_advance_vehicle` останавливает отдельные engines перед их задачами, executor двигает конкретный борт. Snapshot state — observation record, а не полноценный restore snapshot c RNG/очередями/locks.

Необходимо различать гарантии разных путей: у Fleet есть полезный барьер; он не доказывает synchrony OperatorShift и replay/restorability всего продукта.

### V73-12 — P1: честность задач и split ещё не решены [code]

Operator `_opportunity` остаётся производным от required_items/job.kind, success требует `_wrong_awards==0`. Day/WorkPlan экспортируют тип следующего задания; часть one-shot env имеет постоянный «нужно жечь» в obs. Это curriculum/procedure execution, не независимое планирование.

Anomaly и ConjunctionAssessment действительно рандомизируются; Demand YAML действительно различаются. Но Operator holdout seed всё ещё не создаёт независимую новую миссию, а два файла Demand — только два case, не suite. Random action_space в rollout/eval отдельно не seeded. Нужны scenario split manifests и separation pedagogical/oracle/agent interfaces.

### V73-13 — P1: сценарий расширен, но DSL не полная [code]

`version` допускается, но поддерживаемое значение не валидируется и не возвращается как schema version. Demand coordinates проверяются на finiteness, но bounds/epoch/model support и cross-field checks распределены по последующим слоям. Demand step всегда 30 s, horizon не является общим stop condition, deadline может быть перескочен.

Есть два loader family с разными полями; explicit missing shift file возвращает defaults, а mission loader выдаёт error. Cwd override и дубли root/package ещё действуют. Нужны strict canonical schema и явные migrations.

### V73-14 — P1/P2: scheduler пока эвристика, не общий оптимизатор [code]

Workplan выбирает rate по максимальному margin окна, не интегральной capacity; SAR opportunities фильтруются `sunlit`, несмотря на `images_in_eclipse` у SAR в engine. Станция резервируется по физическим пересечениям и большему predicted buffer, даже если winner в итоге не использует весь интервал. Несколько манёвров в forecast не представлены общим списком операций для downstream execution.

В planner нет общей функции mission value/deadlines/quality, rolling horizon при изменении мира и baseline CP-SAT/MILP. Их надо добавлять поверх единого исполнителя, а не писать ещё один независимый buffer/schedule engine.

### V73-15 — P1/P2: data/evaluation pipeline пока не benchmark промышленного качества [code]

Registry benchmark ограничивает многие env 1–4 шагами; Demand до 2 шагов не доходит до стандартной доставки >400 s. Это smoke suite, не сравнение качества миссий. Scale report проверяет success для нескольких маленьких count, без throughput/RSS/error budget.

SFT содержит фиксированные процедуры; keyword recall не проверяет истинность. Некоторые templates независимо семплируют связанные orbital quantities. JSONL не имеет полного scenario/backend/input lineage и уникального глобального episode identity. Необходимы replay-verified demonstrations, distinct splits и dataset validation.

### V73-16 — P1/P2: API, UI, packaging и CI [code + ограниченная проверка сборки]

API — один global session под lock, private fields, нет session persistence/idempotency/revision. POST coerces int и не покрывает все invalid JSON-value types; необходимы строгие schemas и ограничения ресурсоёмких запросов перед внешним доступом.

Studio использует общую карту/глобус, но геодезия JS упрощена, «виток» фиксирован 6000 s; в coast branch меняются r/v, но не весь environment state. Playback/forecast/live должны иметь согласованный timestamp/source. Browser визуально заново не проверялся.

Wheel assets `viz/` вне package tree; CI — editable install+pytest без wheel/out-of-tree/check_env/performance gates. Попытка `python -m build --wheel --no-isolation` остановилась из-за отсутствия `hatchling`; это ограничение окружения, не доказанный build defect. Полная установка wheel не выполнена и не выдаётся за пройденную.

## Проверки

1. `python -m pytest --collect-only -q`: **175 tests collected**.
2. `python -m pytest -q`: **174 passed, 1 failed, 39 warnings**, 390.22 s. Единственный failure — `test_http_server_answers_health`: sandbox `PermissionError` при создании TCP socket. Отдельный повтор вне песочницы: **1 passed in 1.26s**. Все 175 cases прошли между основным прогоном и повтором одного теста; полный прогон вне песочницы не выполнялся. Сбой основного прогона — ограничение окружения, не установленный дефект HTTP-логики.
3. `python docs/audits/v073_reproduce.py`: завершён, 9 групп наблюдений в JSON; это подтверждение текущих дефектов, а не green acceptance suite.
4. `python -m space_env.validation.report`: результат в [v073_validation_results.json](v073_validation_results.json). Четыре аналитических/sanity case не подтверждают access, time roundtrip, frame transforms, subsystem fidelity и миссионные KPI.
5. Wheel no-isolation build: unavailable backend (`hatchling` отсутствует); чистая wheel portability не подтверждена.

## Практический вывод

Не нужно выбрасывать существующую библиотеку: её ширина и демонстраторы полезны. Но следующий апдейт должен связать её части одним корректным миром и проверяемыми контрактами. Первая цель — исправить time/access/frame и ресурсную причинность, затем новая mission-level ДЗЗ-среда с заявками и честным benchmark. Новые классы спутников и MARL добавлять к этому общему ядру, сохраняя старые упражнения с явной маркировкой.
