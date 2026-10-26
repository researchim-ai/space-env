# Space-Env 0.86.0 — аудит перед следующим крупным проходом

Дата: 26 октября 2026. HEAD: `e512037ee1857ea62c4ab7584de4bfc685a9ed6c`. На старте дерево чистое. Текущая очередь реализации — [INDUSTRIAL_ONE_PASS_PLAN.md](../INDUSTRIAL_ONE_PASS_PLAN.md).

## 1. Итог

Проект заметно продвинулся от набора учебных сред к составному симулятору миссий. Уже есть MissionWorld, сценарии v2, экземпляры одной модели, product custody, квитанции, диспетчеризация запросов, FIFO/deadline/priority, outage inputs, точный маленький assignment baseline и тесты 72-часовых прогонов. Следующий шаг — укрепить **единую исполнимую семантику**, а не ещё раз добавлять эти компоненты под новыми именами.

Основной риск: при существующих happy-path тестах side effects прогноза, обход ограничений станции, повторный зачёт данных и заглушки KPI способны давать неправдоподобные результаты обучения. Промышленная достоверность не следует из наличия 72-hour YAML или числа env ID.

## 2. Метод и покрытие

- Полная статическая инвентаризация авторских исходников/конфигов/тестов: **110 Python-модулей пакета, 39 test-модулей, 188 файлов** без vendored JS; пути/хеши/символы — [v086_inventory.json](v086_inventory.json).
- Детально разобраны новые/изменённые пути после 0.76: world, scenario_v2, tasking, TaskingShift, custody/receipts, snapshot/restore, interval handling, новые тесты. Также проверены связи с engine/Fleet, старым Operator, RL/eval/data, API/Studio, packaging/CI и validation profile.
- Инвентаризация всех модулей не является обещанием ручного доказательства каждой строки. Изменения физики и ранее найденные проблемы перепроверялись по коду/регрессиям; новый независимый astrodynamics reference run и полный браузерный e2e здесь не проводились.
- Runtime cases в [v086_audit_probes.py](v086_audit_probes.py), значения — [v086_probe_results.json](v086_probe_results.json). Synthetic fixtures диагностируют контракты, а не точность реального аппарата.
- Собраны 223 pytest cases, полный прогон завершён с `--durations=12`. Фактический результат приведён в разделе проверок. Продуктовый код не менялся; commit/push/PR не выполнялись.

## 3. Что уже реализовано и не должно разрабатываться заново

- `core/`: численные орбиты, J2/drag/SRP/third-body, время/EOP, access, питание/термика, упрощённая ориентация, link, CW, Pc/CDM. Ограничения reference fidelity по-прежнему нужно сохранять явно.
- Исправлены старые UTC roundtrip, far-side access, frame preservation/rejection, overflow, single-step REINFORCE. Текущие тесты также проверяют planned imaging 30–40 s, relay identity, разные dt повтора terminal interval и restart trainer.
- `simulation/engine.py`: продукты с product/request IDs, chunk offsets, custody, receipts, operations sample integration, capture/restore. Наличие методов не означает полноту контракта, см. ниже.
- `simulation/world.py`: одновременное продвижение нескольких engines, noisy reading, truth/estimate/forecast, snapshot/restore.
- `mission/scenario_v2.py`: несколько instance IDs одной model, список станций/заявок, file SHA manifest; `world_from_scenario` собирает модель.
- `mission/tasking.py`: диспетчеризация через физический доступ и storage, relay на борт с наземной связью, приоритеты, ограничения политики для станции, отчёты/квитанции.
- `configs/scenarios/eo_72h{,_16}.yaml`: 4/16 аппаратов, 72 h, 3 станции, **две** заявки. Есть исполняющие эти файлы тесты, а не только проверка YAML. Это полезная long-horizon regression, но ещё не широкая нагрузка по спросу/конкуренции.
- `TaskingShift-v0`: один аппарат, заявки v2, оценка по receipts. Fleet executor существует отдельно; публичная Gymnasium tasking-среда сама запрещает несколько instances.
- Старые procedural environments, linear/factorized REINFORCE, SFT/JSONL и Studio остаются полезны. Их нужно маркировать по назначению и связывать через единые контракты, а не выдавать за полностью однородную платформу.

## 4. Подтверждённые находки

### A01 — P0: прогноз изменяет будущее, snapshot неполный

`MissionWorld.forecast`, `SimulationEngine.capture/restore`.

В synthetic world запланирован impulse +1 m/s на t=0.5. До forecast запас Δv=20; после `forecast(...,1)` часы откатились, но запас стал **19**. При последующем реальном шаге событие манёвра уже **не возникает**, поскольку fired state не откатан.

Capture не включает полный spacecraft state, RNG, fired inject/planned sets, encounter, pending recovery, modes, quaternion/slew, product sequence и ряд counters/cache semantics. World snapshot не включает world RNG: noisy stored reading до/после restore различается (**1029.87 против 910.94** при том же исходном объёме).

Следствие: what-if search, planner preview, branch/replay и dataset reproducibility ненадёжны. Требуются полные versioned snapshots и forecast на изолированной ветке либо строгое restore в `finally`, включая исключения.

### A02 — P0: MissionWorld не обеспечивает общий контракт времени/валидации

`MissionWorld.__init__/step/restore` не проверяют согласованную абсолютную epoch/clock и полный набор actor IDs. `step(1, {'ghost':'imaging'})` молча игнорирует неизвестного actor и двигает мир. Мир с engines на t=0 и t=1 после joint step оставляет **t=1 и t=2**.

`run_policy` начинает локальный elapsed с 0 даже для уже продвинутого мира. На world horizon=2, начатом с t=1, runner заканчивает на **t=3**. Нужен единый now, event boundary handling и preflight до изменения состояния; runtime failure не должен оставлять полумира после частичного шага.

### A03 — P0: outage и capacity — правила политики, не физические ограничения мира

Прямой joint step отдаёт двум аппаратам `comms` на одну станцию: **каждый передаёт 1 Mbit за ту же секунду**, без общей reservation проверки. Политика выбирает winner, но manual/agent/world callers обходят это правило.

Более серьёзный сквозной случай: станция доступна на [0,1), outage на [1,3). `run_policy` при step=1 оставляет квитанции на **t=1,2,3**, всего **3 Mbit**. Передатчик после `comms` armed до LOS; политика переходит на `wait`, но этим не закрывает ранее начатую передачу. Outage не подключён к физическому access/исполнителю.

Relay reservations находятся в Fleet wrapper; `_relay` создаёт новый Fleet при вызове, и relay происходит до выбора/cut следующего world dt. Нет единого shared terminal/resource interval contract для relay/downlink и actual world horizon.

### A04 — P0 для benchmark: scorer принимает дубли и данные до release

`delivery_score`, `_completion_time` суммируют bits без union product ranges. Для заявки 100 bit одна квитанция 60 bit не даёт completion, а **две одинаковые квитанции по 60 bit дают on_time**. Receipt на t=1 для заявки release=20 также засчитывается.

В TaskRequest нет allowed destinations/quality/completeness policy. Поэтому пока нельзя отличить правильную доставку от неправильного endpoint или повторной квитанции. Нужен один authoritative evaluator по уникальным byte/bit ranges, lifecycle, release/deadline, payload/quality и назначению.

### A05 — P1: KPI сообщают неподтверждённую безопасность и смешивают метрики

`mission_kpis` всегда возвращает `safety_violations: 0`. Runtime: SoC=0.01 создаёт constraint event, но KPI остаётся **0**. Некоторые тесты assert ноль для этой константы, не проверяя независимый event accounting.

`mean_delay_sec` берёт первую квитанцию, а не completion time; `quality_full_fraction` усредняет product records, включая relay copies/chunks, вместо оценки требуемого качества заявок. Нужны отдельные first-byte/completion latency, actual safety events/duration и логические уникальные продукты.

### A06 — P1: DSL принимает физически/логически невалидные данные

Runtime loader принимает negative horizon, NaN в nu, lat=120/lon=300, отрицательный size и deadline раньше release. Не все downstream слои обязаны это принять, но ошибка не ловится при загрузке до дорогого моделирования.

Current schema — ручные dict casts. Request destination/outage/fault/resource rules не описываются целиком через один validated config. Scenario manifest хеширует сырой файл, не нормализованную семантику/model/backend/input versions.

### A07 — P1: обучение не имеет общего интерфейса к многоспутниковой миссии

`TaskingShiftEnv` принимает только один instance. Observation содержит SoC, stored, contact, station code; в нём нет геометрии доступных заявок, deadline/release, pending operations и общего времени. Отсутствие поля `correct_action` само по себе не делает такую постановку достаточной для содержательного планирования.

Fleet `run_policy`, Gym env, старый Operator и Studio пользуются разными поверхностями. Новая цель — публичный централизованный fleet-tasking env над тем же executor. Это не требует сразу PettingZoo/MARL и не должно называться MARL только из-за MultiDiscrete.

### A08 — P1: benchmark/data/evidence ещё не отражают industrial mission learning

Есть две TaskingShift fixtures и несколько heuristic policies. `exact_slot_plan` — bounded assignment reference, не глобальный оптимизатор 72-hour mission. Два 72-hour YAML содержат те же две заявки; большинство бортов могут быть idle. Нужны demand/load/fault families, уникальные splits и метрики latency/RSS/constraint adherence.

SFT generators местами игнорируют seed и генерируют фиксированные планы/процедуры; trajectory serializer flatten-ит array obs и не поддерживает будущие typed Dict observations без адаптера. Нет полного replay-verified manifest с scenario/model/source/backend/seed/policy lineage. Counterexample из A04 делает текущие labels ненадёжными, даже если JSON валиден.

### A09 — P1/P2: поставка и UI не объединены с новым ядром

`/api/tasking` отдаёт manifest, а живой пульт работает с global OperatorShift; это ещё не session API новой fleet mission. Studio показывает карточку сценария, но не позволяет управлять request lifecycle того же World. Static assets находятся вне package tree; CI проверяет editable install + pytest без wheel clean-install/check_env/UI e2e/бюджетов производительности.

README ссылается на удалённый `docs/IMPLEMENTATION_PLAN.md`. Предыдущие audit/status файлы удалены между версиями; новый план должен оставлять один стабильный entry point и measured status, чтобы следующий агент не повторял старую работу.

## 5. Ограничения физического профиля, которые не надо скрывать

Validation report содержит energy/angular momentum/J2/drag-sign cases и теперь явно перечисляет Pc not independently validated, shared RK4 stage epoch, отсутствие TEME→numerical transform и полного IAU2000A. Это правильная прозрачность, но не проверка реального digital twin. Проверки access/power/link и external orbit references нужны по заявленной области применения.

В ближайшем проходе не следует одновременно переписывать все forces, делать docking/6DoF, HIL, GNSS и lunar missions. Нужно зафиксировать educational/engineering approximation profile, закрыть execution integrity и добавить узкие независимые reference cases. Расширение fidelity — последующий пакет с конкретным accuracy budget.

## 6. Рекомендуемый существенный результат одного прохода

**Mission Execution & Agent Benchmark v1:** один executor с действующими ограничениями, полные replay/forecast, достоверные products/receipts/KPI, строгая DSL, fleet-tasking Gym env, воспроизводимые baselines/datasets и минимальный локальный session API/Studio. Переиспользовать существующие мир, физику, policies и 72-hour fixtures.

Полная спецификация, порядок, точные negative cases, команды приёмки и границы — [в плане](../INDUSTRIAL_ONE_PASS_PLAN.md). Он нацелен на research/pilot foundation, не выдаёт внутренний тест за customer acceptance или flight certification.

## 7. Проверки этого аудита

- `python -m pytest --collect-only -q`: **223 cases**.
- `python -m pytest -q --durations=12`: **222 passed, 1 failed, 44 warnings**, 962.95 s. Единственный failure — `test_http_server_answers_health`: sandbox запрещает TCP socket. Отдельный повтор вне песочницы: **1 passed in 1.30s**. Все 223 cases прошли между полным прогоном и отдельным повтором; полный suite вне песочницы не запускался.
- Long cases реально выполнены: 4 аппарата/72 h — **83.99 s**, 16 аппаратов/72 h — **313.35 s**. Обе fixtures имеют две заявки; это не подтверждённый performance SLO для промышленного спроса.
- `python docs/audits/v086_audit_probes.py`: выполнен; подтверждает A01–A06 короткими synthetic cases.
- Полная wheel установка, browser e2e, reference astrodynamics backend и customer pilot здесь не выполнялись. Runtime suite и время одной машины не являются performance SLO.
