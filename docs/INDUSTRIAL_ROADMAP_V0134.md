# Space-env 0.134: от набора моделей к промышленной платформе миссий

Дата: 30 октября 2026. База: `1e5a9f6`, версия `0.134.0`, чистое рабочее дерево
до аудита. Подтверждения: [V0134_STATE.md](audits/V0134_STATE.md),
[диагностика](audits/v0134_probe_results.json), [инвентаризация](audits/v0134_inventory.json).
Это текущая очередь работ. Предыдущие планы 0.93/0.98 сохраняют историю требований.

> Разработка части P0 уже выполнена в рабочем дереве: см.
> [EXECUTION_V1.md](acceptance/EXECUTION_V1.md). Закрыты указанные там контракты
> длительности, reset/restore и JSON-экспорта; добавлено управление зарядкой смен.
> P0 и весь промышленный roadmap не объявляются завершёнными автоматически.
> Reference foundation теперь частично реализован отдельно от MissionWorld:
> [GCRS_REFERENCE_V1.md](validation/GCRS_REFERENCE_V1.md). Новые frame/time/orbit
> проверки не закрывают полную P1 миграцию или квалификацию всех сил/подсистем.

## 1. К чему приходим

Промышленный продукт для разработки, обучения и проверки агентов управления
миссиями группировок. Один причинно согласованный мир объединяет орбиту, борт,
связь, наземные ресурсы, полезную нагрузку, заявки и решения оператора.
Планировщики, RL/MARL и LLM получают объявленную информацию и исполняют проверяемые
команды. Итоговая полезность подтверждается продуктами, доставкой и ограничениями.

Первый профиль пилота: неоднородная LEO-группировка ДЗЗ с наземной доставкой и
перепланированием при ограничениях/отказах. Это гипотеза точки входа, а не уже
подтверждённый рынок. Отдельный эксплуатационный ЦУП и flight-qualified simulator
требуют дополнительных критериев и не объявляются результатом этого плана.

## 2. Что уже сделано

144 модуля пакета, 37 зарегистрированных сред, 72 тестовых модуля, 358 тестов
при сборе. Эти числа показывают объём, не квалификацию.

По сравнению с 0.98 существенно расширены:

- Постоянные команды, продолжительные операции, станции/терминалы, snapshot/restore.
- RK4 со временем стадий, ERFA-функции, TEME/ITRS/GCRS utilities и reference report.
- Finite burn, подготовка/slew/settling, safe mode, обработка/сжатие/вытеснение памяти.
- Упрощённые облака/качество, геометрии заявок, SAR duty/energy, независимый спрос.
- Задержки измерений, awareness/FDIR, агентные tools, DTN/routing и orbital operations.
- Dict observation, параллельная обёртка, replay-verified записи, живые сессии Studio.
- Конфигурация CI fast/reference/full и ранее выполненные installed-wheel проверки.

Не создавать эти компоненты снова с нуля. Разделять существующий helper, его
включение в общий сценарий и квалифицированный профиль целой миссии.

## 3. Что пока отделяет от промышленного продукта

**Корректность исполнения.** Подтверждены превышение длительности операции,
неполный reset/restore и несовместимость Dict с rollout writer. Это влияет на
обучение даже при успешных локальных тестах.

**Физическая согласованность.** Метка GCRF сейчас обозначает собственный кадр
Кауэлла с вращением GAST/полюса, не полноценный GCRS. Для одного вектора на эпохе
20 октября 2026 преобразования пакета и c2t06a расходятся примерно на 46 км в ECEF.
Это не оценка ошибки всех траекторий; это доказательство различия соглашений.
ERFA epv00 и старый узел каталога требуют единого frame/time контракта.

**Интеграция сложных моделей.** YAML v2 не описывает весь набор конфигураций борта,
измерений/отказов и fidelity. Geometry helpers ещё не заменяют накопление покрытия
полигона и сертифицированный stereo/revisit outcome. Паспорта реальных аппаратов
требуют provenance и калибровки, а не только опубликованного названия.

**Обучение.** DictEnv управляет первым аппаратом и наблюдает четыре скаляра;
ParallelFleet пока повторяет часть API, но не является проверенным PettingZoo adapter.
Сокращённая SFT-запись без контекста и полных наблюдений не заменяет обучающий корпус.

**Масштаб и доставка.** Ступени 16/64/256 — propagation-only с request IDs вместо
исполняемого спроса. Это не проверка большой mission workload. В CI профиль full
охватывает два файла, все профили вместе — семь из 72 тестовых файлов.

**Продукт.** Нет подтверждённой клиентской приёмки/сравнения, промышленного SLA
и полного описания совместимости. Это отдельные deliverables, не следствие номера версии.

## 4. Пакеты развития и критерии завершения

### P0. Восстановить доверие к контрактам исполнения

Приоритет: блокирующий. Основные пути: `simulation/world.py`, `commands.py`,
`parallel.py`, `snapshot.py`, `mission/session.py`, `rl/trajectories.py`, CI.

1. Превратить `v0134_probes.py` в failing regression tests; исправить причины.
2. Ввести единый event timeline для конца операции, AOS/LOS, outage, release,
   processing-ready/TTL/ACK, fault/recovery. Операция2с внутри step10с исполняется
   только2с; остаток шага следует объявленному idle/running режиму.
3. Snapshot обязан охватывать truth history, SensorStream RNG/cache, task registry,
   command batch, session revision и необходимые накопленные KPI. Rollback и fork
   не оставляют измерения из будущего или скрытое изменение повторного решения.
4. Reset создаёт/восстанавливает исходный мир, seed streams и список agents;
   корректно обработать done, остаток горизонта и пустое множество агентов.
5. Recursive serializer для Gym Dict/Tuple/arrays/scalars; action RNG отдельно
   seeded. Schema и lineage обязательны, молчаливое flatten запрещено.
6. Пересмотреть CI: contract suite, physics reference и полный pytest имеют
   реальные разные области; добавить scheduled/manual long suite с 4/16×72h.
   Не называть два smoke-файла полной регрессией. Не снижать assertions ради зелени.
7. Проверить суточную миссию32: статус сообщает18 сданных, тест требует≥30.
   Если изменение вызвано корректной физикой, обновить версионированный сценарий
   и заранее определённый expected outcome с объяснением. Иначе исправить регрессию.

Gate P0: полная регрессия без скрытых исключений; duration/dt invariance,
reset determinism, restore/fork parity, Dict rollout roundtrip. Накапливаемые
нарушения и дискретные outcomes совпадают; численные поля имеют явные допуски.

### P1. Явные кадры, время и воспроизводимая физика

Зависимость: P0 для сквозного результата. Пути: `core/frames.py`, `time.py`,
`sun.py`, `bodies.py`, `catalog/build.py`, `catalog/geometry.py`, `validation/`.

1. Разделить legacy-training frame и настоящий GCRS/GCRF reference profile.
   Вести явные frame tags. Нельзя выдавать одинаковую строку GCRF за разные системы.
2. Для reference выбрать согласованную цепочку GCRS↔ITRS, время UTC/TAI/TT/UT1,
   EOP и velocity/covariance transforms. Leap-second table и диапазон эпох — часть профиля.
3. Все r/v, Солнце/Луна, силы, атмосфера, освещение, LTAN/LTDN и станции согласованы
   в одном профиле. Опциональная установка ERFA не должна молча менять физику:
   manifest фиксирует backend; недостающая dependency явного профиля даёт ошибку.
4. Исправлять frame/model согласованность независимо от сохранения старого score.
   Старые задания оставить в legacy suite. Для нового профиля заново генерировать
   и версионировать трассы/цели, а не подбирать физику ради12/12.
5. Reference fixtures: несколько эпох/наклонений, r/v/6×6 transforms, наземная точка,
   AOS/LOS, тень, J2/drag/SRP/third-body и finite burn. Error budget задаётся до измерения.
6. Развести analytic sanity, внутренняя сверка формул и внешняя независимая валидация.
   Собственный disk integral полезен, но не заменяет внешний Pc corpus.

Gate P1: frozen reference data с источниками/лицензиями, measured error/tolerance,
capability report и миграция legacy→reference. Допуск короткой двухтелой600с дуги
не распространяется на perturbed сутки. Нужны validation arcs на24/72ч.

### P2. Конфигурируемая и калибруемая миссия

Зависимость: P0/P1. Пути: schema/catalog, capability models, engine, outcome.

1. Версионируемая схема миссии: начальные состояния и profiles, модель борта,
   stations/terminal capabilities, demand, sensors, faults, seeds и внешние данные.
   Строгая вложенная validation, finite/ranges, references и dependency cycles.
2. Все существующие prep/slew/processing/safe/finite-burn/SAR возможности доступны
   через публичный SDK и сценарий; тесты не зависят исключительно от приватных полей.
3. Capability registry моделей аппаратов: source/epoch/units/uncertainty/assumptions,
   profile applicability и ожидаемые traces. Калибровка сначала существующих моделей.
4. Полигон/полоса — накапливаемое покрытие с допустимым разрешением/качеством;
   revisit/stereo — набор связанных acquisitions с временем и направлением взгляда,
   а не только проверка «вмещается в один круг» или разности модулей углов.
5. RF/DTN: rate/latency/error, ACK/retry/idempotent custody, processing/TTL и
   half/full-duplex соединены общим executor; ресурсы/данные не возникают заранее.
6. Поэтапно выделить engine event scheduling, subsystems и product ledger через
   публичные интерфейсы, сохраняя parity. Не начинать с полного переписывания ядра.

Gate P2: три неоднородные модели в одной24ч миссии; ограничение энергии/памяти,
outage, processing, relay и срочная заявка реально меняют полезность и причины.
Balance/conservation и spectrum of feasible/infeasible outcomes проверены.

### P3. Настоящие среды и данные обучения

Зависимость: P0; fidelity promotion после P1/P2.

1. Один MissionEnv с entities vehicles/requests/stations/operations, masks и
   configurable capacities. Observation содержит информацию, достаточную для задачи,
   но не скрытые причины/teacher. Режимы full/partial observation явно разделены.
2. Реальный PettingZoo Parallel adapter optional extra, официальные API tests,
   agent lifecycle, seed/options/spaces, invalid actions и variable dt.
   Централизованный baseline и joint actions используют тот же executor.
3. LLM tools с JSON schemas, action validity, cost/latency/call budgets; ответы
   планов проверяются исполнением. Read-only forecast не мутирует состояние и историю.
4. Run artifact самодостаточен: schema, code/model/data hashes, initial config/state,
   observations/actions/tool calls/events, полные products/receipts, final state/KPI.
   Проверить перенос в другой каталог без исходного YAML и с недоступной сетью.
5. RL/SFT/preferences: входной контекст и доступная телеметрия, ответ/tool actions,
   failures, teacher/provenance, outcome и replay verification. Atomic shards,
   checksums, schema validation, повторяемая сборка и grouped split без leakage.
6. Обучить один воспроизводимый baseline на нескольких независимых сценариях;
   оценить held-out demand/epoch/fleet и fast→reference transfer. Не требовать
   победы RL, но честно публиковать результат и failures.

Gate P3: scenario→agent→dataset→replay→evaluation одной командой; Dict/MARL/LLM
paths проверены на целых эпизодах, включая reset, failure и restart.

### P4. Benchmark широких промышленных задач

Зависимость: P1–P3, семейства вводятся после gates соответствующих моделей.

Начальная продуктовая линейка:

- EO mission scheduling: quality/revisit/coverage и end-to-end delivery.
- Ground network scheduling: terminals, setup, outages и sharing.
- DTN routing: меняющийся граф, очереди, TTL, backlog и справедливость.
- Resource/FDIR: диагностика, safe/recovery и стоимость упущенной работы.
- Dynamic replanning: urgent request, потеря аппарата/ресурса, новые ограничения.
- Orbit maintenance/phasing/disposal: расход ресурса и влияние на расписание.
- Conjunction response: сообщения/неопределённость и независимая safety validation.

Для каждого семейства: generator независимого спроса, declared objective/safety,
feasible/infeasible fixtures, baseline planner и tiny exact/reference subproblem.
Добавить rolling-horizon solver с настоящим constraint model; heuristic сортировка
по сроку сама по себе не доказывает решение задачи оптимизации.

Разбиение по scenario lineage/region/epoch/fleet/demand; hidden holdout не доступен
teacher и подбору гиперпараметров. Все failures в знаменателе. Отдельно utility,
latency, quality, safety, runtime, memory, confidence intervals и sample count.

Gate P4: frozen suite версии, raw artifacts, честный multi-seed report. Smoke
регрессия и learning benchmark раздельны. Изменение физики требует rebaseline/version.

### P5. Масштаб, эксплуатация и Studio

Зависимость: P0/P2/P3.

1. Лестница реальных нагрузок:16 аппаратов/100 заявок/24ч,64/1000/24ч,
   затем256/10000 с горизонтом по измеренной feasibility; отдельно72ч endurance.
   Станции, выдача спроса, связь, planner и outcomes работают, не только request IDs.
2. Hardware/profile/graph/log policy фиксированы. Wall time, peak RSS, event count,
   reset/snapshot/replay, planner time и dataset throughput измеряются отдельно.
3. Сначала profile, затем кэш с invalidation, spatial screening, batching,
   process workers и bounded logs. После оптимизации повторять correctness gates.
4. Session tracks/frames и truth history имеют bounded retention/streaming;
   GET не возвращает бесконечно растущий архив. Пагинация, revisions, cancellation,
   lifecycle/delete сессий, quotas и проверяемое crash recovery.
5. Studio отображает actual planned/running/completed/failed events с правильными
   actor/time; request status не подменяет факт события. Timeline/Gantt,2D/3D,
   footprints, ресурсные конфликты и сравнение run доступны в одном workflow.
6. Installed-wheel matrix на поддержанных Python/OS; snapshot/data migrations,
   CLI/SDK contracts, dependency/profile locks и package/data/license completeness.

Gate P5: воспроизводимые load/endurance/install reports; длительная сессия с
ограниченной памятью, restart без потери outcomes и проверенный браузером workflow.

### P6. Проверка преимущества и клиентский пилот

Ориентиры: [BSK-RL](https://avslab.github.io/bsk_rl/api_reference/index.html)
документирует Gymnasium/PettingZoo, группировки и обмен; [Basilisk](https://avslab.github.io/basilisk/)
— моделирование аппаратов, Monte Carlo/HIL; [STK](https://ansys.synopsys.com/products/missions/ansys-stk)
— инженерный анализ миссий, сенсоров/связи и визуализацию. Это разные классы систем.
Сравнительных прогонов пока нет; отсутствие возможности на странице не доказывает
её отсутствие в продукте. Нельзя считать Python/RL/3D уникальным преимуществом.

Проверяемая гипотеза нашего преимущества: меньше времени на построение и отладку
агентной миссии, более прозрачное исполнение/данные и лучшее решение операционных
задач при согласованной точности и compute budget.

1. Выбрать совпадающий профиль/задачу; зафиксировать версии, physics, inputs,
   constraints, solver/agent budget, hardware и preprocessing до эксперимента.
2. Сравнивать качество при равном budget либо стоимость при равном качестве,
   воспроизводимость и время пользователя. Не сравнивать разные fidelity как равные.
3. Первое исследование —20 paired scenarios; достаточный объём уточнить по variance.
   Публиковать интервалы неопределённости, все failures и ограничения покрытия.
4. Пример целевого преимущества:≥20% в одной согласованной метрике без ухудшения
   safety,≥30% сокращения времени настройки/диагностики. Это цели, не результаты.
5. Провести пилот на разрешённых новых данных оператора, с его baseline и
   критериями. Развести исследовательский replay, offline deployment и hosted service.
6. До shared hosting: ownership/auth/isolation, secrets, quotas, backup/recovery,
   observability и support. Контракты, права на исходные данные и price discovery
   требуют работы пользователя с клиентом, агент не может их объявить выполненными.

Gate P6: pilot acceptance, измеренная полезность/стоимость и подтверждённый workflow.
«Промышленный» и «лучше» применяются к конкретному прошедшему gates профилю.

## 5. Следующее цельное поручение агенту

Выполнить **P0 целиком**: новые diagnostic cases → regression tests → исправления,
полная проверка CI покрытия и суточной миссии → acceptance report. Одновременно
описать P1 frame migration и error budget, не менять незаметно legacy геометрию.
Это один связный проход; P1–P6 — последовательная программа, а не обещание
автоматической квалификации/пилота за один запуск.

Артефакты реализации:

- `docs/acceptance/EXECUTION_V1.md`: gates, commands, версии и результаты P0.
- `docs/validation/FRAME_MIGRATION.md`: legacy/reference semantics и rebaseline.
- `docs/validation/REFERENCE_MATRIX.md`: model→fixture→source→tolerance→limits.
- `docs/benchmarks/MISSION_SUITE.md`: scope/splits/baselines и frozen comparison protocol.
- `docs/product/PILOT_SPEC.md`: согласованная постановка и критерии заказчика.

Эти пути — задания на создание, не существующие доказательства. Статус каждого
этапа вести как implemented/verified/blocked с реальными ссылками. Сохранять
совместимость через versioned profiles/adapters. Не коммитить и не пушить самостоятельно.
