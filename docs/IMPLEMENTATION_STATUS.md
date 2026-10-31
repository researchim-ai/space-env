# Статус пакета исполнения миссии


## B04-D2a — проверенное ядро вращения, 1 ноября 2026

Добавлено standalone rigid attitude ядро: физически допустимый тензор инерции,
Euler/quaternion integration, три идеальных ортогональных маховика, пределы torque
и momentum, DC/heat/mechanical-work ledger, внешний момент и `r cross F`.
21 новых тестов; JSON continuation и пример насыщения/разгрузки.
[Контракт](validation/RIGID_ATTITUDE_CONTRACT.md),
[приемка](acceptance/B04_RIGID_ATTITUDE_KERNEL.json),
[пример](../examples/rigid_attitude/demo.py).

**Это пока отдельный kernel:** MissionRuntime сохраняет rate-limited slew и
held-body burn. B04-D2 и B04 остаются in_progress. Следующий исполняемый блок —
B04-D2b: opt-in mission controller/state, реальная pointing/rate settling,
энергия/тепло и faults/cancel без сброса spin, off-center rotating thrust,
общие ownership/rollback/checkpoints и installed mission acceptance.
Подробные критерии интеграции — в контракте; затем calibration и B04-E/B05/B09.

Дальнейшая реализация ведется крупными пакетами [B01–B12](INDUSTRIAL_AI_SIMULATOR_PLAN.md).
Текущий — B04; B04-A связанного борта проверен как инженерный surrogate.
B04-B1 добавил sampled polygon/AOI coverage;
B04-B2 добавил optical/SAR quality/resources;
B04-B3 добавил exposure-time revisit/stereo и ground bundles;
B04-C добавил реальные managed CPU/payload/radio lanes;
B04-D1 добавил paid body-frame thrust; следующий — B04-D2 rigid attitude/momentum. Этот файл хранит факты
реализации и проверок; список возможностей v1–v10 не закрывает новые пакеты.

## B04-D1 — связанная тяга в системе корпуса, 1 ноября 2026

Добавлены chemical/electric engine schemas и `body_burn`: конечный gimbal,
settling/ignition, фактические quaternion → GCRS force, масса/топливо,
электричество/тепло, faults/cancel и совместная работа с CPU/omni-radio.
Прогноз учитывает только исполненную тягу; дальше — coast. Legacy GCRS команды
в coupled profile не открыты. Нет rigid-body моментов/маховиков и flight calibration.
[Контракт](validation/BODY_THRUST_CONTRACT.md),
[приемка](acceptance/B04_BODY_THRUST.json),
[пример](../examples/actuation/demo.py).

Следующий блок **B04-D2**: инерция, угловая скорость и Euler dynamics;
ограниченные torque actuators/маховики, насыщение и разгрузка momentum;
off-center thrust torque, связь с pointing/энергией/теплом; аналитические
conservation tests, fault/cancel/replay и installed mission acceptance.
B04-D и B04 остаются in_progress; затем B04-E thermal, B05 RF и B09 scale.

> **31 октября: B04-C — resource-lanes/1.**
> CPU/payload/radio действительно работают одновременно при наличии ресурсов;
> runtime claims запрещают конфликтующих владельцев CPU, камеры и attitude.
> Общие peak/SOC/bandwidth и reservations, CPU slowdown/radio throttle, native
> station arbitration, pause/cancel/fault/preemption и copied local grants.
> Serial остается прежним; native in-place CPU в parallel profile unsupported.
> Проверены общий energy/I/O integral, memory peak, SAR RF/heat, ownership,
> three-lane rollback/replay. 34 новых cases; reference 559 passed, fast 110 passed,
> 989 collected (overlap). Wheel 189/189 modules + 3 fixtures, isolated SDK/CLI
> обоих режимов проверены. Full regression не перезапускался. Это deterministic
> engineering model без measured hardware/SLA или optimal scheduling guarantee.
> [Контракт](validation/RESOURCE_LANES_CONTRACT.md), [приемка](acceptance/B04_RESOURCE_LANES.json).

> **31 октября: B04-B3 — revisit/stereo и ground bundles.**
> Фактические sampled exposure times/poses и quality links входят в provenance.
> Разные acquisitions, допустимые time gaps и общая пригодная площадь нужны для
> revisit; stereo дополнительно проверяет Earth-fixed baseline и convergence.
> Явный наземный пакет позволяет двум аппаратам выполнить одно обязательство
> только после всех полных ACK. Cross-satellite onboard processing не имитируется.
> Проверены partial/cloud overlap, invalid lineage, cancel/TTL/deadline, budgets
> и mid-acquisition/partial-radio replay. 37 новых cases; reference 525 passed,
> fast 110 passed, 955 collected (overlap). Wheel 188/188 modules + 3 fixtures;
> isolated SDK/CLI обоих режимов проверены. Full regression не перезапускался.
> [Контракт](validation/OBSERVATION_REQUIREMENTS.md), [приемка](acceptance/B04_OBSERVATIONS.json).

> **31 октября: B04-B2 — optical/SAR quality и ресурсы.**
> Explicit instrument/quality schemas, categorical cloud/unknown masks с AOI hash,
> validity/availability и forecast opt-in. Spatial Sun/GSD и nominal SAR incidence,
> side/resolution проходят в evidence lineage; качество проверяется отдельно по
> каждому request и конкретному полностью доставленному продукту.
> SAR peak/mean power, rolling session duty и DC/RF/amplifier heat согласованы с
> bus; safe/recovery/cancel не обнуляют историю. Есть quality budgets, actor privacy
> и mid-acquisition optical/SAR restore. Это engineering surrogate без measured
> radiometry, focused SAR, stochastic fields или flight calibration.
> 39 новых cases; reference 488 passed, fast 110 passed, 918 collected (overlap).
> Wheel 187/187 modules и 3 fixtures, isolated optical/SAR SDK и CLI проверены;
> full regression не перезапускался.
> [Контракт](validation/PAYLOAD_QUALITY_CONTRACT.md), [приемка](acceptance/B04_PAYLOAD_QUALITY.json).

> **31 октября: B04-B1 — sampled area coverage.**
> Прямоугольный detector, фактическая ориентация и WGS84 ray hits; AOI с отверстиями,
> даталайн/полярные области, LAEA площади, union без повторного зачета и source
> lineage через transform DAG. Полный ACK продукта с недостаточным покрытием
> не выполняет заявку. Есть historical acquired / delivered reports и bounded
> revisit diagnostics. Optional coverage extra и CI, manifest Shapely/GEOS/PROJ,
> atomic sample-budget rollback и mid-acquisition replay.
> 32 новых cases; reference 449 passed, fast 110 passed, 879 collected.
> Наборы пересекаются; full regression не перезапускался.
> Это геометрический surrogate: continuous swath, stereo constraints и качество
> реальных optical/SAR данных остаются открыты. B04 — in_progress.
> [Контракт](validation/AREA_COVERAGE_CONTRACT.md), [приемка](acceptance/B04_AREA_COVERAGE.json).

> **31 октября: B04-A — reference-spacecraft-bus/1.**
> Конечные slew/settling, pointing/keep-out, body-fixed PV, converter losses,
> RC thermal, safe/fault/recovery и persistent degradation связаны с операциями.
> Два synthetic presets, local bus state, energy/heat ledgers, rollback/checkpoint.
> 29 новых tests; reference 417 passed, fast 110 passed, legacy attitude 19 passed;
> 847 collected (наборы пересекаются). Wheel 185/185 modules, 3 fixtures и installed
> SDK/CLI проверены. Full regression не запускался. B04 остается in_progress:
> polygon/AOI, optical/cloud/SAR, реальная параллельность и torque/thrust coupling
> не реализованы. В новом профиле maneuvers запрещены до физической связи с бортом.
> [Контракт](validation/SPACECRAFT_PROFILES.md), [приемка](acceptance/B04_BUS.json).

> **31 октября: B03 — reference-lifecycle/1, инженерный surrogate.**
> Request → sampled acquisition → raw → consuming transform DAG → full receipt.
> Новый objective fulfilled_requests; raw/CPU/partial transmission не дают fulfillment,
> infeasible/cancelled demand не исчезает из denominator. Есть causal request events,
> persistent workspace/switching, TTL/pins, fan-in/out, different-size outputs,
> versioned transformer registry и самодостаточный prefix restore. Не генерирует
> реальные изображения: metadata/JSON geometry features отделены от resource costs.
> Исправлен run manifest GCRS/ITRS для dynamic reference; actor-scoped jobs не
> раскрывают чужие данные. 31 новых tests; reference 388 passed, fast 110 passed,
> 818 collected. Wheel 184/184 модулей и installed SDK/CLI verified; full regression
> не запускался. Parent physical qualification и calibrated pilot остаются открыты.
> [Контракт](validation/PRODUCT_LIFECYCLE_CONTRACT.md),
> [приемка](acceptance/B03_PRODUCT_LIFECYCLE.json).

> **31 октября: B02-Q2 — point acquisition geometry и события.**
> WGS84 targets/ideal cone, illumination/range/elevation/dwell constraints,
> bounded searches с явной sampling incompleteness, readonly conical eclipse
> forecast и root-guarded opportunities. Optional forecast binding обеспечивает
> atomic stale/foreign-plan rejection после изменения state/maneuver/cancel.
> Проверены analytic-orbit/bisection event boundaries и spherical-disc eclipse
> comparison; shared inputs и различие моделей обозначены. Не создает продукты.
> 28 новых tests; reference 357 passed, fast 110 passed, 787 collected.
> Wheel 183/183 modules, 3 fixtures, isolated -I SDK/CLI, checkpoint/resume/replay.
> Full regression не перезапускался. Parent B02 остается needs_external_evidence.
> [Контракт](validation/GEOMETRY_EVENTS_CONTRACT.md), [приемка](acceptance/B02_Q2_GEOMETRY.json).

> **31 октября: B02-Q1 — optional covariance/STM prediction.**
> Строгий GCRS/PSD контракт, Phi и integrated acceleration PSD, impulse velocity
> noise, глобальная композиция через burn/coast/cancel, agent current P,
> privileged STM и prefix checkpoint/rollback. Номинальные состояния остаются
> детерминированными; нет measurement updates или calibrated orbit determination.
> Frozen J2+Sun/Moon дуга другой эпохи имеет общие ephemerides/frames/EOP.
> 24 новых теста; reference 329 passed, fast 110 passed, 759 collected.
> Wheel: 180/180 модулей, 3 orbit fixtures, isolated -I и covariance CLI workflow.
> Full regression не перезапускался. Parent B02: needs_external_evidence.
> [Контракт](validation/NAVIGATION_COVARIANCE_CONTRACT.md),
> [приемка](acceptance/B02_Q1_NAVIGATION.json).

> **31 октября: B02 — управляемый GCRS-профиль, needs_external_evidence.**
> Impulse и finite burn через общий runtime, изменяемая масса, fuel admission,
> cancel, пересчет окон связи, geometry revision, явные сроки данных и mid-burn
> resume/replay. Frozen Boinor J2 24/72h сравнения имеют общие frame/EOP/constants;
> независимый RHS + RK45 проверяет central finite burn. Это не flight qualification.
> 18 новых тестов; reference 305 passed, fast 110 passed, 735 collected.
> Installed wheel: 178/178 модулей, fixtures, dynamic console replay и qualification.
> Full regression не перезапускался. Acquisition geometry, dynamic covariance/STM,
> independent combined-force/events и mission error budgets еще не закрыты.
> [Матрица](validation/PHYSICS_QUALIFICATION_MATRIX.md), [приемка](acceptance/B02_PHYSICS.json).

> **31 октября: B01 — общий runtime-контракт.** Scenario v3 и явные adapters
> для legacy/reference, capability checks, revision/idempotency, атомарный rollback,
> раздельные views, self-contained verified-prefix checkpoint/resume/replay,
> CLI mission-validate/run/resume/replay/schema. 38 runtime tests, reference 287 passed,
> fast 110 passed, 717 collected. Wheel: 173/173 модулей, isolated Python -I smoke
> и установленная CLI. Полный regression этого пакета не запускался.
> [Контракт](architecture/RUNTIME_V1.md), [приемка](acceptance/B01_RUNTIME.json).
> Это интеграция двух профилей; маневрируемая reference-физика и быстрый snapshot
> не реализованы B01. Seed оболочки — метаданные, не новый stochastic sampler.

> **31 октября: explicit CPU dispatch v10.** Добавлены per-step выбор CPU job,
> owner/control validation, admission diagnostics и optional Dict radio/CPU
> action space в обоих Gym facade; dynamic slots и journal/CLI/replay поддержаны.
> [Контракт](validation/REFERENCE_CPU_DISPATCH_V10.md). Новых условий 13:
> reference 249 passed, fast 110 passed, 679 collected. Скрипт без смены приоритетов
> доставляет 2 Mbit против 1 Mbit automatic CPU; replay и RL подтверждены.
> Полный regression этого пакета повторно не запускался; результат v9 ниже
> относится к предыдущему рабочему состоянию.
> Wheel smoke из /tmp с Python -I прошел; 168 модулей совпадают с исходниками,
> CPU selection replay и console verification подтверждены.
> [Проверки](acceptance/REFERENCE_CPU_DISPATCH_V10_VERIFICATION.json).

> **31 октября: calendar qualification v9.** Добавлены независимый subset-DP
> oracle с рациональной арифметикой и проверкой исполнения, matching-class regret,
> сравнение текущей политики с защитой от несопоставимой preemption, portable
> benchmark с runtime/lineage/split checks и полным recomputation.
> [Контракт](validation/REFERENCE_CALENDAR_QUALIFICATION_V9.md). Новых условий 29:
> reference 236 passed, fast 110 passed, 666 collected. Контрпример: optimum
> 4 Mbit против priority-greedy 3 Mbit по throughput objective. Это не общий
> optimum энергетических/DAG/thermal/multi-station миссий.
> Полный набор выполнен: 665 passed, один HTTP-тест получил PermissionError
> из-за запрета socket в песочнице; тот же тест отдельно прошел с loopback-доступом.
> Это не один непрерывный green-run. Суточные/трехсуточные кампании и 16 аппаратов
> на 72 часа прошли. Console entry point space-env добавлен и проверен из wheel;
> 168 модулей совпадают с исходниками. Проверен предел oracle 16 jobs/65536 states.
> [Результаты](acceptance/REFERENCE_CALENDAR_QUALIFICATION_V9_VERIFICATION.json).

> **31 октября: processing dependencies v8.** Добавлены same-actor зависимости,
> проверка циклов/ссылок, fan-in/fan-out, удержание входных байтов, динамическое
> добавление и cancellation semantics, граф в Gym и planner/replay integration.
> [Контракт](validation/REFERENCE_PROCESSING_DEPENDENCIES_V8.md). Добавлено 16
> тестовых условий: reference 207 passed, fast 110 passed, 637 collected.
> Пример: 6 Mbit в срок; родитель обработан в t=1, ребенок начинает в t=1,
> replay подтвержден и RL 9 шагов. Полный pytest и большие нагрузки не запускались.
> Wheel smoke прошел из /tmp с Python -I; 166 модулей совпадают с исходниками.
> [Проверки](acceptance/REFERENCE_PROCESSING_DEPENDENCIES_V8_VERIFICATION.json).

> **31 октября: onboard processing v7.** Реализованы длительность и очередь CPU,
> энергозатраты и workspace, readiness/downlink gating, pause/preemption,
> отмена и rollback, planner/Gym/CLI/replay integration. Это in-place surrogate
> над уже имеющимися байтами, не реальный алгоритм обработки или съемки.
> [Контракт](validation/REFERENCE_ONBOARD_PROCESSING_V7.md). Добавлено 15 тестовых
> условий: reference 191 passed, fast 110 passed, 621 collected. Пример: 6 Mbit
> в срок, CPU 3 s на аппарат, replay подтвержден, RL 10 шагов.
> Wheel smoke прошел из /tmp с Python -I; 165 модулей совпадают с исходниками.
> [Результаты](acceptance/REFERENCE_ONBOARD_PROCESSING_V7_VERIFICATION.json).
> Полный pytest suite и большие нагрузки этим пакетом не квалифицировались.

> **31 октября: transmitter health v6.** Добавлены opt-in однозвенный thermal
> surrogate с trip/resume hysteresis, суммарная TX quota, аналитические границы
> нагрева/охлаждения, heat ledger, quota-aware selection и передача открытой
> станции другому аппарату при перегреве. Оба Gym facade и replay учитывают
> новое состояние. [Контракт](validation/REFERENCE_TRANSMITTER_HEALTH_V6.md).
> Добавлено 15 тестовых условий: reference 176 passed, fast 110 passed,
> 606 collected. Пример: 5 Mbit в срок, TX=4 s, RL=18 шагов, replay подтвержден.
> Wheel smoke из /tmp с Python -I прошел, 165 модулей совпадают с исходниками.
> [Результаты проверок](acceptance/REFERENCE_TRANSMITTER_HEALTH_V6_VERIFICATION.json).
> Полный pytest, реальная thermal-калибровка и большие нагрузки не запускались.

> **31 октября: ground booking v5 — 12 задач.** Реализованы бронирования терминалов,
> station/single-radio conflicts, сохранение idle-reserved capacity, exclusive
> policy, active/future cancellation, boundary-correct execution, planner/RL/
> replay/CLI integration, service windows и utilization. [Контракт](validation/REFERENCE_GROUND_BOOKING_V5.md).
> Добавлено 21 тестовое условие: reference 161 passed, fast 110 passed,
> 591 collected. Сквозной пример доставляет 5 Mbit в срок. Полный pytest suite
> и промышленные нагрузки не запускались. Wheel установлен в отдельный venv,
> smoke/replay проверены из /tmp с Python -I; 164 модуля совпадают с исходниками.
> [Результаты проверок](acceptance/REFERENCE_GROUND_BOOKING_V5_VERIFICATION.json).

> **31 октября: service lifecycle v4 — пакет из 10 задач.** Добавлены отмена,
> discard ledger, изменение deadline, ограничения скорости станции/передатчика,
> rate-aware planner, RL telemetry/cancellation mask, replay/CLI-сценарий и
> исправление численного завершения дробной передачи. [Контракт и полный список](validation/REFERENCE_SERVICE_LIFECYCLE_V4.md).
> Добавлено 18 тестовых случаев: reference 140 passed, fast 110 passed,
> 570 collected. Wheel проверен из /tmp с Python -I в отдельном venv;
> 163 модуля совпадают с исходниками, lifecycle replay smoke прошел.
> [Доказательства](acceptance/REFERENCE_SERVICE_LIFECYCLE_V4_VERIFICATION.json).
> Полный pytest suite и крупные нагрузки не запускались.

> **31 октября: dynamic reference operations v3.** Реализованы атомарные события
> поступления готовых данных, смены приоритета, отказа/восстановления станции;
> журнал /2, `reference-events` / `reference-adapt`, inventory ledger и отдельный
> fixed-capacity Gym environment с появлением продуктов внутри эпизода.
> [Контракт](validation/REFERENCE_DYNAMIC_OPERATIONS_V3.md). Добавлено 20 тестовых
> случаев: reference 122 passed, fast 110 passed, 552 collected. Wheel установлен
> в отдельный venv и smoke проверен из /tmp с Python -I; 163 модуля совпадают
> с исходниками. Полный pytest suite, acquisition physics и большие нагрузки
> этим пакетом не квалифицировались.

> **31 октября: rolling reference operations.** Выбор продукта в mission/recorder/
> Gym, восстановление проверенного prefix-журнала и перепланирование из текущего
> состояния реализованы. CLI `reference-replan` / `reference-resume` проверяет
> отпечаток исходного состояния; энергия, частично переданные данные и история
> сохраняются. [Контракт](validation/REFERENCE_REPLANNING_V2.md). Добавлено 14
> тестовых случаев: reference 102 passed, fast 110 passed; полный suite и wheel
> этого пакета не проверялись. Динамическое добавление заявок/outages и независимая
> оценка оптимальности остаются открытыми.

> **31 октября: reference scheduling.** Добавлен priority/slack greedy-планировщик
> группировки, CLI `reference-plan`, исполняемый план и повторная проверка на новой
> reference-миссии. [Контракт и следующие этапы](validation/REFERENCE_SCHEDULING_V1.md).
> Добавлено 9 тестов: reference 88 passed, fast 110 passed. Полный suite и новая
> сборка wheel в этом пакете не запускались. Оптимальность и промышленная
> квалификация планировщика не заявляются.

> **31 октября: reference operations.** Отдельный reference-fixed-downlink/1 исполняет
> передачу готовых продуктов на GCRS-эфемеридах с терминалами/outages/памятью/энергией.
> Gymnasium facade и portable replay реализованы. [Контракт](validation/REFERENCE_OPERATIONS_V1.md),
> [приёмка](acceptance/REFERENCE_OPERATIONS_V1.md). Новые15, physics79, fast110 и wheel проверки прошли.
> Полная миграция legacy MissionWorld, thermal/ADCS/RF и acquisition не объявляются закрытыми.

> **Reference access:** переносимая эфемерида, AOS/LOS, range/range-rate/Doppler и
> Boinor formula/LSODA cross-check реализованы в рабочем дереве.
> [Контракт](validation/REFERENCE_ACCESS_V1.md), [приёмка](acceptance/REFERENCE_ACCESS_V1.md).
> Новые20, physics64, fast110 и installed-wheel проверки прошли. Общие EOP/ephemerides/density явно
> отмечены; это не независимая квалификация всей миссионной физики.

> **Физический фундамент:** отдельный gcrs-reference/1 реализует двухчастное время,
> явный EOP, GCRS/ITRS r/v/covariance и DOP853. MissionWorld остаётся legacy-training/1.
> [Физический контракт](validation/GCRS_REFERENCE_V1.md), [приёмка](acceptance/GCRS_REFERENCE_V1.md).
> Новые28, physics44, fast110 и integration93 прошли; wheel проверен. Полная миграция и физическая
> квалификация возмущённых миссий не объявляются завершёнными.

> **Агентный протокол:** schemas, snapshots сессии/каналов, записываемые вызовы/ошибки,
> serial relay и verified tool-dialogue доступны в рабочем дереве.
> Исправлены общий sensor cache и выдача subsystem truth в новом протоколе.
> [Контракт и ограничения](AGENT_PROTOCOL.md), [приёмка](acceptance/AGENT_PROTOCOL_V1.md).
> Новые38, CI fast110 и интеграция82 прошли; установленные tool/relay/replay проверены.
> Это не квалификация полной физики/DTN.

> **Переносимые эксперименты:** world schema2 хранит эффективную конфигурацию и журнал
> без исходного YAML; run/1 проверяет прямые команды и экспортирует transitions JSONL.
> [SDK/CLI и ограничения](PORTABLE_EXPERIMENTS.md), [приёмка](acceptance/PORTABLE_RUN_V1.md).
> CI fast72, интеграция20 и installed-wheel проверка прошли; полный P3 этим не закрыт.

> **Следующее дополнение:** [Resource timeline v1](acceptance/RESOURCE_TIMELINE_V1.md)
> исправляет границы бронирований и отключений внутри шага. Предыдущие wheel/72ч
> результаты ниже относятся к прежнему срезу. Следующая очередь: [PRODUCT_EXECUTION_NEXT.md](PRODUCT_EXECUTION_NEXT.md).

> **Исполнение следующего пакета (рабочее дерево):** добавлены границы длительности
> команд, откат ленты истины/RNG измерений, reset параллельных эпизодов и Dict JSONL.
> Исправлен профиль CI `full`: он включает все тестовые файлы. Суточный baseline
> зафиксирован как18/32 с объяснением энергозависимости, без изменения физики.
> Проверка поставки wheel вне репозитория прошла. Полный перечень проверок и
> оставшиеся ограничения: [EXECUTION_V1.md](acceptance/EXECUTION_V1.md).
> Ожидание в DayShift/DishAward/OperatorShift теперь использует реальные команды
> зарядки/возврата в надир; семь сценариев смен и восемь проверок пульта/API прошли.
> Остаточный набор прошёл:18 тестов, включая16 аппаратов за72 часа; проверки
> выполнялись раздельными прогонами, единого полного зелёного запуска нет.
> Миграция кадров пока описана спецификацией: [FRAME_MIGRATION.md](validation/FRAME_MIGRATION.md).

> **Проверка30октября2026:** актуальная очередь — [промышленный roadmap0.134](INDUSTRIAL_ROADMAP_V0134.md).
> [Аудит](audits/V0134_STATE.md) подтверждает прогресс и фиксирует открытые duration,
> reset/restore и Dict-rollout дефекты. Все CI-профили прошли, но отдельный суточный
> тест падает:18/32 при требовании≥30. Строки verified ниже относятся к указанным
> локальным проверкам, не к полной промышленной приёмке.

> Следующие этапы после среза M1: [roadmap моделирования](MODELING_ROADMAP_V098.md).
> [Новая проверка](audits/V098_STATE.md) подтверждает прогресс ниже и фиксирует
> оставшиеся проблемы batch atomicity, request execution, acquisition scoring
> и восстановления станций. Поэтому отдельные `verified` не означают закрытие M0.

> Уточнение по аудиту 27 октября 2026: `verified` ниже относится к конкретным
> тестам, а не ко всем комбинациям плана A–H. Утечка health из прогноза и
> конфликт продолжающегося радиосеанса закрыты в 0.94.0.
> См. [аудит](audits/V093_AUDIT.md), [пакет исправлений](DEVELOPMENT_PLAN_V093.md)
> и [конкурентный план развития](COMPETITIVE_DEVELOPMENT_PLAN.md).

Пакет `0.134.0`. Короткий набор печатает строку обучения по каждому случаю. Отказ станции в повторе остаётся «станция недоступна», а не «срок вышел». Снимок после сдачи, лишний шаг ожидания и возврат снимка оставляют ту же сдачу, что непрерывный прогон. Шестнадцать аппаратов крутятся четыре шага по 30 с и доходят до 120 с без станций и без сдачи заявок. Строка обучения появляется только если повтор Кируны сошёлся. Глобус и журнал сессии читают кадры прожитого мира: два кадра после шага 30 с, в кадре вектор из трёх чисел. Живая сессия отдаёт трассу и аппараты своего мира: повторный отчёт трассу не удлиняет, шаг на 30 с добавляет точку и сдвигает широту или долготу. Выработка батарей и внешний нагрев считаются по отсчётам шага вместе с косинусом в каждой точке. Поток на батарею — среднее по отсчётам шага: вход в тень не обнуляет освещённую часть. Тень, световое давление, освещённость и прилив читают одно Солнце ERFA epv00. Узел каталога остаётся на ряде Миуса, потому что записанные цели лежат на этой трассе: сдвиг узла на epv00 оставляет короткий набор на 5 из 12. Короткий набор снова сдаёт 12 заявок. Параллельный флот видит у каждого аппарата только его заряд и контакт. Совместный шаг сдачи совпадает с прямым шагом мира. Бюджет в два вызова останавливает третий. Реестр помечает `mission_holdout` отдельно от Кируны. Ступени 16/100, 64/1000 и 256/10000 делают шаг 1 с без станций: сданных заявок 0, это не сутки и не 72 часа. Колесо `0.127.0` установлено в отдельное окружение, `validate` читает из него Кируну. CI гоняет профили fast, reference и full. Короткий набор сдаёт 12 заявок.

| Проверка | Статус | Доказательство |
| --- | --- | --- |
| Прогноз не списывает импульс и не съедает будущее событие | verified | `tests/test_mission_contract.py` |
| Шум после restore повторяет тот же поток | verified | тот же файл |
| Неизвестный аппарат и разъезд часов не двигают мир | verified | тот же файл |
| Продолжение плана идёт от общих часов, а не с нуля | verified | тот же файл |
| Одна станция не сдаёт два аппарата в один шаг | verified | мир и `FleetTasking-v0` |
| Отказ станции останавливает уже включённый передатчик | verified | квитанций внутри закрытого интервала нет |
| Повтор одной квитанции не закрывает заявку | verified | объединение отрезков |
| Квитанция раньше выпуска не считается | verified | тот же тест |
| Нарушение ограничения увеличивает safety | verified | низкий заряд |
| Кривой сценарий отвергается до пропагации | verified | горизонт, NaN, координаты, срок |
| Среда одного аппарата и среда двух аппаратов | verified | `TaskingShift-v0`, `FleetTasking-v0` |
| 72 часа, 4 и 16 аппаратов | verified | Заявки сдаются. Счётчик нарушений больше нуля: долгий полёт реально задевает ограничение, ноль больше не подставляется. |
| Команда `python -m space_env.cli validate` | verified | читает файл без полёта |
| Снимок JSON и повтор шага в новом процессе | verified | `tests/test_mission_roundtrip.py` |
| Повтор команды и отмена | verified | тот же файл |
| Станция, качество и статус заявки | verified | чужая станция и partial не закрывают заявку |
| Короткая миссия на 12 заявок и суточная на 32 | verified | два семени дают разные координаты |
| `MissionFleet-v0` видит сроки и маски | verified | неверная съёмка отклоняется |
| CLI, API и отчёт дают один исход Кируны | verified | `kiruna-pass` |
| Повтор запечатанного прогона | verified | записанные команды дают те же квитанции |
| Короткая миссия из 12 заявок в CLI, Gym и API | verified | один список сдачи и причин. На шаге 60 с в срок уходят все 12 |
| Квитанция хранит интервал передачи | verified | `start_sec` ≤ `end_sec` = `time_sec`, и не раньше появления кадра |
| Раннее окно снимается раньше более позднего | verified | полоса радара над Матерой не перескакивает ближайшую точку |
| Конец прохода не стирает уже открытую сдачу | verified | радио, закрывшееся внутри шага, оставляет квитанцию за открытый префикс |
| Сутки, 4 аппарата, 32 заявки | verified | На шаге 120 с в срок 18 из 32. Остальные 14 — «срок вышел». Срез 0.119.0 это число не сдвигает |
| Шестнадцать аппаратов, 16 заявок на реальных проходах | verified | 15 из 16 сданы в срок на шаге 60 с. Не сдана `fleet-02` |
| Три семьи: обычная, отказ станции, ранний срок | verified | причины различаются; обычная повторяется по записанным командам |
| Отложенное семя не равно учебной короткой миссии | verified | `mission_holdout.yaml` |
| Одна книга команд для плана, повтора и среды | verified | тот же id не меняет тело; вторая сдача на одну станцию отклоняется |
| Студия показывает статус заявки и причину | verified | список в `#mission-statuses` |
| Прогноз отказа не оставляет прибор сломанным | verified | health, очередь восстановления, ориентация и RNG входят в снимок |
| Уже включённое радио занимает станцию | verified | второй аппарат не сдаёт параллельно с armed-сеансом |
| Шаг не выходит за горизонт | verified | `world.step` длиннее остатка отвергается до движения |
| Разные снимки не складываются в одну заявку | verified | два `product_id` по половине объёма не закрывают заявку |
| validate отвергает модель, эпоху и NaN-отказ | verified | до сборки движка |
| Книга команд живёт в мире | verified | тот же id не исполняется второй раз, другое тело отвергается до шага |
| Ретрансляция идёт на укороченном шаге | verified | полудуплекс не совмещает сдачу и ретрансляцию; книга занятости переживает новую Fleet |
| Повтор прогона сверяет причины | verified | `unmet` входит в сравнение replay |
| Сбой одного аппарата откатывает весь шаг | verified | часы обоих бортов остаются на месте |
| Семантический отпечаток не зависит от комментариев | verified | сырой sha256 разный, semantic одинаковый |
| Две сессии миссии не делят часы | verified | устаревшая ревизия отвечает 409 |
| FleetWatch-v0 принимает 1–4 аппарата | verified | 16 аппаратов отвергаются до сборки мира |
| Короткий набор миссий | verified | `python -m space_env.cli benchmark mission_v1_smoke.yaml`: обычная сдача, отказ станции, ранний срок и 12 из 12 |
| Студия и сценарии входят в колесо | verified | `studio_assets` и `configs/benchmark_suites` собираются hatchling |
| Конфликтная пачка команд не оставляет первую запись | verified | одинаковый id с разным телом не меняет книгу |
| Неизвестная заявка не становится выполненной | verified | статус `rejected`, причина «неизвестная заявка» |
| Ранняя полная съёмка закрывает заявку | verified | поздний больший продукт не отменяет сдачу в срок |
| Снимок JSON сохраняет Кируну | verified | после load станции не пустые |
| Плохой шаг сессии не пишет команду | verified | ревизия и книга остаются прежними |
| Стадии RK4 не делят один момент | verified | моменты 0, h/2, h/2 и h; широта стадии пересчитывается, когда плотность её читает |
| Короткий эталонный отчёт | verified | `reference_report`: позиция, скорость, сходимость шага, полдень J2000, внутренняя точка тени, конверт расстояния Луны, пушечное ядро, импульс, граница сеанса. Допуски из профиля. Эйлер на той же дуге допуск 50 м не проходит |
| Подготовка съёмки и занятый передатчик | verified | `prep_sec` 15 с не открывает затвор в первые 10 с и открывает на следующем шаге. При `exclusive_with_downlink` включённое радио и плановая экспозиция не принимают сдачу. Пустая батарея не начинает съёмку и радио |
| Конечное включение отдельно от импульса | verified | Без тяги, Isp и сухой массы манёвр мгновенный и масса не меняется. С ними импульс набирается по шагам, масса убывает, снимок возвращает недожжённый остаток. Нехватки топлива и обрывка карточки двигателя импульс не подменяют |
| Очередь памяти | verified | По умолчанию лишний кадр режется, а продукт доступен сразу. `drop_oldest` снимает целый старый продукт. Доля сжатия уменьшает записанный объём. Очередь обработки не отдаёт биты в радио раньше готовности и списывает свою мощность |
| Разворот с ускорением | verified | Нулевое ускорение по-прежнему сразу берёт предельную скорость. Заданное ускорение за первую секунду даёт 2 °/с и 1 ° пути, а не предел 10 °/с. Новая команда гасит набранную скорость. Съёмка молчит, пока остаток больше `slew_settle_deg` |
| Гистерезис питания | verified | Нулевые пороги не включают безопасный режим. Заряд не выше входа переводит в safe, гасит радио и импульс и не принимает работу. Между порогами режим держится. На выходе возвращается прежняя нагрузка |
| IAU 2000A | verified | `gast_rad` совпадает с ERFA `eraGst00a` и крутит ECEF. Десять членов IAU 1980 остаются в `gast_iau1980_rad` |
| Центрированный F10.7 | verified | 81 сутки вокруг даты, если в снимке есть 40 суток с каждой стороны. На 20 октября 2026 функция возвращает пусто, учебный индекс остаётся скользящим средним |
| Оболочка 66 и линия 1 Вт | verified | 66/6/2 на 780 км. Ближайшая пара 797 км, замыкание линии 1 Вт ближе 343 км, замкнутых линий 0 |
| Срок сведения | verified | Оценка H/\|da/dt\|. Низкая лёгкая орбита короче года, высокая плотная длиннее 25 лет. Это не интегрирование на 25 лет |
| Спрос вне удобного прохода | verified | Точки семени не выбираются у открытой станции. Полоса 5000 км длиннее кадра и недостижима. Доля срока считается и от всего выданного спроса |
| Качество оптического кадра | verified | Облачность 0.8 не создаёт продукт, 0.5 даёт partial, радар облачность не наследует. Ясная цель по-прежнему full. Короткая миссия сдаёт 12 заявок |
| Подготовка антенны | verified | `setup_sec` 0 отдаёт биты в том же контакте. 20 с не отдают ничего в первые 15 с и начинают после 20 с. Снимок возвращает момент готовности |
| Несколько терминалов, частота, TTL и custody | verified | Один терминал оставляет второго без сдачи, два терминала сдают оба. Чужая несущая не даёт контакт. TTL 3 с сжигает продукт до прохода. Ack 4 с ставит квитанцию позже конца передачи. Повтор не плодит квитанцию. Короткая миссия сдаёт 12 заявок |
| Измерение отдельно от истины | verified | Отказ payload виден на борту и отсутствует в пакете. Задержка 10 с отдаёт старый заряд. Период держит прежний отсчёт. Пропуск не выдумывает число. Один seed даёт один поток. Сдача и safety лежат в разных полях |
| Инструменты агента и три политики | verified | forecast не сдвигает часы. cancel снимает pending. Неизвестная заявка rejected. Ревизия 0 после шага отвечает ошибкой. fifo берёт ранний выпуск, priority и deadline — срочную. База на family_nominal делает 4 шага |
| Геометрия заявки | verified | Полоса длиннее кадра, полигон шире кадра, мало повторов, нет стереопары и несданный предшественник дают разные причины. Старый файл Кируны остаётся точкой |
| Операционный SAR | verified | Угол падения 90°−угол места. Ниже горизонта, сверх скважности и без энергии взгляд не начинается и ресурс не тратится |
| Следующий хоп DTN | verified | Хоп идёт на короткую очередь по замкнутой линии. Выключенное ребро меняет хоп. Больший приоритет уходит первым. Повтор того же интервала биты не двигает |
| Знание орбиты | verified | Сдвиг 1 км вдоль трека меняет дальность сильнее, чем поперёк, больше чем в 5 раз. `operational_od` остаётся ложью и для двух станций |
| Уклонение и компенсация торможения | verified | Нехватка резерва оставляет промах и Pc. Принятый импульс 0.9 м/с снижает Pc. Сутки низкого лёгкого аппарата дороже 1 м/с и дороже высокого тяжёлого больше чем в 10 раз |
| Экспозиция по геометрии заявки | verified | Точка по-прежнему пишет кадр. Полоса 5000 км, широкий полигон, несданный предшественник, один проход повтора и одна стереопара кадр не пишут. Второй проход повтора и второй угол стереопары пишут |
| Скважность радара в шаге | verified | Доля витка 1e−8 не пишет кадр и не копит ресурс. Полная доля пишет и копит. Энергия выше заряда отказ даёт до траты скважности |
| Погода линии и цепочка DTN | verified | 80 дБ оставляют контакт и обнуляют сдачу. Два хопа доходят до третьего аппарата. Выключенное ребро останавливает цепочку. Подтверждённая опека не возвращается. Просроченная возвращается только по живой линии |
| Скрытый отказ | verified | До инжекта сдача совпадает с исправным бортом. После него скорость 0.25, health остаётся nominal, причина отсутствует в измерении. Ложный safe 15 с не пишет кадр и не сажает заряд; снятие открывает съёмку следующим шагом |
| Уклонение в миссии | verified | Промах сообщения 200 м. Принятые 0.8 м/с оставляют 79.2 м/с и меняют численный промах больше чем на 1 м. Резерв 79.5 м/с оставляет 80 м/с. После траты манёвр 79.5 м/с не проходит |
| Расписание компенсации торможения | verified | Сутки низкого лёгкого аппарата при бюджете 1 м/с не ставят включений. Высокий тяжёлый ставит четыре, сумма меньше 1 м/с. Включение в шаге съёмки кадр не пишет. Резерв на весь остаток план не добавляет |
| Книга станции и общий модем | verified | Один терминал берёт приоритет 5 и отвергает пересечение. Два терминала и соседние окна принимают оба. Пустое окно отвергается. Шаг с книгой сдаёт только аппарат внутри окна, причина «вне окна». Общий модем отдаёт каждому половину одиночной сдачи |
| Фазирование вдоль трека | verified | Импульс равен ошибке / (3T). Вековой член на горизонте сходится до 0.0001 м. 100 км и горизонт 5000 с импульс не тратят. Принятое включение меняет скорость вдоль трека больше чем на 0.5 м/с. Резерв 79 м/с оставляет бюджет |
| Ёмкость прохода | verified | Запас 10 дБ за 5 с — 5 Мбит, запас 1 дБ — 1.25 Мбит. Контакт на 10° сдаёт меньше половины зенита за ту же секунду. Сданные биты не выше ёмкости модема |
| Смаз и запись прогона | verified | Порог 0 не портит кадр. 2 °/с при пороге 1 °/с дают partial, радар и облачность 0.8 этот порог не переписывают. Два одинаковых шага дают одну запись, ожидание вместо сдачи — другую |
| Деградация, длительность и облако | verified | Деградировавшая связь сдаёт половину. Деградировавшая камера пишет partial. Команда на 10 с идёт два шага и на третьем гаснет; отмена оставляет уже сданные биты. Одинаковое семя повторяет облако. Выдержка 5 с не пускает пачку дальше |
| Узкое окно, дуплекс и семена | verified | rolling берёт окно 30 с против срока 100 с и уступает приоритету только внутри своего правила. Полудуплекс соседу не отдаёт, полный дуплекс отдаёт. Три семени дают долю 1/3, сбой остаётся в строке. Варианты одного задания в одном куске |
| Переход TEME | verified | При общих UT1 и полюсе ITRS совпадает с astropy до 0.001 м. Численный вектор отличается от сырого TEME больше чем на 1 м. `use_tle` собирает GCRF. Сырой TEME в `SimulationEngine` по-прежнему отвергается |
| Схождение и наблюдаемость | verified | Уравнение da/dt с плотностью по высоте сводит 200 км / 10 кг / 1 м² меньше чем за год и не сводит 800 км / 1000 кг / 0.01 м² за 25 лет. Угол видит поперечные 1 км, дальность меняется меньше чем на 50 м. Чен и интеграл по диску расходятся меньше чем на 5 % |
| GCRS и Луна IAU | verified | TEME→GCRS через c2t06a совпадает с astropy до 0.001 м и отличается от численного кадра больше чем на 1 м. Прилив читает moon98. Миус и IAU оба в конверте 350–410 тыс. км, разница 0.1–20 тыс. км. Солнце epv00 и Миус расходятся меньше чем на 1° |
| Прилив Солнца и живой хоп | verified | Ускорение прилива совпадает с Луной IAU и Солнцем epv00 и не совпадает с Солнцем Миуса. Два живых хопа по 30 с ставят часы на 60 с и сдвигают аппарат больше чем на 1 км. Замороженная цепочка часы не двигает |
| Интервал терминала, снимок и полоса | verified | Пересечение 0–8 и 4–10 с одним слотом оставляет второму 8–10. Снимок возвращает цель после очистки. Полоса 10 км пишет покрытие 10 км, полоса 5000 км кадр не пишет, полигон из трёх вершин пишет три. Повтор сверяет конечный заряд и сдачу. Выключенное ребро уводит живой маршрут на другой аппарат |
| Сессия, набор и живая студия | verified | Два шага с ревизией 0 оставляют ревизию 1 и время 30 с. Сходящийся повтор входит в набор, подмена сдачи остаётся в отказе. Варианты одного задания в одном куске. Страница студии открывает сессию и делает шаг |
| Словарная среда и совместный шаг | verified | `MissionDict-v0` наблюдает словарь clock, contact, open_requests, soc. Шаг ожидания двигает долю времени. Совместный шаг совпадает с прямым шагом мира, оба аппарата сдают биты. Замер 4 аппаратов пишет время стены и шаг 1 с |
| Параллельный флот, бюджет и лестница | verified | Наблюдение аппарата `a` не содержит `b`. Награды совпадают со сданными битами прямого шага. Третий вызов при бюджете 2 получает «бюджет вызовов исчерпан», часы остаются на нуле. 16, 64 и 256 аппаратов проходят 1 с, сданных заявок 0. Holdout отмечен в реестре. Профили CI указывают на существующие файлы |
| Колесо вне дерева | verified | Сборка `space_env-0.132.0` поставлена в `/tmp/space-env-clean-132`. Импорт идёт из этого окружения. `validate` по файлу из колеса печатает `sentinel-a` и `kiruna-pass`. Страница студии в колесе содержит живую сессию |
| Одно Солнце для тени и сил | verified | Тень аппарата на антисолнечной стороне обнуляет световое давление, освещённая сторона толкает от Солнца. Прилив совпадает с этим вектором. Узел каталога совпадает с рядом Миуса и расходится с epv00. Короткий набор сдаёт 12 заявок |
| Поток по отсчётам шага | verified | Шаг, который начинается на свету и кончается в тени, оставляет заряд выше полностью тёмного и ниже полностью освещённого. Короткий набор сдаёт 12 заявок |
| Косинус и тепло по отсчётам | verified | Надирные панели, которые за шаг уходят от Солнца, оставляют заряд между аппаратом у терминатора и аппаратом, который остался на ярком участке. Короткий набор сдаёт 12 заявок |
| Трасса живой сессии | verified | Отчёт сессии содержит аппарат и одну точку трассы. Шаг на 30 с добавляет точку с тем же именем и сдвигает координату больше чем на 0,05°. Вторая сессия остаётся на нуле. Повторный отчёт трассу не растит |
| Глобус сессии | verified | После шага 30 с в эпизоде два кадра, времена 0 и 30 с. В кадре у `sentinel-a` вектор из трёх чисел. Радиус Земли в эпизоде больше 6 000 км. Подпись эпизода — `session` |
| Снимок, минуты и строка обучения | verified | Возврат снимка после лишнего ожидания оставляет ту же сдачу и те же часы, что непрерывные 10 с. 16 аппаратов за четыре шага доходят до 120 с, сданных заявок 0. Чужая запись строку обучения не даёт. Сошедшаяся Кируна даёт `kiruna-pass` |
| Набор из коротких миссий | verified | Четыре случая печатаются и проходят повтор. Обычная сдаёт `kiruna-pass`, отказ станции оставляет эту заявку в unmet, ранний срок не сдаёт ничего, учебные 12 заявок сданы и есть в командах |

Не входит в этот пакет: Kubernetes, авторизация, полный набор данных для обучения. Библиотека PettingZoo не ставится: параллельный шаг повторяет её контракт на том же мире. Лестница 16/64/256 записана на шаге 1 с без станций и без сдачи заявок, не на 24 и не на 72 часа. Pc без независимого эталона. Переход TEME не считается. Учебный профиль не переносит допуск 50 м reference-дуги на сутки и 72 часа. По умолчанию съёмка совмещается со сдачей и начинается сразу: это не паспорт конкретного аппарата. Сжатие — доля потока, не кодек. Ускорение разворота и пороги безопасного режима по умолчанию нулевые. Линия 1 Вт на оболочке 66 не замыкается, пороги ради этого не поднимались. Интегратор остаётся в кадре звёздного времени и полюса: перевод того же вектора через c2t06a сдвигает наземную точку примерно на 45 км и снял бы записанные проходы. Центрированный F10.7 на 20 октября пуст. Линия 1 Вт на оболочке 66 разомкнута. Суточная сдача остаётся 18 из 32.

B04-D1 validation: 39 новых cases, reference 598 passed (13 warnings), fast 110 passed (2 warnings), 1028 collected. Counts overlap; full regression не перезапускался. Wheel: 190/190 Python modules и 3 frozen fixtures. Chemical/electric installed CLI validate/run/resume/replay verified; обе миссии 2/2 delivered requests.

B04-D1 isolated extended SDK verification passed: chemical/electric, mid-gimbal и mid-firing restore, balances и console replay.

B04-D2a validation: 21 новых cases; reference 619 passed (13 warnings), fast 110 passed (2 warnings), 1049 collected. Counts overlap; full regression не запускался. Wheel 191/191 Python modules byte-matched, 3 frozen fixtures; isolated standalone demo passed.

Расширенный isolated installed SDK/CLI smoke также passed, включая прежние chemical/electric missions и mid-gimbal/mid-firing replay. Это проверка совместимости, не интеграция rigid kernel с MissionRuntime.
