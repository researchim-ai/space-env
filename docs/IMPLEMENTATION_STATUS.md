# Статус пакета исполнения миссии

## Текущий пакет — B11-Q5-D2a: независимая геометрия и capacity audit

Добавлен отдельный fixed-ephemeris delivery compiler: independently specified
initial planes/phases/radii и synthetic fixed WGS84 sites. Станции не строятся
из subsatellite point; все actor×station products и zero-contact pairs сохраняются.
Профиль ограничен ready onboard data, constant-rate geometric downlink и shared
station terminals. Closed-loop EO/pointing этим не объявляется проверенным.

Conditional fractional delivery relaxation учитывает actor single-TX и station
slots. Weak-duality bound пересчитывается exact Fraction arithmetic для binary
coefficients, отдельно от solver status и native conservative sanity ceiling.
Native witnesses проверяют receipts/contact/outage/rate, occupancy, product bits,
energy/inventory и explicit minimum service. Diagnostic LP не передаётся policy.

Контракт: [INDEPENDENT_GEOMETRY_V1](validation/INDEPENDENT_GEOMETRY_V1.md).
Приёмка: `acceptance/B11_INDEPENDENT_GEOMETRY.json`. C2 measured calibration,
D2b coupled EO/pointing, Q6 benchmark и Q7 durable/OS isolation открыты.


### Приёмка D2a — 2026-11-04

30 новых geometry/capacity tests,110 targeted geometry/restart/ensemble,
1611 reference и110 fast прошли; наборы пересекаются. 2041 collected,
expanded full suite не запускался. Reference прошёл одним запуском с localhost
permission для существующего HTTP test.

Installed small/full suites проверили2/4 actors,2 planes и6000s:
nominal3/6 и6/12 delivered соответственно; outage/no-contact0 retained.
Stress с1000Mbit products и той же геометрией: priority baseline1/12 при одном
terminal и2/12 при двух. У stress explicit min service0: это observed behavior,
не экспертный SLA. Все16 logical native executions,8 separate-process restores
и semantic pair comparisons прошли объявленные gates.

253 Python modules и286 source/data files byte-matched; wheel SHA:
`e17b266d3557c71446e48d8dc8194a3ace85fb62928b3c0c63b825c93628d66a`.
Installed compiler/bundle audit и прежние materials34/calibration14/pointing612/
envelopes119 прошли. Venv system-site-packages, без dependency-clean/remote CI.
[Приёмка](acceptance/B11_INDEPENDENT_GEOMETRY.json) сохраняет полные native
результаты, exact input/certificate/packet bundles и19 linked artifact hashes.

## Предыдущий пакет — B11-Q5-D1b: process-isolated checkpoint continuation

Добавлен opt-in resume_in_new_process для ScaleSpec/ensemble: prefix producer
успешно выходит до запуска другого interpreter для continuation. Packet связан
с workload/scenario/task/seed/policy, code/data/Python/dependency identity и nonce.
Parent-observed PIDs/exit codes сверяются с consumer evidence и отдельным
native-integrity gate. Restore использует audited prefix replay существующего
PlanningRun API, без передачи Python graph/caches и без pickle.

Deadline общий для двух фаз; byte-budget, corrupted packet, terminated prefix,
worker failure и timeout остаются error rows. Restart packets/progress сохраняются
и проверяются offline audit. Default false/legacy serialized inputs сохранены.
Это recovery evidence конкретных frozen workloads, не durable hosted job queue.

Контракт: [PROCESS_RESTART_V1](validation/PROCESS_RESTART_V1.md).
Приёмка: `acceptance/B11_PROCESS_RESTART.json`. C2, independent closed-loop D2,
Q6 benchmark/teacher и Q7 hard quotas/durable jobs остаются открытыми.


### Приёмка D1b — 2026-11-03

41 новый process test, 91 targeted process/ensemble/progress, 1581 reference и
110 fast прошли (наборы пересекаются), 2011 collected. Reference прошёл одним
запуском с разрешённым localhost HTTP; expanded full suite не запускался.

Установленный wheel проверен на6 пакетах: nominal EO, prefix после3 decisions,
random-masked policy continuation, structural rejection, prefix byte-budget failure
и shared timeout. Всего30 logical native attempts и9 successful process-boundary
restores. Nominal/midflight cases выполнили обе заявки во всех direct/restored
вариантах. Random probe имеет explicit min completion0 и проверяет RNG/continuation
integrity, а не экспертность. Отрицательные qualification сохранили passed=false.

Сверены251 Python modules и284 source/data files; wheel SHA:
`9e6ffa7b22eb130ecdb107d1acc97d0c4837a141c1987a7f6c57f83fe6994412`.
Все6 offline/CLI audits прошли; прежние materials34/calibration14/pointing612/
envelopes119 также прошли на этой сборке. Venv system-site-packages, без clean
remote CI и без qualification independently located large mission.

[Приёмка](acceptance/B11_PROCESS_RESTART.json) содержит exact-byte packet/progress
bundles, negative outcomes, source hashes и test logs. Исторические receipts
сохранены; обновлённый код ими автоматически не квалифицируется.

## Предыдущий пакет — B11-Q5-D1: native qualification ансамблей параметров

Sensitivity compilation теперь связана с isolated scale runner: frozen baseline,
каждый structural candidate, repeats, warm/cold restart comparison и стандартные
native replay/quality/resource/completion gates. Все structural rejections,
worker errors, timeouts и missing rows остаются в общем результате; baseline
отделён от variant denominator. Completion lower bound требуется задать явно.

Offline verifier сверяет frozen compilation, artifact identity, native manifest,
progress и агрегацию raw execution rows; это не новый запуск физики и не подпись
автора. Конкретные масштабы workload/budgets заданы до исполнения. Проверка
отрицательной qualification не превращает её в passed mission.

Контракт: [NATIVE_ENSEMBLE_QUALIFICATION_V1](validation/NATIVE_ENSEMBLE_QUALIFICATION_V1.md).
Приёмка: `acceptance/B11_ENSEMBLE.json`. Q5-D2 independently located closed-loop
missions, C2 measured hardware calibration и Q6 teacher/holdout benchmark открыты.


### Приёмка D1 — 2026-11-03

39 новых targeted tests, 1540 reference и 110 fast прошли; наборы пересекаются.
1970 collected; expanded full suite не запускался. Reference прошёл в одном
запуске с разрешённым localhost HTTP для существующего теста пульта.

Установленный wheel проверен на четырёх пакетах: positive (6 native executions,
baseline + 2 variants, completion gates passed), structural rejections (2 baseline
executions + 2 retained rejected variants), checkpoint-budget failure (6 executions,
service completed but budget gate failed), subprocess timeout (6 retained failures).
Всего20 executions. Offline audit всех четырёх пакетов прошёл; отрицательные
qualification сохранили passed=false и CLI verify exit1.

Прежние numerical/workflow проверки также прошли: materials34, calibration14,
pointing612, envelopes119. Wheel SHA:
`f4ba929c97778ab6b4f7ffe02beaeb7299285e5cd26afbc0f23c52722d42f7a3`.
250 Python modules / 283 source-data files byte-matched. Venv system-site-packages;
remote clean CI, dependency-clean установка и большая independently located
миссия на этой сборке не запускались. Restore выполняется внутри отдельного
worker; OS-процесс между prefix и continuation не меняется.

[Приёмка](acceptance/B11_ENSEMBLE.json) содержит полные результаты, исходные
frozen manifests/progress/receipts в UTF-8 bundles, test logs и hashes.

## Предыдущий пакет — B11-Q5-C1b: material cp(T) и энтальпия тепловой сети

Добавлен opt-in material_capacity для узлов: опубликованная корреляция NIST
Al6061-T6, масса и homogeneous-lump assumption задаются явно. C(T) применяется
на каждом solver step, stored energy — через m∫cp(T)dT в kernel, bus balance
и observations. Диапазон 4–300 K не экстраполируется; native domain failure
откатывает всю транзакцию. Постоянная C сохраняется для прежних сценариев.

Frozen stdlib Decimal/Simpson oracle отделён от native Horner/QUADPACK/ODE:
34 numerical comparisons, плюс native replay, paid heater, PV/battery energy
и domain rollback tests. Источник — material correlation, не новый measurement
series: Q5-C2 аппаратная калибровка и Q5-D whole-mission qualification открыты.
Контракт: [MATERIAL_ENTHALPY_V1](validation/MATERIAL_ENTHALPY_V1.md).
Приёмка этой сборки: `acceptance/B11_MATERIALS.json`.


### Приёмка C1b — 2026-11-03

31 новый material test и 158 targeted material/thermal/calibration tests прошли;
110 fast прошли, 1931 collected. Наборы пересекаются. В reference-профиле
1500 passed и один PermissionError на bind локального HTTP-сервера; именно этот
тест повторён с разрешением локального сервера и прошёл. Все 1501 distinct
reference tests проверены, но единого полностью зелёного reference-run не было.
Оба исходных результата сохранены в acceptance artifacts.

Установленный wheel прошёл 34 material comparisons, native cold replay/domain
rollback и предыдущие 119 envelope / 612 pointing / 14 calibration checks.
249 Python modules и 282 source/data files byte-matched. Wheel SHA:
`eddf4f1947c244349fc957887ca33ce8825a3e63a28867a461f76ea555560687`.
Venv system-site-packages; clean dependency install, remote CI, expanded full
suite и большая миссия на этой сборке не запускались.

Приёмка: [B11_MATERIALS.json](acceptance/B11_MATERIALS.json), включая source audit,
полные numerical reports, native scenario/prefix/final и журналы HTTP recheck.

## Предыдущий пакет — B11-Q5-C1: hardware evidence и conditional calibration workflow

Добавлены шесть manufacturer reference-point profiles с primary PDF hashes,
условиями и SI; native wheel/solar adapters оставляют host assumptions явными.
Три identifiable scalar fits требуют typed experiment/frame/units, known-noise
условий, canonical measurement export и независимых train/holdout experiments.
Synthetic/declared measured origin не превращается автоматически в calibration.

Counter-keyed sensitivity ensembles сохраняют все draws/rejections, проверяют
structural native contracts и снимают stale calibration label после perturbation.
Native execution/teacher quality такой manifest не утверждает. 73 новых targeted
checks проверяют provenance, leakage, units, bounds, failure retention и CLI.

Контракт: [HARDWARE_EVIDENCE_CALIBRATION_V1](validation/HARDWARE_EVIDENCE_CALIBRATION_V1.md).
Приёмка: `acceptance/B11_CALIBRATION.json`. Q5-C2 реальные измерения/error model/
coverage и Q5-D closed-loop coupled mission qualification остаются открытыми.


### Проверенная приёмка C1 — 2026-11-03

73 targeted tests, 1470 reference и 110 fast прошли; наборы пересекаются.
1900 тестов собрано, полный расширенный suite не запускался. Установленный wheel
прошёл 14 workflow checks, предыдущие analytic/pointing проверки, три native
sensitivity-варианта с cold restore/replay и ground smoke с verified RL/SFT.
Отдельный witness подтвердил изменение реальной angular velocity при изменении
инерции и применение battery capacity/solar efficiency в загруженных миссиях.

Wheel SHA: `ac8b13750bfc4e2dc3b177893cc5d49a59ff56233952e3e020ad79cb113efc45`;
247 Python modules и 279 source/data files сверены. Venv использует system site
packages: dependency-clean/remote CI и большая миссия на новой сборке не проверены.
Оригинальные два PDF сверены по bytes SHA; это спецификации, не измерительные ряды.

Отчёт: [B11_CALIBRATION.json](acceptance/B11_CALIBRATION.json).
Следующий исполнимый план: [OPEN_DATA_CALIBRATION_NEXT.md](OPEN_DATA_CALIBRATION_NEXT.md).
Q5-C2 и Q5-D открыты; hardware calibration/industrial readiness не заявляются.

## Предыдущий пакет — B11-Q5-B: off-axis pointing, actuation и fixed-site contacts

Добавлены 16 frozen cases и 612 численных сравнений: symmetric gyrostat вне
главной оси, signed wheel saturation/reversal, asymmetric external unloading,
centered/offset rotating burns и восемь комбинаций двух orbital planes с четырьмя
заранее заданными станциями. Есть семь окон и четыре корректных no-contact случая.

CLI `reference-qualify-pointing` пишет ошибки, предзаданные budgets, точные
input/time domains, source provenance и unqualified gaps. Oracle отделён от
production: analytic matrix solution, independent matrix/Cartesian RK4 и
contact bisection; Earth frames/EOP явно общие. Wheel saturation timing и
refinement проверяются отдельно. Это numerical verification объявленных моделей,
не hardware calibration или qualification arbitrary mission/controller.

Приёмка Q5-B: **1397 reference**, **110 fast**, **52 новых targeted tests**,
**1827 collected**; наборы пересекаются. Усиленная проверка всех границ fixed-site
контактов также прошла отдельно. Final installed wheel: **241/241** modules и
fixture побайтно совпали, README актуален; **612 Q5-B** + **119 Q5-A** numerical
checks и existing external J2/ERFA проверки прошли. Chemical/electric/off-center
mission demos, nominal/limited RCS, replay и verified RL/SFT прошли. Limited
RCS сохраняет ожидаемый failed unloading operation (service 2/2), а не превращает
его в completed. Venv system-site-packages; remote clean CI/expanded full и
большая миссия на новой сборке не запускались.

Контракт:
[POINTING_ACTUATION_CONTACTS_V1](validation/POINTING_ACTUATION_CONTACTS_V1.md).
Текущая приёмка — `acceptance/B11_POINTING.json`. Q5-C calibration и Q5-D реальные
closed-loop mission workloads остаются следующими уровнями.

## Предыдущий пакет — B11-Q5-A: независимые аналитические векторы

Добавлены 10 frozen cases: две наклонные эллиптические орбиты, conduction с
нулевым/ненулевым нагревом, radiative cooling, principal-axis coast/wheel/external
торк и S/X-band link budgets. CLI `reference-qualify-envelopes` измеряет 103
проверки, `--long` — 119 с дугами до 72 часов и отдельной Hermite interpolation
ошибкой; `--external` дополнительно исполняет существующие J2/ERFA проверки.

Заданы заранее фиксированные допуски, pinned input/time domains, offline stdlib
генератор, отчёт с provenance/checks/build inventory/evidence registry. Подписанный
неверный expected vector не считается физическим эталоном. Отказы решателя
остаются failed evidence. `industrial_readiness=false`: это numerical verification,
не аппаратная калибровка или квалификация произвольной миссии.

Приёмка Q5-A: **42 новых targeted tests**, **1339 reference**, **110 fast**;
**1775 collected**. Наборы пересекаются: reference запуск включал первые 36 новых
тестов; после provenance-only исправления версии отдельно прошли все финальные 42.
Окончательный installed wheel: **240/240** Python modules byte-matched, **119**
analytic checks + existing external J2/ERFA проверки, small coupled native replay
и verified RL/SFT прошли. Venv system-site-packages; dependency-clean/remote CI и
большая миссия на новой сборке не запускались.

Контракт и оставшиеся Q5-B/C/D: [ANALYTIC_ENVELOPES_V1](validation/ANALYTIC_ENVELOPES_V1.md).
Историческая большая Q4 mission относится к прежнему wheel SHA и не переносится
на эту сборку. Текущая приёмка записана в `acceptance/B11_ENVELOPES.json`.

## Предыдущий пакет — B11-Q4: ground windows и causal ACK release

Добавлены versioned ground control windows 2–8 с (старый default 4 с сохранён),
опциональные cancellation actions и opt-in controller по наблюдаемому ground ACK.
Новый product для замены должен быть owned, видимым, ready и иметь remaining>0.
До ACK, для чужих products, CPU/relay или при incoming relay отмена запрещена.
Role slots, joint limits, bookings и native cancel→re-admission/rollback сохранены.
Старый frozen manifest нормализуется с прежним hash; defaults не переписаны.

Conditional preflight описывает narrow controller-window bound и assumptions,
не является physical infeasibility proof и не подавляет прогоны. В результатах
разделены runtime, native integrity/witness и mission expectations; общий `ok`,
failed denominators и formal semantic group не ослаблены.

**1303 reference**, **110 fast**, **1733 collected**, **27 новых ground checks**;
наборы пересекаются. Native-проверки включают разную геометрию станции, slow RF,
setup/ACK за пределами окна, future-seed/outage causality, joint ACK cancellation,
invalid replacement rollback и cold prefix replay. Шесть fault/recovery повторов
прошли, включая exact parity ACK-reclaim. Installed wheel: **239/239** modules,
paired short windows + реальный ACK cancel/resubmit + verified RL/SFT прошли.
Venv использует system-site-packages; remote clean CI и expanded full не запускались.

Новый frozen профиль **прошёл все gates в трёх независимых процессах**:
12 аппаратов / 40 заявок / 120 физических секунд, **40/40** в каждом повторе;
одна native/trace signature pair у straight×2 и prefix recovery×1. Наблюдённые
максимумы: execution 1164.7 с <1200, step p95 26.47 с <60,
peak RSS 1.76 ГиБ <6, checkpoint 72.2 МиБ <512.
Физическое definition совпадает с историческим baseline; task diff — только
окна downlink 4→2 с. Старый 4-секундный случай 28/40 остаётся failed.
Это приёмка конкретного synthetic formation profile, не всех космических миссий. Физические
модели, cadence, tolerances, RF rate/setup/ACK и terminal capacity не изменялись.

[Контракт](GROUND_SCHEDULING_CONTRACT.md) ·
[Приёмка](acceptance/B11_GROUND.json).

## Исторический пакет — B11-Q3: native copies и диагностика workers

Убран повторный science deepcopy внутри locked runtime observation. Owned scalar/
dict/list trees копируются с сохранением memo, cycles, aliases и fallback для
NumPy/subclasses/hooks. Native capture сохраняет полный независимый mutable graph;
внешние transformers используют прежний standard capture. Расчёт награды использует
те же scalar outcomes вместо полного научного отчёта; summation order сохранён.
Compact audit переиспользует validated canonical bytes. Ошибка подготовки event copy
откатывает физику/revision и не расходует idempotency-key.

Workers атомарно сохраняют bounded stage/PID/input-digest/decision/CPU/RSS diagnostics
в `workers/000000/progress.json`; snapshots остаются при timeout. Они не являются
native replay proof, не попадают в policy и не могут сделать failed repeat успешным.

Финальная source-проверка: **1276 reference**, **110 fast**, **1706 collected**,
из них **24 новых copy/progress checks**; наборы пересекаются. Same-build A/B на
пяти малых сценариях подтвердил точное совпадение всех переходов, trace и полного
native report. Четыре frozen fault cases прошли replay/quality/export gates:
recovery 4/4, ACK loss 6/6 с 9 потерянными ACK/9 повторными фрагментами,
полный outage и короткий TTL — ожидаемые 0. Installed wheel: **237/237** modules,
малый paired recovery/native replay/RL-SFT smoke прошёл. Venv с system-site-packages;
remote clean CI и expanded full не запускались.

Крупный frozen профиль завершён во всех трёх процессах. Технические gates
(execution 856.4–889.2 с, step p95 ≤26.65 с, peak RSS ≤1.90 ГиБ, checkpoint
≤71.8 МиБ), native replay, accounting/witness и export прошли. **Overall приёмка
не пройдена: 28/40 заявок**, при этом acquisition/CPU/custody — 40/40.
Все три raw native/trace пары совпали; formal comparison group осталась failed.
У current baseline 4-секундные non-overlapping ground windows и один terminal:
после начального pipeline 8 с остаётся максимум 28 таких окон. Это ограничение
controller/profile, не общий physical infeasibility proof.
[Следующий пакет — окна и ground scheduling](GROUND_DELIVERY_NEXT.md).
Старые frozen inputs/failed receipts сохранены; пороги не снижались.
Физические cadence/tolerances/quality/admission не менялись. Immutable/shared history
и отказ от вложенных snapshots не реализованы; произвольные глобальные миссии не
квалифицированы.

[Контракт](NATIVE_COPY_AND_WORKER_DIAGNOSTICS.md) ·
[Приёмка](acceptance/B11_NATIVE_COPY.json) ·
[Дальнейшая оптимизация](COUPLED_PERFORMANCE_NEXT.md).

## Исторический пакет — B11-Q2: локальные чтения и проекции

Native observe одного аппарата больше не строит телеметрию всей группировки.
Один `_refresh` переиспользует owned public reads между центральной и локальными
проекциями; registries ролей независимы. Кэш не живёт между refresh/мутациями,
а joint admission после cancellation по-прежнему читает реальный candidate.
Физика, cadence, tolerances, capture/history и публичный ownership не изменены.

Same-build A/B на четырёх малых сценариях подтвердил точное совпадение всех
переходов, trace и полного native backend report. Новые проверки покрывают
scope, отсутствие устаревшего кэша, alias isolation и fork. **1251 reference**
прошёл в sandbox; единственный запрет localhost-сокета отдельно перепроверен:
**1 HTTP test passed** при разрешённом сокете. **110 fast**, **1682 collected**;
наборы пересекаются. Installed wheel: **235/235** Python-модулей byte-matched,
малый paired restore/replay/RL-SFT smoke прошёл. Venv использует system-site-packages.

Крупный профиль завершён и **не прошёл** прежние frozen gates.
Все три worker превысили 2400 с; итоговые completion/replay/export неизвестны.
Ошибки сохранены, readiness не подтверждена.
Прежние B11-Q1 timeouts сохранены. Следующие участки —
стоимость копирования native evidence/report, capture/trial и рост истории;
промышленная масштабируемость не следует из малых тестов или числа чтений.

[Реализация и границы](PROJECTION_READ_OPTIMIZATION.md) ·
[Приёмка](acceptance/B11_PROJECTION_READS.json) ·
[Дальнейшая оптимизация](COUPLED_PERFORMANCE_NEXT.md).

## Исторический пакет — B11-Q1: измеренная coupled-миссия

Добавлены генератор formation workload, реальные цепочки EO → onboard CPU →
DTN relay → shared ground, отказы, native witnesses, префиксное восстановление
и сравнение семантики независимых процессов. Supervisor ограничивает параллелизм,
фиксирует входы и сохраняет промежуточные результаты без исключения неудачных повторов.

**Крупный профиль не прошёл:** 12 аппаратов, 40 заявок, 120 физических секунд;
все три повтора превысили заранее установленный wall timeout 2400 секунд.
Итоговые completion/replay/export показатели этого профиля не получены.
Порог не повышался; промышленная масштабируемость не подтверждена.

Четыре малых fault-сценария прошли: recovery — 4/4 заявок, ACK loss — 6/6
с 9 потерянными ACK и 9 повторно сериализованными фрагментами; полный ground outage
и короткий TTL дали ожидаемые 0 выполненных заявок при проверке промежуточных стадий.
Малый paired probe подтвердил совпадение native семантики после cold prefix restore.
Геометрия благоприятная синтетическая, clear-cloud declarations явные; это не
валидация произвольных многоплоскостных группировок или многосуточных миссий.

Проверки финального исходного пакета: **1247 reference**, **110 fast**,
**1677 collected** (наборы пересекаются). Installed wheel: **235/235** Python-модулей
совпали; малый native smoke с replay/export и двумя fresh processes прошёл.
Venv использует system-site-packages; remote clean CI и expanded full не запускались.

Следующий приоритет — убрать повторное копирование и построение task-проекций,
сохранив owned observations, role isolation, rollback и физические cadence/tolerances;
затем повторить **тот же** frozen крупный профиль. Диагностический cProfile малого
случая указал на deepcopy; это не измерение production latency.

[Контракт](COUPLED_CAMPAIGN_QUALIFICATION.md) ·
[Фактическая приёмка](acceptance/B11_CAMPAIGN.json) ·
[План ускорения ядра](COUPLED_PERFORMANCE_NEXT.md).

## Исторический пакет — B10-C2: ограниченный streaming transport

Реализован process-local ArtifactStore: закрытый дисковый spool, 32 МиБ на файл,
128 МиБ reservations и до 8 transfers; exact-range retry, SHA-256, status/cancel,
кооперативный TTL и защита занятого файла lease. HTTP upload идёт PUT-фрагментами
до 1 МиБ с ограниченным чтением и whole-chunk deadline; download читает файл по
64 КиБ. Старый JSON limit 8 МиБ сохранён. Native interactive bounds не увеличены.

Архивы/checkpoints импортируются через прежние replay/provenance guards. Успешный
commit имеет cached receipt; неясный/ошибочный restore терминален для transfer и
не запускается повторно вслепую. ZIP RL/SFT строится на диске через export_to(),
с проверенным native episode и operator provenance. UI пульта использует новые
routes и проверяет digest; браузер и native JSON parser не объявлены constant-RSS.

Приёмка: **124 operator/API/durable/cursor/artifact**, из них **30 новых**;
**5 legacy Studio API**, **1229 reference**, **110 fast**, **1659 collected**
(наборы пересекаются).
Источник и installed wheel прошли настоящий HTTP import >8 МиБ (JSON padding
fixture), native restore, chunk retry и verified ZIP. Chromium проверил restart,
роли, большой импорт, ZIP и mobile; **234/234** modules и UI assets byte-matched.
Installed использует system-site-packages; remote clean CI и expanded full не запускались.

[Контракт и протокол](OPERATOR_ARTIFACT_CONTRACT.md) ·
[Приёмка](acceptance/B10_ARTIFACTS.json).

### Следующий большой блок — B10-C3 / B11

1. Durable artifact/event metadata, indexed event storage и retention policy;
   восстановление незавершённых transfers после process crash сейчас не реализовано.
2. Измеренные long-run latency, RSS, recovery и disk-growth gates на frozen workloads;
   payload с JSON-padding проверяет транспортный порог, не большой mission workload.
3. Worker isolation, quotas, cancellation/monitoring и deployment-level identities.
4. B12 независимая калибровка и пилотные отраслевые benchmark-задачи.
B10-C/B11 целиком не закрыты. Сайт проекта не регенерируется.

## Исторический пакет — B10-C1: история по курсору и online recovery

Добавлены role-scoped journal/navigation pages: capability/session/channel binding,
подписанный prefix cursor, продолжение после compatible-build restart, явный
конфликт при изменении прочитанного префикса. Последовательность локальная для
проекции роли; физическое время и ревизии чтением не изменяются. Старый state API
сохранён; UI продолжает использовать прежний контракт.

Добавлены OS-owner SDK/CLI online SQLite backup, verify и restore в новый файл.
Копия живой WAL-базы проходит integrity и active-session native replay до публикации;
атомарная exclusive publication не перезаписывает существующие outputs, включая
гонку создания файла. Проверка live-базы тоже использует частный snapshot.
Restore сохраняет IDs/capability hashes/control epochs и не вращает credentials.
Source/dependency guards не ослаблены; автоматической миграции между builds нет.

Приёмка: **94 operator/API/durable/maintenance**, из них **38 новых**;
**1199 reference**, **110 fast**, **1629 collected** (наборы пересекаются).
SDK и настоящий loopback HTTP backup/restore/cursor прошли из исходников и
installed wheel вне checkout. **233/233** Python modules и UI assets byte-matched.
Installed использует system-site-packages; remote clean CI и expanded full не запускались.

[Контракт и команды](OPERATOR_MAINTENANCE_CONTRACT.md) ·
[Приёмка](acceptance/B10_MAINTENANCE.json).

### Следующий пакет — B10-C2

1. Streamed artifact upload/export, bounded disk spooling, отмена и cleanup.
2. Индексированный event storage, retention/backup policy, измеренные recovery
   latency и disk growth на frozen workloads; существующие лимиты пока сохраняются.
3. Далее B11 workers/quotas/monitoring/identities; B12 независимая калибровка и пилоты.
B10-C целиком этим пакетом не объявлен закрытым. Сайт не изменяется.

## Исторический пакет — B10-B: восстановление и управление operator ↔ agent

Реализовано постоянное локальное SQLite/WAL-хранение с atomic snapshot/journal
commit, generation CAS, проверкой целостности и native replay при восстановлении.
Подтверждение шага выдаётся после commit; сбой записи откатывает live-сессию,
неопределённый результат блокирует handle до восстановления. Добавлены тесты
аварийного завершения процесса до/после commit и конкурирующих writers.

Координатор разделяет каналы operator/agent и проверяет control epoch вместе
с native revision. Передача управления не двигает физическое время. Архив
сохраняет наблюдённые трассы и историю, а connection-файл позволяет подключиться
после рестарта. В базе хранятся хеши capabilities; локальное recovery вращает
секреты и эпоху управления. Scripted checkpoint требует явного takeover,
проверяется native replay и сохраняет исходный provenance. Training ZIP содержит
отдельный operator-provenance.json, без назначения оператору статуса expert.

Приёмка: **56 operator/durable**, **5 legacy Studio API**, **1161 reference**,
**110 fast**; наборы
пересекаются. **1591 collected**. Native relay закрыл 2/2 заявки за 10 решений
с restart/handover/replay/export в исходниках и installed wheel. Chromium
проверил restart/reconnect/archive import на desktop/mobile и вне checkout.
Wheel: **231/231 modules**, UI byte-matched. Remote clean CI и expanded full
не запускались; установочная среда использует system-site-packages.

[Контракт, запуск и ограничения](OPERATOR_DURABILITY_CONTRACT.md).
[Приёмка B10-B](acceptance/B10_DURABILITY.json).

### Следующие gates — B10-C / B11 / B12

1. Event cursors, streamed uploads/exports, retention/backup и нагрузочная
   квалификация cold recovery и bounded snapshot storage.
2. Изолированные workers, квоты, отмена операций, мониторинг и deployment-level
   identities/SSO/TLS. Сейчас это локальный сервис с capabilities, не публичная SaaS.
3. Длинные coupled/fault/arrival миссии, увеличенные fleet profiles и проверяемые
   estimator/forecast overlays; не повышать лимиты без измерений и replay gates.
4. B09-M2 performance, пилотные отраслевые задачи и независимая B12 калибровка
   физики/механик. Внутренние тесты не устанавливают flight qualification.

## Исторический пакет — B10-A: Mission Control на общем native runtime

Новый `/operator` объединяет пульт, текущую телеметрию, 3D и 2D пройденные трассы,
ресурсы, продукты/заявки/CPU jobs, журнал и native replay/export на одной странице.
Прежняя `/studio` сохранена. Клиент не считает физику: команды идут через B06
TaskTools/TaskSession, с теми же masks/admission/revision/idempotency и стоимостью
отклонённого действия. Native SDK parity проверена для fixed и coupled сценариев.

Оператор, read-only viewer, artifact-admin и локальные controller roles имеют
разные session bearer capabilities. Авторизация и actor visibility проверяются
сервером; названия локальных ролей не перекрывают системные capability labels.
Checkpoint restore проверяет native replay, восстанавливает журнал и выдаёт новые
токены. Прошлая геометрия при restore не выдумывается. RL/SFT ZIP проверяет replay,
сохраняет external provenance и не объявляет действия оператора expert teacher.

Демо доступны в selector: четыре отдельных J2 дуги/готовые продукты и существующая
**EO→CPU→ISL→Ground ACK** coupled миссия. Последняя завершила **2/2 заявки за 10 шагов
и 20 с** в SDK, браузере и installed wheel; replay и training export прошли.
Навигация использует текущие native providers и их sealed EOP, GCRS→ITRS/WGS-84;
страница показывает только уже наблюдённые точки, без нового sensor/forecast model.

Приёмка: **31 old/new API cases**, **1131 reference**, **110 fast**, **1561 collected**
(наборы пересекаются), **26 новых operator cases**. Дополнительно coupled API/SDK
переходы и итоговый checkpoint совпали точно. Wheel: **229/229** Python modules,
3 boinor fixtures, operator fixture и UI assets byte-matched. Installed вне checkout
прошёл native цепочку, replay и ZIP; environment использует system-site-packages,
не dependency-clean install. Первый installed smoke завершился SIGTERM без итогового
результата; последующие повторные и инструментированный (36 с) прогоны passed.
Причина первого внешнего обрыва не установлена и не выдана за успешную проверку.
Expanded full и remote clean CI в этом пакете не запускались.

[Приёмка](acceptance/B10_OPERATOR.json), [контракт и запуск](OPERATOR_CONTRACT.md),
[installed smoke](../examples/operator/smoke.py).

### Следующий большой блок — B10-B / B11

1. Durable session/checkpoint/journal storage, event cursors, crash recovery и
   artifact upload/export streaming. Сейчас состояние в памяти процесса; большой
   checkpoint может превысить HTTP upload budget. Схемы и source guards не ослаблять.
2. Явный coordinator/ownership и provenance для передачи SDK/agent→operator→agent.
   В B10-A живой adapter — единственный writer; agent/human clients используют его
   revision-bound API. Прямые SDK mutations содержащегося TaskSession не считаются
   поддержанной согласованной передачей управления.
3. Multi-user identities/SSO, TLS/proxy и quota/worker lifecycle, несколько владельцев
   и session isolation на deployment level. Текущий сервис — bounded local console,
   не публичная production API. Legacy routes не объявлены enterprise-protected.
4. Длинные coupled/arrival/fault missions, большие grouping/operator profiles,
   estimator/forecast overlays и native attitude display — с отдельной квалификацией.
5. B09-M2 mutable history/performance и B12 pilot/calibration остаются gates;
   browser smoke и число тестов не заменяют внешние industrial evidence.


## Исторический срез — B09-M1: память task evidence и replay-once RL/SFT export

Task history, соседние observation frames и idempotency-cache теперь используют
общие внутренние append-only evidence objects. Публичные ответы, verification
receipts и строки экспорта остаются owned copies. Изменение полученного результата
не меняет runtime, будущие строки или повторные idempotent ответы.

`VerifiedEpisode` фиксирует входной checkpoint и делает один cold replay;
planning-вариант дополнительно проверяет policy/RNG/trace и native quality gates.
Повторные RL/SFT/local-role exports используют проверенный снимок. `export_bundle`
потоково пишет в staging и публикует каталог атомарно; byte budget включает receipt,
ошибка второго файла не оставляет частичного target. Старые single-export API сохранены.
Фабрика использует один verified planning episode для всех role/format exports.
Scale worker освобождает исходный run перед независимым replay и не повторяет его
для каждой выходной формы данных. Уменьшение расходов не заменяет QA unchecked state.

На прежнем frozen **64/1000**, два повтора, **1000/1000** и exact native/trace parity:
peak RSS **1,85 ГБ → 504 МБ**; полный QA pipeline **181–196 → 93–122 с**.
Запись RL/SFT после проверки — около **1,2 с**, без двух дополнительных cold replay.
Сравнение на общем стенде с регрессией, не SLA; replay/quality и запись измерены
отдельно. Full checkpoint bytes/форматы и физический sampling не упрощались.

Проверки: **234 targeted**, **1105 reference**, **110 fast**, **1535 collected**
(наборы пересекаются), **18 новых export/evidence cases**. Wheel **228/228** модулей
и **3/3** fixtures byte-matched. Installed вне checkout: native/legacy-byte parity
replay-once SDK, корпус **36 эпизодов, 48 RL/20 SFT** и coupled EO **2/2, 10 RL/10 SFT**.
Expanded full regression и чистая remote CI в этом пакете не запускались.

[Приёмка](acceptance/B09_M_MEMORY_EXPORT.json),
[контракт и API](VERIFIED_EXPORT_CONTRACT.md),
[сквозной пример](../examples/scaling/verified_export.py).

### Дальнейшая последовательность

1. **B10**: Studio/оператор на общем runtime: causal telemetry и роли, единые
   admission/revision/idempotency, объяснения rejection и ресурсы/обязательства/
   timeline; replay действий оператора. UI не создаёт собственный physics state.
2. **B09-M2**: mutable native history/capture и полные JSON checkpoints всё ещё
   требуют RAM; cold replay линейный. Нужны отдельные измерения и event/history
   storage без нарушения rollback/ledger/RNG contracts, durable phase receipts.
3. **B09-Q2/Q3**: длинные heterogeneous coupled arrivals/fault missions и отдельный
   контракт 256/10000; 64/1000 fixed workloads не квалифицируют эти постановки.
4. **B11/B12**: dependency-clean release/CI и независимый промышленный пилот с
   данными, калибровкой и tolerances. Эти gates не заменяются internal test counts.


## Исторический срез — B09-Q1: отдельные орбиты и измеряемые actor/task нагрузки

Добавлен self-contained `FleetWorkloadSpec`: отдельные J2/DOP853 GCRS дуги,
несколько орбитальных плоскостей, индивидуальные продукты/команды и controller
roles в существующих лимитах. Порог native/task профиля остаётся 64 аппарата /
1023 action templates; 256/10000 не объявлены поддержанными.

Профиль выявил повторное построение native outcome report для чтения horizon
при каждой проверке action mask. Добавлены лёгкие synchronized readers горизонта
и команд; causal masks/planning input больше не считают итоговый отчёт. На 16/100
native reports и traces до/после совпали. Физика и sampling grids не изменены.

Manifest фиксирует repeats, thread environment и latency/execution/RSS/checkpoint
budgets. Каждый повтор исполняется отдельно; CPU model/affinity/cgroup limits
попадают в receipt. Ошибки и невыполненные задачи сохраняются. Проба 64/1000 с
32 candidates выполнила 544/1000 и осталась failed; исправлен только явный бюджет
политики (64), без снижения completion threshold и изменения физического сценария.
Исходный прерванный hotspot probe также сохранён. Frozen manifests хранятся как
сжатые JSON вместе с результатами, а не заменяются описанием удачного запуска.

[Приёмка B09-Q1](acceptance/B09_Q_FLEET.json), [контракт](SCALING_CONTRACT.md),
[запуск разных орбит и hour arc](../examples/scaling/README.md).

Финальная приёмка: **55 scaling cases**, включая **24 новых**, **1087 reference**,
**110 fast**, **1517 collected**; наборы пересекаются. Полная expanded-регрессия
в этом пакете не повторялась. Wheel **228/228** модулей и **3/3** fixtures byte-matched;
installed SDK/CLI **64/64**, два повтора вне checkout. Clean remote CI не запускалась.

Заданные профили прошли по два повтора: **16/100 → 100/100**, **64/1000 → 1000/1000**,
часовая fixed-дуга **4/4 → 4/4**. Native replay/балансы/экспорт и repeat hashes passed.
64/1000: execution **42–43 с**, step p95 **2,98–3,11 с**, checkpoint **11,10 МБ**,
peak RSS **1,85 ГБ** на общем стенде. Это не статистический p95/SLA или длительная
coupled-квалификация. Память и replay/export стоимость остаются gates B09-M.

### Следующие крупные блоки

1. **B09-M**: mutable task/native history, индексы/incremental accounting без смены
   результатов, устранение повторного cold replay для trusted multi-format export
   при сохранении независимого QA. На крупных случаях RAM и replay всё ещё дороги.
2. **B09-Q2**: длительные *coupled* EO/CPU/DTN/ADCS/thermal миссии, непрерывные
   поступления задач, альтернативные станции/ограничения и штатные/аварийные режимы;
   повторные испытания на закреплённом CPU/quota envelope. Hour fixed arc с четырьмя
   предзагруженными продуктами это не заменяет.
3. **B09-Q3**: отдельный контракт 256/10000: динамический каталог/очередь задач,
   bounded observations, agent scheduling и ресурсный бюджет. Не поднимать лимиты
   неподтверждённого native профиля ради маркетингового числа.
4. **B10**: Studio/оператор на том же runtime: causal telemetry, ревизии и роли,
   native admission/rejection, обязательства/ресурсы/timeline и replay. UI не создаёт
   вторую независимую физику; затем B11 clean-install/release/CI и B12 pilot/calibration.


## Исторический срез — B09: streaming и измеряемое масштабирование, 2 ноября 2026

Реализованы потоковые RL/SFT exports с атомарной записью и совместимыми хешами,
SQLite spool с ограниченными page cache/disk/row budgets и транзакционным rollback
строк/дедупликации. Runtime fork больше не копирует содержимое накопленного audit/cache;
mutable backend, команды и RNG по-прежнему изолированы. Compact runtime/task/planning
checkpoints `/2` восстанавливаются через независимый cold replay и хеши полного
operation/transition evidence; неподтверждённой загрузки физического state нет.

Exact ERFA LRU не округляет время/аргументы и возвращает независимые матрицы.
Spawn workers исполняют по одному эпизоду, с bounded pending work, стабильным
порядком и сохранением ошибок в знаменателе. Локальный cross-bundle registry
проверяет native replay и транзакционно блокирует split contamination/повторные
траектории между новыми корпусами. Фабрика по умолчанию использует compact audit.

Typed `scale-benchmark` замораживает manifest до исполнения и измеряет каждый
workload в свежем интерпретаторе: full-step p50/p95, fork Python peak, checkpoint,
replay, verified RL/SFT rows/sec и process-lifetime RSS с hardware/thread metadata.
Это измерения конкретных workloads, не универсальная промышленная квалификация.
[Контракт](SCALING_CONTRACT.md), [примеры](../examples/scaling/README.md),
[точная приёмка](acceptance/B09_SCALING.json).

Проверки финальных исходников: **192 targeted**, **1063 reference**, **110 fast**,
**1493 collected** (наборы пересекаются). Расширенная регрессия: **1463 passed**
до validation-only Python-3.10 guard; после guard повторены targeted/reference/fast,
а не вся расширенная регрессия. 29 длительных legacy cases не перезапускались.
Wheel: **227/227** модулей и **3/3** boinor fixtures byte-matched. Installed corpus:
36 эпизодов, 48 RL/20 SFT; coupled EO: 2/2 заявки, 10 RL/10 SFT; serial/2-worker
native parity, registration/idempotency и CLI smoke passed вне checkout. Локальный
venv использует system-site-packages и изолирует user-site; чистая remote CI не запускалась.
Восемь synthetic probes passed, в том числе fixed-fanout 64/64. Paired EO checkpoint
**73,28 → 2,93 МБ**, native report/trace identical. Один baseline fork замер:
**4,308 → 0,155 с**, Python peak **211,18 → 9,30 МБ**; это не гарантированный SLA.
Параллельные workers требуют Python 3.11+, serial сохраняет контракт Python 3.10+.

### Оставшиеся gates B09 и следующий пакет B10

1. **B09-Q**: независимые multi-plane орбиты, 16/100 и 64/1000 actor/task нагрузки,
   длинные coupled horizons и отдельный бюджет для 256/10000. Текущие короткие
   fixed-fanout probes не закрывают эту квалификацию. Нужны повторные p50/p95,
   CPU/quota/thread envelope, исходный frozen manifest, memory/latency thresholds,
   partial packet/cancellation/fault stress и сохранение conservation/replay.
2. **B09-M**: убрать транзиентное построение полных operation reports и копирование
   mutable native history; при необходимости добавить versioned snapshot+event tail.
   До этого нельзя обещать постоянную RAM или мгновенный restore на длинных миссиях.
3. **B10**: Studio и оператор на общем B01/B06 runtime: роли/доступ, causal telemetry,
   команды через те же admission/masks/revision, объяснение rejection, timeline
   обязательств/ресурсов и native audit/replay. Начать со сквозной существующей миссии,
   без второго независимого физического state в браузере.
4. **B11/B12**: clean install/release/CI matrix и независимый промышленный pilot:
   данные, калибровка, tolerances, устойчивость к сбоям и внешняя приёмка. Эти gates
   не заменяются количеством функций и не считаются закрытыми по smoke-прогонам.


## Исторический срез — B08: versioned mission/data factory, 2 ноября 2026

Реализована bounded фабрика поверх B06/B07: native-шаблоны, typed distributions с
единицами и synthetic provenance, детерминированные draws без resample-to-success,
геометрические группы и label/seed-independent fingerprints. Разделение фиксируется
до исполнения; одинаковые публичные траектории между splits блокируют готовность.

Каждый draw/case/policy остаётся в аудите, включая ошибки и infeasible missions.
Корпус строится атомарно в новый каталог: train/holdout/OOD RL/SFT, action maps,
SFT tool schemas, native replay/ACK/ledger/mask gates. Строгие SFT gates требуют
подтверждения именно выбранного teacher; статусы ready/quarantined не означают
expert/flight qualification. Native raw audit отделён от policy-facing data.

Целевые проверки: 161 passed (61 новый B08 case), reference — 1032, fast — 110.
Wheel: 221 модуль и 3 boinor fixtures побайтно совпадают с исходниками.
Расширенная регрессия — **1433 passed**; 29 неизменённых длительных legacy cases
в этот раз не перезапускались и остаются отдельными CI jobs. Installed-corpus:
36 эпизодов, 48 RL / 20 SFT строк, полный native/data replay и SDK/CLI parity;
повторная сборка даёт тот же content hash. Coupled EO/CPU/ISL/ground: 2/2 заявки,
10 RL / 10 SFT, полный пересчёт корпуса passed (явный non-expert teacher opt-out).
Quarantine без teacher evidence проверен через installed CLI: exit 1, audit сохранён.
[Точная приёмка](acceptance/B08_FACTORY.json), [контракт](FACTORY_CONTRACT.md),
[пример](../examples/factory/README.md). Массовая генерация/streaming не заявлены.

### Исходный checklist B09: память, streaming и измеряемое масштабирование

1. Заморозить workload manifests для fixed/coupled/EO/CPU/DTN/fault profiles,
   нескольких размеров группировки и коротких/длительных horizons. Проверять
   fulfillment, native conservation, partial packets, deadline/TTL и rejection cost.
2. Ввести streaming RL/SFT JSONL и bounded corpus buffers с backpressure, лимитами,
   deterministic ordering и атомарным commit/quarantine. Не держать все строки в RAM.
3. Убрать копирование всей истории/cache при каждом task fork: текущая physics state
   и immutable journal должны разделяться без alias/RNG ошибок. Rollback сохраняет
   source/receiver custody, queues, active commands, spin, CPU, heat и ledger totals.
4. Версионировать compact checkpoint с явными source/dependency/seed descriptors,
   event cursors и проверкой restore. Сохранять cold replay как независимый oracle;
   не менять verified replay на бесконтрольную deserialization ради скорости.
5. Добавить кеширование geometry/frame computations только с доказанной границей
   ошибки; не менять physical/safety sampling grid без numerical acceptance.
6. Изолировать workers по процессам/seed/manifest, внедрить cross-bundle group и
   trajectory index. Несколько независимых корпусов сейчас не образуют общего
   дедуплицированного registry автоматически; merge требует проверки групп/splits.
7. Измерять p50/p95 decision wall time, end-to-end QA rows/sec, checkpoint bytes,
   isolated-worker peak RSS и ошибки. Process-lifetime RSS не выдавать за delta RAM.
   Пропускную способность квалифицировать на фиксированном hardware manifest.
8. Расширить CI gates для streaming/restore/fleet workloads и reproducible worker
   scheduling. Поддержка learned-policy evaluation и независимая калибровка B12
   остаются явными gates; увеличение max_cases само по себе не доказывает масштаб.

## Исторический срез — B07: causal baselines и проверяемое планирование, 2 ноября 2026

Реализованы пять baseline-политик поверх B06: wait/random/priority/EDF и bounded
route-aware current-window search. Вход политики — immutable public value, без
scenario/truth/future rollout. Один runner исполняет решения через TaskSession,
сохраняет RNG/trace и холодно проверяет native episode вместе с решениями политики.
Operator override требует явного capability, логируется и маркирует данные external.
Текущие бронирования/provider revisions учитываются без скрытых будущих событий.

Benchmark фиксирует manifest до запуска, исключает общие mutable ссылки, сохраняет
ошибки/невыполнимые cases и делит по lineage/OOD family. Teacher gates сравнивают
held-out результаты с priority/EDF, но не выдают их за экспертную квалификацию.
Есть повторное semantic verification и tiny native masked-tree oracle с бюджетом.
CI full разбит на other/campaign/tasking-base/tasking-sixteen без изменения состава.

Приемка bounded profile: **57 новых cases**, 100 вместе с B06; reference — 971,
fast — 110, expanded regression — **1372 passed**. 29 неизменённых длительных
legacy cases в этот раз не перезапускались и выделены в отдельные CI jobs.
Installed wheel: 214 модулей и 3 boinor fixtures совпадают с исходниками побайтно;
nominal/loss/reroute — 2/2, TTL — 1/2, conservation и cold policy/runtime replay passed.
Проверены SDK/CLI native parity и prefix RL/SFT exports. Portable benchmark:
10 эпизодов, пять blocked случаев сохранены, teacher gate ожидаемо failed.
[Результаты и границы проверок](acceptance/B07_PLANNING.json),
[контракт](PLANNING_CONTRACT.md), [сквозной пример](../examples/planning/README.md).
Это bounded engineering controller: current-window subset search, ideal local
telemetry и конечные task templates, без orbital/RF forecast service, доказательства
глобального оптимума, calibrated teacher или квалификации massive fleet.

### Следующий пакет — B08: воспроизводимая фабрика задач и данных

1. Версионировать task families и distributions: EO/CPU/relay/ground, отказы и
   recovery, shared terminals/booking, finite mission resources. Физику не дублировать.
2. Вводить явные диапазоны/единицы и synthetic/calibrated provenance параметров;
   не объявлять случайные числа калибровкой реального аппарата.
3. Генерировать feasible и infeasible случаи; checking через native execution,
   независимые small oracles и teacher gates, без удаления трудных кейсов из оценки.
4. Дедуплицировать по семантической миссии, а не id/seed; lineage и OOD families
   фиксировать до обучения. Простое переименование не создает independent evidence.
5. Версионировать RL/SFT exports: role views, action mappings, dt/discount, feedback,
   provenance, teacher status и held-out результаты. Никакого truth в prompts.
6. Ввести quality gates до публикации: schema/finite values, causal masks, replay,
   conservation, duplicate/leakage checks, rejected teacher actions и native ACK.
7. Измерять diversity/coverage и baseline/learned-policy качество на замороженных
   manifests с ошибками в знаменателе. Не называть synthetic scripts экспертами.
8. Прежде чем обещать массовую генерацию, закрыть streaming/history/RSS и throughput
   gates B09: B06 audit-checkpoint короткой миссии может занимать около 60 МБ.
   B12 independent calibration/operator evidence остаётся отдельным пакетом.

## Исторический срез — B06: обучение на едином MissionRuntime, 2 ноября 2026

Реализован versioned `space-env.task/1`: конечные action templates, роли аппаратов
и оператора, causal entity/relation views и masks без future admission oracle,
атомарный joint step, физические elapsed/discount, пять целей, horizon termination
и budget truncation. Gymnasium, optional PettingZoo и structured LLM tools используют
один executor. Checkpoint холодно переигрывает решения; RL/SFT export проверяет replay,
фильтрует роли и группирует split по lineage. Контрольные артефакты с seeds/truth
не являются наблюдениями политики.

43 новых cases; reference — 914, fast — 110. Полный набор: **1344/1344 passed**
в непересекающихся частях (1315 + 14 + 14 + 1), включая 16 аппаратов / 72 часа.
Объединенные попытки прерывались SIGTERM, поэтому не зачтены как успешные; все
итоговые части имеют exit 0. Installed wheel проверен вне checkout: 207 модулей
и 3 boinor fixtures побайтно совпадают с исходниками; optional Parallel API passed.
Сквозные nominal/reroute/loss выполняют 2/2 заявки, TTL — 1/2; native checkpoint
совпадает для SDK/CLI-plan/Gym/tools, а nominal — также для PettingZoo.
Точные результаты и прежний CLI import-order debt — в [приемке](acceptance/B06_AGENTS.json).
[Контракт](AGENT_RUNTIME_CONTRACT.md), [пример](../examples/agent_task/README.md).

Это bounded reference task profile: ideal current local telemetry, finite declared
commands, shared team reward и synthetic scripted teacher. Нет estimator/belief,
qualified expert policy, universal legacy adapters или massive fleet qualification.
B06 закрыт в этой границе; расширенная наблюдаемость и streaming scale — отдельные gates.

### Следующий крупный пакет — B07: планирование и baseline qualification

[Подробный исполняемый план и критерии приемки](PLANNING_BASELINES_EXECUTION.md).

1. Добавить rolling-horizon planner на публичной проекции: release/deadline,
   CPU precedence, custody locks, конечные ресурсы и terminal windows. Прогнозы —
   только явно разрешенные nominal estimates; future realized loss/faults недоступны.
2. Реализовать wait/random-masked/priority/earliest-deadline/route-aware baseline
   на одном task executor; неизвестные ресурсы не считать гарантированно доступными.
3. Перепланирование после reject/fault/TTL/SOC/booking revision; event причинность
   и стоимость recovery включены в реальные transitions.
4. Сравнить baseline по fulfillment/on-time Mbits/energy/violations/rejections,
   с lineage-separated held-out missions и несколькими заранее фиксированными seeds.
5. Добавить infeasible/blocked cases без ложного success; сохранять trace решений,
   assumptions и native checkpoint для независимого replay каждого результата.
6. Квалифицировать teacher до B08: execution replay сам по себе не доказывает
   оптимальность, экспертность или пригодность SFT demonstrations.
7. B08 — scenario/data factory с held-out параметрами, duplicate checks, quality gates,
   RL/SFT provenance и reproducible evaluation. B09 — замер p50/p95 wall time/RSS,
   streaming datasets/checkpoints и fleet scaling; B12 — независимая калибровка.

## Исторический срез — B05-B: конкурентные ISL и custody, 2 ноября 2026

Добавлены `operations.dtn_network` и runtime-команда `relay`: единые RF/modem,
сферическая Earth occultation, реальная BODY-направленность TX/RX, half-duplex
admission обоих аппаратов, receiver buffer reservation, TX/RX DC/SOC/peak/quota,
RX heat routing и очереди serialization → propagation → receive → ACK/timeout.
Store-and-forward сохраняет source copy до полного custody ACK; данные нельзя
использовать до прихода/подтверждения. Есть bounded retries/fixed backoff, realized
seeded data/ACK loss, deduplication, packet interruption cost, immutable lifetime
между hop/reroute, отмена с drain и атомарный перенос transport owner без потери
acquisition provenance. Прямые и независимые relay-каналы исполняются конкурентно.

44 новых cases; три аппарата без preloaded payloads в `examples/relay_mission/`:
nominal/reroute/loss выполняют 2/2 заявки, TTL — 1/2 без ложного completion.
[Контракт](validation/CUSTODY_NETWORK_CONTRACT.md),
[приемка](acceptance/B05_CUSTODY_NETWORK.json),
[инструкция](../examples/relay_mission/README.md).

B05-A/B закрывают B05.01–10 **в объявленном bounded engineering profile**.
Это не BPv7/BIBE/LTP wire implementation, multi-radio RF/interference model,
remote managed CPU или calibrated flight qualification. Использованы whole-product
reassembly, per-attempt acquisition и ideal out-of-band ACK с заданными loss/delay.
Наблюдения пока централизованные с идеальным control plane. B09 fleet scale и B12
independent hardware/operator evidence остаются отдельными gates.

### Исходный scope пакета B06: единый task layer для обучения агентов

1. Зафиксировать versioned entities/relations для actors, stations, products,
   transfers, requests, buffers и contact resources; разделить truth, belief и
   actor observation, включая явную наблюдаемость/возраст telemetry.
2. Добавить causal action catalogue и masks, включая next-hop/ground routing,
   custody locks, reserved buffers, safe recovery и текущие resource windows.
   Полный будущий seed/loss outcome агенту не передавать.
3. Дать Gym facade непосредственно поверх MissionRuntime с padded/sparse
   observations, bounded entity/action schemas и deterministic reset/seed.
   Реализовать optional MARL facade, endpoint/operator roles и shared validation.
4. Отделить decision epoch от внутренних physical/safety steps; transitions
   содержат elapsed_sec и discount по физическому времени. Wait/packet arrival/
   custody ACK не должны превращаться в одинаковые фиктивные RL ticks.
5. Создать objective registry с immutable task definitions: throughput/SLA,
   energy/latency/robustness, buffer loss и safety; reward не может выдавать ISL
   custody за ground fulfillment или повторно награждать duplicate fragments.
6. Привести LLM structured tools и human/API control к одному action adapter:
   schemas, revision/idempotency, rejection reason, budgets и replay.
7. Добавить проверяемый export SFT/RL transitions: provenance, dt, legal action,
   feedback, realized outcome, split/lineage и checkpoint replay verification.
8. Закрыть SDK/CLI/Gym/MARL/LLM parity на одинаковой EO→CPU→relay→ground миссии,
   с fault/TTL/retry и без future truth leakage. Затем B07 planner baselines,
   B08 datasets/evaluation и отдельная B09 scale qualification.


## Исторический срез — B05-A: RF и наземный сегмент, 2 ноября 2026

Добавлен opt-in `operations.radio_network`: SI Friis/EIRP/G/T/C/N0/информационный
Eb/N0, явные режимы modem/code/overhead/rolloff, passband/Doppler, BODY pointing,
synthetic seeded weather, persistent ground acquisition/slew/handover и отдельные
per-terminal/aggregate rate/bandwidth policies. Фактические TX/DC/heat остаются
связаны с mission/lifecycle; ground setup/tracking оплачены в отдельном ledger.
Подтверждение продукта задержано на ideal round-trip + declared ACK processing;
customer credit до ACK не начисляется. Local/external bookings используют общий
admission с revisions, stale/conflict rejection, immutable history и replay.

42 новых cases; сквозные nominal/outage examples выполняют 2/2 заявки, blocked RF —
0/2 без ложной доставки. [Контракт](validation/RADIO_GROUND_CONTRACT.md),
[приемка](acceptance/B05_NETWORK.json), [пример](../examples/network_mission/demo.py).
B05 **in_progress**: B05-A — bounded engineering package, не полная сеть/flight
qualification. Пороговый modem, constant G/T, ideal ground tracking/ACK и aggregate
capacity pools не означают waveform/interference/ITU rain/custody моделирование.

### Следующий исполняемый пакет — B05-B: concurrent ISL/DTN на том же runtime

1. Отделить origin/product identity от transport owner и custody. Ввести per-actor
   tx/rx/in-flight/reserved/discard/duplicate ledgers; каждый бит имеет одну
   проверяемую позицию, промежуточный прием не считается ground delivery.
2. Добавить ISL commands и endpoint admission: обе стороны занимают radio,
   half-duplex/resource lanes, power/SOC, thermal, pointing и memory bandwidth.
   Конкуренция с downlink использует тот же allocator и атомарные claims.
3. Реализовать ограниченную очередь событий сериализации/распространения/приема;
   receiver buffer резервируется заранее, данные нельзя использовать до arrival.
   Горизонт/отмена/fault не телепортируют и не возвращают уже потраченные ресурсы.
4. Добавить стабильные fragment IDs, TTL, partial transfer, seeded realized loss,
   bounded retries/backoff, deduplication и custody ACK. Уточнить retained source
   policy: текущий direct профиль освобождает сериализованные байты, retries не обещает.
5. Дать агенту explicit route/next-hop actions и причинные наблюдения очередей;
   truth/unknown future outages не использовать как скрытого оракула маршрутизации.
6. Закрыть B05.10 смешанными direct/relay missions: outage + reroute, buffer pressure,
   конкурирующие терминалы, TTL/drop, поздний/дублированный ACK, checkpoints во время
   передачи и in-flight. Проверить exact-once request credit и все балансы.
7. Повторить installed SDK/CLI и полную регрессию на финальных исходниках; отдельно
   измерить B09 масштабирование и получить B12 внешние RF/ground calibration evidence.

Критерий закрытия B05-B — одна связная миссия от capture/CPU через конкурентную
ISL-цепочку до подтвержденного ground product, с отказами и восстановлением.
Существующие последовательные legacy relay/DTN helpers этот критерий не закрывают.


## Исторический срез — B04-E: многозонная тепловая модель, 2 ноября 2026

Добавлен opt-in `thermal_network`: до 16 узлов, симметричная теплопроводность,
нелинейное T⁴-излучение, Sun/shadow/Earth IR/albedo/sky, маршрутизация фактического
CPU/payload/radio/ADCS/propulsion/conversion heat, PV extraction/curtailment без
двойного учета и paid thermostatic heaters с peak/SOC/hysteresis.
Есть node/global energy ledgers, thermal crossing history, actor-safe/recovery,
атомарные budgets/rollback и канонический cold-start/mid-operation replay.
[Контракт](validation/THERMAL_NETWORK_CONTRACT.md),
[приемка](acceptance/B04_THERMAL_NETWORK.json),
[пример](../examples/thermal_network/demo.py).

54 новых инженерных теста; nominal/cold EO examples выполняют 2/2 заявки, hot CPU
корректно блокирует финальный продукт (0/2). Старый one-node профиль сохранен.
Полная аппаратная/flight calibration не заявлена: lumped nodes, sampled forcing,
uniform spherical Earth/Lambert phase, aggregate EPS routing и global protection
остаются явными предположениями. Generic nonthermal DC export виден в ledger.

**Следующий крупный пакет — B05:** RF/link budgets, частоты/полосы/антенны,
EIRP/G/T/modulation/rate, погода и конкуренция ground/relay ресурсов на том же
runtime; независимые link calculations, conservation/admission/receipt/replay
и installed acceptance. B09 scale и B12 external qualification остаются gates;
B04 целиком не закрыт из-за открытых sensor/uncertainty/calibration требований.


## Исторический срез — B04-D2b3: разгрузка завершена в инженерном профиле

Добавлены explicit `definition.unloading`, ресурсно учитываемая команда
`momentum_unload` и coordinated wheel/balanced-RCS allocator. Учтены
startup, per-axis/total authority, общий бак/segment budget с основным
двигателем, DC/heat, SOC, faults/cancel/recovery, residual momentum,
pointing/rate settling, replay и atomic rollback. Никакого сброса h/omega.
[Контракт](validation/MOMENTUM_UNLOADING_CONTRACT.md),
[приемка](acceptance/B04_MOMENTUM_UNLOADING.json),
[пример](../examples/momentum_unloading/demo.py).

B04-D2a/b1/b2/b3 **доставлены как bounded engineering profile**. Физическая
qualification остается needs_external_evidence: ideal balanced continuous RCS,
shared lumped tank, fixed COM/inertia и sampled checks не заменяют flight model.
Nominal unloading завершается; limited case честно сохраняет residual и failed
status. Обе последующие EO-миссии выполняют 2/2 заявки.

**Следующий крупный пакет — B04-E:** multi-node thermal/coupling, явные
источники параметров и applicability, независимые балансы/limit tests,
mission replay и installed acceptance. Отдельные physics qualification gates:
variable COM/inertia и несколько tanks; calibrated nozzle/pulse/pressure,
disturbance/sensors/estimation; B05 RF, B09 scale и B12 industrial pilot.
B04 целиком не закрыт; отсутствие внешних данных не маскируется количеством тестов.


## Исторический срез — B04-D2b2, 1 ноября 2026

Добавлена explicit `engine.rigid_coupling`: вращающаяся BODY→GCRS тяга
по sealed RK4 quaternion path с SLERP внутри орбитального RHS; off-center
`lever_arm cross force` в той же rigid динамике; linear/angular impulse и
external mechanical-work ledger. Fuel/DC/heat, geometry revisions, полный
thrust manifest для nominal covariance, cancel/fault/rollback и mid-burn replay
согласованы. Нет скрытого wheel hold во время манёвра.
[Контракт](validation/ROTATING_THRUST_CONTRACT.md),
[приемка](acceptance/B04_ROTATING_THRUST.json),
[пример](../examples/rotating_thrust/demo.py).

Инерция и COM постоянны только в объявленном small-mass-loss режиме:
расход выше cumulative fraction прекращает дальнейшую тягу. Angular-step
applicability отвергает слишком редкую дискретизацию. Это не variable-inertia
rocket/exhaust/flight-qualified model. Старые held-body actors сохранены.
Chemical/electric EO demos выполняют 2/2 заявки, включая mid-gimbal/mid-firing
restore и действительное качество/доставку после манёвра.

**Следующий блок — B04-D2b3:** ресурсно учитываемая разгрузка momentum и
распределение wheel/external actuator authority при controlled burn;
явные fuel/DC/heat и остаточный momentum, без бесплатного external torque.
Затем variable COM/inertia/exhaust applicability и независимая calibration.
B04-D2/B04 остаются in_progress; B04-E/B05/B09/B12 открыты.


## Исторический срез — B04-D2b1, 1 ноября 2026

Rigid attitude подключен к MissionRuntime опционально через
`spacecraft_bus[actor].attitude.rigid`: реальная angular velocity/quaternion,
маховики и насыщение, команда `wheel_torque`, sampled PD `slew` и tracking,
совместные power/SOC/thermal, faults/cancel без сброса spin, actor-local telemetry,
atomic rollback и mid-wheel/CPU/tracking replay. Legacy actors не изменены.
[Контракт](validation/RIGID_MISSION_CONTRACT.md),
[приемка](acceptance/B04_RIGID_MISSION.json),
[сквозной пример](../examples/rigid_mission/demo.py).

Tuned controller выполняет 2/2 заявки; weak controller доставляет raw, но выполняет
1/2 из-за AOI coverage. Критерии покрытия и payload-rate одинаковы; отличаются
только gains. Это инженерный пример, не квалифицированный benchmark.

**Следующий блок — B04-D2b2:** вращающееся направление тяги и off-center thrust
момент на общем bounded timeline; запрет ideal-hold обходов; fuel/DC/heat и
геометрия/covariance, независимые conservation tests, replay и installed mission.
Потом B04-D2b3 — явный actuator для unloading с реальными затратами и остаточным
momentum. Сейчас body_burn на rigid actor отвергается, external unloading
не предоставлен. B04-D2/B04 остаются in_progress; B04-E/B05/B09/B12 открыты.



## Исторический срез B04-D2a — ядро вращения, 1 ноября 2026

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

B04-D2b1 validation: 33 новых mission cases; reference 652 passed (13 warnings), fast 110 passed (2 warnings), 1082 collected. Counts overlap; full regression не перезапускался. Wheel: 192/192 Python modules byte-matched, 3 frozen boinor fixtures. Installed tuned/weak demos: 2/2 и 1/2 requests соответственно, с mid-wheel/CPU и mid-tracking replay.

B04-D2b1 extended installed SDK/CLI smoke passed: tuned/weak, conservation ledgers, cancelled wheel, mid-wheel/CPU + mid-tracking replay, console validate/run/resume/replay. CLI dependency visibility выровнена с Python -I через PYTHONNOUSERSITE=1; strict fingerprints не ослаблены.

B04-D2b2 validation: 36 новых cases; reference 688 passed (13 warnings), fast 110 passed (2 warnings), 1118 collected. Counts overlap; full regression не перезапускался. Wheel: 192/192 Python modules byte-matched, 3 frozen boinor fixtures. Extended isolated SDK smoke passed; installed chemical/electric demos по 2/2 requests, mid-gimbal/mid-rotating-firing restore и conservation ledgers verified.

B04-D2b2 installed console mission-validate/run/resume/replay passed для chemical и electric. Строгая проверка runtime/dependency fingerprints сохранена.

B04-D2b3 validation (2026-11-02): 43 новых cases, reference 731 passed (13 warnings), fast 110 passed (2 warnings), 1161 collected. Counts overlap. Wheel 193/193 Python modules byte-matched и 3 frozen boinor fixtures; extended installed SDK/CLI и оба mission validate/run/resume/replay passed. Nominal unload completed; limited unload failed с residual 0.024 Nms; обе EO-миссии 2/2 requests.

B04-D2b3 final full regression (2026-11-02): **1161 passed, 58 warnings, 1513.13 s**, exit 0. Непрерывный полный прогон на финальных исходниках с разрешенным loopback, включая прежние многосуточные/16-actor сценарии; это не отдельная нагрузочная квалификация нового rigid/RCS profile. Старые отметки «full regression не запускался» выше — исторические срезы.

B04-E validation (2026-11-02): 54 новых cases; reference 785 passed (13 warnings), fast 110 passed (2 warnings), 1215 collected. Counts overlap. Wheel 195/195 Python modules byte-matched и 3 frozen boinor fixtures; extended installed SDK smoke passed, включая nominal/hot/cold и cold-start/mid-capture/mid-CPU replay.

B04-E installed console: `mission-validate`, `mission-run`, `mission-resume`, `mission-replay` passed для nominal/hot/cold; строгие source/dependency fingerprints сохранены.

B04-E final full regression (2026-11-02): **1215 passed, 58 warnings, 1512.72 s**, exit 0. Непрерывный прогон на финальных исходниках с разрешенным loopback; reference/fast counts пересекаются с full. Installed SDK и console validate/run/resume/replay passed для nominal/hot/cold. Это инженерная регрессия, не аппаратная квалификация и не отдельная B09 нагрузочная приемка thermal profile.

B05-A validation (2026-11-02): **42 новых cases; reference 827 passed (13 warnings), fast 110 passed (2 warnings), 1257 collected**. Counts overlap. Wheel 197/197 Python modules byte-matched, 3 frozen boinor fixtures. Extended isolated installed SDK и console mission-validate/run/resume/replay passed для nominal/outage/blocked; checkpoint mid-acquisition/transmission. Полная регрессия завершена успешно (итог ниже); B05-B/аппаратная qualification не закрыты.

B05-A final full regression (2026-11-02): **1257 passed, 58 warnings, 1511.44 s**, exit 0. Непрерывный прогон на финальных исходниках с loopback, включая прежнюю 16-actor/72h миссию. Wheel 197/197 source modules и 3 boinor fixtures byte-matched; installed SDK/CLI checkpoints для nominal/outage/blocked полностью совпали. Это инженерная регрессия и совместимость, не отдельная B09 нагрузочная квалификация RF и не B12 hardware/flight calibration. B05-B остается следующим пакетом.

B05-B validation (2026-11-02): **44 новых cases; focused 45 с legacy regression; reference 871 passed (13 warnings), fast 110 passed (2 warnings), 1301 collected**. Counts overlap. Wheel 198/198 source Python modules byte-matched и 3 frozen boinor fixtures; extended isolated installed SDK и CLI mission-validate/run/resume/replay passed для nominal/reroute/loss/ttl. SDK/CLI final checkpoints совпали; source/installed demo results совпали; content hash сохранился при смене owner. Полная регрессия завершена успешно (итог ниже).

B05-B final full regression (2026-11-02): **1301 passed, 58 warnings, 1554.35 s**, exit 0. Непрерывный полный прогон на финальных исходниках с loopback, включая 16-actor/72h tasking scenario. Source payload 198/198 modules и 3 boinor fixtures byte-matched; installed SDK/CLI и четыре миссии nominal/reroute/loss/ttl passed с identical checkpoints и сохраненным content hash. Финальный wheel обновил только README metadata/RECORD; идентичность исполняемого payload доказана, reinstall/replay всех четырех режимов passed. Это не отдельная B09 scale qualification и не B12 hardware/flight/protocol qualification. Следующий пакет — B06.
