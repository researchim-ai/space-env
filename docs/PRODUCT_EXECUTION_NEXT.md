# Следующие крупные пакеты реализации

## Текущий пакет — B10-B: восстановление и управление operator ↔ agent

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

Актуально на 1 ноября 2026.

**Единая очередь:** [INDUSTRIAL_AI_SIMULATOR_PLAN.md](INDUSTRIAL_AI_SIMULATOR_PLAN.md).
В ней находятся B01–B12, задачи, зависимости, критерии приемки и журнал закрытия.
Этот файл — короткая навигация, а не конкурирующий roadmap.

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


## Последовательность

1. B01: общий runtime и сценарий.
2. B02: управляемая reference-физика и независимая квалификация.
3. B03: request → acquisition → processing → delivery.
4. B04: связанный аппарат, ADCS/payload/энергия/здоровье.
5. B05: RF, наземные ресурсы и конкурентная ретрансляция.
6. B06: одна задача для RL/MARL/LLM/оператора.
7. B07: планирование, objectives и честный benchmark.
8. B08: воспроизводимая фабрика RL/SFT/preference данных.
9. B09: исполняемые нагрузки 16/100 → 64/1000 → отдельно 256/10000.
10. B10: Studio/оператор на общем runtime.
11. B11: release engineering и эксплуатация.
12. B12: данные промышленного пилота, калибровка и приемка.

## Реализованная база и доказательства

[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) фиксирует v1–v10 и проверки.
[Исходный аудит/roadmap](INDUSTRIAL_ROADMAP_V0134.md) сохраняет историю разрывов.
[Контракты и ограничения моделей](validation/) и [acceptance artifacts](acceptance/)
нужны для приемки; они не заменяются количеством функций или тестов.

v10: reference 249 passed, fast 110 passed, 679 collected, installed-wheel smoke.
Предыдущий полный v9: 665 passed, один sandbox socket failure, тот же HTTP-тест
отдельно passed с loopback-доступом. Это не непрерывный green-run на v10.

B01: runtime 38 passed, reference 287 passed, fast 110 passed, 717 collected;
installed wheel и console resume/replay подтверждены. Полный B01 regression не запускался.
B02: 18 новых тестов, reference 305 passed, fast 110 passed, 735 collected;
wheel 178/178 модулей и установленная dynamic CLI подтверждены. Полный regression
не перезапускался. B02 needs_external_evidence (исторический срез на завершении B02).
B02-Q1: 24 новых теста, reference 329 passed, fast 110 passed, 759 collected;
wheel 180/180 Python modules, 3 orbit fixtures, covariance CLI workflow проверены.
B02-Q2: 28 новых tests, reference 357 passed, fast 110 passed, 787 collected;
wheel 183/183 modules и installed SDK/CLI с binding/stale rejection проверены.
B03: 31 новых tests, reference 388 passed, fast 110 passed, 818 collected;
wheel 184/184 modules, installed EO lifecycle SDK/CLI и полная receipt проверены.
B04-A: 29 новых tests, reference 417 passed, fast 110 passed, legacy attitude
19 passed; 847 collected. Wheel 185/185 modules и 3 fixtures, installed SDK/CLI.
B04-B1: 32 новых cases, reference 449 passed, fast 110 passed, 879 collected;
wheel 186/186 Python modules и 3 frozen fixtures, isolated SDK/CLI acceptance.
B04-B2: 39 новых cases, reference 488 passed, fast 110 passed, 918 collected;
wheel 187/187 modules и 3 frozen fixtures, isolated optical/SAR SDK/CLI verified.
B04-B3: 37 новых cases, reference 525 passed, fast 110 passed, 955 collected;
wheel 188/188 modules + 3 fixtures; isolated SDK/CLI обоих режимов — в артефакте приемки.
B04-C: 34 новых cases, reference 559 passed, fast 110 passed, 989 collected;
wheel 189/189 modules + 3 fixtures; isolated resource-lanes SDK/CLI обоих режимов verified.
Full regression не запускался. Следующее «продолжаем» выполняет B04-D.
Commit/push/deployment — только по отдельной команде.

B04-D1: 39 новых cases; итоговые проверки — в acceptance artifact. Следующий шаг B04-D2.

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
