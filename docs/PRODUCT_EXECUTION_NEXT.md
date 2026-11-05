# Следующие крупные пакеты реализации

## Рабочая страница миссии и улучшенный fleet scheduler

Реализован единый путь: редактирование орбит/станций/заявок → план → native
выполнение → фактические результаты → RL/SFT export. Главная страница сервера
открывает этот интерфейс; SDK `MissionWorkspace` использует ту же сессию.
Планировщик сравнивает исходное расписание с ограниченным поиском порядка
и компактным резервированием; общий бюджет узлов ограничен. Локальные проверки:
128 passed; стартовая миссия доставляет 2/2 заявки, включая прогон в браузере.
[Инструкция и точные ограничения](MISSION_WORKSPACE.md). Профиль ограничен
4 аппаратами; общей оптимальности,
автоматического перепланирования и квалификации реальной аппаратуры не заявляем.

## Следующий блок — автоматическое назначение заявок

Добавлен ограниченный поиск пары аппарат/станция для свободных полей заявки.
Ручные назначения сохраняются; исходная и разрешённая конфигурации разделены,
история вариантов и общий расход бюджета доступны оператору. SDK, API и форма
используют один механизм; успех подтверждается физическим исполнением.
Это поиск начального плана, без автоматического перепланирования после отклонения.
Локальный объединённый набор: 139 тестов прошли (138 в основном запуске;
HTTP loopback-тест отдельно с разрешённым сокетом после sandbox PermissionError).
В браузере: 7 вариантов назначения, фактическая доставка 2/2 за 120 с.

## Текущий порядок разработки: без CI

По решению пользователя GitHub Actions/workflows удалены. Для разработки и
выпуска используем локальные тесты и ручной запуск примеров. Remote/full CI,
матрицы и обязательные кампании приёмки не являются открытыми задачами или
блокерами. [Установка и запуск](LOCAL_RUN.md), [актуальный план](RELEASE_FINISH_PLAN.md).
Ниже сохранена история выполненных пакетов; прежние упоминания CI описывают
прошлые планы и не задают текущие требования.

## B12 — release qualification, общий forecast и явные ML splits

Реализованы bundled scope manifest и [область выпуска](RELEASE_SCOPE_V1.md),
[32-case benchmark](validation/RELEASE_BENCHMARK_V1.md), одинаковая forecast
информация для трёх методов и bounded ordering search. Frozen train/development/
test geometry groups, независимые процессы, полный знаменатель и error rows.
Measured step p50/p95, RSS, checkpoint/replay/export и storage-capacity probes.

Wheel `39d1db7680ea…`:283 modules/324 source+data files и13 assets совпали с
checkout/installed payload. **1911 reference и110 fast passed** (пересекаются);
22 benchmark tests с clean dependencies,4 orchestration и4 split-guard cases.
Чистый Python3.12 venv:63 точные зависимости, pip check passed. Восемь зависимостей
обновлены после88 initial advisory records; повторный audit не нашёл известных
записей. Исходные findings сохранены. Это не remote/full CI и не legal review.

32 физических сценария ×3 policies:4GiB profile дал96 native-verified rows.
93 прошли strict64 MB per-result resource gate; три long exports заняли около
64.36 MB. Их повтор с128 MB сохранил эпизоды побайтно и прошёл resource gates.
Homogeneous full128 MB profile локально не выдаётся за выполненный; он задан
как необязательный локальный профиль.1GiB profile:72 native rows,24 storage-budget rejects.
Оба профиля и capacity retry относятся к тем же32 cases, не к новым holdouts.
Cold recovery проверен в отдельных PID на idle и active-command prefix.

Важный split finding:legacy lineage hash маркирует одну test geometry group как
train. Raw native benchmark bundles остаются evaluation/provenance.
[Новый exporter](validation/RELEASE_DATA_SPLITS_V1.md) создаёт явные partitions:
3600 train,3600 development,7200 test RL rows. Approved-SFT отсутствует:teacher
service/uncertainty gates не пройдены; expert_teacher_qualified=false.
Исходные observations/actions не меняются, source row/file SHA проверяются.

Приёмка и точные receipts: B12_RELEASE_ACCEPTANCE.json. Архивы сохраняют source
bundles, исправленное ML-разбиение, wheel, lock и recipes. Большие generated archives
исключены из обычного git add; receipts/inputs/recipes остаются reviewable. Restored archive audit
проверяет consistency; native replay выполнен на originals. CI удалён из плана;
дополнительные measurements и benchmarks необязательны. C2, long rigid, Q7 и
экспертная teacher qualification относятся к отдельным областям применимости.


## Исторический пакет D2h-B2b-B — incoming demand и версия каталога

Реализованы native request_added, append-only journal, request_ref для повторных
съёмок существующих целей, explicit actor rebind и cumulative demand certificate.
Полный знаменатель включает новые невыполненные заявки; SFT разрешён только при
выполнении mission gates. [Контракт](validation/VERSIONED_DEMAND_V1.md).
Приёмка: B11_DEMAND_ACCEPTANCE.json (точный wheel и завершённые проверки).
Installed cases:4/4 batch,4/4 two updates,2/3 tight deadline с RL-only;
fresh-process checkpoint/data equality и Gym/Parallel physical equality подтверждены.
1889 reference/110 fast passed (профили пересекаются);19 новых demand cases.
Числа и wheel предыдущего пакета ниже относятся только к предыдущему коду.

Дальнейшая разработка ограничена [конечным планом выпуска](RELEASE_FINISH_PLAN.md):
independent workload/baselines → stress envelope → clean CI → RC. Equal-information
teacher qualification, C2, long rigid и Q7 не объявляются закрытыми.


## Историческая приёмка D2h-B2b-A — live actors и flat incident history

Общий projector связывает actor datasets и live SDK. Добавлены atomic coordinator
с binding/idempotency/authority checks, Gym joint Dict и optional PettingZoo
Parallel adapters. Checkpoint восстанавливает reference и actor trace через
native replay; actor policy может materialize реально выбранный prefix как plan.
[Контракт](validation/ACTOR_POLICY_JOURNAL_V1.md).

Flat journal хранит origin один раз, до8 epoch nodes с delta actor edits и native
source witnesses. Committed history неизменна, новые operator injections не
разрешены в actor edits. Actor RL и условный completed-goal SFT проверяются
построчно. New demand/catalogue mutation, measured calibration, rigid и hosted
изоляция остаются открыты. Весь D2h-B2b не объявляется закрытым.


Приёмка: [B11_ACTOR_ACCEPTANCE](acceptance/B11_ACTOR_ACCEPTANCE.json), wheel
`dc388d030ae6…`;278 modules/318 source+data files совпали с wheel/installed payload.
**1870 reference,110 fast,3 adapter smoke passed** (пересекаются);22 новых actor
cases включены в reference. Полная/remote CI и dependency-clean окружение не
запускались: installed venv с system-site-packages, Python -I вне checkout.
Три synthetic mechanism cases:2 epochs с actor wait вместо TX →2/24;
failed policy →0/24 с RL-only;8 native epochs →2/24. Journal8=1077272 bytes.
SDK/Gym/Parallel по27 actor steps дали одинаковый physical report. Отдельный
процесс128→180s восстановил полный checkpoint и data manifest побайтно;
CLI start/observe/step совпал с SDK, третья CLI-эпоха прошла native prefix audit.
Сохранены206 actor RL+130 SFT-кандидатов (включая повторный replay-вариант,
не новый независимый workload).61-file archive восстановлен и прошёл consistency
аудиты; native audits выполнены на originals. Expert qualification остаётся false.

## Исходный scope D2h-B2b-B — mechanism реализован; independent baselines остаются

1. Typed incoming customer requests/jobs на decision boundary; строгая graph/
   ownership/release/deadline validation и сохранённый полный demand denominator.
2. Versioned action catalogue с immutable executed templates и явным rebind
   actor policy observation/action mapping после обновления. Новый epoch Env
   вместо скрытого изменения spaces внутри episode.
3. Увязать новые заявки с journal native witnesses, checkpoint/replay и actor
   dataset lineage, включая rejects/неполучившие обслуживание заявки.
4. Frozen surge/deadline/capacity cases и equal-information baseline до promotion
   teacher quality. Не заменять реальные service gates количеством API/tests.

Bounded replay caching/throughput отдельно от correctness; C2 measured calibration,
long rigid и Q7 hosted isolation не объявлять закрытыми за счёт нового catalogue.

## Предыдущий пакет D2h-B2a — incident и phase-aware recovery

Добавлен один committed operator incident исходного B1 proposal: отмена активных
control windows, station outage и новый suffix по native остаткам данных/CPU.
Final повторно не снимается, paused CPU не сбрасывает прогресс, потерянный capture
при max_attempts=1 остаётся terminal. Обе границы prefix проверяются cold replay.
[Контракт](validation/INCIDENT_RECOVERY_V1.md).

Actor suffix данные отделены от operator_trace: только собственный catalogue,
пересчитанная mask/binding, team reward, условный SFT action prediction. Инъекции
fault не являются agent targets. Native audit регенерирует каждую derived строку.
Экспертность, measured calibration, arbitrary incidents, новые demand и несколько
incident epochs не заявлены. Следующий пакет — D2h-B2b: versioned demand/catalogue
и bounded multi-epoch history с сохранением prefix/conservation.


Приёмка: [B11_INCIDENT_ACCEPTANCE](acceptance/B11_INCIDENT_ACCEPTANCE.json), wheel
`fa04a5c476f3…`;271 modules/311 source+data files совпали с wheel и installed payload.
**1848 reference,110 fast,6 compatibility passed** (пересекаются);26 новых incident
cases включены в финальный reference. Full regression/remote CI/dependency-clean
окружение не запускались: installed venv с system-site-packages и Python -I.
5 зафиксированных synthetic cases: partial TX2/24, paused CPU2/24, interrupted
slew2/24; lost capture0/24 и blocked station0/24 сохранены с failed goals/RL-only.
6 actor dataset native audits, включая CLI/new-process continuation104→180s,
совпавшее с direct по physical report.564 actor RL+406 SFT-кандидатов;
540 operator RL+360 operator SFT отдельно, не agent-training entry points.
148-file archive (187616024 uncompressed bytes) восстановлен и consistency-аудирован;
полные native audits выполнены на originals. Pre-hardening interrupted evidence
и прежний wheel сохранены отдельно, не включены в финальную квалификацию.

## Карта D2h-B2b: online/journal закрыты в bounded профиле, new demand/benchmark открыты

1. Общая online actor projection/coordinator: те же catalogue/mask/binding, что
   в actor dataset, action-ID decoder и equivalence tests на live TaskSession.
2. Versioned new demand/catalogue на реальной decision boundary, typed budgets,
   release/deadline/ownership bindings и native conservation без подмены прошлого.
3. Bounded multi-epoch incident journal без рекурсивного копирования полного
   record; checkpoint replay и независимый restore на каждом committed epoch.
4. Frozen нагрузки/отказы/новые заявки, сохранённые failed cases и baseline с
   равным information budget. Expert/teacher promotion только по отдельным gates.

Закрытие B2b не заменяет C2 measured calibration, long rigid qualification и Q7
hosted resource/auth isolation; они остаются отдельными критериями готовности.

## Предыдущий пакет D2h-B1 — связанные цепочки и idle-boundary replan

Прогнозный планировщик без preload формирует paid slew → acquisition → metadata
transform → downlink. Новый явно разрешённый `forecast_pointing` даёт будущий
геометрический quaternion без admission oracle/rollout. Поиск value-only,
консервативно резервирует actor/time/terminal/energy/TX/memory; native bundle
по-прежнему является проверкой declared goal, не optimality/expert certificate.
[Контракт](validation/FORECAST_CHAINS_V1.md).

Добавлен один idle-boundary suffix replan: source checkpoint проходит cold replay,
каждая затронутая цепочка должна завершиться, прошлые steps/templates сохраняются,
новый prefix исполняется заново и сравнивается по physical native report.
Mid-operation interrupts, новые operator events/demand, recursive epochs,
rigid tracking и equal-information teacher benchmark ещё не реализованы в этом
профиле. Весь D2h-B не объявляется закрытым; следующий пакет — D2h-B2.


Приёмка: [B11_CHAINS_ACCEPTANCE](acceptance/B11_CHAINS_ACCEPTANCE.json), wheel
`6a75da954b59…`;268 modules/308 source+data files совпали с wheel и installed payload.
**1822 reference,94 targeted,110 fast passed**;30 новых cases, наборы пересекаются.
Полная/remote/dependency-clean CI не запускалась; venv с system-site-packages.
Installed nominal2/24, shifted0/24, outage0/24; failed goals сохранены с RL-only.
Fresh-process continuation110→600s совпало с direct replan по physical report.
Отдельный синтетический pending-suffix case4/26 проверил2 completed+2 future chains
и новые forecast bindings;90deg/s fixture не объявляется реальным hardware/holdout.
5 export cold replays,3 дополнительных full native audits;570 RL и330 SFT-кандидатов.
Архив восстановлен и прошёл exact inventory + consistency audits; native audits
выполнялись на оригинальных bundles. Ни superiority, ни expert quality не заявлены.

## Предыдущий пакет D2h-A — forecast-first ready-data delivery

Добавлены value-only operator-authorized forecast, bounded deterministic scheduler,
fixed MissionPlan до rollout, CLI proposal/paired benchmark и независимый audit.
Учитываются ready/deadline, actor/terminal capacity, initial conservative energy,
TX quota и полный знаменатель. [Контракт](validation/FORECAST_DELIVERY_V1.md).

Frozen benchmark: train, два разных географических holdout и full-outage negative
control, одинаковые физические ресурсы для forecast/priority/earliest_deadline.
Information budgets различны и явно записаны. Limited teacher gates включают
native replay, service/rejection thresholds, prediction realization и paired
noninferiority. `expert_teacher_qualified=false` независимо от passing gates.

Следующий D2h-B: acquisition/processing/paid pointing dependencies, rolling
replanning, forecast baselines с равной информацией и более сложная конкуренция.
Новый scheduler не закрывает long rigid, measured calibration C2, hosted Q7
или универсальное промышленное моделирование. Проверки и receipt — ниже.


Приёмка D2h-A: [B11_FORECAST_ACCEPTANCE](acceptance/B11_FORECAST_ACCEPTANCE.json),
wheel `4223f2f462e6…`, 266 Python modules / 306 source+data files byte-matched.
**1792 reference, 31 targeted, 110 fast passed** (пересекаются); full regression
и remote/dependency-clean CI не запускались. Installed venv использует system-site-packages.
12 paired native runs + 12 обязательных cold replays + 12 дополнительных native
audit replays. Candidate/оба baseline:4/16 на каждом из3 positive cases,
0/16 на outage,0 rejection. Passing minimal declared gates означает только
noninferiority в этом наборе, не superiority или expert quality. Сохранены400 RL
и300 SFT-кандидатов;62-file archive восстановлен и проверен по inventory,
после восстановления — consistency audit, не повторный native rollout.

## Дорожная карта D2h-B: B1 реализован в ограниченном профиле, B2 остаётся

Порядок реализации:

1. Расширить value-only contract release/capture/processing/pointing dependencies;
   запретить чтение private state и native admission при поиске плана. Разделить
   идеальные/кинематические/rigid предположения, неподдержанное отвергать.
2. Построить resource calendar для payload/CPU/attitude/radio и release/deadline;
   явно учитывать длительность и энергозатраты наведения. План до исполнения,
   diagnosis невыполнимости с полным request denominator.
3. Replan по новому наблюдению: immutable executed prefix, новые bindings,
   отказ от stale forecasts; interruption и fresh-process continuation.
4. Добавить forecast-aware baselines с теми же входными прогнозами, frozen
   contention/deadline/release/load cases, разные широты станций и отказы.
   Заранее объявить service floors и сравнивать каждый paired case без pruning.
5. Сохранить планы/неуспехи/датасеты и native replay receipts. Для SFT отдельно
   указывать goal feasibility, benchmark selection и expert qualification.

Критерии закрытия: native capture→processing→delivery без preload, оплаченные
ресурсы наведения, корректный forecast invalidation, terminal/resource conflicts,
negative controls без SFT и не менее двух независимых positive holdout на
семейство. Long rigid и measured calibration закрывать отдельными receipts;
прохождение kinematic tests не повышает автоматически уровень fidelity.

## Предыдущий пакет D2g-A — исполнимый план и сохраняемый RL/SFT bundle

Добавлены sealed `MissionPlan`, fixed `PlanExecution` поверх TaskSession,
build/seed/scenario/task bindings и native checkpoint continuation. Публикация
сохраняет plan, execution, goal certificate и RL/SFT вместе; failed goal сохраняет
RL, не получает SFT. Atomic staging, byte limits, строгий inventory и typed
canonical JSON audit покрывают повреждения и подмены. Native mode пересоздаёт
каждую строку из independently replayed episode. [Контракт](validation/MISSION_PLAN_BUNDLE_V1.md).

Recorded candidate из existing causal rolling baseline — первая реализация
proposal, не новый forecast optimizer. Компиляция чужого record не доказывает
causality его source policy. Certificate доказывает declared model-goal
feasibility фиксированного плана, не optimality, measurement calibration или
expert teacher quality. Последний статус всегда false. Remaining D2g/D2h ниже.

Приёмка: [B11_MISSION_PLAN](acceptance/B11_MISSION_PLAN.json), wheel `2f7811dbad94…`.
1761 reference,129 targeted,110 fast passed (пересекаются),33 новых tests.
3 installed workflows с fresh-process continuation и native export replay:
positive2/24, intentionally infeasible goal0/24 (RL only), full12600s8/32.
Сохранены1440 RL+1350 SFT строк,68-file exact archive и17 linked artifacts.
Short bundles прошли отдельный native audit, в том числе после archive restore;
full — mandatory native export replay и integrity audit. Source policy causality,
expertise и forecast optimizer этим не квалифицируются. Следующий пакет — D2h.

## Предыдущий пакет D2f — bounded full-orbit missions и geometry refresh

Добавлен отдельный `FullOrbitSpec`/`full_orbit_workload`, CLI
`compile-full-orbit`, cycle-specific requests без preload и явно выбранные
kinematic/rigid ADCS tiers. Geometric diagnostic измеряет actual J2 winding
и сохраняет все actor×station пары. Epoch service gate требует release,
acquisition и delivery в одном периоде, сохраняя весь request denominator.

После maneuver пересчитывается suffix на прежней scan lattice, past windows
сохраняются. Для dense gravity-only balanced RCS добавлен explicit opt-in
coast reuse; legacy default остаётся eager. Drag/SRP/navigation/active thrust
не попадают в fast path. Исправлен ulp endpoint bug repeated decimal burns.
[Контракт и запуск](validation/FULL_ORBIT_MISSIONS_V1.md).

Приёмка: [B11_FULL_ORBIT](acceptance/B11_FULL_ORBIT.json), wheel `894d6d435420…`.
4 full-mission attempts по12600s прошли: nominal8/32, outage0/32 при8 captures;
2 fresh-process restores на6300s совпали с direct. Winding обоих actors>2.
Dense-coast envelope40→0 propagation calls и2 native actuator programs прошли.
1728 reference,176 targeted,110 fast passed (пересекаются); findings сохранены.
Long rigid qualification и independent executable feasibility certificate
не объявлены закрытыми. Следующий самостоятельный раздел — D2g ниже.

Это первые bounded full-cycle workloads, не универсальный calibrated mission
simulator. Геометрический diagnostic не заменяет executable feasibility schedule.
Критерии remaining D2f ниже остаются актуальными для teacher/feasibility,
long rigid refinement и realistic station/network deployment.

## Предыдущий пакет D2e — fixed-clock actuator coupling

Unloading получил собственный joint sample-and-hold wheel/RCS loop с immediate
guards, owned latch, capture/fork/rollback/replay. Добавлена frozen18-member
RK/partition программа rotating chemical thrust + unloading, shared tank,
analytic axial impulse и retained limited-authority failure. Extended600s
nominal/outage direct/restored profiles имеют заранее заданные service minima.
[Контракт](validation/FIXED_CLOCK_ACTUATORS_V1.md).
Это synthetic hardware verification, не flight calibration, активный engine
attitude hold или полный orbital mission. Full-cycle service/C2/Q6/Q7 открыты.

Офлайн-аудит `verify-coupled-actuators` пересчитывает witnesses и comparisons,
проверяет frozen recipe, полный member/file inventory, scenario identity и
checkpoint digests. Integrity и qualification разделены: корректно сохранённый
failed member остаётся неудачей. Отдельный weekly/manual installed-wheel CI job
сохраняет артефакты независимо от результата. Remote CI локально не заявляется.

Приёмка D2e: [B11_ACTUATORS](acceptance/B11_ACTUATORS.json), wheel
`87228be247b2…`.18 actuator members и audit прошли;2 midphase и4 extended
installed executions прошли с3 fresh-process restores. Nominal600s2/24,
all-outage0/24 при2 acquisitions. Reference1705 passed+1 HTTP test отдельно
с localhost permission;21 actuator tests+1 CLI test,110 fast (пересекаются).
Подробные границы и полный archive описаны в контракте. D2f ещё не реализован.

### D2f — следующий большой пакет: полный цикл миссии и производительность

1. **Сначала стоимость geometry refresh.** Измерить число propagation/access
   calls на фиксированной actuator программе для partitions0.1/0.025/0.01s.
   Оптимизировать пересчёт будущих окон после maneuver без изменения прошедшей
   истории; отдельно доказать допустимость fast path для ideal balanced RCS.
   Нельзя пропускать изменение массы при drag/SRP, забывать active burns или
   использовать stale contact cache. Сверить с eager reference на общей сетке.
2. **Full-orbit compiler.** Текущий coupled compiler ограничен600s; не выдавать
   снятие bound за готовую long-horizon модель. Добавить явно bounded новый
   профиль минимум на два оборота с независимыми fixed sites и несколькими
   contact cycles, лимитами памяти/числа действий/segments и closed-loop
   acquisition→CPU→delivery. Зафиксировать входы до оценки политики.
3. **Положительная проверка feasibility.** Раздельно считать геометрическую
   возможность, paid pointing/settling, storage/CPU/energy и occupied terminals.
   Минимальный service floor обосновать воспроизводимым executable schedule,
   затем сравнить baseline с теми же ресурсами. Все requests остаются в denominator.
4. **Восстановление на физических границах.** Direct/fresh-process пары перед
   acquisition, внутри transform и между AOS/LOS; повреждение и остановка
   worker сохраняются как failed rows. Измерять replay cost отдельно от step cost.
5. **Приёмка и границы.** Зафиксировать mesh/partition/event envelopes,
   conservation ledgers, installed-wheel identity и все failed attempts.
   Затем Q6: frozen train/dev/test task families и baselines; Q7: durable jobs,
   OS quotas, tenancy/auth. C2 требует измерений; открытые specifications
   продолжают маркироваться как specifications, синтетические данные — synthetic.

Критерий завершения D2f: два полных оборота и несколько доставок в разные
контактные окна, обоснованный положительный service floor, matching process
continuation, regression относительно eager geometry и retained negative controls.
Это ещё не flight qualification или универсальная модель всех spacecraft systems.

### D2g — исполнимый mission plan и сохраняемый teacher dataset

1. Ввести bounded JSON `MissionPlan`: scenario/task/source/data hashes,
   время выпуска, горизонты, actor/command IDs, planned start/end и declared
   assumptions. Отделить прогноз от фактических execution receipts.
2. Сделать детерминированный planner на доступных forecasts и публичных
   constraints: paid slew/settling → dwell → CPU/storage → ground opportunity.
   Нельзя планировать с truth-only admission oracle, перемещать targets/sites
   или удалять неудобные requests. First feasible plan не объявлять optimal.
3. Исполнять план через native revision/idempotency API, сохраняя rejections,
   interruptions, actual times и resource ledgers. Никакой принудительной смены
   attitude, inventory или request status для получения service floor.
4. Сохранять полный executed checkpoint/trace и независимый feasibility
   certificate. Проверять два release epochs, разные contact windows,
   acquired→CPU→delivery lineage, resource occupancy и direct/restored equality.
   Geometric capacity ceiling и feasible native lower bound считать отдельно.
5. Сравнить plan executor и rolling baseline на одних frozen inputs и ресурсах;
   retained infeasible/outage/weak-hardware controls обязательны. Holdouts
   зафиксировать до разработки эвристики; calibration claims не добавлять.
6. Сохранить verified RL/SFT export рядом с certificate, вместо временного
   удаления export после benchmark timing. Offline audit должен восстанавливать
   связь строки dataset с native trace, plan, code/data identity и split.
7. Критерий закрытия: installed-wheel executable positive-service plan,
   два full cycles, separate-process continuation, повторный offline dataset
   audit из сохранённого bundle и сохранённые failed attempts. Long rigid
   sensitivity/refinement остаётся отдельным физическим пакетом.

### D2h — forecast planner, teacher gates и frozen benchmark

1. Сначала зафиксировать independent train/dev/holdout inputs и budgets:
   actor planes/phases, fixed sites, releases, outages и explicit service quotas.
   Known D2f/D2g development cases не выдавать за unseen generalization.
2. Добавить value-only public forecast adapter: interval/trajectory-token/source/
   validity bindings и declared geometric/RF assumptions. Запретить private
   truth, future stochastic outcomes и native admission oracle в policy input.
3. Реализовать bounded deterministic interval scheduler для paid pointing/
   settling, dwell, CPU/storage и ground occupancy. Schedule должен явно
   включать waiting/holds; нельзя получать результат принудительным изменением
   physical state или удалением необслуженных requests.
4. Передавать candidate schedule новому fixed executor, сохраняя rejected и
   failed operations. First feasible lower bound и independent conditional
   capacity upper bound считать отдельно, не называть эвристику optimal.
5. Сравнить с rolling/priority baselines на одинаковых frozen resources.
   Teacher gates должны явно требовать service/rejection/robustness thresholds,
   а не только native replay. Infeasible/outage controls обязательны.
6. Использовать retained bundles для central и затем local-role RL/SFT с
   episode-family split и duplicate audit; исключать failed/weak teachers по
   declared gates. Metadata/native certificates не вкладывать в prompts.
7. Приёмка: installed-wheel forecast proposal → native plan execution →
   retained dataset → independent native audit, positive heldout service и
   сохранённые все failures. Long rigid refinement/C2/Q7 остаются отдельно.


## Предыдущий пакет D2d — independent controller clock

Opt-in fixed-tick/1 удерживает motor torque между global ticks и реагирует на
команды/protection без ожидания tick. Сохранены legacy event-driven semantics,
энергоучёт, saturation, capture/rollback/fork/replay. Добавлены frozen native
command partition program и mandatory event-resolution gate в fixed-clock study.
[Контракт](validation/FIXED_CONTROLLER_CLOCK_V1.md).
Разные digital controller periods остаются разными моделями; controller sensitivity
не объявляется закрытой одним event-invariance gate. Long-horizon service/C2/Q6/Q7 открыты.
Приёмка: [B11_FIXED_CLOCK](acceptance/B11_FIXED_CLOCK.json), wheel `062b7b4d9bd0…`.
1685 reference,154 targeted,110 fast passed (наборы пересекаются).18 final installed
attempts:13 mission+3 frozen programs+2 mid-tick process runs. Event gate обязателен
и пройден; q difference на контрольной сетке≤2.431e−12deg при прежнем budget0.1deg.
Fresh-process prefix5.15s сохранил last sample5.1s. Multi-period robustness gate
по-прежнему false (0.2↔0.1s:5.1738deg); разные periods — разные controllers.
Следующий D2e: disturbances/engine qualification, fixed-input RK oracle, затем
independent feasible long-horizon positive-service benchmark. Calibration C2/Q6/Q7 открыты.



## Предыдущий пакет D2c — refinement и unseen geometry studies

Добавлены bounded evaluator-only numerical samples, max/RMS comparison,
условная трёхсеточная kernel self-refinement проверка и отдельные controller/
lifecycle cadence sensitivity reports. Frozen study compiler объединяет7
resolution configurations и direct/restored пары3 геометрически новых случаев.
Все неуспешные attempts и service outcomes сохраняются. SDK/CLI плюс offline
checksum/recomputation audit. [Контракт](validation/MISSION_REFINEMENT_STUDY_V1.md).
Это model verification, не calibrated physical truth и не ML generalization.
Fixed-input/orbit/contact refinement, измерительная калибровка C2 и Q6/Q7 открыты.
Приёмка: [B11_MISSION_STUDY](acceptance/B11_MISSION_STUDY.json), источник257 modules/
290 files, wheel `89bf4fc8cb1e…`. 1671 reference,198 targeted,110 fast passed
(пересекаются);13 corrected installed attempts и3 restore pairs. Первые13 попыток
с дефектом witness и failed reference log сохранены в B11_STUDY_FINDING.
Kernel self-refinement прошёл, **temporal_stable=False**, strict release gate
не пройден: controller max attitude difference4.1571°,event-resolution0.6558°.
Unseen cases имеют0 deliveries и declared min service0; SLA не квалифицирован.
Следующий D2d: explicit latched controller clocks/safety interrupts, fixed-input
replay refinement, затем feasible long-horizon service и новый untouched набор.



## Предыдущий пакет D2b — independent coupled metadata mission

Реализованы bounded compiler независимых planes/targets/stations, rigid paid
slew/tracking, acquisition→CPU→ground без preload; полный actor×target denominator.
Public-only rolling heuristic использует current pointing и даёт время на settling.
Nominal/weak-wheels/all-outage direct/restored workloads, coupled lineage/resource
witnesses интегрированы в scale worker и ensemble audit. Это metadata acquisition,
не изображение реального сенсора. [Контракт](validation/COUPLED_INDEPENDENT_MISSION_V1.md).
Numerical refinement, independent mission holdout и C2/Q6/Q7 остаются открытыми.
Приёмка: [B11_COUPLED.json](acceptance/B11_COUPLED.json), wheel `ccd258b24a00…`.
1641 reference,162 targeted,110 fast passed (пересекающиеся наборы),6 installed
attempts и3 separate-process restore. Nominal2/24; weak0/24; outage0/24 при2
acquired+processed. Full expanded suite/remote dependency-clean CI не запускались.


## Предыдущий шаг D2b — reference pointing

Добавлены snapshot-at-admission команды наведения на registered target/station,
current ideal pointing telemetry и opt-in causal masks. Legacy quaternion/task
serialization сохранена. [Контракт](validation/REFERENCE_POINTING_V1.md).
Prerequisite использован в последующем coupled profile; ограничения его
приёмки и независимой физической достоверности описаны отдельно.

## Предыдущий пакет — B11-Q5-D2a: независимая геометрия и capacity audit

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
[Приёмка](acceptance/B11_GROUND.json) ·
[Следующие крупные уровни](INDUSTRIAL_NEXT_AFTER_GROUND.md).

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
