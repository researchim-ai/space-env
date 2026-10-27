# Space-Env после 0.73.0: чёткий план следующей разработки

> **Актуальная очередь после 0.93:** [план развития](DEVELOPMENT_PLAN_V093.md), [аудит текущего кода](audits/V093_AUDIT.md). Ниже сохранён исторический backlog: компоненты World/DSL/tasking/72-hour missions уже частично реализованы, не создавать их повторно без сверки.

Актуально на 25 октября 2026, HEAD `fcab948`. Основание — [новый аудит](audits/V073_STATE_AUDIT.md), [инвентаризация](audits/v073_inventory.json) и [runtime probes](audits/v073_probe_results.json).

Этот документ — актуальная очередь реализации для следующего агента. [STARTUP_DEVELOPMENT_PLAN.md](STARTUP_DEVELOPMENT_PLAN.md) сохраняет глобальную продуктовую стратегию, [старый master plan](../SPACE_ENV_AGENT_MASTER_PLAN.md) — дальний backlog. При конфликте состояния/приоритетов использовать этот документ. Номер 0.74 или 0.80 не является критерием готовности: пакеты закрываются доказательствами.

## 0. Правила для исполнителя

- Ничего не пушить, не создавать PR и не публиковать без отдельного поручения владельца. Не коммитить автоматически. Не сбрасывать, не переписывать и не переключать пользовательское рабочее дерево.
- Проверить фактическую версию, инструкции и diff перед началом: после аудита код мог измениться.
- Первые действия — regression cases для найденных ошибок; не менять физические допуски, чтобы скрыть их.
- Оставить 32 существующих env доступными. Учебные задачи пометить как tutorial/procedure; новые mission semantics получать новый env ID/version.
- Не создавать ещё один самостоятельный buffer, clock или success evaluator. Исправления должны соединять существующие реализации.
- Держать `docs/IMPLEMENTATION_STATUS.md`: пакет, зависимости, status, evidence, changed contracts, остающиеся ограничения. Начальный статус всех работ ниже — `todo`.
- Сохранять configuration/backend/input fingerprints и dirty-tree fingerprint. Данные, построенные с некорректным UTC/access, версионировать как legacy; не смешивать с исправленным benchmark.

## 1. К чему приходим

Общий **MissionWorld**, в котором satellite models, orbit/physics profile, operations, ground network, requests и observation model задаются конфигом. Человек в Studio, classical planner, RL/MARL policy и LLM tools используют один executor и получают одинаковый результат одной команды.

Первый законченный продуктовый срез: **гетерогенная группировка ДЗЗ на 72 часа с новыми заявками, несколькими станциями, отказом сети и перепланированием**. Числа масштаба ниже — цели приёмки, не нынешняя производительность.

Полезный итог для клиента: сравнить свою политику с baseline по доставленным заявкам в срок, качеству, стоимости, безопасности, latency и runtime, воспроизвести каждое решение и получить проверенный обучающий dataset.

## 2. Релизный пакет R1 — исправить доказанные дефекты

Порядок: R1.1 → R1.2 → R1.3 → R1.4. RL fix R1.5 может делаться независимо от physics, но не снимает ограничений на данные.

### R1.1. UTC, координаты и физическая видимость — P0

Связь с аудитом: V73-01/02/03. Файлы: `core/time.py`, `core/orbit.py`, `core/frames.py`, `core/access.py`, `integrations/sgp4_prop.py`, все access callers и exports.

Работы:

1. Исправить calendar/JD roundtrip, определить supported range/precision, UTC semantics и политику leap seconds; использовать устойчивую проверенную time representation при необходимости.
2. Golden cases: midnight/noon, дробные секунды, переход дня/месяца/года, timezone offsets. Сверка с независимой библиотекой; погрешность должна быть объявлена, а не обещана наносекундной при одном float JD.
3. Ввести `target_access`: Earth LOS/ellipsoid visibility, FOV/pointing, target illumination при оптике; SAR не должен наследовать optical illumination rule.
4. Убрать возможность imaging antipodal target; единое правило в engine, opportunities, env observations и planner.
5. Сохранять frame на no-transform операциях. На границах mismatched frame — явная ошибка либо validated transform, не смена строки.
6. Задать frame/time contract для GCRF/TEME/ITRF/EME2000. Проверить r, v и covariance transformations; отдельно геодезическую/геоцентрическую широту.
7. Перегенерировать fixtures с объяснением причин изменения, старые outputs не выдавать за новые.

Приёмка:

- Probe UTC больше не даёт −43200 s; независимые roundtrip fixtures проходят.
- Far-side target никогда не создаёт продукт, near-side и limb cases проходят.
- Нулевой Δv не меняет frame; неподдерживаемая frame не принимается silently.
- Map/JSON timestamps совпадают с epoch сценария и backend data.

### R1.2. Recorder и единый data/product ledger — P0

Связь: V73-04/08. Файлы: `engine.py`, `fleet.py`, `catalog/traffic.py`, acquisition/workplan/campaign/demand envs, state, API/export.

Работы:

1. Уточнить понятия total produced, accepted/stored, lost, transmitted и uniquely delivered. Canonical unit — bit; обозначения в API однозначны.
2. Создать product ID с request ID, generated time, size, quality и owner; initial preload как отдельное происхождение, не новая съёмка.
3. Интегрировать store_rate по фактическому времени допустимой съёмки. Discrete image mode имеет explicit image size. Один target не должен бесконечно создавать один и тот же продукт без repeat-acquisition policy.
4. Overflow: total produced учитывает отброшенный объём; политика partial product/drop явная. Full product success отличается от «какой-то бит принят».
5. Fleet collect/relay/downlink и env-local counters становятся adapters к единому ledger, без независимых объёмов.
6. Убрать прямые записи `_stored_bits/_generated_bits` из DishAward и evaluation. Учебные preload cases используют публичный validated initial state.
7. Relay/downlink происходят в операции с capacity/time accounting; repeat request не расходует один интервал повторно. Учесть receiver capacity и unique delivery при репликах.

Приёмка:

- `initial + produced = buffered + delivered_unique + lost` в модели без копирования, отдельный accounting физических копий при репликации.
- После relay receiver engine/Fleet/API показывают один объём.
- Empty-buffer, overflow, interrupted pass, duplicate command и relay→ground cases проверены.
- Один и тот же acquisition duration даёт согласованный объём в planner и executor.
- Нет успеха доставки при пустом/неполном продукте, неверной станции или пропущенном deadline.

### R1.3. Временная причинность и физические балансы — P0

Связь: V73-05/06/10. Файлы: engine, propagate, passes, encounter, workplan.

Работы:

1. Шаги разбиваются на event boundaries: operation start/end, inject/recovery, AOS/LOS и другие значимые переходы.
2. Planned event t=0 выполняется один раз; событие внутри dt не исполняется в конце с ретроспективным доступом к данным.
3. Реализовать duration, interruption/cancel и resource locks. Скорость связи интегрируется внутри реального окна и по действующему режиму.
4. Время и географические координаты сил обновляются по стадиям RK4 либо честно определить и проверить другой integration scheme.
5. Устранить повторное использование начального when в encounter prediction; унифицировать force context для forecast/access/execution.
6. Убрать двойной heater heat, явно моделировать load ledger payload/radio/actuators и разделение electrical dissipation/полезной излучаемой энергии по профилю.
7. Guard finite dt/state/mass; явно обрабатывать выход за применимость атмосферы/thermal solver/орбитальной области.

Приёмка:

- Одна и та же continuous command schedule на dt-разбиениях 1/10/60 s даёт согласованный объём, события и энергию в заранее заданных допусках.
- Шаг через LOS не передаёт за закрытую часть; ступень rate не берётся по одному endpoint для всего интервала.
- Heater power добавляется к тепловому узлу один раз.
- События и labels несут фактическое время возникновения, а не только timestamp конца шага.

### R1.4. Mission outcome и CDM input correctness — P0/P1

Связь: V73-08/09/12.

Работы:

1. Общий mission evaluator: product completeness/quality, destination, delivery time, unmet reason, safety violations и cost.
2. Исправить `PassFault` delivery-without-contact и уточнить семантику Day/Operator comms success.
3. Отделить procedure fidelity от mission success; альтернативный допустимый keeper/план не должен автоматически проваливать миссию.
4. CDM: explicit unit parsing, consistent covariance SI, per-object frame/epoch, PSD/rank/applicability diagnostics и controlled unsupported-profile errors.
5. Не выдавать текущий Pc как эксплуатационную оценку до reference validation.

Приёмка:

- Диагностическая последовательность PassFault больше не сообщает физическую доставку без битов/contact.
- `100 [m]` не превращается в 100000 m, `1 [km²/s]` преобразуется согласно размерности; независимые message fixtures проходят.
- Широкий regression suite различает accepted command, operation completed, request delivered и mission succeeded.

### R1.5. Работоспособное обучение — P0 для ML

Связь: V73-07/15. Файлы: rl/reinforce, report, CLI и training tests.

Работы:

- Заменить baseline текущего single-step episode на корректный running/value/batch baseline; не вводить зависимость baseline от выбранного действия.
- Добавить bandit test с нетривиальной reward gap: веса должны меняться, preferred-action probability расти на нескольких seeds.
- Linear и MultiDiscrete trainers возвращают policy artifact, RNG, normalization/config и поддерживают evaluation/resume.
- Отделить train return от held-out success. Поле `learned` не определять простым сравнением noisy training tail и random reward.
- Учебный REINFORCE оставить baseline; production training adapter к внешней библиотеке — позже, после честной среды.

Приёмка: один шаг действительно способен обучать; сохранённая/загруженная политика даёт воспроизводимый результат; OOD evaluation не участвует в обновлении/подборе модели.

## 3. Релизный пакет R2 — объединить компоненты

Зависимости: критичные исправления R1.1–R1.3; улучшение CLI/packaging может идти параллельно.

### R2.1. MissionWorld и общий executor — P1

- World registry: unique spacecraft instances, model references, абсолютная epoch/time scale, ground/terminal resources, operation queue, product ledger.
- Command schema: ID, actor, target, parameters, requested start, duration, expected world revision и idempotency key.
- Lifecycle submitted/accepted/queued/running/completed/failed/cancelled и structured failure reason.
- Общий observation barrier; разные численные substeps разрешены, но joint decisions смотрят на один world timestamp.
- Snapshot/restore включает RNG streams, события, ops, ресурсы, model/input versions; clone/branch/replay для what-if.
- Эволюционная миграция: SimulationEngine остаётся physics/subsystem component, Fleet/Operator/Day становятся adapters, не ещё одним мировым состоянием.

Приёмка: два параллельных действия и resource conflict детерминированно разрешаются; replay/restore совпадает с uninterrupted run; абсолютные эпохи бортов проверяются; order permutation не меняет joint-step результат при одинаковых правилах tie-break.

### R2.2. Scenario DSL v2 — P1

- Один canonical schema/version/migration для spacecraft instances, models, orbit/backend, stations/antennas, requests, faults, observation model, objective и fidelity.
- Поддержать одинаковые модели в разных instance IDs, произвольные орбиты и много запросов; не создавать новый класс env на каждый аппарат.
- Version/epoch/frame/model IDs/bounds/horizon/deadline/cross-field validation. Explicit missing file — error, defaults только явно.
- Canonical resource loading и один источник конфигов; provenance каждого assumed/published/derived/calibrated параметра.
- Generator с seed streams для geometry, demand, faults, telemetry и policy, scenario fingerprint и feasibility class.

Приёмка: одна DSL задаёт 1/4/16 бортов без правок Python, duplicate instance IDs запрещены, один model ID разрешён многим instances, invalid/unsupported fields диагностируются до expensive simulation.

### R2.3. Agent observation contract — P1

- Truth, measured telemetry, state estimate и forecast раздельны; noise/latency/dropouts/stale timestamps configurable.
- План/requests доступны в том виде, в каком их реально знает диспетчер; optimal action/required_items не входят в миссионные признаки.
- Action masks означают допустимость по известным ограничениям, а не оптимальность.
- info/tools/API не дают скрытый oracle state обучаемой политике; privileged baselines/CTDE state явно отдельные.
- Variable-size fleet graph/set/masked representation и версионирование observation schema.

Приёмка: agent не извлекает правильную команду из hidden labels; шум/задержка действительно меняют наблюдение; permutation/telemetry-age tests; procedural env сохраняются с явным назначением.

## 4. Релизный пакет R3 — законченная миссия и benchmark

### R3.1. Request-driven EO Mission v1 — P1

Развивать идею DemandShift, а не переносить его single-request ограничения в весь продукт.

- Много заявок с release/deadline, AOI, повторяемостью, required size/quality, приоритетом, payload requirements и delivery endpoint.
- Heterogeneous optical/SAR, slew/settle, duty cycle, energy, memory, thermal и station/antenna reservations.
- Новые заявки и отказ станции во время исполнения; rolling horizon с commitment window и стоимостью изменений плана.
- Planner выдаёт operations в общий executor, а результат проверяется outcome evaluator.
- Baselines: wait, FIFO, earliest deadline, priority greedy, rolling-horizon heuristic; CP-SAT/MILP на малых экземплярах с feasibility/timeout.

Сценарий приёмки: 72 h, 4–16 гетерогенных аппаратов, не менее трёх станций, конкуренция за антенну, срочная новая заявка и outage. Требуются успешные и невозможные запросы с объяснимым исходом. Сначала измерить стоимость на 4 бортах, затем расширять.

KPI: weighted complete delivery on time, lateness distribution, unique useful bits, quality/coverage, Δv/energy/ground cost, constraint violations, planner latency и missed/infeasible reasons. Не сводить всё к «все заранее назначенные дела закрыты».

### R3.2. Физическая validation matrix — P1

Сохранить полезные четыре core cases и дополнить:

- UTC/frame/velocity/covariance reference fixtures; несколько эпох/орбит и границы EOP.
- Двухтельная позиция/скорость и convergence, J2 с matching force models, drag/SRP/third-body независимые profiles.
- Ground target LOS/lighting, AOS/LOS, интегральная link capacity, slew и energy/thermal analytic balances.
- TCA refinement и multiple minima; Pc PSD/conditioning/zero-speed/domain-of-validity cases.
- Отдельные accuracy budgets для EO planning и conjunction assessment; backend/model version и source hash.

Приёмка: каждый опубликованный capability имеет область применимости и измеренный error report. Reference suite не должна быть тем же алгоритмом под другой обёрткой. Пропуск optional backend в CI отражается как unverified, а не passed.

### R3.3. Benchmark v1 — P1

- Smoke suite переименовать/документировать отдельно от full mission evaluation; убрать искусственные 2-step caps из mission benchmark.
- Versioned scenario sets: in-distribution, unseen geography/epoch/model/network, disturbances, scale, impossible cases.
- Grouped train/val/test splits, scenario fingerprints/near-duplicate checks, неизменяемый test suite; несколько policy seeds.
- Сравнение classical/rule/learned/oracle раздельно, confidence intervals, per-task failures, runtime/compute budgets.
- Никаких «holdout» только из замены seed без изменения мира.

Приёмка: повторяемый report по полной миссии с policy/config artifacts, regression thresholds и доказанным отсутствием пересечения split. Не ставить обязательное превосходство RL над оптимизатором как заранее гарантированный результат.

## 5. Релизный пакет R4 — данные и мультиагентность

### R4.1. Dataset factory — P1/P2

- Schema transitions/episodes/products/events: IDs, timestamp, observable input, actions, operation outcome, reward components, constraints, terminal reason, lineage.
- Replay-verified demonstrations от classical planner, процедур и человека; preference pairs с проверяемыми критериями, failure/recovery/invalid inputs.
- SFT tool trajectories/structured plans; explanation только из доступных фактов, без выдуманных инженерных норм и hidden reasoning.
- Dedup, split isolation, checksums, shards/columnar format, atomic writes/resume, dataset/model cards и license/provenance.
- Template narratives отделить от physics-derived truth; связанные orbital parameters выводить из общего sampled state.

Приёмка: каждый sample позволяет восстановить сценарий/источник label; демонстрация достигает указанного исхода при replay; тестовые миссии не попадают в обучение.

### R4.2. MARL и масштабируемое обучение — P2

- Central dispatcher остаётся Gymnasium; satellite/ground agents — PettingZoo Parallel, AEC только для выбранного последовательного протокола.
- Local observations, communication costs/delays, resource arbitration, team/individual reward, CTDE-only truth.
- Dynamic agents, heterogeneous capabilities, fault removal, permutation-aware policy и unseen fleet size.
- External RL adapter, vector rollout, checkpoint/resume, recurrent policy для partial observability, bounded workers.

Приёмка: API conformance и world-order invariants; несколько агентов действительно принимают независимые решения. MultiDiscrete одного dispatcher не переименовывать в MARL.

### R4.3. LLM mission tools — P2

Инструменты inspect/query opportunities/propose/validate/branch/reserve/submit/cancel, те же sessions и права, structured schemas, budgets/timeouts, audit trail. CI с mock policy, внешние модели optional. KPI: feasible plan, correct tools, mission outcome, latency/cost, recovery и factual consistency.

## 6. Релизный пакет R5 — продуктовая поставка

### R5.1. SDK/API — P1/P2 после R2

Public session SDK; session IDs, isolation, revision checks, idempotency, persistence/recovery, async jobs и cancellation; versioned schemas/OpenAPI, request limits и structured errors. Auth/RBAC/tenant isolation до общего внешнего deployment. Не связывать публичный contract с private job fields.

Приёмка: две независимые миссии не влияют друг на друга, retry не дублирует burn, stale revision отвергается, restart восстанавливает операцию/ledger.

### R5.2. Studio — P2 после world stream

Одна страница: 2D/3D, actual/forecast/alternative labels, timeline, заявки, операции, resources/KPI, validation failures и журнал. Scenario editor и branch comparison; клиент использует SDK/API contracts. Map coordinates/UTC согласованы с backend; actual revolution duration вместо константы 6000 s; никакой замены только r/v без освещения/contact выбранной ветки.

Приёмка: пользователь объясняет любой unmet request по событиям/ресурсам, manual/solver/agent run сравниваются; UI e2e, errors/loading/empty states и bounded rendering.

### R5.3. Packaging/CI/observability — P1 параллельно, финал P2

- Изолированная сборка wheel/sdist, assets внутри distributable resources, запуск из чистого каталога.
- Supported dependency policy, optional backends отдельно, smoke examples, lock/constraints strategy.
- CI: lint/type baseline, contract/unit/integration/property, wheel install, Gymnasium checker, short UI e2e; heavy reference/performance отдельно.
- Structured telemetry run/session IDs, runtime metrics, crash reports, memory/log retention limits, backup/restore/migration/rollback instructions.
- Licence/SBOM/data rights и dependency checks до коммерческой поставки.

Приёмка: новый пользователь запускает установленный пакет без checkout, CI проверяет именно эту конфигурацию; восстановление проверено, ошибки имеют диагностику.

### R5.4. Performance — P2, только после correctness baseline

Измерять propagation/access/screening/planning/export отдельно. Убрать повторный all-pairs O(N²) из внутренних циклов, кэшировать по world revision/model/input hash, добавить screening broad phase с проверенным narrow phase. Bounded snapshots/logs и streaming datasets; батчинг и независимые workers.

Ступени: 4/16/64/256 бортов, 1/7/30 суток; 1000+ сначала geometry/screening profile. Для каждой ступени публиковать hardware, fidelity/error, wall time, throughput, RSS и failures. Performance SLO выбрать по измерениям/пилоту, не обещать real-time произвольного флота.

## 7. Новые миссии: порядок расширения после базового EO

1. **Ground network/SLA и FDIR:** спрос, booking/cost/outage, degraded mission и восстановление. Максимально переиспользуют world/resources.
2. **ISL/DTN:** очереди, TTL, store-and-forward, contact graphs, link model/terminal contention, real ground delivery. Устранить награду за бессмысленную циркуляцию пакета.
3. **Formation/orbit maintenance:** phase/relative geometry, differential drag, finite burn по необходимости, fuel/lifetime, deployment/replacement.
4. **SSA/OD/CAM:** sensor observations → estimate/covariance → TCA → manoeuvre → перескрининг и mission impact. Требует отдельного accuracy gate R3.2.
5. **Responsive/stereo/revisit tasking:** AOI, quality/information gain, heterogeneous collaboration и новые заявки.
6. **GEO operations:** долгота/наклонение/эксцентриситет, appropriate perturbations, eclipse seasons и payload service. Semimajor correction одного case не закрывает весь класс.
7. **Proximity/service/disposal:** navigation sensors, rotating-frame dynamics, keep-out/abort, 6-DoF при необходимости; EOL с uncertainty и проверенным требованием конкретного профиля.
8. **MEO/GNSS/cislunar/interplanetary:** отдельные physics/measurement/communication specs и reference suites после подтверждённой потребности клиента.

Каждое новое семейство принимается только при наличии scenario generator, независимых control cases, baseline, KPI/failure cases, observation contract и documented fidelity. Не добавлять только бинарный выбор «включить правильную кнопку» под новым industrial названием.

## 8. Непосредственная очередь следующему агенту

1. R1.1: time roundtrip, antipodal access, frame preservation/rejection. Первые regression tests брать из diagnostic probes.
2. R1.2 + R1.3: общий ledger, длительность и event ordering, heater/radio energy, planner/executor parity.
3. R1.4 + R1.5: outcome truth/CDM units и исправление обучаемости одношаговых задач.
4. R2: общий World/operations/snapshots, DSL v2 и честные наблюдения.
5. R3: новая полная EO-миссия, classical baseline, validation и benchmark.
6. R4/R5: datasets, MARL/tools, API/Studio и воспроизводимая поставка.

После каждого пакета: regression report, версия contracts, compatibility notes и короткий owner-facing результат. Не отмечать весь этап выполненным потому, что прошло несколько happy-path тестов.

## 9. Когда это действительно стартап-продукт

**Исследовательский релиз:** R1/R2 + full mission/validation/benchmark из R3, воспроизводимые policies/data и явные ограничения.

**Пилотный продукт:** исследовательский релиз + нужные SDK/API/Studio и один согласованный с партнёром сценарий, независимая приёмка, onboarding, compute-cost measurements и runbooks.

**Эксплуатационный продукт для конкретной миссии:** дополнительно customer requirements, data rights, независимая проверка модели, security/deployment/support, restoration/SLO и согласованные границы ответственности. Агент не может объявить клиентскую приёмку выполненной без клиента.

Следующая бизнес-задача владельца — выбрать ICP и одну измеримую пилотную боль. Универсальность архитектуры сохраняем, а полезность и надёжность доказываем на конкретной миссии.

## 10. Источники контрактов

- [Gymnasium environment checking](https://gymnasium.farama.org/v1.2.1/api/utils/) — проверка API, не физической реалистичности.
- [Предыдущий глобальный план](STARTUP_DEVELOPMENT_PLAN.md) — продуктовый scope и дальние классы задач.
- [Текущий кодовый аудит](audits/V073_STATE_AUDIT.md) — точные исходные находки и границы доказательств.
