# Space-Env: Mission Execution & Agent Benchmark v1

## 0. Задание следующему агенту

**Реализовать этот пакет целиком за один последовательный проход**, от воспроизведения до итогового отчёта. Не останавливаться после одного исправления, новой версии, заглушки API или предложения «продолжить следующим шагом». Под «одним проходом» понимается одно завершённое поручение с внутренними этапами и gates, а не одна огромная правка без проверок.

Исходная версия: **0.86.0**, HEAD `e512037ee1857ea62c4ab7584de4bfc685a9ed6c`, аудит 26 октября 2026: [V086_AUDIT.md](audits/V086_AUDIT.md). Проверить состояние перед реализацией: оно могло измениться. Старый [план 0.73](NEXT_DEVELOPMENT_PLAN_V073.md) — дальний backlog, а не актуальная очередь создания компонентов с нуля.

### Правила

- **Не коммитить, не пушить, не создавать PR, не публиковать и не отправлять сообщения вовне без отдельного поручения владельца.** Сохранить его изменения; не переключать и не сбрасывать дерево автоматически.
- Прочитать локальные инструкции, текущий код и тесты. Существующие MissionWorld, custody/receipts, tasking policies, DSL и 72-hour cases переиспользовать и довести, а не дублировать.
- Сначала negative regression cases из аудита, затем реализация. Не удалять проверки и не менять допуски для сокрытия ошибок.
- Различать implemented, regression-verified, independently-validated и pilot-accepted. Реализация method name не равна подтверждённому контракту.
- Вести `docs/IMPLEMENTATION_STATUS.md` для этого пакета: WP ID, зависимости, status `todo/in_progress/blocked/verified`, evidence commands/exit codes, changed contracts, ограничения. Не переносить старые отметки «сделано» без повторной проверки.
- Устанавливать только необходимые зависимости и действовать по правилам окружения. При внешнем блокере сохранить код/источники, зафиксировать точную причину, продолжить независимые этапы. В финале не объявлять заблокированное выполненным.
- Старые env IDs/imports и documented adapters сохранить; новое поведение benchmark получать с новой версией среды/schema. Не молча менять смысл старого `success`.

## 1. Результат прохода

После исполнения должна работать одна цепочка:

**Validated scenario → MissionWorld → typed commands → resource-aware execution → product custody → unique delivery receipts → mission KPI → Gymnasium/CLI/API/Studio → replayable training data.**

Человек, heuristic и агент подают команды одному executor. Они не могут обходить station outage, терминалы, память, deadlines или safety rules через другой entry point. Forecast/branch не меняют исходный мир. Один и тот же snapshot+command stream воспроизводит результат.

Демонстрация: гетерогенная группировка ДЗЗ, реальные физические окна текущего профиля, несколько станций, заявки с release/deadline/объёмом/приоритетом/назначением, outage и срочный запрос, ручное/автоматическое исполнение, объяснимый результат и данные для обучения.

### Граница объёма

В этом проходе обязательны WP-01…WP-08 и все gates ниже. Это существенная research/pilot foundation, не объявление универсального flight-grade simulator.

Не включать: новый высокоточный интегратор, полную IAU2000A/TEME conversion, новый OD/Pc solver, 6-DoF/docking, GNSS/lunar missions, HIL, Kubernetes, облачную billing-платформу, полный authentication stack или обучение большой нейросети. Для неподдерживаемого — явная граница, а не псевдореализация. PettingZoo/MARL отложить: сначала полноценный централизованный fleet environment.

## 2. Исходные подтверждённые ошибки

Воспроизводитель: `python docs/audits/v086_audit_probes.py`. Результаты аудита: [JSON](audits/v086_probe_results.json). Не превращать этот скрипт в тест, который ожидает старое ошибочное поведение; добавить правильные assertions в `tests/`.

- A01: forecast оставляет расход Δv 20→19 и consumed planned-event; world RNG не восстанавливается.
- A02: unknown actor игнорируется; engines с разными clocks принимаются; runner после resume превышает горизонт.
- A03: два ручных downlink на одну станцию одновременно проходят; открытая линия продолжает передавать во время outage.
- A04: повтор одной 60-bit receipt закрывает заявку 100 bit; delivery до release принимается.
- A05: safety KPI всегда 0 даже при реальном constraint event; completion latency подменена первой квитанцией.
- A06: DSL принимает negative horizon, NaN, invalid coordinates/size/deadline.
- A07–A09: fleet runner, single-satellite Gym и Studio не объединены; нет надёжного benchmark/data/portable delivery contract.

## 3. Архитектурные договорённости

- `simulation/engine.py`: физика и подсистемы аппарата; публичные export/import state, execution hooks и typed read models. Не добавлять туда обучение или HTTP.
- `simulation/world.py`: общая абсолютная эпоха и monotonic now, engines, операции, station/terminal resources, event queue, RNG streams, revisions, snapshot/branch. Это источник правил исполнения, а не только цикл `for engine.step`.
- `mission/`: validated requests/scenarios, policy decisions, единый delivery/outcome/KPI evaluator. Policies предлагают действия и не меняют private engine state.
- `envs/`, `api/`, `viz/`: адаптеры одного публичного session/executor interface. Никакого второго ledger/scheduler ради интерфейса.
- Data records: typed/schema-versioned, finite SI values, explicit units/frame/timescale/model profile. IDs детерминированы от run/instance/sequence и восстанавливаются при replay.
- Неприменённая команда, конфликт ресурсов, safety rejection, operational failure и fatal simulation error — разные состояния.

Рекомендуемые новые модули: `simulation/contracts.py`, `simulation/resources.py`, `simulation/snapshot.py`, `mission/delivery.py`, `mission/metrics.py`, `eval/mission_benchmark.py`, `data_generators/mission_dataset.py`, `api/sessions.py`. Имена можно адаптировать после чтения кода, границы ответственности — сохранить.

## 4. WP-01 — Полный state contract, clock и безопасный forecast

**Зависимости:** нет. **Основные файлы:** world/engine/Fleet, новые contracts/snapshot; тесты world/replay.

### Реализация

1. `MissionWorld.now_sec` и общая absolute epoch. При создании проверить совпадение эпох, frame profiles, текущих clocks и уникальность instance IDs. Проверять horizon/dt на finite/positive; нельзя идти назад или выходить за горизонт без explicit mode.
2. Preflight всего joint action batch: неизвестные actors/parameters, неверный тип/shape, NaN, stale revision — structured error до изменения мира. Runtime failure при исполнении не оставляет частично продвинутый world: rollback последнего committed snapshot либо fail-stop state с запретом продолжения и явной семантикой.
3. Полный snapshot engine: orbit/frame/epoch, весь spacecraft state (включая Δv/health), modes, attitude/slew, RNG, faults/recovery, encounters/covariances, fired events, operation state, counters, products/chunks/receipts, sequences, reservations, telemetry accumulators, event log cursor. Derived caches пересобираются или сериализуются с проверкой совместимости.
4. Полный snapshot world: engines, named RNG streams, now/epoch/horizon, requests/lifecycle, queue, resources/outages, event sequence/revision, policy-independent state, scenario/model/backend/input fingerprints.
5. Versioned portable serializer: JSON metadata + явное кодирование массивов; без pickle и скрытых Python object references. Проверять размерность, finiteness, registry/hash/schema до restore; несовместимое состояние отвергать атомарно.
6. `clone/branch/forecast` исполняются на изолированной ветке. Если применяется temporary restore — обязателен `try/finally`, включая исключения; исходные RNG, logs/revision и future events не меняются.
7. Runner продолжает от `world.now_sec`, а не от локального 0. Snapshot resume продолжает оставшийся горизонт.

### Приёмка G1

- Forecast перед planned burn не меняет запас Δv, очередь, RNG, event log и будущий реальный burn.
- Restore после burn/fault/slew/reboot/cancel/partial relay возвращает полное состояние; результат следующего шага совпадает с uninterrupted run.
- Restore в новый процесс через file snapshot воспроизводит receipts/reward/KPI, за исключением wall time.
- Noisy observations после restore повторяют seed stream; repeated forecast не меняет observations исходной сессии.
- Unknown actor/mismatched epochs/clocks/invalid dt не приводят к частичному шагу.

## 5. WP-02 — Общий executor и реальные ограничения ресурсов

**Зависимости:** WP-01. **Основные файлы:** world, engine step/communication hooks, Fleet relay, tasking runner, resource contracts.

### Контракты

Typed command минимум: `command_id`, `actor_id`, `kind`, `request_id/product_id`, `destination_id`, `start_sec`, `duration_sec`, параметры и `expected_revision`. Для команд без части полей — явная схема, не произвольный dict. Lifecycle: submitted→accepted/queued→running→completed/failed/cancelled; хранить reason и фактический интервал.

### Реализация

1. Общие ресурсы: station antennas (default capacity=1), spacecraft terminals и capability-specific одновременность payload/downlink/relay/slew. Half/full duplex определяются профилем, а не догадкой frontend.
2. Capacity/outage проверяются в executor для любого caller, включая ручные команды и агент. Policy winner — выбор стратегии, не единственная защита мира.
3. Общая reservation книга для relay и downlink; пересоздание Fleet wrapper не сбрасывает занятость. Команды с одинаковым ID/body идемпотентны, same ID/different body — конфликт. Overlap определяется по абсолютным полуинтервалам [start,end).
4. Мир режет шаг по release/deadline, operation start/end/cancel, outage start/end, fault/recovery. Доступ по геометрии уточняется в задокументированном допуске текущего physics profile.
5. `wait` имеет определённую семантику. Для persistent radio — отдельные stop/cancel; outage всегда прекращает фактическую передачу независимо от последней команды. Передача возобновляется только по правилам lifecycle.
6. Settling data происходит только после прошедшего интервала и доступного rate integral. Будущие биты relay нельзя мгновенно downlink-нуть в том же интервале до завершения ingress. Учесть store-and-forward causal ordering.
7. In-step conflict разрешать детерминированно по опубликованной tie-break policy. Dictionary iteration order не определяет физический результат.
8. Planned operations и agent actions используют одинаковые ресурсы и lifecycle. Сегментировать subsystem updates и health на фактическое время действия; исключить ретроспективную съёмку/передачу после fault/cancel.
9. Сохранить legacy helper adapters, но запретить обход ресурса и повторное начисление времени; обозначить deprecation старого ambiguous dt API.

### Приёмка G2

- Два аппарата на одной однолучевой станции физически не передают одновременно; на двух антеннах/станциях могут.
- Outage [1,3) после arming на t=0 не создаёт delivery за закрытый интервал. Проверить outage, начинающийся внутри внешнего шага.
- Relay + downlink на одном half-duplex terminal конфликтуют; требования full duplex проходят отдельно.
- Одинаковый control schedule при dt 1/5/10/60 s даёт согласованные объёмы/события/энергию в заранее объявленных допусках. Дискретные запреты/release/deadline не размывать большим tolerance.
- Несколько повторных/overlapping команд не удваивают capacity; retry после snapshot restore также идемпотентен.

## 6. WP-03 — Product lifecycle, уникальная доставка и честные KPI

**Зависимости:** WP-01/02. **Основные файлы:** engine custody/receipts, tasking score/KPI, outcome, новые delivery/metrics.

### Реализация

1. Immutable logical product ID и request ID, acquisition identity, custodian, размер/bit ranges, origin/time, quality profile. Куски сохраняют identity; serialization/replay не создают новые продукты автоматически.
2. Упростить и формализовать custody: default — transfer без копирования, каждый полезный range имеет одного owner до ground delivery; если вводятся реплики, unique goodput отделён от physical transmitted bits.
3. Receipt содержит ID, product/acquisition/request IDs, уникальный range, receiver, фактическое начало/конец передачи, завершённый объём, profile/quality и provenance. Не ставить всем кускам только timestamp конца большого шага.
4. Delivery — union ranges по логической acquisition, без двойного зачёта повторов/overlap и разных retransmission IDs. Не складывать половины разных наблюдений в один снимок без явной aggregation policy.
5. Request contract минимум: required size, payload, release/deadline, allowed destinations и minimal quality profile. Default — один complete logical product; поддержку mosaics/complex fusion не выдумывать в этом проходе.
6. Unknown request/product, nonfinite/negative ranges, impossible timestamps, forged duplicates или conflicting receipt body — ошибка/диагностика без зачёта. Доставка до release, после deadline или не в allowed destination не закрывает заявку.
7. Outcome evaluator один для CLI/Gym/API/Studio. Держать раздельно procedure success, operation completion, request status и mission objective. Старые `success` сохранить только в compatibility/tutorial профиле.
8. Lifecycle запросов: unreleased→available→assigned/in_progress→delivered/expired/cancelled/unmet, явные reasons. Завершённую заявку не снимать бесконечно; initial/preloaded data отражать отдельно.
9. KPI из событий/ledger: unique requested on-time bits/value, completion/first-byte latency отдельно, late/unmet reason, actual violations (count и duration по типу), Δv, energy по subsystem, station utilization, retransmitted/relay traffic, planner/runtime/RSS. Unsupported measurement — `null` с причиной, не константа 0.
10. Балансы проверяются глобально и по бортам. Выдать public immutable read models вместо чтения `_products`/правок private buffers из policies и eval.

### Приёмка G3

- 60-bit receipt дважды не закрывает заявку 100 bit; overlap 0–60 и 40–100 даёт ровно 100, а не 120.
- Две несовместимые частичные acquisitions не складываются в полный снимок. Cross-relay identity и offset сохраняются через несколько hops.
- Перед release/не на тот endpoint/не того payload/неполное качество — корректный unmet, даже при positive raw bits.
- Deadline boundary оценивается по фактическому completion time, не времени первого receipt или конца всего эпизода.
- Low SoC/thermal violation повышает реальный safety KPI; отсутствие измерителя не выдаётся за нулевые нарушения.
- Все публичные counters согласуются с ledger после partial delivery/overflow/relay/cancel/restore.

## 7. WP-04 — Строгая DSL и разнообразные воспроизводимые миссии

**Зависимости:** contracts WP-01/02/03; читать/специфицировать можно параллельно. **Файлы:** scenario_v2, shift adapters, configs, catalog integration.

### Реализация

1. Validated typed schemas с unknown fields forbidden и finite checks. Проверять exact schema version, horizon>0, координаты, size>0, release>=0, deadline>release, enum payload/mode, существование instances/models/stations/endpoints, уникальные IDs и cross-field compatibility до сборки engines.
2. Разделить model ID и instance ID; разрешить много экземпляров одной модели. Модель описывает published/assumed/derived параметры и допустимые capabilities; источник/uncertainty не превращать в маркетинговый «паспорт».
3. Добавить requests из WP-03, station antennas, outages/faults, observation noise/latency policy, scenario seed streams, initial state и operation conflict rules. Поддержать legacy v2 strings/stations через явную normalization/migration; новый breaking format — новая schema version.
4. Horizon и deadline могут отличаться только с явной политикой: запрос за горизонтом получает documented `not_released/horizon_exhausted`, а не тихо исчезает. Фиксировать denominator mission KPI.
5. Stable normalized config hash + raw file hash, generator version, model/backend/input data hashes, code/dirty diff hash и named seeds. Изменение YAML comments не должно менять semantic hash.
6. Один packaged config source of truth; cwd overrides только явным параметром. Удаление дублирующих файлов — лишь с совместимым adapter и тестом.
7. Seeded scenario factory: спрос/география, release/deadline/priority, capacity/memory/energy stress, outages и urgent arrivals. Не создавать вариативность только переименованием объектов.
8. Примеры: короткая 4-bus mission с >=12 заявками, competing access и outage; средняя day mission с >=32 заявками; сохранить существующие 4/16-bus 72h fixtures как regression. Feasible-by-construction cases маркировать отдельно от свободно сгенерированных/infeasible.

### Приёмка G4

Невалидные случаи A06 отвергаются до expensive propagation. Одинаковый seed/normalized config воспроизводится; разные scenario seeds действительно меняют миссию. Policy RNG не меняет geometry/fault streams. Файлы работают из installed wheel вне checkout. Есть специально невыполнимые заявки с ожидаемыми причинами.

## 8. WP-05 — Единый runner, baselines и полноценный fleet Gym env

**Зависимости:** WP-01…04. **Файлы:** tasking runner/policies, новый env adapter, cards/registry, expert.

### Реализация

1. Выделить policy interface `decide(observation) -> action batch`. FIFO/deadline/priority только предлагают команды; не меняют ground_targets/private buffers и не исполняют relay вне world.
2. Runner использует world now, events и public outcome; поддерживает resume/cancellation и завершает оставшийся горизонт. «Replans» — фактические решения/изменения плана; не выдавать tick count за полноценную оптимизацию расписания.
3. Существующий exact assignment сохранить как reference ограниченного subproblem. Добавить tiny exhaustive/DP reference для явно сформулированного интервал/ресурс scheduling case с upper bound по числу candidates и deterministic timeout/error. Не называть его глобальным оптимумом 72h без доказательства.
4. Зарегистрировать новую среду `space_env/FleetTasking-v0` (имя можно уточнить при проверке registry): один централизованный агент, много spacecraft и requests. Не ограничивать одним бортом. TaskingShift-v0 остаётся совместимым учебным adapter.
5. Typed Dict observation: now/remaining horizon, spacecraft telemetry/resources/capabilities, доступные запросы (deadline/priority/required size/status), accessible windows/forecast с time/source, active operations и valid masks. Никакого optimal action/job.kind/hidden fault label в agent-visible части.
6. Action contract адресует actor/request/destination/partner и режим wait/image/downlink/relay/cancel. Для фиксированного scenario задать конечные slot maps и masks, padding не должен означать реальную заявку. Синтаксически invalid action отвергать; ресурсный conflict возвращать как documented rejection.
7. Noise/latency/dropped telemetry отделены от truth и явно воспроизводимы; privileged critic/oracle получает отдельный interface. Не возвращать скрытый ответ через `info`.
8. Reward по actual changes mission objective/cost/violations, не за совпадение с heuristic. Mission success/reward/safety раздельны. Terminated для конечного mission MDP, truncated для внешнего cutoff; remaining time доступно.
9. Добавить простой flatten/mask-aware wrapper для текущих учебных trainers; не переписывать RL algorithms или заставлять Dict serializer молча терять структуру.

### Приёмка G5

- Direct runner, policy через Gym и recorded commands через world дают одинаковые операции/receipts/KPI при равном decision schedule.
- 1/4/16 instances задаются конфигом; API spaces и masks корректны. Joint dictionaries с другим порядком не меняют физический результат.
- Random, wait, greedy и reference baseline исполняются до реального исхода; пример, где разные допустимые планы успешны.
- Gymnasium `check_env` и repeated reset/seed проверяются. Старые env IDs проходят smoke и не переименованы в MARL.

## 9. WP-06 — Benchmark и dataset factory как проверяемый результат

**Зависимости:** WP-03…05. **Файлы:** eval/mission_benchmark, data_generators/mission_dataset, CLI/examples/config suites.

### Реализация

1. Отделить registry smoke от полноценной mission suite. Нельзя оценивать day delivery за 2–4 шага и называть это mission benchmark.
2. Versioned suite: nominal, station contention, outage/recovery, urgent arrival, scarce memory/energy, relay, infeasible; минимум два качественно разных family. Tiny smoke — десятки минут; day/72h — отдельный long job.
3. Scenario group train/validation/test manifests, no exact/near duplicates; ID/OOD по географии/network/load/model combination. Test scenarios не участвуют в подборе эвристик/параметров.
4. Report per scenario/seed/policy: actual KPI, feasibility, confidence intervals по реально выполненным cases, wall time, peak RSS, backend/profile, commands/receipts reference. Не объявлять статистическую значимость по одной миссии.
5. Dataset schema: run/episode/scenario IDs, observation schema, source time, action/masks, command result, event refs, reward decomposition, termination, mission outcome, seed/model/source/input hashes, behavior policy.
6. JSONL сохраняет Dict observation целиком; typed arrays имеют явный codec/schema. Для больших прогонов sharding, atomic finalize и resume/manifest checksums. Не писать весь dataset в память.
7. SFT demonstrations только из фактически replay-verified policy/solver/human commands; structured plans/tool calls + доступные факты, без выдуманных инженерных нормативов. Template text отдельно от verified simulation-derived data.
8. Replay verifier восстанавливает scenario+snapshot и проверяет actions→events→receipts→outcome в допусках. Failures/invalid commands/recovery — отдельные dataset labels, не выкидывать всё кроме успеха.
9. Physics qualification не заменять benchmark: добавить analytic timing/capacity/conservation fixtures и matching-force two-body/frame sanity reference с текущими доступными библиотеками; сохранить явные unsupported precision claims.

### Команды, которые должны существовать после реализации

Предлагаемый единый CLI (структуру можно адаптировать, команды в docs должны быть реальными):

```bash
python -m space_env.cli validate configs/scenarios/mission_execution_smoke.yaml
python -m space_env.cli run configs/scenarios/mission_execution_smoke.yaml --policy priority --output artifacts/run
python -m space_env.cli benchmark --suite configs/benchmark_suites/mission_v1_smoke.yaml --output artifacts/benchmark
python -m space_env.cli dataset --suite configs/benchmark_suites/mission_v1_smoke.yaml --policy priority --output artifacts/dataset
python -m space_env.cli replay artifacts/run/manifest.json
```

### Приёмка G6

Каждый sample traceable и replayable, duplicate receipts не улучшают score, разные seeds не повторяют один baked scenario. Два запуска дают одинаковые semantic outputs/hashes без runtime fields. Train/test overlap check падает на специально внесённом дубле. Dataset export и replay одного sealed shard входят в CI.

## 10. WP-07 — Локальный session API и Studio одного мира

**Зависимости:** WP-01…05, schemas WP-06. Это bounded local product slice, не облачная production платформа.

### Реализация

1. Публичный `MissionSession` фасад: create/reset/observe/submit/advance/snapshot/restore/report/event_log. CLI, API и Gym используют его либо тот же immutable world contract.
2. Локальный session registry с IDs, per-session lock, limits/cleanup. Две сессии не делят глобальную `_ENV`. Stale revision→409; invalid schema→400/422; idempotent retry не повторяет impulse/image/receipt.
3. Versioned endpoints минимум: create session из validated scenario, get observation/report, submit command batch, advance ограниченного интервала, save/restore snapshot, bounded event pagination. Не принимать произвольный файловый путь клиента без controlled resource resolution.
4. Request/body/time/step limits, structured errors, finite JSON. Не запускать 72h внутри необозначенного blocking GET и не делать бесконечный worker pool.
5. На существующей Studio добавить mission mode: scenario selection, requests/status/deadlines, operations+rejection reasons, resources, actual KPI, event timeline. Manual и auto policy работают с той же session/revision. Существующие globe/map и legacy Operator доступны.
6. UI явно различает executed/forecast/alternative, telemetry timestamp/source/profile; raw untrusted labels рендерятся как text. Не перерисовывать визуальный стиль и спутники в этом проходе без необходимости.
7. Статические assets и packaged scenario работают из wheel; snapshot одной local session можно восстановить в новом процессе. Auth/tenant/cloud storage — явно следующий пакет, сервис по умолчанию localhost.

### Приёмка G7

API две сессии/повтор/stale revision/restart проходят интеграционные tests. UI smoke: открыть миссию, выполнить/отклонить команду, увидеть фактический request status и тот же KPI, что CLI. Browser e2e проверяет пользовательский путь; отсутствие браузерного инструмента фиксируется как непроверенный gate, не как успешный screenshot test.

## 11. WP-08 — Поставка, performance evidence и финальная интеграция

**Зависимости:** все предыдущие, подготовка packaging/CI возможна параллельно.

- Wheel/sdist build, install в чистое окружение/каталог, imports/CLI/registry/scenarios/static assets без repository root. CLI entrypoint и python -m mode документированы.
- Fix README broken links и описания: текущий scope/профили, tutorial vs mission env, реальные команды, ограничения. Стабильные docs entry points не удалять при следующих версиях без замены ссылок.
- CI quick: lint/types для новых/изменённых contracts, unit/integration/negative/property tests, Gym checker, dataset replay, package install и short API/UI gates.
- Long suite отдельно: существующие 4/16-bus 72h tests, новый loaded day scenario и performance report. Не удалять long coverage ради зелёного quick CI; expensive job имеет явный запуск/расписание.
- Profile propagation/access/tasking/receipt scoring, cache по world revision/model/config hash, bounded logs/preview caches. Оптимизировать только подтверждённые hot spots, не снижая fidelity незаметно.
- Измерять hardware, backend, sample/step, wall time, throughput, peak RSS, event/receipt counts. 16 idle satellites с двумя запросами не выдавать за industrial-demand scale. Установить regression budgets после baseline, не придумывать real-time SLO заранее.
- Финал: `docs/MISSION_EXECUTION_V1_REPORT.md`, status по G1…G8 с evidence, changed contracts/migration, observed numeric accuracy, performance, известные ограничения, reproduction command.

### Приёмка G8

Чистый install воспроизводит smoke mission, dataset/replay и local UI. Полный совместимый suite и long cases либо реально прошли, либо перечислены как blockers с конкретными outputs; не выдавать частичный прогон за полный. Новые тесты проверяют поведение/инварианты и контрпримеры, а не только presence ключей и finite reward.

## 12. Порядок выполнения внутри одного прохода

1. Зафиксировать baseline HEAD/dirty tree, capabilities, Python/dependencies и результаты проверок; воспроизвести A01–A06.
2. Сначала WP-01 (state/clock), затем WP-02 (executor), затем WP-03 (ledger/outcome). Здесь проходит основной путь достоверности.
3. Завершить WP-04 schema/generator поверх утверждённых contracts; перенести legacy fixtures/adapters.
4. WP-05 — объединить policies/Gym; WP-06 — benchmark/dataset/replay, не до исправления score.
5. WP-07 — тонкий API/Studio adapter; WP-08 — install/CI/performance/report.
6. После каждого WP проверять ближайший gate и обновлять status; при провале искать причину, не перепрыгивать к UI ради демонстрации.
7. Финальный end-to-end run, независимый replay, package install, все gates и owner-facing краткий итог. Завершать только после конкретного reviewable результата или точного внешнего блокера.

## 13. Сквозной критерий «пакет выполнен»

- [ ] Forecast, restore и resume не меняют/не теряют будущее миссии.
- [ ] Общие ограничения выполняются независимо от caller/policy.
- [ ] Unique delivery, SLA и safety вычисляются из фактических событий.
- [ ] Строгая DSL создаёт различные конфигурируемые fleet missions.
- [ ] Агент принимает содержательные решения через новый fleet env без скрытой правильной команды.
- [ ] CLI, Gym, API и Studio показывают один и тот же исход одной миссии.
- [ ] Benchmark и dataset versioned, split-safe, воспроизводимы и replay-verified.
- [ ] Пакет устанавливается вне checkout; long/performance evidence приложены.
- [ ] Никаких автоматических commit/push/PR; ограничения профиля и невыполненные gates честно перечислены.

## 14. После прохода: следующий уровень, а не часть скрытого объёма

Далее — независимая qualification выбранного EO profile на данных партнёра, PettingZoo/MARL, richer weather/quality models, более точные orientation/finite-burn/navigation backends, SSA/Pc, GEO и proximity missions. Их добавлять на общий executor и benchmark infrastructure этого пакета.

Стартап-пилот требует отдельного согласования ICP, данных/прав, baseline и customer acceptance. Этот план должен дать техническую основу и демонстрацию для такого пилота, но агент не может объявить коммерческий спрос или эксплуатационную квалификацию доказанными собственными тестами.

## 15. Короткое поручение для запуска агента

> Прочитай `docs/INDUSTRIAL_ONE_PASS_PLAN.md` и аудит `docs/audits/V086_AUDIT.md`. Перепроверь актуальный код и реализуй весь пакет Mission Execution & Agent Benchmark v1 последовательно, WP-01…WP-08, за одно поручение. Не останавливайся после одного подпункта. Переиспользуй существующие компоненты; начни с негативных regressions и contracts, затем общий executor/ledger, DSL, fleet env, benchmark/data, local API/Studio и package/CI. Проверяй G1…G8, веди status и evidence. Не заявляй промышленную достоверность без соответствующих доказательств. Не коммить, не пушь, не создавай PR и не публикуй ничего самостоятельно. Сохрани изменения владельца. В финале приложи реально работающие команды, report и все оставшиеся ограничения.
