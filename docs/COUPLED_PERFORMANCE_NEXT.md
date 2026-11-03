# Следующий профиль: копирование и наблюдения coupled runtime

Основание — B11-Q1, а не предположение о необходимости уменьшить физическую точность.
Инструментированный малый coupled case (2 аппарата, 1 заявка, 20 физических секунд,
10 решений) дал 21.18 с под cProfile. Cumulative `deepcopy` — 17.41 с, 13.8 млн
вызовов; вложенные cumulative строки пересекаются, их нельзя складывать. Это
диагностические измерения с overhead profiler, не unprofiled latency qualification.
Полный top-25 и scope сохранены в `acceptance/B11_CAMPAIGN_PROFILE.*`.

## Первый безопасный участок — построение task-проекций

1. Проследить `TaskSession._refresh` → `Projection.build` → native actor/inventory/
   station reads и masks. На одной post-physics ревизии central и локальные роли
   повторно строят одни и те же публичные actor views; science evidence копируется,
   хотя числовой граф использует лишь часть полей.
2. Ввести внутренний, ограниченный одним `_refresh` снимок публичной телеметрии и
   inventory. Передавать в проектор только область текущей роли. Каждый role registry
   и его индексы остаются независимыми. Не использовать global entity registries.
3. Отдельно проверить reuse при построении single-action masks. Маска не должна
   вызывать backend admission или будущую физику. Реальный допуск после cancellation
   в trial должен заново читать актуальное состояние; snapshot до cancellation
   не подходит. Не сохранять mask cache между мутациями/ветками runtime.
4. Проверить, что все используемые getters read-only: повторное/сокращённое число
   вызовов не может менять RNG, sample age, погоду, очередь или историю. При любом
   сомнении оставить текущий native reader и собрать доказательство отдельно.
5. Публичные наблюдения/отчёты остаются owned mutable copies. Не отдавать общие
   science dictionaries или mutable NumPy aliases пользователю ради ускорения.

## Второй участок — trial и committed history

6. Разобрать повторные capture внутри TaskSession trial и native `_mutation`.
   Оптимизация вложенной транзакции допустима только с доказанным discard/rollback
   на любой validation/physics/projection failure, включая RNG и idempotency cache.
7. Отдельно аудировать действительно append-only native histories. Нельзя делить
   между forks произвольные изменяемые строки/буферы. История, command/event queues,
   physical state и public report copies требуют разных правил владения.
8. Не менять heat/RF/geometry sampling, solver tolerances, power/thermal thresholds,
   masks, operation admission или payload/quality требования ради меньшего wall time.

## Приёмка

- Golden сравнение всех task transitions, masks/role indices, reward/discount,
  controller trace и native backend report до/после на одинаковых frozen inputs.
- Positive EO/CPU/DTN/ground cases, loss/recovery/TTL/outage, concurrent resource
  lanes, cancellation, rollback/projection overflow и corrupt replay negatives.
- Mutation-alias checks: caller меняет возвращённые вложенные dict/list/arrays;
  parent/fork/cache/следующие наблюдения не должны измениться.
- Повторить B11-Q1 paired uninterrupted/prefix-recovered профили. Не сравнивать
  top-level checkpoint SHA между разными builds: source manifest закономерно меняется.
- Измерить unprofiled execution, step p50/p95, fork, replay/export, checkpoint bytes
  и fresh-process peak RSS. Profiler применяется отдельно, не внутри readiness timing.
- Сохранить исходный baseline и любые failures. Пороги не поднимать после результатов.

B11-Q2 реализует первый ограниченный участок: локальное построение native actor
observation и переиспользование owned public reads в пределах одного `_refresh`.
Native joint admission после cancellation по-прежнему читает реальный candidate.
B11-Q3 сохраняет полное клонирование, ускоряет owned trees и capture lifecycle
через общий memo; immutable/shared history ещё не реализована; точность физики
и ownership/rollback контракт сохранены. См. `PROJECTION_READ_OPTIMIZATION.md` и
`acceptance/B11_PROJECTION_READS.json` для фактических границ и результатов.

## Диагностика долгих workers перед следующей оптимизацией

Текущий `progress.json` публикует только полностью завершённые повторы. Его `0/N`
не говорит, находится ли worker в native execution, prefix restore, fork,
checkpoint, replay или export. Нельзя при timeout приписывать весь wall time
физическому интегратору или заявлять неизвестное число выполненных заявок.

Следующий отдельный участок — ограниченные atomic stage receipts каждого worker:
PID, frozen-input digest, monotonic stage start/end и счётчик решений, без science
payload и без утечки privileged данных в policy. Отдельно сохранять последние
наблюдённые RSS/CPU с явным статусом partial. Эти файлы диагностические, не replay
proof, не финальный результат и не основание исключать неудачный повтор из gates.
Тесты должны покрыть exception, timeout, неполную запись, стабильный порядок
агрегации и отсутствие изменения native/controller semantics при включённой
телеметрии. В B11-Q3 stage receipts реализованы: `NATIVE_COPY_AND_WORKER_DIAGNOSTICS.md`.

## Конкретные границы следующего copy-аудита

- `Lifecycle.observation` копирует каждый большой artifact, затем
  `MissionRuntime.observe` повторно копирует итоговый actor view. Проверить отдельный
  внутренний borrowed builder, используемый только внутри синхронного защищённого
  runtime read, с одной финальной owned-копией. Публичный lifecycle/default builder
  не должен внезапно начать возвращать native aliases. Нельзя кэшировать borrowed
  значения между revisions или передавать их в публичные ответы.
- `Lifecycle.report` отдельно копирует artifacts, operations и evidence histories;
  `ControlledBackend.capture` копирует связный mission/lifecycle/bus граф. Изменять
  правила shared history допустимо только после явной immutable representation и
  тестов вложенных mutations, fork isolation, discard и rollback. Отсутствие
  присваивания верхнему dict не доказывает неизменяемость вложенных science rows.
- Compact audit по-прежнему строит полную evidence-строку и хеширует её. Ускорение
  сериализации должно сохранять canonical encoding, все validation failures и
  точный evidence digest. Удалять поля из хешируемого evidence ради быстродействия
  нельзя. Отдельно измерять построение строки, копирование и canonical hashing.

B11-Q3 реализует borrowed construction с одной финальной owned-копией, узкий
scalar reward, memoized native capture и reuse validated canonical bytes.
Immutable/shared histories и отказ от вложенных rollback snapshots не реализованы. Прежде чем заменять общий deepcopy, нужно проверить repeated
references, cycles, tuple с mutable children, NumPy ownership, subclasses и
нестандартные `__deepcopy__`/reduction hooks. JSON round-trip не является общей
заменой: он меняет типы, aliases и допустимые ключи.

## Результат B11-Q3 и следующий приоритет

Прежний крупный workload завершён во всех трёх процессах: технические
latency/RSS/checkpoint gates и native replay/accounting/export прошли, но 28/40
заявок нарушили исходный completion target. Все 40 acquisition/CPU/custody
завершены. Следующий приоритет — versioned control windows и causal ground
scheduling, с сохранением старого failed результата и без ослабления порогов.
Подробный исполнимый пакет: `GROUND_DELIVERY_NEXT.md`.
