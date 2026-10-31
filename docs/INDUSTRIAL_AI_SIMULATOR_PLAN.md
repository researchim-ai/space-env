# План промышленного симулятора космических миссий для AI


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

Дата базовой сверки: 31 октября 2026. Версия пакета: 0.134.0.
Статус документа: **основная очередь реализации**. Старые roadmap сохраняют историю
аудита; при выборе следующего пакета использовать этот документ.

## 1. Целевой результат

Настраиваемая open-source библиотека, в которой один и тот же сценарий группировки
можно исполнять через Python SDK, CLI, Gym/MARL, LLM tools и Studio. Действия агента
изменяют причинно согласованное состояние миссии. Полезность подтверждается
полученными и доставленными продуктами, соблюдением ограничений, воспроизводимым
повтором и сравнением с объявленными baseline.

Первый промышленный продукт: **планирование и исполнение неоднородной LEO-миссии
ДЗЗ с динамическими заявками, бортовой обработкой и наземной доставкой**. Включает
перепланирование при отказах, обучение агентов и генерацию проверяемых данных.
Базовая измеряемая ступень — 16 аппаратов/100 заявок; следующая — 64/1000.
256/10000 — расширенная квалификация после предыдущих ступеней, а не условие
первого пилота или уже подтвержденная возможность.

Уровень моделирования выбирается явно: быстрый обучающий, reference и
калиброванный под конкретную миссию. Цифровым двойником профиль становится после
калибровки по данным миссии и согласованной приемки, а не после включения флага.
Flight-qualified управление реальными аппаратами — отдельная область требований.

## 2. Точка старта и подтверждения

Сверены код и проверки существующих компонентов:

- `simulation/world.py`, `commands.py`, `snapshot.py`, `parallel.py` — общий legacy
  мир, длительные команды, состояние и параллельная обертка.
- `reference/time.py`, `frames.py`, `orbit.py`, `access.py` — самостоятельные
  GCRS/ITRS, время/EOP, адаптивная пропагация и геометрические окна связи.
- `reference/operations.py`, `events.py`, `booking.py`, `thermal.py`,
  `dependencies.py` — исполняемые downlink/CPU/ресурсные ограничения, события,
  бронирования, отмены и зависимости на фиксированных эфемеридах.
- `reference/operations_env.py`, `event_env.py` — Gym, динамические продукты,
  опциональное раздельное управление radio/CPU.
- `reference/scheduling.py`, `optimal.py`, `benchmark.py` — эвристика и точный
  oracle для узкого static zero-power/nonpreemptive класса до 16 продуктов.
- `data_generators/portable_run.py`, `agent_run.py`, `reference_operations.py` —
  переносимые записи, replay, поведенческие RL-переходы и tool-dialogue.
- `api/server.py`, `api/console.py`, `viz/` — Studio/операторские сценарии,
  пока преимущественно привязанные к legacy session/состоянию процесса.
- `eval/load_report.py` — явно propagation-only замеры; отдельно есть legacy
  EO-регрессия 16 аппаратов/16 заявок/72 часа в `tests/test_tasking.py`.

На срезе CPU dispatch v10: 168 Python-модулей, 679 собранных тестов;
reference 249 passed, fast 110 passed, installed-wheel smoke прошел.
Это пересекающиеся наборы, их нельзя складывать. Полный прогон предыдущего
среза v9: 665 passed и один HTTP socket PermissionError песочницы; тот же тест
отдельно прошел с loopback-доступом. Полный набор на v10 повторно не запускался.

Подтверждения: [текущий статус](IMPLEMENTATION_STATUS.md),
[v10 verification](acceptance/REFERENCE_CPU_DISPATCH_V10_VERIFICATION.json),
[v9 full-run evidence](acceptance/REFERENCE_CALENDAR_QUALIFICATION_V9_VERIFICATION.json),
[frame migration](validation/FRAME_MIGRATION.md).

**Главные разрывы:** два несведенных пути исполнения; reference-операции над
фиксированными траекториями; обработка над уже загруженными байтами; неполная
связь съемки/ADCS/RF/ресурсов; ограниченная независимая физическая квалификация;
отсутствие подтвержденной плотной mission-нагрузки и общего dataset/service пути.
Существование helper или optional backend не закрывает интеграцию всей миссии.

## 3. Как будем выполнять

Работа идет крупными вертикальными пакетами **B01–B12**, а не серией отдельных
полей. Один пакет дает новый законченный workflow, код, сценарии, документацию
и доказательства. Следующий запрос «продолжаем» означает следующий незакрытый
пакет с выполненными зависимостями. Первым выполняется B01.

Статусы: `planned`, `in_progress`, `verified`, `needs_external_evidence`.
`verified` ставится только после приемки. Не называть частичную реализацию
закрытым пакетом. Если внешние данные отсутствуют, выполнить подготовку/измерения
и явно оставить соответствующий критерий незакрытым; не придумывать калибровку.
Размер пакета не обещает завершение за один ответ независимо от сложности.
Если независимая evidence для B02 пока недоступна, после реализации его контрактов
можно развивать B03/B06/B08 на явно ограниченном fixed-ephemeris профиле. Их
приемка в составе динамической промышленной миссии и соответствующие milestones
при этом остаются открытыми. Независимая подготовка не требует повторного
согласования уже разрешенных действий и не должна маскировать незакрытый gate.

Для каждого пакета обязательны:

1. Короткий ADR: контракты, границы модели, совместимость и зависимости.
2. Использование имеющегося кода через адаптеры/рефакторинг; новая реализация
   только при доказанном отсутствии нужной механики.
3. Сквозной воспроизводимый сценарий со случайными и scripted агентами.
4. Проверки успеха, невыполнимости, отмены/отказа, rollback и повторного запуска.
5. Отчет проверки с version/profile/schema, seeds, hashes, допусками и командами.
6. Обновленные SDK/CLI-примеры и ограничения. Сохраненные policies/журналы
   получают явную политику совместимости, без скрытого relabeling или fallback.
7. Обновление статуса ниже со ссылкой на артефакт приемки.

Короткие профили запускаются на измененном коде; полный suite — на milestone,
перед release и после существенного изменения общего исполнения. Интеграционные
HTTP-тесты запускать в среде с loopback; запрет сокета фиксировать как ограничение
среды, а не маскировать skip или ослаблением проверки. Проверять wheel из другого
каталога без импорта из checkout. Численные допуски объявлять до измерений.

Без запроса пользователя не делать commit/push, публикацию или deployment.

## 4. B01 — единый контракт сценария, runtime и эксперимента

Статус: **verified (2026-10-31), в пределах runtime-контракта**. Зависимости: существующие v1–v10.
Результат: одну reference-миссию можно запустить через общий сценарий, сохранить,
восстановить и наблюдать через одинаковый API; legacy остается явным backend.

Задачи:

- [x] B01.01. Описать ADR границ `ScenarioSpec`, `Runtime`, `Command`, `Event`,
  `ObservationView`, `Checkpoint`, `RunManifest`; закрепить SI/frame/time/revision.
- [x] B01.02. Ввести строгую scenario schema v3: актеры, аппараты, станции,
  заявки, profile/backend, источники данных, горизонт, seeds, fidelity и KPI.
- [x] B01.03. Реализовать общий runtime protocol: capabilities, observe,
  submit/advance, checkpoint/restore, iter_events(cursor, limit).
- [x] B01.04. Обернуть MissionWorld и ReferenceDownlinkMission отдельными
  адаптерами. Не переписывать все ядра и не переименовывать legacy-векторы в GCRS.
- [x] B01.05. Добавить capability negotiation: неподдерживаемая команда/profile
  отвергается до продвижения времени; dependencies/EOP не подменяются молча.
- [x] B01.06. Единые admission/result/error schemas, expected_revision и
  idempotency ключи. Повтор команды не создает вторую физическую операцию.
- [x] B01.07. Единый порядок одновременных событий: команда, завершение,
  availability, fault/cancel; принять порядок в ADR и проверить границы.
- [x] B01.08. Разделить truth, agent observation и privileged forecast; описать,
  какие capabilities/права нужны каждому представлению.
- [x] B01.09. Сохранить исходную и эффективную конфигурацию, profile/backend/data
  hashes. Неподдерживаемая миграция snapshot дает явную ошибку.
- [x] B01.10. CLI `mission-validate`, `mission-run`, `mission-replay` поверх
  runtime; существующие команды продолжают работать. Реализован также mission-resume.
- [x] B01.11. Добавить self-contained run/checkpoint для общего reference пути,
  включая события, брони, DAG, CPU, энергию и thermal hysteresis.
- [x] B01.12. Зафиксировать legacy/reference integration contracts и
  reference пример scenario → commands → checkpoint → resume → replay.

Код: существующие `mission/scenario_v2.py`, `simulation/world.py`,
`simulation/snapshot.py`, `reference/operations.py`, `cli.py`; предлагаемый новый
слой `runtime/` и schema v3 реализованы. Новые пути следующих пакетов остаются проектируемыми.

Приемка: одинаковые дискретные исходы через CLI и SDK; восстановление без старого
YAML; invalid/duplicate/stale команды не меняют состояние; coarse/fine исполнение
согласовано в объявленном допуске; agent view не содержит скрытой truth.
Артефакты: `docs/architecture/RUNTIME_V1.md`, `docs/acceptance/B01_RUNTIME.json`,
`examples/unified_runtime/`.

## 5. B02 — управляемая reference-физика и независимая квалификация

Статус: **needs_external_evidence (2026-10-31)**. Управляемый runtime реализован;
полная физическая квалификация не закрыта. Зависимости: B01. Результат: орбита reference-профиля может
реагировать на команды, а точность и область применимости измерены.

- [x] B02.01. Составить fidelity/error-budget matrix: режим, горизонт,
  выходные величины, допуски, данные, стоимость. Разделить ошибку интегратора,
  физической модели, EOP/атмосферы и исходных условий.
- [x] B02.02. Общий state provider для фиксированной эфемериды и динамической
  reference-пропагации; зафиксировать сохранение интерполяционных границ.
- [x] B02.03. Перенести импульсные маневры через публичный command protocol:
  масса, топливо, delta-V, epoch/frame и изменение последующих доступностей.
- [x] B02.04. Подключить имеющийся finite-burn helper к reference-профилю с
  определенной моделью тяги/ориентации/массы; не выдавать helper за уже связанный путь.
- [x] B02.05. Обновлять окна связи/съемки после маневра; старые планы инвалидируются
  по ревизии и физическому состоянию, а не продолжают использовать старую дугу.
- [x] B02.06. Унифицировать Sun/Moon, terrestrial gravity/drag, eclipse и
  атмосферное вращение в объявленных кадрах. Использовать текущие ForceModel switches.
- [x] B02.07. Ввести явные сроки годности leap/EOP/weather/ephemeris и flags
  предсказанных данных; проверить missing/out-of-range без тихой экстраполяции.
- [x] B02.08. Получить независимо построенные perturbation fixtures с provenance,
  units/frame/epoch/constants и лицензией. Общие inputs перечислить явно.
- [ ] B02.09. Проверить разные наклонения/эксцентриситеты, эпохи, 24/72 часа,
  отдельные/совместные силы, маневры и convergence по настройкам интегратора.
- [ ] B02.10. Проверить r/v/covariance, AOS/LOS/eclipses и миссионные исходы;
  не переносить короткий two-body tolerance на всю возмущенную миссию.
- [ ] B02.11. Добавлять higher-order gravity/другие эффекты по error budget
  конкретного профиля; отложенные эффекты перечислить, не обещать «всю физику».
- [x] B02.12. Выдать applicability report; эталонный профиль без внешних
  fixtures остается `needs_external_evidence`, даже если integration tests зеленые.

Код: `reference/{time,frames,orbit,access}.py`, `core/gravity.py`,
`core/propulsion.py`, runtime adapter, `validation/`.
Приемка: заданные заранее tolerances на независимых fixtures; command-driven
траектория/окна/расход топлива воспроизводятся; mixed-frame входы отвергаются.
Артефакты: [матрица](validation/PHYSICS_QUALIFICATION_MATRIX.md),
[приемка](acceptance/B02_PHYSICS.json), versioned frozen Boinor J2 fixtures
24/72h с shared EOP/frame/constants и generator/library/license provenance.
Это ограниченная независимость force formula/integration family, не orbit truth.

B02.05: выполнен инженерный geometry contract для окон связи/съемки и
forecast binding/revision invalidation; исполнение acquisition и создание данных
остаются B03. Частично: B02.09 — центральные/J2 дуги, short
all-force и burn/convergence проверки есть. В Q1 добавлены другая эпоха/наклонение
и frozen J2+Sun/Moon shared-input comparison; independent drag/SRP combined arcs
остаются открытыми. B02.10 — r/v, frame covariance, измененный LOS и optional
Cartesian STM/P propagation проверены. В Q2 есть analytic-orbit/bisection AOS/LOS
и point-acquisition сравнения, conical-versus-apparent-disc eclipse contacts с
явным model-comparison budget. Независимые frames/ephemerides, perturbed covariance
и калиброванные миссионные исходы еще не закрыты.
B02.11 — список deferred physics опубликован; выбор higher-order effects требует
error budget целевой миссии. Не считать эти пункты выполненными автоматически.

## 6. B03 — замкнутый жизненный цикл заявки и продукта

Статус: **verified (2026-10-31), point/resource engineering surrogate**. Зависимости: B01 и проверенные
инженерные контракты B02-Q1/Q2. Открытая физическая квалификация B02 продолжает
блокировать промышленную приемку, но не разработку B03 под объявленным surrogate
профилем. Производственный gate B02.09/.10/.11 не закрывается этим разделением.
Результат: заявка действительно порождает данные при выполнении съемки, затем
обработку и доставку; заранее загруженные байты остаются отдельным режимом.

- [x] B03.01. Отделить customer request, acquisition opportunity, operation,
  raw/intermediate/final product и delivery obligation; стабильные IDs/provenance.
- [x] B03.02. Состояния released/accepted/scheduled/running/completed/failed/
  cancelled/expired и проверенная таблица переходов, включая частичный результат.
- [x] B03.03. Прием, приоритет, дедлайн, отмена и независимый поток заявок во время
  миссии; не создавать спрос из уже найденных удобных окон.
- [x] B03.04. Выполнять acquisition на фактической геометрии и ресурсах;
  не создавать успешный продукт по одному наличию команды/планового окна.
- [x] B03.05. Материализовать raw data после допустимой съемки; учитывать
  bytes, quality metadata, coverage, время доступности и энергозатраты.
- [x] B03.06. Processing DAG с потребляемыми входами, сохраняемыми выходами,
  изменением размеров, fan-in/out и admission атомарной записи результата.
- [x] B03.07. Разделить resource surrogate и плагин реального преобразования;
  synthetic compression/quality параметры не представлять научным алгоритмом.
- [x] B03.08. Persistent workspace/checkpoint/switching cost, потери/повторение
  работы и политика хранения; сохранить явное управление CPU из v10.
- [x] B03.09. Retention, TTL/expiry и выбор eviction; не удалять pinned input
  или еще не подтвержденный output без объявленной политики.
- [x] B03.10. Баланс: acquired/injected/processed-in/out/transmitted/acked/
  expired/discarded; преобразование не означает произвольного сохранения bit count.
- [x] B03.11. Связать delivery receipt с исходной заявкой и конечным destination;
  разные стадии не получают одну и ту же награду за конечный результат.
- [x] B03.12. Сценарии feasible/infeasible/deadlock/cancel/failed materialization
  с причинами результата для каждой заявки.

Код: `mission/tasking.py`, `simulation/commands.py`, `reference/operations.py`,
продуктовые модели в `core/`, runtime events и dataset recorder.
Приемка: request → acquisition → raw → processing → final → delivery в одном
run; rollback не создает двойных выходов; невыполнимые заявки остаются в знаменателе.
Артефакты: [пример](../examples/end_to_end_eo/demo.py),
[контракт](validation/PRODUCT_LIFECYCLE_CONTRACT.md),
[приемка](acceptance/B03_PRODUCT_LIFECYCLE.json).

Область закрытия: sampled point/dwell acquisition, declared resource work/energy,
JSON/resource artifacts и reliable fixed-rate transport. Не утверждается создание
научных изображений, непрерывная geometry completeness, реальный kernel checkpoint,
packet ACK/retransmission или calibrated spacecraft behavior. Raw/final отделены,
а plugins имеют version/fingerprint и отдельный materialize interface. Полная
промышленная приемка остается B02/B04/B05/B09/B12.

## 7. B04 — связанный аппарат: ориентация, payload, энергия и здоровье

Статус: in_progress — B04-A coupled bus, B04-B1 sampled area coverage и
B04-B2 optical/SAR quality/resources, B04-B3 revisit/stereo/bundles и B04-C
resource lanes и B04-D1 body thrust verified в пределах объявленных инженерных surrogate contracts.
Зависимости: инженерные контракты B02/B03;
открытая квалификация B02 остается production gate.
Результат: окна/качество съемки и связи зависят от состояния конкретного аппарата.
Отмеченные пункты приняты в пределах инженерного B04-A surrogate; детализация
ограничений — в SPACECRAFT_PROFILES.md. Физическая qualification не закрыта.

- [x] B04.01. Versioned spacecraft presets с происхождением параметров, units,
  неизвестными значениями и меткой synthetic/published/calibrated.
- [x] B04.02. Связать attitude quaternion/rates, slew/settling, ограничения
  pointing/keep-out с операциями и фактическими окнами; переиспользовать helpers.
- [ ] B04.03. Footprint/AOI intersection, площадь/полоса, revisit/stereo timing
  и накопление покрытия; не заменять полигон удобной центральной точкой.
  B04-B1 verified: sampled rectangular footprints, polygon/holes/dateline/polar,
  area union, DAG lineage и delivered threshold. Revisit — bounded diagnostics
  по completion times. B04-B3 verified: actual exposure records, revisit chains,
  sampled stereo convergence/baseline и quality-qualified common area;
  partial ground delivery не дает credit. Continuous swath/limb clipping и
  scientific reconstruction остаются открыты.
- [ ] B04.04. Оптические illumination/cloud/quality и SAR duty/energy — отдельные
  объявленные модели с воспроизводимыми неопределенностями и applicability.
  B04-B2 verified: categorical cloud/unknown validity, sampled optical GSD/Sun,
  nominal SAR incidence/side/resolution, pulse/session duty и DC/RF/heat ledgers.
  Stochastic uncertainty, radiometry и flight calibration остаются открыты.
- [x] B04.05. Pointing солнечных панелей, shadow, нагрузка payload/CPU/radio,
  battery efficiency/capacity и low-power/safe-mode поведение.
- [x] B04.06. Общая ресурсная arbitration: реально допускаемая параллельность
  CPU/payload/radio, buses, memory bandwidth; serial surrogate остается вариантом.
  B04-C verified: один managed CPU/payload/radio lane, shared peak/SOC/bandwidth,
  реальные pause/throttle, reservations и native station arbitration. Native
  in-place CPU сохраняется serial-only; multicore/hardware/SLA не квалифицированы.
- [ ] B04.07. Thermal resource nodes/CPU-payload coupling там, где есть данные;
  параметры и погрешность отделить от измеренной температуры.
- [ ] B04.08. Топливо, момент/запас исполнительных органов и health limits,
  если включены соответствующие модели; пропущенное явно объявлять.
  B04-D1 verified: body-frame finite thrust, paid gimbal/ignition, mass/fuel,
  electric DC/jet/heat balance, faults/cancel/replay. Torque/wheels — B04-D2.
- [x] B04.09. Fault/FDIR/recovery и degradation без скрытого восстановления
  ресурса; команды safe mode отменяют/приостанавливают работу причинно.
- [x] B04.10. Квалифицировать несколько различных resource presets одной
  сценарной схемой; проверять не названия, а различающееся поведение.

Приемка: slew занимает время и энергию; неверное pointing/keep-out блокирует
результат; интегральные ledgers сходятся; outages/thermal/fault меняют исход.
Артефакты: `docs/validation/SPACECRAFT_PROFILES.md`, `docs/acceptance/B04_BUS.json`.

## 8. B05 — наземная сеть, RF и ретрансляция

Статус: planned. Зависимости: B01, B03, B04.
Результат: доставка — исполняемая сеть ресурсов с конечной пропускной способностью.

- [ ] B05.01. Отделить visibility, link eligibility, setup/lock, throughput
  и успешное подтверждение доставки.
- [ ] B05.02. Декларируемый link-budget interface: частоты, pointing, потери,
  margins, режимы modem и источники параметров; constant-rate baseline сохранить.
- [ ] B05.03. Per-terminal и aggregate ограничения не смешивать; реализовать
  sharing/spectrum/bandwidth как отдельные policies с явной областью применимости.
- [ ] B05.04. Подготовка/перенацеливание наземной антенны, переключение аппаратов
  и стоимость handover; бронь не означает мгновенно готовый канал.
- [ ] B05.05. Общий booking/admission adapter: локальные брони v5 и импорт
  внешних reservation records; история, версии, конфликты и восстановление.
- [ ] B05.06. Интерспутниковые каналы: visibility/pointing, radio resource,
  serialization, задержки и хранение у промежуточных узлов.
- [ ] B05.07. Конкурентный event-driven relay вместо только последовательного
  blocking вызова; ledger не допускает раннего использования полученных бит.
- [ ] B05.08. DTN queues/routing, TTL, retries, dropped/duplicate fragments;
  ACK и exactly-once credit конечного продукта отделить от факта передачи.
- [ ] B05.09. Неопределенность outage/weather/loss, наблюдаемая агентом в
  положенный момент, с seeded truth/observation раздельно.
- [ ] B05.10. Смешанный сценарий direct/relay, отказ станции, переполнение
  буфера, reroute, late ACK и восстановление с checkpoint.

Код: `simulation/relay.py`, `core/ground.py`, `core/link.py`, `simulation/dtn.py`,
`reference/booking.py`, runtime resource events.
Приемка: данные проходят всю цепочку во времени, не дублируются в reward,
конкурентные radios/терминалы не превышают capacity; replay совпадает.
Артефакты: `examples/network_mission/`, `docs/acceptance/B05_NETWORK.json`.

## 9. B06 — единая задача для RL, MARL, LLM и человека

Статус: planned. Зависимости: B01, B03; интеграционная приемка после B04/B05.

- [ ] B06.01. Разделить truth state, estimator/belief и actor observation;
  latency, шум, пропуски, обнаружение faults и forecast uncertainty.
- [ ] B06.02. Общие entities/actions/masks для аппаратов, станций, заявок,
  продуктов и операций. Маска не должна раскрывать скрытое будущее.
- [ ] B06.03. Стабильные padded/sparse representations и versioned action
  contracts; переполнение capacity отклоняется, а не скрыто обрезает задания.
- [ ] B06.04. Gym facade поверх Runtime, официальный optional MARL adapter
  с lifecycle agents/reset/seed/termination по проверяемому API.
- [ ] B06.05. Joint/centralized и decentralized control; station/operator agents
  используют тот же диспетчер ресурсов, без второй физики.
- [ ] B06.06. Политика решений и внутренние safety thresholds разделены;
  симулятор не вызывает policy повторно скрыто внутри advance.
- [ ] B06.07. Variable-duration transitions: elapsed_sec, discount per physical
  time, termination/truncation и retry semantics для обучения.
- [ ] B06.08. LLM tool schemas, structured validation, budget, revisions,
  idempotency, recovery после отказанного вызова; отсутствие фиктивной truth.
- [ ] B06.09. Task/objective registry: throughput, SLA/priority utility, latency,
  energy/risk, constrained reward; infeasible requests остаются в отчетах.
- [ ] B06.10. Replay одного сценария через SDK/CLI/Gym/tools/operator с
  одинаковыми командами и исходами; random/scripted/learned результаты различать.

Код: `simulation/parallel.py`, `mission/agent_protocol.py`, `mission/tools.py`,
`rl/`, reference Gym facades; адаптация вместо параллельной новой среды для каждого кейса.
Артефакты: `docs/AGENT_RUNTIME_CONTRACT.md`, `docs/acceptance/B06_AGENTS.json`.

## 10. B07 — планирование и честное сравнение AI

Статус: planned. Зависимости: B03, B05, B06.

- [ ] B07.01. Общая objective specification между reward, planner и отчетом;
  throughput optimum не называть optimum по priority/SLA.
- [ ] B07.02. Оставить v9 oracle как узкий подтвержденный класс; добавить
  ресурсные малые задачи с независимо проверяемым optimum/bounds.
- [ ] B07.03. Rolling horizon: warm start, commitments/бронь, отмена, измененные
  окна после маневра, runtime budget и гарантированно выполнимый fallback.
- [ ] B07.04. Совместный выбор acquisition/CPU/radio/relay с переменной
  параллельностью, memory, deadlines и setup costs.
- [ ] B07.05. Не маркировать timeout, feasible incumbent и lower/upper bound
  как доказанный optimum; gap и сравнимость классов обязательны.
- [ ] B07.06. Seeded robustness для outage/demand/forecast/model uncertainty;
  information budget одинаков для сравниваемых политик.
- [ ] B07.07. Набор greedy/fifo/deadline/constraint baselines и learned policy,
  explicit hardware/runtime budget, policy/data hashes и независимые seeds.
- [ ] B07.08. Отчеты utility/violations/coverage/latency/resource cost/regret;
  средние, разброс и хвостовые неуспехи вместо одного удачного эпизода.
- [ ] B07.09. Известные infeasible cases и adversarial corner cases;
  дать причины unmet, не убирать их из метрик.
- [ ] B07.10. Benchmark report self-contained и recomputable; кросс-классное
  сравнение допускается только с объявленными ограничениями интерпретации.

Артефакты: `docs/BENCHMARK_PROTOCOL.md`, `docs/acceptance/B07_PLANNING.json`,
набор versioned small/medium benchmark instances.

## 11. B08 — фабрика RL/SFT/preference данных

Статус: planned. Зависимости: B01, B03, B06; качество teacher оценивается через B07.

- [ ] B08.01. Scenario families и seeded generators: география/орбиты/аппараты,
  demand, deadlines, ресурсы, неопределенности, feasible/infeasible доли.
- [ ] B08.02. Полный manifest: effective spec, seeds, профили, данные, policy,
  observation/action schema и objective; semantics и byte hashes разделены.
- [ ] B08.03. Append-only event/transition journal, cursors, ограниченная RAM,
  sharding/checksums и atomic completion. Не сохранять весь префикс в каждой строке.
- [ ] B08.04. Crash/resume/corrupt shard detection; экспорт не читает незавершенный
  shard как полный эпизод. Квалифицировать schema migrations.
- [ ] B08.05. RL transitions с Dict/MultiDiscrete/sparse actions, elapsed_sec,
  masks, terminal observations и reward components.
- [ ] B08.06. SFT conversations: task, действительно доступные observations,
  tool call/result и завершение; не выдумывать chain-of-thought эксперта.
- [ ] B08.07. Teacher tiers: scripted, heuristic, feasible solver, proven-in-scope,
  calibrated operator; replay integrity отдельно от качества демонстрации.
- [ ] B08.08. Preference пары с одинаковыми исходными условиями и объективным
  ranking; safety trade-offs/несравнимые варианты маркировать явно.
- [ ] B08.09. Train/validation/test по mission family/lineage, epoch/geography/
  hardware где нужно; exact и near-duplicate/correlation audit.
- [ ] B08.10. Dataset card: лицензии входов, provenance, profile limitations,
  distribution/отказы, coverage и leakage checks.
- [ ] B08.11. Replay выборок и end-to-end training smoke; одинаковые seeds
  воспроизводят данные, независимые seeds не схлопываются в один шаблон.

Код: `data_generators/`, `rl/trajectories.py`, portable recorder APIs.
Артефакты: `docs/DATASET_CONTRACT.md`, `docs/acceptance/B08_DATA_FACTORY.json`,
малый публикуемый только по явному решению пользователя sample dataset.

## 12. B09 — масштабирование настоящей нагрузки

Статус: planned. Зависимости: B03, B05, B08.

- [ ] B09.01. Новый load runner с исполняемыми заявками, станциями, CPU, сетью
  и fault events; старые propagation-only отчеты не переименовывать в mission load.
- [ ] B09.02. Зафиксировать CPU/RAM/versions и warm/cold режим; отдельно мерить
  propagation, access, planner, event kernel, serialization и artifact I/O.
- [ ] B09.03. Ступень A: 16 аппаратов, 100 заявок, 24 часа, несколько станций,
  online updates; outcome/reason для каждой заявки и сверка ledgers.
- [ ] B09.04. Ступень B: 64 аппарата, 1000 заявок, 72 часа; heterogeneous buses,
  shared capacities, conflicts, downlink+relay и решения policy.
- [ ] B09.05. Ступень C: 256 аппаратов, 10000 заявок — только после B;
  расширить текущие caps через квалификацию, не простым удалением проверок.
- [ ] B09.06. Измерить wall/simulated time, events/sec, decision P50/P95/P99,
  peak RSS, bytes/transition, artifact growth и restart/replay cost.
- [ ] B09.07. Зафиксировать целевые бюджеты после исходного измерения на заданной
  машине до оптимизации; thresholds хранятся в versioned performance spec.
- [ ] B09.08. Найти профиль O(T^2)/O(N^2) history/scan/copy costs; индексировать
  события/продукты/окна, bounded views и sparse observations по измеренным hot spots.
- [ ] B09.09. Параллельная генерация независимых эпизодов с независимыми RNG,
  стабильными IDs и воспроизводимым merge; не распараллеливать причинный мир наугад.
- [ ] B09.10. Endurance/recovery и проверка выхода памяти на плато при streaming;
  не доказывать это только запуском без OOM.

Приемка: A и B выполняются в объявленном бюджете с измеренными исходами,
балансами и воспроизводимым отчетом; C отдельно обозначается verified/unverified.
Артефакты: `docs/PERFORMANCE_SPEC.md`, `docs/acceptance/B09_SCALE.json`,
per-rung reports и profiler evidence.

## 13. B10 — Studio и оператор на общем runtime

Статус: planned. Зависимости: B01, B03, B05, B06, B08.

- [ ] B10.01. API sessions поверх runtime adapter; profile/capabilities и
  текущая revision видны пользователю, исчезает отдельная «демо-истина».
- [ ] B10.02. Сценарий/seed/аппараты/цели/политика выбираются на одной странице;
  invalid combination объясняется до запуска.
- [ ] B10.03. 2D ground track и 3D показывают один timestamp/selection из run,
  GCRS/ITRS корректно конвертированы; antimeridian/LOS/AOS обработаны явно.
- [ ] B10.04. Mission timeline: acquisition, CPU, slew, radio, relay, bookings,
  faults; запланированное отделено от выполненного.
- [ ] B10.05. Пульт управления: команды, admission reasons, ожидаемые ресурсы,
  ревизии, безопасная отмена/повтор, подтвержденное изменение состояния.
- [ ] B10.06. Объяснения unmet, dependency block, power/thermal/memory issues,
  delivered/acked/late/expired и model limitations обычным языком.
- [ ] B10.07. Event cursor/pagination, ограниченная история, pause/seek/replay,
  восстановление сессии после перезапуска и экспорт того же run artifact.
- [ ] B10.08. Read-only agent view и truth inspection разделены; диагностический
  UI не становится неявным каналом передачи будущего агенту.
- [ ] B10.09. API concurrency/resource limits; переписать process-global
  session storage через storage interface, а не объявить словарь production DB.
- [ ] B10.10. Browser E2E: создать миссию → выполнить/отменить команду →
  увидеть исход → скачать run → replay вне Studio, включая ошибки/перезапуск.

Код: `api/server.py`, `api/console.py`, `mission/session.py`, `viz/`, runtime API.
Артефакты: `docs/STUDIO_RUNTIME_GUIDE.md`, `docs/acceptance/B10_STUDIO.json`,
скриншоты и browser E2E report. Внешний hosting не является разрешенным deployment.

## 14. B11 — release engineering и эксплуатация

Статус: planned. Зависимости: B01–B10 для выбранного release scope.

- [ ] B11.01. Public API/schema/version policy, deprecations и migration fixtures;
  модель, артефакт и версия пакета имеют разные явно описанные версии.
- [ ] B11.02. Core/optional dependencies и воспроизводимые qualification locks,
  dependency/model data manifests, licenses/attribution; offline data setup.
- [ ] B11.03. Wheel/sdist, console/Gym entry points, packaged assets/configs,
  исходники и установленные examples; другие рабочие каталоги без checkout.
- [ ] B11.04. CI fast/reference/full/install и отдельный load/endurance schedule;
  обеспечить loopback для HTTP integration и сохранять machine-readable reports.
- [ ] B11.05. Python/platform support matrix по реально выполненным проверкам,
  без экстраполяции одного Linux/Python запуска на все платформы.
- [ ] B11.06. API input limits, path boundaries, ownership/auth/isolation для
  hosted multi-user варианта; локальная библиотека не обязана содержать весь SaaS.
- [ ] B11.07. Session/job quotas, durable storage, backup/recovery, cancellation,
  structured logs/metrics и понятные operational failures для hosted профиля.
- [ ] B11.08. Reproducible release checklist, changelog, upgrade guide и demo
  выполненные на release candidate; support policy и известные ограничения.
- [ ] B11.09. Все заявленные обязательные gates зеленые на финальном коде;
  средовые ограничения не скрываются, исключения явно отражены в протоколе.
- [ ] B11.10. Подготовить reviewable release artifacts. Публикация, push и
  deployment выполняются только по отдельному решению пользователя.

Артефакты: `docs/RELEASE_POLICY.md`, `docs/OPERATIONS_GUIDE.md`,
`docs/acceptance/B11_RELEASE.json`, versioned release candidate manifests.

## 15. B12 — промышленный пилот и продуктовая приемка

Статус: planned; калибровочная часть требует внешних данных.
Зависимости: B02–B11 в пределах профиля пилота.

- [ ] B12.01. Выбрать один сценарий заказчика и измеряемое улучшение: SLA,
  latency, utility, доля выполненных заявок, труд оператора или robust recovery.
- [ ] B12.02. Получить законно используемые орбиты, demand, bookings, ресурсы,
  telemetry/операционные исходы; анонимизация и права на dataset отдельно.
- [ ] B12.03. Калибровать параметры на train-периоде, закрыть applicability
  и error budget, заморозить модель до holdout испытаний.
- [ ] B12.04. Сравнить с реальным incumbent/manual baseline на одинаковой
  информации и вычислительном бюджете; synthetic baseline не подменяет incumbent.
- [ ] B12.05. Включить неудобные/невыполнимые дни, ошибки прогноза и отказы;
  зафиксировать uncertainty, safety violations и реальную цену улучшения.
- [ ] B12.06. Проверить workflow пользователя: setup → experiment → training/
  evaluation → объяснение результата → повтор и экспорт данных.
- [ ] B12.07. Разделить open-source core, поддерживаемые model packs,
  enterprise integration/hosted compute и услуги валидации; проверить спрос.
- [ ] B12.08. Собрать acceptance report с согласованными критериями и
  ограничениями. До этого использовать формулировку «pilot candidate».

Артефакты: `docs/pilots/EO_PILOT_PROTOCOL.md`, data access/license manifest,
`docs/acceptance/B12_PILOT.json`. Отсутствующие данные/клиентская приемка не могут
быть заменены программным тестом или придуманным числом точности.

## 16. Порядок и контрольные результаты

**Milestone M1 — общий воспроизводимый runtime:** B01, B02.
Новая reference-миссия управляется публичными командами в объявленных frame/time;
legacy-профиль сохраняется явно. B02 без независимой физической evidence не verified.

**M2 — замкнутая полезная миссия:** B03, B04, B05.
Новые заявки приводят к съемке, обработке, передаче и подтвержденным outcomes;
маневры/pointing/ресурсы/сеть действительно влияют на результат.

**M3 — AI benchmark и данные:** B06, B07, B08.
Агенты разных интерфейсов решают одну задачу; learned/heuristic/solver/teacher
различаются, данные воспроизводятся и имеют split/quality provenance.

**M4 — поставляемый pilot candidate:** B09(A/B), B10, B11.
Плотные нагрузки, Studio/SDK/CLI, wheel/recovery и release protocol подтверждены.
Ступень 256/10000 не блокирует первый пилот, если она не заявлена в release scope.

**M5 — промышленный профиль:** B12 и выполненный B02 error budget выбранной миссии.
Именно данные/сравнение/приемка дают право заявлять промышленную применимость
этого профиля. Обобщение на другие орбиты/аппараты требует новой квалификации.

Основная последовательность: B01 → B02 → B03 → B04 → B05 → B06 → B07 → B08 →
B09(A/B) → B10 → B11 → B12. Контракты наблюдений/артефактов B06/B08 проектируются
в B01, но их полная приемка выполняется в своем пакете. Подготовка внешних fixtures
и pilot data может идти раньше; она не закрывает зависимые gates автоматически.

## 17. Расширение после первого промышленного профиля

Следующие отдельные профили, а не скрытое обещание «моделируем весь космос»:

1. Коммуникационная группировка: маршрутизация/traffic/service quality и
   совместное управление спектром, антеннами, энергией и наземным сегментом.
2. Orbit operations/SSA: OD/навигационные наблюдения, covariance, conjunction
   risk, uncertainty и маневры с независимой проверкой физических допущений.
3. Formation/RPO/servicing: относительная динамика, proximity/safety,
   ориентация и контактные/сервисные операции в отдельной fidelity matrix.
4. GEO/MEO/PNT-постановки: подходящие force/time/clock/payload models и
   stationkeeping/service objectives с отдельной калибровкой.
5. Транспортные/лунные/межпланетные миссии: новые тела, кадры, environment,
   связи и propulsion regimes. LEO-валидация не переносится на них автоматически.

Каждый профиль проходит тот же цикл: постановка → контракты/модели → независимые
fixtures → end-to-end задачи → baseline/AI/data → нагрузка → applicability report.

## 18. Журнал закрытия крупных пакетов

B01 verified в пределах runtime-контракта; B02 needs_external_evidence; B03 verified как point/resource surrogate; B04 in_progress; B05–B12 planned. v1–v10 — исходная реализованная база,
а не автоматическое закрытие новых пакетов.

Для каждой записи использовать:

- Package ID, статус и реализованный пользовательский workflow.
- Измененные API/schema/model profiles и policy миграции.
- Выполненные команды проверок и measured outcomes; skips/environment failures.
- Ссылка на acceptance JSON, примеры и known limits.
- Следующий пакет и реально невыполненные зависимости.

### B01 — verified, 2026-10-31

Реализованы scenario v3, два адаптера, capability negotiation, revision/idempotency,
атомарные мутации и локальная сериализация вызовов, agent/truth/forecast views,
self-contained verified-prefix checkpoints, CLI validate/run/resume/replay/schema.
38 новых runtime tests; reference 287 passed; fast 110 passed; всего 717 collected.
Installed wheel проверен из /tmp с Python -I, CLI resume/replay и 173/173 модулями.
Полный regression B01 повторно не запускался: это не заявление all-tests-green.
[ADR](architecture/RUNTIME_V1.md), [acceptance](acceptance/B01_RUNTIME.json),
[пример](../examples/unified_runtime/demo.py).
Ограничения: фиксированные reference-эпhemerиды, metadata seed, replay-prefix
restore растет с журналом; общий обучающий facade и промышленная физическая
квалификация не закрыты этим пакетом.

### B02 — needs_external_evidence, 2026-10-31

Реализован `reference-dynamic-downlink/1`: общий state provider, импульсные и
конечные включения через runtime, расход массы/топлива, cancel, пересчет окон
связи, geometry revision, mid-burn checkpoint/resume/replay, все ForceModel
switches и явная data validity. Frozen external-formula J2 24/72h и отдельный
RK45 finite-burn RHS прошли заранее заданные бюджеты. Общие входы перечислены.

18 новых tests; reference 305 passed; fast 110 passed; всего 735 collected.
Изолированный wheel: 178/178 Python modules, frozen fixtures, установленная CLI
и динамическая миссия с восстановлением. Full regression не перезапускался.
[Матрица применимости](validation/PHYSICS_QUALIFICATION_MATRIX.md),
[приемка](acceptance/B02_PHYSICS.json), [пример](../examples/unified_runtime/dynamic_demo.py).

### B02-Q1 — verified в пределах navigation prediction, 2026-10-31

Optional GCRS Cartesian STM/P, continuous white-acceleration PSD, impulse
velocity noise; global STM composition across burn/coast/cancel, current agent
covariance и privileged STM. Строгие frame/PSD/force-model guards и solver budget;
prefix checkpoint/rollback включают неопределенность. Калиброванный фильтр и
случайная фактическая ошибка исполнения этим не реализованы.

Дополнительно frozen J2+Sun/Moon 24h comparison на другой эпохе/наклонении,
с явно общими ephemerides/frames/EOP/constants. Qualification CLI включает
сравнение covariance с 12 deterministic sigma points через отдельный RK45 RHS.
24 новых теста; reference 329 passed; fast 110 passed; 759 collected.
Wheel: 180/180 Python modules, 3 frozen orbit fixtures, изолированный Python -I,
установленная CLI validate/run/resume/replay с covariance. Full regression
не перезапускался. [Контракт](validation/NAVIGATION_COVARIANCE_CONTRACT.md),
[приемка](acceptance/B02_Q1_NAVIGATION.json),
[пример](../examples/unified_runtime/navigation_demo.py).

### B02-Q2 — verified в пределах geometry/event contract, 2026-10-31

Point-target WGS84/cone acquisition forecasts, solar/range/elevation/dwell
ограничения, availability и root-guarded usable windows; readonly eclipse
forecast использует ту же conical shadow модель, что power/SRP. Все forecasts
имеют optional command binding по scenario/actor/trajectory/revision. Stale,
foreign или перестроенный после maneuver/cancel прогноз отвергается атомарно.
Checkpoint/replay и идемпотентный retry сохраняют контракт. Это не acquisition
execution и не улучшение flux-модели или resource integration.

Отдельные analytic orbit/WGS84 equations + bisection проверяют AOS/LOS и окна
съемки; eclipse contacts сравниваются с spherical apparent discs, budget 1s
обозначен как различие моделей. Sampling completeness не обещается.
28 новых tests; reference 357 passed; fast 110 passed; 787 collected.
Wheel 183/183 Python modules, 3 frozen fixtures, isolated -I SDK/CLI workflow
с binding и stale rejection. Full regression не перезапускался.
[Контракт](validation/GEOMETRY_EVENTS_CONTRACT.md),
[приемка](acceptance/B02_Q2_GEOMETRY.json),
[пример](../examples/unified_runtime/geometry_demo.py).

### B03 — verified в пределах point/resource lifecycle, 2026-10-31

Новый явный backend/profile reference-lifecycle/1 и objective fulfilled_requests.
Customer demand/release/priority/deadline/cancel, sampled geometry/resource
acquisition без preloaded raw, consuming transformation DAG с меняющимися размерами,
fan-in/out, persistent workspace/switching/TTL, atomic materialization и
request-linked full-station receipts. External unmanaged injection не выполняет
заявку; impossible/failed/cancelled demand сохраняется в denominator.

ProductTransformer registry отделен от declared CPU/resource costs; встроены
metadata surrogate и JSON geometry features. External plugins требуют явных
version/fingerprint и регистрации при restore, без fallback. Артефакты не являются
rendered/scientific images. Исправлен ошибочный legacy frame tag в dynamic
run manifest: все reference-* имеют GCRS/ITRS.

31 новых tests; reference 388 passed; fast 110 passed; 818 collected.
Wheel 184/184 Python modules, 3 frozen orbit fixtures, isolated Python -I;
CLI validate/run/resume/replay и mid-acquisition/mid-processing checkpoints.
Demo: 1 fulfilled / 2 total, acquired 1Mbit → transformed final 0.5Mbit →
full ACK, нулевой inventory residual. Full regression не перезапускался.
[Контракт](validation/PRODUCT_LIFECYCLE_CONTRACT.md),
[приемка](acceptance/B03_PRODUCT_LIFECYCLE.json),
[пример](../examples/end_to_end_eo/demo.py).

**Следующее: B04-D — actuator/thrust/attitude/health coupling в общем spacecraft profile.**
B03 verified как инженерный surrogate, B04 in_progress, B05–B12 planned. Полный B02 остается
needs_external_evidence: независимые физические данные/модели и calibrated mission
budgets обязательны для production claim. Commit/push не выполнялись.


### B04-A — verified в пределах coupled-bus contract, 2026-10-31

Явный профиль reference-spacecraft-bus/1: body→GCRS quaternion, конечная скорость
разворота, settling/ADCS energy, pointing/keep-out и rate-bound tracking. Body-fixed
PV/shadow, two-port battery с efficiencies, единый electronics RC thermal node,
load limits, latch safe/fault/recovery и монотонная degradation. Два synthetic
presets дают различные исходы при одинаковой нагрузке. Resource/thermal ledgers,
atomic rollback, local observation и prefix replay включают bus state.

B04.01/.02/.05/.09/.10 покрыты инженерными surrogate contracts; B04.07 частично
покрыт одним синтетическим RC узлом. Это не flight-calibrated models и не
независимая квалификация. B04.03/.04/.06/.08 остаются открыты. Маневры в новом
профиле намеренно запрещены до thrust/attitude/health coupling; trajectory
объявлена propagated-gcrs-coast. Другие dynamic/lifecycle профили сохранены.

29 новых tests; reference 417 passed, fast 110 passed; отдельно legacy slew/
operations/pass-record 19 passed; 847 collected. Счетчики пересекаются.
Wheel 185/185 modules и 3 frozen fixtures совпадают с исходниками; isolated -I
SDK и установленный CLI validate/run/resume/replay проверены. Demo: явный разворот
и restore посреди него, затем acquisition→transform→full receipt; 1/2 requests,
невыполнимая заявка без raw. Full regression не перезапускался.
[Контракт](validation/SPACECRAFT_PROFILES.md), [приемка](acceptance/B04_BUS.json),
[пример](../examples/spacecraft_bus/demo.py). Commit/push не выполнялись.


### B04-B1 — verified в пределах sampled area coverage, 2026-10-31

Rectangular body camera rays → first WGS84 intersections, edge sampling;
local ellipsoidal LAEA polygon/AOI clipping и union без двойного зачета площади.
AOI holes/dateline/polar/invalid geometry guards, явные spatial/sample/vertex
budgets. Source footprints остаются lon/lat; clipped union — GeoJSON в метрах
объявленной local projection. Failed/cancelled acquisition не создает coverage.

Area requirement на customer request проверяется по конкретному продукту после
полного station ACK, а не по глобальной площади других заявок. Transform DAG
наследует union inputs; fan-in/out не удваивает площадь или unique source samples.
Историческое acquired и фактическое delivered coverage разделены. Есть bounded
pairwise revisit diagnostics и actor/release-scoped AOI/sensor observation.
Geometry native versions зафиксированы в effective scenario; budget rollback
и mid-acquisition restore сохраняют bus aliases и накопленные sample geometry.

B04.03 закрыт частично. Revisit/stereo requirements, continuous swath, limb clipping,
optical/cloud/SAR quality не реализованы. B04 remains in_progress; external B02
qualification, parallel scheduling, actuator/thrust и calibrated thermal открыты.
[Контракт](validation/AREA_COVERAGE_CONTRACT.md),
[приемка](acceptance/B04_AREA_COVERAGE.json),
[пример](../examples/area_coverage/demo.py). Commit/push не выполнялись.

Проверки B04-B1: 32 новых cases, reference 449 passed, fast 110 passed,
879 collected; наборы пересекаются. Full regression не перезапускался.
CI устанавливает coverage extra; отсутствие geometry dependencies не дает
fallback к point coverage. Installed-wheel и CLI evidence — в артефакте приемки.


### B04-B2 — verified в пределах payload quality/resource contracts, 2026-10-31

Discriminated optical/SAR instrument settings и per-request quality criteria.
Пространственные clear/cloud/unknown mask cells связаны с AOI hash, источником,
validity/availability; ambiguity и скрытый forecast отвергаются. Quality regions
клипуются по реальным sampled footprints и маскам, проверяются по vertices и
representative point. Неизвестное/просроченное/будущее не считается ясным; ночная
пассивная оптика не дает usable area. Optical GSD/Sun и nominal SAR resolution
даны как формулы инженерной модели, без научного изображения.

SAR incidence/side, pulse mean DC и peak admission, rolling session on-time
ограничивают физическое исполнение. DC проходит общий bus/battery ровно один раз;
thermal поправка отделяет RF radiation от amplifier heat. Cancel/safe/recovery
не стирают on-time history или потраченный ресурс. Quality evidence наследуется
через fan-in/out с дедупликацией; usable union оценивается заново для каждого
request только по его доставленному продукту. Бюджеты и mid-acquisition replay
покрывают evidence и RF state, actor view не раскрывает чужие/future masks.

B04.04 закрыт частично: deterministic engineering surrogate, не calibrated
radiometry/SAR focusing или stochastic field. B04.03 revisit/stereo requirements
остаются следующими, B04 parent in_progress. Подробный thermal/parallel scheduler/
actuator/thrust и независимая квалификация B02 открыты.
[Контракт](validation/PAYLOAD_QUALITY_CONTRACT.md),
[приемка](acceptance/B04_PAYLOAD_QUALITY.json),
[пример](../examples/payload_quality/demo.py). Commit/push не выполнялись.

Проверки B04-B2: 39 новых cases; reference 488 passed, fast 110 passed,
918 collected; counts overlap. Wheel 187/187 Python modules + 3 frozen fixtures
совпадают с исходниками. Installed Python -I SDK и CLI validate/run/resume/replay
для optical и SAR проверены; оба demo дают 1 fulfilled / 2 total, в том числе
restore посреди acquisition. SAR demo: amplifier DC 40 J = radiated RF 20 J +
amplifier heat 20 J. Full regression не перезапускался.


### B04-B3 — verified в пределах sampled revisit/stereo contract, 2026-10-31

Каждый площадной raw сохраняет pose/time/epoch/trajectory/AOI observation records;
quality evidence привязано к конкретной экспозиции. Revisit считает разные
acquisitions в допустимых time gaps на общей пригодной площади. Stereo проверяет
Earth-fixed photogrammetric baseline и convergence по spatial samples; совместные
критерии требуют общей области, не раздельного успешного union по разным кадрам.
Копии/ветки DAG, completion times и глобальная история не создают наблюдения.

Одноаппаратный fan-in сохранен. Для группировки добавлен named ground bundle из
primary + managed support products: полные ACK одной станции до deadline,
собственные bits/ancestry не удваиваются. Потеря обязательного продукта и отмена
еще не произведенного raw причинно проваливают зависимость. Бюджеты и кэш оценки
fail closed; pose/evidence/partial radio включены в checkpoint/replay.

37 новых cases; reference 525 passed, fast 110 passed, 955 collected; counts overlap.
Полный regression не перезапускался. B04.03 закрыт только в пределах sampled
engineering модели: не stereo reconstruction/DSM, не nominal-position uncertainty
qualification, не coherent SAR interferometry или continuous footprint guarantee.
Parent B04 remains in_progress; B04-C parallel scheduling и оставшиеся thermal/
actuator/thrust/independent-calibration gates открыты.
[Контракт](validation/OBSERVATION_REQUIREMENTS.md),
[приемка](acceptance/B04_OBSERVATIONS.json),
[пример](../examples/observations/demo.py). Commit/push не выполнялись.

B04-B3 installed acceptance: wheel 188/188 Python modules + 3 frozen fixtures
совпадают с source bytes; Python -I SDK и CLI validate/run/resume/replay для
revisit и stereo-fleet проверены. Оба demo: 2 fulfilled / 2 total, joint fraction 1;
mid-acquisition и partial-radio restore. Fleet schedule ждет reopening станции,
не считает command completion подтверждением передачи. Full regression не запускался.


### B04-C — verified в пределах resource-lanes/1 contract, 2026-10-31

Явная per-actor capacity/priority, несколько control windows без конфликта
CPU/payload/radio/attitude claims. Batch admission также исключает двух владельцев
одного managed request/job между actors. Приоритет независим от submit order;
бортовое время, peak/mean power, SOC, PV/thermal и memory reservations общие.

CPU progress масштабируется по доступной I/O bandwidth, radio rate ограничивается
оставшейся bandwidth, sensor stream требует полной емкости. Native station/
booking/outage/terminal selector переиспользован без отдельного transport двойника.
Native in-place processing в opt-in parallel явно unsupported; serial modes прежние.
Switching, pause/resume/cancel/fault/preemption сохраняют потраченную работу,
без выдуманного непрерывного raw. Invalid imaging не снимает нагрузку соседнего CPU.

Readonly per-actor grants и operator intervals содержат work phase/rate, actual
radio bits, memory I/O и reservations, полезную энергию и причины блокировки.
Unexpected fault откатывает все lanes и scheduler; restore сохраняет shared bus.
34 новых cases; reference 559 passed, fast 110 passed, 989 collected; counts overlap.
Full regression не перезапускался. Пример без preloaded data выполняет три заявки
в wide/constrained вариантах, включая restore посреди трех незавершенных lanes.
[Контракт](validation/RESOURCE_LANES_CONTRACT.md),
[приемка](acceptance/B04_RESOURCE_LANES.json),
[пример](../examples/resource_lanes/demo.py). Commit/push не выполнялись.

Полный B04 остается in_progress: detailed thermal, actuator/thrust coupling,
continuous sensor geometry и независимая calibration открыты. B04-D должен
возвращать маневры именно в coupled profile, не наследовать несвязанный burn
из другого backend без ресурсов, attitude и health.

B04-C installed acceptance: wheel 189/189 Python modules и 3 frozen fixtures
совпадают с source bytes. Python -I SDK и CLI validate/run/resume/replay проверены
для wide/constrained; 3 fulfilled / 3 total, 7 Mbit modeled memory I/O в обоих
сценариях, mission time 8/10 s при разных control-window plans. Это не scheduler
benchmark. Prefix restore включает три незавершенных канала; full regression
не перезапускался.

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


B04-D1 validation: 39 новых cases, reference 598 passed (13 warnings), fast 110 passed (2 warnings), 1028 collected. Counts overlap; full regression не перезапускался. Wheel: 190/190 Python modules и 3 frozen fixtures. Chemical/electric installed CLI validate/run/resume/replay verified; обе миссии 2/2 delivered requests.

B04-D1 isolated extended SDK verification passed: chemical/electric, mid-gimbal и mid-firing restore, balances и console replay.

B04-D2a validation: 21 новых cases; reference 619 passed (13 warnings), fast 110 passed (2 warnings), 1049 collected. Counts overlap; full regression не запускался. Wheel 191/191 Python modules byte-matched, 3 frozen fixtures; isolated standalone demo passed.

Расширенный isolated installed SDK/CLI smoke также passed, включая прежние chemical/electric missions и mid-gimbal/mid-firing replay. Это проверка совместимости, не интеграция rigid kernel с MissionRuntime.
