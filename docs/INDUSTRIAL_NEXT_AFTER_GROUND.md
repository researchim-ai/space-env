# Следующие крупные уровни после B11-Q4

Актуализация 2026-11-04: D2a independent fixed-site delivery и conditional
capacity audit реализованы. [Контракт](validation/INDEPENDENT_GEOMETRY_V1.md).
D2b теперь должен связать эту независимую геометрию с EO/pointing/control;
C2/Q6/Q7/Q8 и промышленный пилот остаются открытыми.


Актуализация 2026-11-03: реализованы также C1b material enthalpy,
D1 native ensemble bridge и D1b process-isolated continuation. Контракты:
[материалы](validation/MATERIAL_ENTHALPY_V1.md),
[ансамбли](validation/NATIVE_ENSEMBLE_QUALIFICATION_V1.md),
[recovery](validation/PROCESS_RESTART_V1.md). C2, независимый D2, Q6–Q8
и промышленный пилот остаются открытыми; предыдущие результаты не переносятся
на изменённую сборку автоматически.


B11-Q4 закрывает конкретный operational-control пакет. Его фактический итог — в
`acceptance/B11_GROUND.json`; текущий статус нельзя заменять обещанием готовности
произвольного промышленного симулятора. Ниже — следующие пакеты. В Q5 реализованы Q5-A, Q5-B и foundation Q5-C1;
остальные пункты не закрыты.

## Q5 — независимая валидация и расширение mission envelopes

**Q5-A реализован:** [аналитические векторы, CLI и контракты](validation/ANALYTIC_ENVELOPES_V1.md),
10 frozen cases, 103/119 numerical checks, independent equations/shared parameters,
явные gaps и отрицательные проверки. Это частичное закрытие пунктов 1–2 ниже;
внешняя калибровка и coupled-envelopes остаются открытыми.

**Q5-B реализован:** [off-axis/actuation/contact contract](validation/POINTING_ACTUATION_CONTACTS_V1.md),
16 frozen cases / 612 comparisons, saturation refinement, asymmetric external
moment, centered/offset rotating force и fixed-station contacts с positive/negative
окнами. Это самостоятельная numerical qualification точечных synthetic cases,
не closed-loop controller или flight truth. Приёмка: `acceptance/B11_POINTING.json`.

**Q5-C1 реализован:** [evidence/calibration contract](validation/HARDWARE_EVIDENCE_CALIBRATION_V1.md):
6 published reference-point profiles, explicit native adapters, 3 identifiable
known-noise fit families, independent holdout and assumed sensitivity ensembles.
Только open specifications и явно synthetic workflow; real hardware calibration
не заявлена. Приёмка: `acceptance/B11_CALIBRATION.json`.

Следующие подпакеты: Q5-C2 открытые measurement datasets, protocol/error model,
identifiability/coverage review; Q5-D nonconstructive multipane/multistation mission
qualification с actual closed-loop pointing/control plans.

1. Зафиксировать поддержанные режимы отдельно для orbit/attitude/power/thermal/
   payload/RF/DTN и их coupled-комбинаций. Указать проверенные диапазоны, допущения,
   параметры и источники; unit tests и conservation не равны внешней калибровке.
2. Сверять численные trajectories/frames/contacts/energy/pointing с независимыми
   эталонными данными и заранее заданными допусками. Внести случаи на границах
   режимов, неопределённость параметров и regression vectors с provenance.
3. Перейти от конструктивных AOI и исходного неподвижного boresight к реальным
   pointing/control plans. Отдельно квалифицировать slew/wheel/propulsion, eclipse,
   multi-plane geometry, несколько stations и changing shared resources.
4. Расширять время/группировки ступенчато на frozen workloads. Каждый новый режим
   имеет native quality, mission objective, latency/RSS/checkpoint/recovery gates.
   Случаи с невыполненной целью сохраняются; thresholds не меняются после outcome.

## Q6 — обучение и массовые rollout

1. Добавить быстрые уровни fidelity с измеренным расхождением относительно native
   reference, domain of validity и проверенными counterexamples. Не выдавать
   упрощённый режим за тот же самый physical reference.
2. Проверить throughput/latency/memory для batches и длительной работы, включая
   failure, cancellation и restart. Убрать повторную работу только с сохранением
   causal observation, ownership, RNG, rollback и replay contracts.
3. Ввести typed immutable science evidence/delta snapshots с доказанным владением.
   Сейчас native mutable graph полностью копируется; долгие миссии ещё не
   квалифицированы и рост истории не исчезает от короткого успешного теста.
4. Разнообразить task families и teacher quality: feasible plans, service gaps,
   safety/constraint violations, independent holdout/OOD mission families и
   anti-leakage. Baselines не маркировать expert; privileged oracle-bound не
   смешивать с causal teacher data без явной аннотации доступной информации.

## Q7 — долгие jobs и эксплуатация

1. Принудительные CPU/RSS/disk/time quotas и cancellation. Текущие max-RSS gates —
   измеренная приёмка, а не OS enforcement; stage files — диагностика, не job queue.
2. Durable job state, ownership/epochs, crash-safe retries, partial artifacts,
   event/artifact retention, bounded history и проверяемое восстановление.
3. Диагностика blocking stages и explicit terminal causes без изменения native
   timing/semantics. Не превращать partial telemetry в replay evidence.
4. Аутентифицированный deployment, tenant isolation, clean installed CI, release
   provenance и pinned-hardware long-run qualification. Локальный system-site venv
   и loopback-пульт не являются такой приёмкой.

## Q8 — прикладной независимый benchmark

1. Зафиксировать несколько внешних постановок и критерии успеха до реализации
   controllers; измерять model validity и mission service отдельно.
2. Проверить generalization за пределами исходных scenario templates и nominal
   weather/formation. Оценивать baseline gaps и качество генерируемых RL/SFT данных.
3. Опубликовать воспроизводимые конфигурации, ограничения, raw failures и receipts.
   Сопоставление с другими инструментами допустимо по одинаковой задаче/точности/
   ресурсам; количество модулей или тестов не доказывает превосходства.
