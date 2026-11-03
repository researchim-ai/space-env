# B11-Q5-C1: evidence, conditional fits и assumed sensitivity

Дата: 2026-11-03. Версия исходников 0.134.0; приёмка привязана также к wheel SHA.
У проекта пока только открытые данные. Этот пакет создаёт проверяемую основу
для параметризации/калибровки, но **не завершает Q5-C аппаратной калибровкой**.
Физическое ядро и defaults существующих миссий не меняются.

## Что добавлено

- Шесть published reference-point profiles: четыре CubeSpace wheel и два
  AZUR SPACE solar technology points. Source bytes SHA, locator, условия,
  исходные единицы, SI и неизвестная неопределённость хранятся раздельно.
- Explicit adapters в RigidAttitudeModel и VehicleResources. Полный host vehicle
  не становится published/calibrated от установки одного компонента.
- Known-noise scalar calibration трёх идентифицируемых величин с отдельными
  независимыми train/holdout experiment summaries, checksum и заранее заданными gates.
- Детерминированные uniform/truncated-normal sensitivity ensembles с retained
  failures. Это предположения для исследования чувствительности, не измеренные
  manufacturing tolerances или автоматически найденные distributions.
- CLI, Python API, packaged offline data, native smoke и installed CI workflow.

## CLI и API

Установить `.[reference]`. Runtime не скачивает datasheet и не обращается в сеть.

```bash
space-env hardware-catalog /tmp/hardware.json
space-env calibration-check /tmp/calibration-workflow.json
space-env calibration-fit /path/to/measurement-spec.json /tmp/fit.json
space-env parameter-ensemble /path/to/scenario.json /path/to/sensitivity-spec.json /tmp/ensemble.json
```

`calibration-fit` пишет полный отчёт и возвращает1 при failed quality gates.
Ошибки входного контракта возвращают2. Input/output aliases, включая hardlinks,
отвергаются; atomic write выполняется после расчёта. Чтение ограничено2MB.
Для ensemble любой structurally rejected member даёт CLI exit1, но все попытки
остаются в JSON. Нативные quality/mission gates этим не объявляются пройденными.

Основные Python entry points: `hardware_catalog`, `hardware_profile`,
`wheel_model`, `solar_resources`, `verify_source_bytes`, `measurement_payload`,
`calibrate`, `sample_parameters`, `compile_ensemble` из `space_env.calibration`.

## Первичные источники и смысл параметров

[CubeSpace reaction wheels](https://www.cubespace.co.za/uploads/ee37948b-0f02-4651-96e3-c6274f1d8797.pdf),
страница2, колонки CW0017/CW0057/CW0162/CW0500; оригинальный PDF SHA-256:
`f787e2fb8899c0dcfbf50eb3abaaf005fcc6485610b6b08fe48248c136b177ca`.
Сохранены rated momentum, соответствующая speed/torque, масса и две power points.
Это не полная torque-speed-temperature curve. Для CW0500 используется torque
именно этой ревизии/точки; цифры из других таблиц/версий не смешиваются.

[AZUR SPACE DB 00010890-01](https://www.azurspace.com/media/uploads/file_links/file/bdb_00010890-01-00_generalinformationazur.pdf),
08-Apr-2025, страница7, generic BOL table и footnote3; PDF SHA-256:
`c931ea2ac58ba357327b666e9716e25d70669d12dc8e827f9d076b3785f9f22f`.
Выбраны bare/CIC 3G30-Advanced reference points с одной irradiance normalization.
Температурный диапазон испытания не является efficiency error bar. Между
различными irradiance normalizations не строится выдуманная temperature/IV curve.

Обе таблицы проверены по отрисованным страницам PDF. Полные документы/изображения
не распространяются в репозитории: только небольшой curated набор фактов и
ссылки. Copyright производителей сохраняется. Packaged catalog byte hash и
canonical body hash проверяются offline. `verify_source_bytes` может сравнить
загруженный отдельно PDF с recorded SHA; совпадение bytes не аутентифицирует автора.
Отчёт workflow не заявляет повторную загрузку/проверку PDF в каждом запуске.

## Mixed-evidence adapters

`wheel_model` требует три explicit profile IDs для идеальных orthogonal wheels
и caller-supplied host inertia, efficiency, idle power, torque loss и rationale.
Rated momentum выбирается как **предел сценария**, torque в rated point — как
постоянный cap модели. Hardware maximum/полная conservative envelope из этой
точки не выводятся. Spin inertia выводится h/ω при постоянной инерции.
Power в steady-at-speed не подставляется как idle, peak — как полный bus budget.
Power/friction/jitter вне этих точек остаются допущениями.

`solar_resources` требует base VehicleResources, reference area, packing factor
и explicit operating assumption. Эффективность постоянна и масштабируется
packing factor. Геометрия, MPPT/wiring, degradation и температурная зависимость
не становятся известными автоматически. Native solar constant и source reference
irradiance записаны отдельно; линейное irradiance scaling является допущением.

Оба адаптера возвращают model payload плюс provenance, mapping assumptions,
`qualification_status=mixed-published-derived-assumed`, `industrial_readiness=false`.
Исходный объект не меняется. Native validators проверяют форму/положительность/
physical inertia, но не объявляют реальные hardware operating envelopes доказанными.

## Контракт измерений и физическая идентифицируемость

Поддерживается **одна независимая aggregate observation на эксперимент**:
`id`, `experiment_id`, `split`, `x`, `y`, `sigma_y`. Нельзя разрезать один
временной ряд на соседние train/holdout точки. IDs уникальны; producer обязан
обеспечить реальную независимость и корректное измерение величин.

Три zero-intercept модели y=θx:

- `pv_efficiency`: x — effective incident power W, y — measured converted power W,
  θ — effective efficiency. Experiment `constant-effective-efficiency`, scalar frame.
- `inverse_inertia`: x — независимо известный net angular impulse N·m·s,
  y — BODY principal-axis Δω rad/s, θ=1/I. Требуются
  `isolated-principal-axis-impulse`, BODY и конкретная x/y/z axis. Нельзя одновременно
  определять неизвестный actuator gain и inertia по одной такой регрессии.
- `inverse_heat_capacity`: x — независимо известная net absorbed energy J,
  y — ΔT K, θ=1/C. Experiment `isolated-known-net-heat`, scalar frame. Неизвестные
  thermal losses нельзя спрятать в heat capacity и выдать за идентифицированную C.

`input_error_assumption=negligible-independent-x` и
`noise_assumption=independent-known-gaussian-sigma-y` обязательны. Коррелированные
ошибки, неизвестная sigma, errors-in-variables и nonlinear/dynamic fits этим
методом не поддерживаются. Conditions должны явно описывать reference state,
physical boundary/датчики и фиксированные nuisance assumptions.

`measurement_payload(spec)` — canonical **prepared measurement export** schema
`space-env.measurement-rows/1`, включая predeclared split, frame, axis и conditions.
Producer должен записать его digest в source.content_sha256 и указать точный scope
`space-env.measurement-rows/1 canonical JSON`. Это hash подготовленного export,
не автоматически hash upstream binary/CSV/приборной записи. Source URI/locator
должны идентифицировать export и его происхождение. Неявного парсинга, conversion,
resplit или пересчёта подписи в `calibrate` нет; изменение строки требует нового
явного artifact. Gates в raw measurement hash не входят, но входят в spec hash.

Datasheet не принимается вместо measurement series. Synthetic/laboratory/flight
origin остаётся явным. Наличие URI, digest и declared experiment IDs не доказывает
истинность измерений, независимость ошибок или право выдавать calibration certificate.

## Оценивание и holdout

Known-noise weighted least squares, fixed zero intercept:

θ̂ = Σ(xᵢ yᵢ/σᵢ²) / Σ(xᵢ²/σᵢ²),
SE(θ̂) = 1/√Σ(xᵢ²/σᵢ²),
conditional interval = θ̂ ± Φ⁻¹(.975) SE.

Расчёт использует scaling x для численной устойчивости; all-zero training x
отвергается как unidentifiable. Holdout не меняет θ̂, SE, training range или
параметры модели. Нет outlier deletion, подгонки по holdout, автоматического
расширения bounds или искусственного covariance inflation для прохождения gates.
Residuals всех экспериментов сохраняются.

Gates заранее задают positive parameter bounds, train/holdout standardized RMSE,
absolute holdout error и минимальные counts. Требуются θ̂ и весь conditional
interval в bounds. Для efficiency upper bound≤1. Holdout обязан лежать в training
input range; хороший residual вне диапазона не считается qualified extrapolation.

Для inverse inertia/capacity выводится reciprocal estimate и **нелинейно
преобразованный** interval [1/upper,1/lower]. Он не называется Gaussian interval
в reciprocal space и не подставляется автоматически в произвольную миссию.

Это условная статистика при заданных предположениях, не полная physical uncertainty.
Fit status — `synthetic_workflow_verified`, `declared_measurements_holdout_checked`
либо `rejected_by_gates`. Ни один из них не выдаёт native `parameter_status=calibrated`.
`independent_source_authentication=false`, `industrial_readiness=false` сохраняются.

## Sensitivity ensembles

`EnsembleSpec`:1–64 members,1–16 параметров, positive ratio ranges. Bindings:
wheel torque/momentum, isotropic body inertia, battery capacity, solar efficiency,
scalar transmitter thermal capacity. Активная node thermal network не изменяется
scalar binding: такой member отвергается вместо silent no-op.

Counter RNG `space-env.counter-uniform/1` основан на SHA-256(id,seed,member,latent).
Добавление параметра, перестановка списка, увеличение sample count не меняют уже
существующие draws. Python/NumPy global RNG не трогаются. Uniform/truncated normal
используют open-interval quantiles; unsupported finite-precision tail сохраняется
как ошибка, а не clipping/resampling физического значения.

Разные latent IDs независимы в declared construction. Общий `latent_group`
означает **common-rank sensitivity assumption**, не измеренную covariance.
Нельзя записать `manufacturer_tolerance` вместо `assumed_sensitivity` без данных.
Статистический confidence interval fit также не превращается автоматически в
эпистемическую/алеаторную distribution для domain randomization.

`compile_ensemble` копирует base scenario, сохраняет environment seed для paired
common-random-number опытов. Draws сначала генерируются полностью, затем применяются.
Каждая попытка остаётся: structural candidate либо rejection с reason и draws.
Проверяются native schema/target contracts, supply envelope и initial momentum.
Validation здесь **не выполняет native admission/replay и не доказывает feasible
mission service**. Неподдержанные backend/actor sets и budget превышения явны.

У изменённого bus снимается унаследованная published/calibrated метка, добавляется
assumed provenance. Derived rigid source помечается как sensitivity variant.
Base object, environment seed и agent metadata не меняются. Manifest с latent draws
является diagnostic artifact, не готовым actor observation/teacher prompt.
Необходимы отдельные public-view/holdout проверки при сборке RL/SFT corpus.

Compilation ограничена1MB base scenario и16MB суммарного JSON output. Эти guards
не являются OS quotas для выполнения native simulation; Q7 остаётся отдельным этапом.

## Приёмка

`space-env calibration-check` проверяет catalog adapters, три synthetic fits,
отрицательный holdout, repeatability, known-noise information и применение
оценённых I/C в native rigid/thermal kernels. Synthetic workflow fixture имеет
pinned bytes; штатная проверка его не генерирует.

`examples/calibration/demo.py` собирает mixed-evidence host, компилирует три
sensitivity members и отдельно запускает их в MissionRuntime. Wheel command,
paid energy, cold checkpoint continuation и load/replay проверяются. Это короткий
integration smoke, не customer mission/teacher benchmark.

Полная текущая приёмка — `docs/acceptance/B11_CALIBRATION.json`. Исторические Q5-A,
Q5-B и Q4 receipts остаются привязанными к своим wheel SHA и не переписываются.

## Что остаётся в Q5-C2 и Q5-D

1. Подобрать открытые **измерительные** datasets с instrument/protocol/units,
   upstream bytes, лицензией и кодом подготовки export. Номинальные datasheet
   строки и температурные условия не заменяют repeated measurements/error bars.
2. Оценить input/output errors, correlations, offsets, nuisance parameters и
   identifiability; добавить подходящий estimator только после выбора протокола.
3. Выполнить grouped/specimen/operating-condition holdout и uncertainty coverage,
   сохранить failures и потребовать независимый review перед physical calibration claim.
4. Q5-D: nonconstructive multi-plane/multi-station closed-loop mission workloads,
   реальные slew/tracking plans, shared resources, native replay и resource gates.
5. Q6: обучение на таких вариантах только после native quality/teacher/public-view
   gates; не выдавать structural candidates за готовый industrial dataset.

## Результаты текущей сборки

Приёмка сохранена в [B11_CALIBRATION.json](../acceptance/B11_CALIBRATION.json)
с хешами девяти связанных артефактов: workflow, источники, wheel, предыдущие
численные проверки, native receipt, ensemble, parameter witness и ground smoke.
73 targeted / 1470 reference / 110 fast прошли (наборы пересекаются), 1900 collected.
14 workflow checks и 3/3 коротких native вариантов прошли; отдельная проверка
сохранённых native состояний подтвердила влияние I на ω и применение battery/PV
параметров. Большая миссия, dependency-clean установка и remote CI не запускались.
Следующий пакет: [открытые измерения и связанная миссия](../OPEN_DATA_CALIBRATION_NEXT.md).
