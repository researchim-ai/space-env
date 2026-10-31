# B02-Q2: геометрия съёмки, затмения и привязка прогнозов

Дата: 2026-10-31. Профиль reference-dynamic-downlink/1; eclipse forecast и
geometry binding также доступны fixed reference. Parent B02 по-прежнему
needs_external_evidence. Это геометрическая основа B03, не исполнение съёмки.

## Конфигурация и смысл окна

В dynamic definition добавлены optional acquisition_targets и acquisition_sensors.
Target: уникальный id, WGS84 lat/lon в градусах, alt_m (0–20000), availability
[from, until) в elapsed SI seconds. Sensor задается по actor ID: max_off_nadir_deg
(0–85), min_elevation_deg (0–90), min_sun_elevation_deg (-90–90), optional
max_range_m и minimum_dwell_sec. Неизвестные actors/поля, дубли и availability
вне horizon отвергаются; конфигурация и defaults входят в effective scenario.

Модель: point target, геодезическая нормаль WGS84, идеальный конус наведения
относительно **геоцентрического** nadir. Elevation и range считаются по текущему
GCRS state через явный GCRS/ITRS rotation. Solar elevation — направление от цели
к ERFA Sun в том же terrestrial frame. -90 для solar minimum отключает ограничение.
Нет области покрытия, swath/terrain/облачности, refraction или фактической attitude.
Смысл похож на геометрические задачи [Orekit ground observation detectors](https://test.orekit.org/site-orekit-11.0.2/apidocs/org/orekit/propagation/events/FootprintOverlapDetector.html),
но наш point/cone surrogate не реализует их полноценную FOV/area/attitude модель.

Runtime.forecast_acquisition(actor, target_id, start_sec=None, end_sec=None,
settings=None) возвращает сырые windows и opportunities с минимальным dwell.
start по умолчанию — текущее время, end — mission horizon. Явный интервал должен
быть положительным и помещаться в horizon; данные не экстраполируются.

Около найденных корней usable_start/end сдвигаются внутрь на root_tolerance_sec.
Известные clipped границы запроса/availability не сдвигаются.
Корень непосредственно на границе запроса не помечается как clipped interior;
для него численный запас тоже сохраняется. Dwell проверяется
по usable_duration_sec; это численный запас, **не физическая гарантия** контакта.
Opportunity IDs зависят от trajectory/target/sensor/window и usable limits;
это идентификаторы версии прогноза, а не постоянные request/product IDs.

## Readonly, права и revision binding

Capability forecast.acquisition.geometry есть только при configured targets и
sensors. acquisition_actors явно перечисляет оснащенных актеров. Для вызова нужны
AccessPolicy(forecast=True); доступ одного AgentView не открывает truth/forecast.
Вызов не продвигает время, не пишет журнал и не меняет ревизию/продукты/ресурсы.

Каждый acquisition/eclipse и существующий station forecast содержит binding:
scenario_sha256, trajectory_sha256, actor_id, revision, geometry_revision.
Команда может передать этот объект в optional forecast. Admission проверяет его
**атомарно** под runtime lock, до любых физических изменений batch. После маневра,
cancel, другого изменения состояния или для чужого сценария/актера/траектории
привязка дает stale_forecast даже при корректном новом expected_revision.
Повтор ранее принятого idempotency request возвращает старый ответ, без операции.

Это optional stale-plan guard для команд, построенных по прогнозу. Непривязанные
ручные команды остаются допустимыми. Binding не является authorization token,
бронированием, доказательством возможности операции или сертификатом качества
поиска. Он не доказывает корректность выбранных caller search settings.
Исполнитель B03 обязан заново проверять геометрию и ресурсы во время acquisition.
Между разными library/data/dependency manifests прогнозы могут иметь другие hashes;
планы с binding строятся и исполняются в согласованном runtime окружении.

## Поиск событий и численные границы

Общий positive_windows ищет положительные части непрерывного margin между
sampling nodes и выполняет Brent refinement. В сетку включаются границы smooth
trajectory arcs. Touch нулевой длительности не образует операцию; соседние окна
с общим точным touch объединяются. Дискретные availability отрезаются до поиска,
а не моделируются ложным непрерывным margin.

SearchSettings: scan_sec >=1e-6 и <=120 (default 5), root tolerance 1e-6…1 sec,
не больше scan interval, max_evaluations default 200000 / max 1000000.
Не более 100000 базовых grid intervals; дополнительный budget охватывает
refinement. Для eclipse общий cache ограничивает реальные geometry evaluations
обоих searches. Невалидный/nonfinite margin или budget exhaustion дают ошибку,
частичный прогноз не возвращается. Empty target availability дает пустые списки
и search_details/search_coverage_sec=None.

**Гарантии полноты нет.** Короткий положительный интервал или короткий отрицательный
разрыв между узлами может быть пропущен. max_grid_gap_sec и completeness записаны
в результате; отдельный тест показывает coarse miss и fine detection. Нельзя
приравнивать root tolerance к error budget физики, интерполяции или покрытия.

## Затмения и согласованность с текущей физикой

Runtime.forecast_eclipses использует существующий core.shadow_state:
model core.spherical-conical-linear-penumbra/1. Возвращаются umbra и
partial_or_full windows (вторые **включают** umbra), clipped flags и coverage.
Sun — ERFA epv00, кадр GCRS. Проверяются finite state/Sun и допустимая Earth-bound
геометрия. Trusted низкоуровневый SDK допускает явный callback Sun для специальных
validation cases; runtime не подменяет ERFA произвольным callback из сценария.

Модель Земли сферическая, границы конусов приближенные, flux в полутени линейный.
Это **не новая apparent-disc illumination модель** и не утверждение эквивалентности
[Orekit EclipseDetector](https://www.orekit.org/site-orekit-12.2.1/jacoco/org.orekit.propagation.events/EclipseDetector.html),
который различает apparent radii и поддерживает иные occulting модели.
Нет ellipsoidal/terrain/atmospheric/Moon occultation или antumbra qualification.

Поэтому readonly forecast не меняет физическую модель SRP/питания. Он также не
подменяет существующее absolute-grid midpoint интегрирование solar resources
точным событийным интегратором: отдельный energy/outcome error budget остается
нужен. Высокая точность времени корня не дает такой гарантии автоматически.

## Сравнения и приемка

28 новых tests: roots/touches/clipping, resolution and budgets, sensor constraints,
availability/dwell guard, invalid config, readonly/permissions, foreign/stale bindings,
atomic cached retry, maneuver/cancel revision invalidation, checkpoint, fixed profile,
poles/antimeridian и nesting eclipse contacts.

geometry_event_report использует analytic circular orbit, отдельно записанные
WGS84/normal/cosine условия и bisection без production search/Brent. AOS/LOS и
acquisition boundaries сравниваются с заранее заданным budget 0.02 sec.
Измерено примерно 1.76e-5 и 2.22e-4 sec соответственно.
Общие EOP/ERFA matrices, константы и derived station coordinates перечислены.
Это независимость уравнений поиска/номинальной орбиты, не frames/flight truth.

Umbra/penumbra contacts дополнительно сравниваются с угловыми радиусами сферических
видимых дисков при постоянном Sun, используя отдельный bisection. Budget 1 sec
покрывает **различие моделей**; измерено около 0.0218/0.0227 sec. Это не независимая
калибровка реального солнечного потока. Qualification CLI включает geometry_events,
а общий status остается needs_external_evidence.

Reference 357 passed, fast 110 passed, 787 collected; полный regression не
перезапускался. [Машинная приемка](../acceptance/B02_Q2_GEOMETRY.json) содержит
измерения, installed-wheel evidence и ограничения. Предыдущие artifacts исторические.

## Воспроизводимый SDK/CLI workflow

```sh
python examples/unified_runtime/geometry_demo.py /tmp/geometry-demo
space-env mission-validate /tmp/geometry-demo/scenario.json
space-env mission-run /tmp/geometry-demo/scenario.json /tmp/geometry-demo/plan.json /tmp/geometry-demo/cli-prefix.json
space-env mission-resume /tmp/geometry-demo/cli-prefix.json /tmp/geometry-demo/resume.json /tmp/geometry-demo/cli-final.json
space-env mission-replay /tmp/geometry-demo/cli-final.json
space-env reference-qualify-orbits /tmp/physics-qualification.json
```

Один environment должен использоваться для генератора и CLI из-за versioned binding.
Пример показывает forecast → bound command → checkpoint → maneuver → stale
rejection → fresh forecast → resume/replay. Он **не создает снимки или продукты**.
Следующий B03 связывает opportunity с исполняемой заявкой, raw data, processing и delivery.
