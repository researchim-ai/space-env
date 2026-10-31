# B04-B1: площадная съемка и подтвержденное покрытие

2026-10-31. Backend `reference-spacecraft`, профиль `reference-spacecraft-bus/1`.
Это opt-in геометрический контракт поверх B03/B04-A. Точечные сценарии сохраняют
прежнюю семантику. Родительский B04 остается in_progress.

## Задание сцены

Установить `pip install '.[reference,coverage]'`. Геометрический extra содержит
Shapely и pyproj; базовый импорт и точечная миссия не требуют этих библиотек.
При запросе area-модели без зависимостей нет fallback к центральной точке:
исполнение сообщает `area coverage requires space-env[coverage]`.

`definition.lifecycle.areas_of_interest` содержит до 128 AOI. Каждая задает:

- уникальный `id`, `anchor_lon_deg`, `anchor_lat_deg`;
- явно замкнутый `shell` и опциональные `holes`;
- координаты **[longitude, latitude]**, градусы WGS84;
- `max_segment_m` для геодезической детализации ребер, по умолчанию 1000 м;
- `source_status` и `source`; значения по умолчанию объявлены synthetic.

AOI — ограниченный Polygon, не MultiPolygon. Отверстия, невыпуклость, даталайн
и полярные области поддерживаются в локальной проекции. Области не исправляются
автоматически: самопересечения, незамкнутые кольца, внешние отверстия, пустая
площадь и последовательные дубли отвергаются. Каждый узел AOI находится в пределах
1000 км от anchor; общий vertex budget 10000 входных / 20000 детализированных
точек. Сегменты трактуются как кратчайшие геодезические WGS84, не линии постоянной
широты. Прямой ввод 179°→−179° не означает полигон размером в остальную Землю.

Для площадной заявки `CustomerRequest.coverage` задает `aoi_id` и `min_fraction`
в (0,1], по умолчанию 0.95. `target_id` остается явной точкой наведения; она не
заменяет геометрию AOI и сама по себе не выполняет площадную заявку.
Каждый eligible actor обязан иметь `PayloadSettings.footprint` с
`half_angle_x_deg`, `half_angle_y_deg` в (0,45], `edge_samples` 1..64.
Эти модели допускаются только с явным coupled spacecraft bus: идеальная
мгновенная ориентация point-only профиля здесь запрещена.

## Камера и поверхность

Модель — прямоугольный pinhole detector: body-frame лучи `[x,y,1]`,
`x=±tan(half_angle_x)`, `y=±tan(half_angle_y)` на границе детектора.
Quaternion body→GCRS и тот же GCRS→ITRS transform, что в reference, переводят
каждый луч в Earth-fixed систему. Стабильное решение квадратного уравнения
находит **первое положительное** пересечение с WGS84 ellipsoid.

Периметр детектора дискретизируется; его проекции образуют footprint. Если
любой выбранный луч не пересекает видимую поверхность, вся площадная операция
отклоняется с `footprint_horizon`. Частично видимая за лимбом полоса не заменяется
фиктивным замкнутым полигоном; полноценное limb clipping пока отсутствует.
Нет DEM/terrain, рефракции, lens distortion, rolling shutter и pushbroom timing.
Поверхность — нулевая высота WGS84, не высота цели наведения.

Перед успешным подшагом применяются текущие geometry/pointing/Sun keep-out,
settling, power/memory/health ограничения B03/B04-A. Для каждого подшага берутся
начало, середина и левая граница конца; используется фактический прогноз
rate-limited quaternion текущего подшага. Только успешно завершенная acquisition
материализует геометрическое покрытие raw. Cancel/fault/failed acquisition не
банкуют частичное покрытие без продукта.

Это **объединение выбранных кадров**, не доказательство непрерывного swath и не
гарантия dwell для каждой точки AOI. Промежутки между samples не заполняются
искусственным convex hull. Есть предел `max_acquisition_samples` (4096 по
умолчанию): превышение атомарно откатывает текущий runtime advance, не оставляя
потраченной энергии, quaternion или половины metadata. Для длинной acquisition
надо явно выбрать поддерживаемые step/budget, а не отключать защиту.

## Площади и происхождение

AOI проецируется в эллипсоидальную Lambert azimuthal equal-area WGS84 с объявленным
anchor. Геодезические ребра AOI детализируются перед проецированием; footprint
периметр детализируется лучами сенсора. Shapely выполняет пересечение с AOI и
union выбранных footprints. Площади — м², не квадратные градусы. Зависимость от
детализации границ остается численной погрешностью, а не flight-calibrated error
budget. Footprint vertices вне 2000 км от anchor отклоняются как неподдерживаемая
область проекции с явной причиной; глобальные области не обещаются.

`raw.area_coverage[aoi_id]` содержит hash определения AOI, проекцию, площадь AOI,
покрытую площадь, fraction, sample count и GeoJSON геометрию **в локальных метрах**.
Этот GeoJSON не является EPSG:4326 и не предназначен для прямой отрисовки как
долгота/широта! Исходные ray footprint rings в longitude/latitude сохраняются
отдельно в `acquisition.footprint_samples`. Sources связывают геометрию с raw ID,
actor и временем материализации.

Явный transform DAG наследует объединение покрытия своих реальных inputs,
включая fan-in/out. Повторный вход или две копии одного финального продукта
не увеличивают геометрическую площадь; source sample counts также дедуплицируются
по raw ID. Совместимы только одинаковые AOI hashes. В этой модели transform
сохраняет пространственную поддержку inputs; crop/registration/quality losses
не моделируются. Это геометрическое происхождение synthetic metadata, не
доказательство покрытия реальным научным изображением.

Fulfillment проверяется **по покрытию конкретного доставленного продукта** после
полного ACK правильной станции и проверки deadline. Глобальное покрытие от другой
заявки не выполняет текущую. При полном ACK с недостаточной площадью продукт
остается delivered, но заявка становится failed с `coverage_requirement` и не
получает fulfillment credit. Чтобы собрать несколько полос, задаются отдельные
acquisitions/requests и fan-in в общий delivery product; бесконечных повторных
raw с одним и тем же ID нет.

## Отчет, доступ и повтор

Operator lifecycle report добавляет `area_coverage`:

- `acquired` — исторический union материализованных raw, в том числе позднее
  потребленных/утраченных; это не текущая доступная память;
- `delivered` — union полностью доставленных продуктов, независимо от того,
  достигли ли они индивидуального min_fraction;
- `revisits` — пары raw с положительной площадью пересечения и интервалом между
  **временами материализации**; также actor IDs и overlap area;
- `areas` — определения и площади AOI.

Revisit diagnostics ограничены 10000 pair checks / записей с явным truncated
флагом; это не полная глобальная статистика огромной миссии. Интервал не является
гарантией revisit deadline, стереобазой, углом stereo или stereo image quality.
AOI definitions в local observation видны только eligible actor после release
его заявки; также есть его footprint settings и geometry manifest.

Все накопленные samples и metadata включены в rollback/checkpoint/replay.
Effective scenario фиксирует версии Shapely/GEOS и pyproj/PROJ; изменение native
geometry manifest не допускает молчаливого восстановления другой геометрией.
Общий runtime manifest также фиксирует Python package versions. Checkpoints
восстанавливаются повторением журнала, а не O(1). SLA и большое масштабирование
этим пакетом не квалифицированы.

## Запуск и приемка

```bash
python -m pytest -q tests/test_area_coverage.py
python -m space_env.eval.ci_profiles reference
python examples/area_coverage/demo.py /tmp/space-env-area-demo
space-env mission-validate /tmp/space-env-area-demo/scenario.json
space-env mission-run /tmp/space-env-area-demo/scenario.json /tmp/space-env-area-demo/plan.json /tmp/space-env-area-demo/run.json
space-env mission-resume /tmp/space-env-area-demo/run.json /tmp/space-env-area-demo/resume.json /tmp/space-env-area-demo/resumed.json
space-env mission-replay /tmp/space-env-area-demo/resumed.json
```

Проверки: analytic equatorial ray solution; отдельно geodesic area vs equal-area
projection (разные формулы одной PROJ dependency, не независимая flight truth);
dateline/polar/holes/invalid geometry; duplicate union и fan-in/out; partial
coverage/full ACK; actor privacy; mid-acquisition replay; sample budget rollback;
projection applicability и geometry manifest drift. Installed-wheel smoke и
CLI запускаются отдельно от source checkout. [Артефакт](../acceptance/B04_AREA_COVERAGE.json).

Технические основания: [pyproj examples](https://github.com/pyproj4/pyproj/blob/main/docs/examples.rst),
[Shapely set operations](https://shapely.readthedocs.io/en/2.0.4/set_operations.html).
Эти библиотеки обеспечивают операции над геометрией, не квалификацию миссии.

## Следующий пакет

B04-B2 добавил [optical/cloud/quality и SAR duty/energy](PAYLOAD_QUALITY_CONTRACT.md)
как отдельные synthetic surrogates. B04-B3 добавил [revisit/stereo requirements](OBSERVATION_REQUIREMENTS.md)
по общей пригодной площади. Далее parallel scheduler, limb clipping и сенсоры иного
типа при объявленной применимости. B04.06/.07/.08 (параллельность, thermal detail,
actuator/thrust coupling) и внешняя квалификация B02 остаются открыты.
