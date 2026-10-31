# B04-B2: оптическое качество и SAR applicability

2026-10-31. Opt-in расширение `reference-spacecraft-bus/1` / schema v3.
Зависит от [B04-A](SPACECRAFT_PROFILES.md) и [B04-B1](AREA_COVERAGE_CONTRACT.md).
Это проверяемые инженерные surrogates; калибровка реальных инструментов и полный
B04 остаются открытыми.

## Контракт и совместимость

`PayloadSettings.imaging_model` — явная discriminated модель `optical` или `sar`.
Обе требуют model_id, source, source_status и площадную acquisition с footprint
и coupled bus. Пустые параметры не заменяются идеальными значениями. Published
или calibrated — декларация автора сценария, не аттестация библиотеки.
Отсутствующий imaging_model сохраняет геометрическую семантику B04-B1.

`CustomerRequest.quality` задает kind и явные требования. Для optical обязательны
`min_sun_elevation_deg` и `max_gsd_m`; для SAR —
`max_ground_range_resolution_m` и `max_azimuth_resolution_m`. Параметры другого
типа запрещены; каждый eligible actor должен иметь соответствующий инструмент.
Без `coverage` quality requirement не допускается. Заявка без quality по-прежнему
требует только геометрический результат: ее выполнение не означает пригодность
научного изображения; квалифицированный продукт требует явного quality criterion.

Выбран evidence-подход вместо наследования одного quality score: raw содержит
пространственные измерения модели, transform сохраняет их с дедупликацией ID,
а каждая заявка проверяет собственные критерии после доставки общего продукта.
Это позволяет одному продукту выполнить менее строгую заявку и не выполнить
более строгую. Изменение записи с тем же ID вызывает ошибку lineage.

## Пространство, время и маски облаков

`LifecycleSpec.cloud_masks` содержит до 4096 клеток. Клетка объявляет ID, AOI ID,
hash полного определения AOI, bounds_m=[xmin,ymin,xmax,ymax] в **локальной LAEA
проекции этого AOI**, state clear/cloud/unknown, source/status и временной контракт:

- valid_from_sec <= t < valid_until_sec;
- available_from_sec <= t;
- forecast требует allow_forecast=true;
- observed не может быть доступен раньше начала своей validity.

Это входные категориальные маски, не облачный классификатор и не автоматически
загружаемый метеопродукт. Synthetic маски задают сценарные условия явно. Stale,
еще недоступные, явно unknown и отсутствующие клетки не становятся clear.
Перекрывающиеся по площади и времени клетки одного AOI отвергаются без скрытого
приоритета или оптимистичного объединения. Соседние клетки и интервалы допустимы.
Новая маска не пересчитывает качество уже снятого raw задним числом.

Ray footprint пересекается с AOI, затем с сеткой quality_sampling.cell_size_m
(1000 м по умолчанию) и доступными прямоугольными cloud masks. Граница маски
сохраняется геометрически: ясная центральная точка не делает ясным весь AOI.
Остаток площади получает unknown. Для каждого региона проверяются вершины и
representative point. Используются начало, середина и левая граница конца
integration substep; mask validity/availability дополнительно разделяют подшаги.

Это **пространственно-временная дискретизация**, не доказательство экстремумов
между точками. Бюджеты cells/sample, points/region и evidence/product явные;
превышение при acquisition атомарно откатывает текущий advance. При слишком
большом fan-in ожидаемый materialization failure сохраняет потраченный CPU и
исходные продукты, как другие ошибки materialization B03.

## Optical

Пассивная daylight-модель объявляет instantaneous_fov_rad, max_incidence_deg и
min_sun_elevation_deg (по умолчанию 5°, минимум 0°). Необычные night-vision,
thermal IR и активная оптика этим kind не моделируются. Ясная маска ночью не
дает пригодную площадь. Неудачная по качеству экспозиция может потратить ресурс
и породить raw metadata, но не дает quality fulfillment.

Для каждого spatial sample рассчитываются WGS84 нормаль, slant range, угол
падения и геометрическая высота Солнца в тех же reference frames/epochs.
Первая оценка GSD: `slant_range * instantaneous_fov_rad / cos(incidence)`.
Регион хранит максимальный sampled GSD и минимальную sampled солнечную высоту.
Зачет требует clear mask, applicability инструмента и критериев заявки.

GSD здесь — локальная геометрическая оценка, не измеренное пространственное
разрешение: нет MTF/SNR, photon budget, атмосферы, DEM/shadows, smear, lens PSF,
экспозиции или спектральной пригодности. Эти эффекты нельзя вывести из одного
положительного usable_fraction.

## SAR

Модель объявляет bandwidth, along-track antenna length, диапазон incidence,
look_side, peak RF power, pulse duty fraction, amplifier efficiency, rolling
window и max_on_sec. Incidence/side проверяются на точке наведения перед
подшагом, затем на sampled покрытых регионах. Look side определяется относительно
орбитальной плоскости: right вдоль v×r, left противоположно; это не ground-track
heading с учетом вращения Земли.

Номинальные stripmap-приближения:

- ground-range resolution = c / (2 * bandwidth * sin(incidence));
- azimuth resolution = antenna_length / 2.

Они не обещают восстановленный SAR image. Нет coherent aperture integration,
Doppler/squint processing, PRF ambiguity, polarization, speckle, noise-equivalent
sigma-zero, terrain layover/shadow или ScanSAR/spotlight processing. Прямоугольный
footprint — effective illumination surrogate, не реальная диаграмма антенны.
Солнечная высота и optical cloud masks SAR quality не ограничивают; дождевые
потери и частотная атмосферная модель не реализованы.

Ресурсная модель различает pulse duty и session duty:

- mean RF = peak RF * pulse duty;
- amplifier DC = mean RF / amplifier efficiency;
- electrical payload load = существующий payload.power_w + amplifier DC;
- amplifier heat = DC − radiated RF.

DC входит в общий bus/battery ledger **один раз**. Тепловая поправка заменяет
generic electronics heat fraction на amplifier heat именно для DC-ветви;
радиоизлучение не остается теплом внутри аппарата. Converter losses продолжают
учитываться общим bus. Report содержит DC input, radiated RF и amplifier heat
с явным равенством баланса, а также payload_heat_adjustment_j. Температура все
еще рассчитана одним synthetic RC узлом, не отдельной конструкцией радара.

В любом скользящем rolling_window длительность активной SAR session не превышает
max_on_sec. Интегратор находит границу разрешенной работы; если acquisition
упирается в лимит, она завершается sar_duty без raw, сохраняя уже потраченную
энергию и on-time. Отмена, смена команды и safe/recovery не стирают историю.
Остывание/ожидание естественно освобождает окно, без фиктивного пополнения ресурса.
Admission консервативно проверяет также peak RF / efficiency вместе с остальной
нагрузкой против bus max_load: средняя мощность не позволяет обойти пиковый лимит.
Накопитель импульсной энергии не моделируется.
Импульсы RF во времени и аккумуляторная электрическая схема не разрешаются
по отдельным pulses; интегрируются средние мощности.

## Lineage и fulfillment

Raw quality_evidence содержит AOI hash, геометрию в локальных метрах, время,
raw ID, actor, параметры инструмента, mask ID/state и sampled geometry metrics.
Повторная ветка DAG не удваивает evidence или usable area. Transform в этом
контракте сохраняет spatial support; crop, blur и потери качества не моделируются.
Произвольный scientific processing этим наследованием не валидируется.

Usable coverage пересчитывается по критериям каждой заявки, объединяется и
ограничивается фактической геометрией доставленного продукта. Denominator —
**вся площадь AOI**, а не только ясная/известная часть снятого кадра.
И geometric fraction, и usable fraction должны достигнуть coverage.min_fraction.
Проверка выполняется после полного ACK нужной станции в deadline. Если геометрия
достаточна, а качество нет, продукт остается delivered, заявка failed с
quality_requirement. Глобальная ясная площадь другой заявки ее не выполняет.

Local observation показывает собственный instrument, sampling, доступные маски
только released/eligible AOI и собственную SAR history. Operator report добавляет
payload_quality с per-request usable area/fraction и fulfillment, моделями и SAR
intervals. Forecast geometry не обещает выполнения quality/resource constraints.
Samples, evidence и RF history входят в rollback/checkpoint/replay; версии native
geometry dependencies фиксируются прежним B04-B1 manifest.

## Запуск и приемка

```bash
python -m pytest -q tests/test_payload_quality.py
python -m space_env.eval.ci_profiles reference
python examples/payload_quality/demo.py /tmp/quality-optical optical
python examples/payload_quality/demo.py /tmp/quality-sar sar
```

Оба примера — synthetic variants. Optical явно переводит сцену в дневную
геометрию; SAR использует наклонную ночную геометрию. Prefix checkpoint создан
посреди acquisition, затем transform и полная доставка. Сохраненные
scenario/plan/resume JSON пригодны для mission-validate/run/resume/replay.
Installed wheel проверяется через Python -I из другого каталога.
[Приемка](../acceptance/B04_PAYLOAD_QUALITY.json).

Формулы и границы области сверены с [NASA SAR Handbook](https://earthdata.nasa.gov/s3fs-public/2025-04/SARHB_FullRes_2019.pdf?VersionId=XU9mWLgLcPtEZiL3.m72A2L8lcJOxub4),
[ESA SAR introduction](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Onboard_Data_Processing/Introduction_to_a_SAR_System)
и [Copernicus S2 processing](https://sentiwiki.copernicus.eu/web/s2-processing).
Это основания постановки, не независимая калибровка реализации или поддержка
полного формата Sentinel-2/SAR продуктов.

B04-B3 добавил [revisit/stereo требования](OBSERVATION_REQUIREMENTS.md)
по реально перекрывающейся пригодной площади и временам наблюдений.
B04-C добавил [parallel resource scheduler](RESOURCE_LANES_CONTRACT.md).
Далее B04-D: actuator/thrust/attitude coupling. RF scheduling/parallelism, actuator/thrust coupling,
расширенный thermal model и внешняя физическая qualification остаются открыты.
