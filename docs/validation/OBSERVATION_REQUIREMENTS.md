# B04-B3: revisit/stereo и наземные пакеты наблюдений

2026-10-31. Opt-in требования schema v3 в `reference-spacecraft-bus/1`.
Зависимости: [area coverage](AREA_COVERAGE_CONTRACT.md) и
[payload quality](PAYLOAD_QUALITY_CONTRACT.md). Статус — инженерный sampled
geometry contract; полный B04 и независимая физическая qualification открыты.

## Что считается наблюдением

Каждая успешная площадная acquisition сохраняет `observations`: sampled времена,
UTC и two-part TAI epoch, raw ID, actor, AOI/hash, trajectory fingerprint,
GCRS state, ITRS camera position, body→GCRS quaternion и clipped footprint.
Геометрия footprint остается в локальной WGS84/LAEA проекции AOI, метры.
Положение камеры — ITRS метры; эти массивы нельзя смешивать с координатами LAEA.

Времена — начало, середина и левая граница конца каждого действительного
подшага экспозиции. `created_sec` продукта и время ACK не являются временем
наблюдения. Поза берется из того же propagated state и rate-limited attitude,
которые проверялись исполнителем. Нет дополнительного идеального steering.

Observation ID связывает содержимое записи целиком. Один raw ID обозначает
одну acquisition: десятки samples одного raw, копии и ветки transform DAG
не превращаются в новые независимые наблюдения. Fan-in/out наследует записи
с дедупликацией, проверкой fingerprint и запретом разных поз одной raw/time.
Это provenance исполнения симулятора, не криптографическая аттестация камеры.

Quality evidence связано с конкретным observation ID, raw, actor и временем.
Для каждой заявки заново применяются ее optical/SAR критерии. Качественный
union всего raw не делает качественными другие его samples. Если в первом
наблюдении ясно слева, а во втором справа, общая ясная площадь может покрыть AOI,
но два пригодных вида одной и той же поверхности этим не получаются.
Без `quality` условие относится только к геометрии, без обещания scientific quality.

## Заявка

`CustomerRequest.observations` требует explicit `coverage` и содержит хотя бы
один критерий `revisit` или `stereo`. `coverage.min_fraction` относится ко всей
площади AOI, не только к снятой или известной части.

Revisit задает min_observations (2..8), положительный min_interval_sec,
max_interval_sec и опциональный different_actors. В каждой цепочке acquisitions
различны; интервалы между **последовательными sampled временами** должны попадать
в заданный включительный диапазон. При different_actors все actors цепочки
различны. Каждая цепочка дает пересечение пригодных footprints, затем площади
разных допустимых цепочек объединяются без повторного зачета.

Stereo задает min/max_convergence_deg, min/max_interval_sec, min_baseline_m,
опциональный max_baseline_m и different_actors. Пара всегда состоит из двух
разных acquisitions, даже если разные записи одного raw имеют разные времена.
Одновременная пара разных аппаратов допустима при min_interval_sec=0;
нулевой baseline не создает положительного convergence.

Для общей поверхности пересекаются пригодные footprints пары. Поверхность
разбивается на локальные клетки; угол проверяется в вершинах и representative
point каждого региона. Обе камеры рассматриваются относительно **одного
статичного WGS84 ground point**. Угол вычисляется между surface→camera rays.
Baseline — расстояние между ITRS camera positions, привязанными к своим эпохам:
это фотограмметрическая база относительно Земли, не одновременная дальность
между спутниками. Различие принципиально для одного спутника в разные моменты.

Есть ограничения пространственной дискретизации: проверки samples не доказывают
экстремумы между ними. Нет DEM, scene motion, RPC/lens error, covariance-aware
chance constraints, matching, triangulation accuracy, DSM, SAR interferometry
или coherent stereo reconstruction. Даже пригодная геометрия пары не обещает
научный stereo product. Угловые пределы — критерии сценария, не универсальная
рекомендация для всех сенсоров и высот.

Если заданы оба критерия, берется пересечение их пригодных областей. Недостаточно
получить revisit на одной части AOI и stereo на другой. Обычные геометрические
и quality thresholds также остаются обязательными.

## Бортовой DAG и группировка

Для одного аппарата несколько raw можно обработать fan-in job в один delivery
product. Ограничение B03 на принадлежность inputs одному actor сохранено.
Кросс-аппаратная бортовая обработка и межспутниковая передача не имитируются.

Для группировки `observations.support_product_ids` задает до 64 дополнительных
**managed** продуктов. Они вместе с delivery_product_id образуют явный наземный
пакет. Каждый должен полностью прийти на ту же destination до deadline заявки.
Unknown/unmanaged, повторный или совпадающий с primary support ID отвергается;
несогласованная destination одного продукта также запрещена.

Проверка использует metadata/evidence только этих именованных продуктов.
Другие снятые и доставленные кадры из общей истории ничего не добавляют.
Критерии заявки переоцениваются для всего названного пакета; более строгая
заявка источника может не выполниться, а ее продукт оставаться пригодным по
другому явно заданному критерию получателя. Artifact request_ids сохраняют
реальное acquisition ancestry; bundle membership не переписывает происхождение.

Это сборка ground evidence для проверки обязательства, без создания фиктивного
pixel mosaic или бесплатного научного processing. Время/мощность ground fusion,
распределенное хранилище и межспутниковый RF остаются отдельными задачами.

Receipt проверяется по полному количеству bits каждого продукта и правильной
станции. Partial ACK любого support не выполняет primary. Время завершения
обязательства — максимум времен нужных полных receipts. Поздняя часть пакета
не оживляет expired request. Связи receipt→obligation расширены на явные supports,
но physical ACK и transmitted bits остаются единственными: учета байтов дважды нет.

Отмена еще не выполненной acquisition или потеря обязательного support
причинно делает зависимую заявку unreachable. Уже созданный независимый продукт
не исчезает только от отмены его customer demand и может служить явно заданному
обязательству. Никакого скрытого восстановления продукта нет.

## Бюджеты, отчет и повтор

`LifecycleSpec.observation_settings` задает record/product, frame/evaluation,
pair/chain/cell check и spatial-point budgets. Слишком большой capture атомарно
откатывает текущий advance вместе с ресурсом, позой и накопленными samples.
Слишком большой transform остается ожидаемой ошибкой materialization после
оплаченного CPU. Наземная комбинация/оценка сверх бюджета возвращает
`status=budget_exceeded`, joint_fraction=0 и не дает fulfillment. Уже переданные
байты не откатываются. Это отказ квалификации по вычислительному бюджету,
не доказательство физической невозможности задачи.

Report добавляет `observation_requirements`: joint area/fraction, число разных
acquisitions с ненулевой пригодной площадью, component areas, counters, full_delivery и delivered_and_fulfilled.
До ACK можно увидеть пригодную геометрию, но fulfillment еще false. До 32 witness
цепочек/пар выводятся для аудита; обрезается только список примеров, не площадь
оценки. При превышении бюджета ни частичная площадь, ни первые хорошие witnesses
не превращаются в частично одобренную заявку.

После полных ACK недостаточный критерий дает observation_requirement;
превышение вычислительного бюджета — observation_budget. Coverage/quality
сохраняют собственные причины отказа. Все demand остаются в denominator B03.
Две служебные acquisition-заявки в demo означают total_requests=2, а не тайную
смену объективной функции на одну campaign.

Capture, evidence links, delivery dependencies и промежуточная передача входят
в checkpoint/replay/rollback. Readonly evaluation имеет ограниченный pure cache
на 32 ключа; результаты выдаются копиями, не изменяющими исполнение. Это не SLA
и не O(1) checkpoint restore. Local observation сохраняет доступ только к своим
артефактам и released/eligible demand; наличие support ID не раскрывает peer poses.

## Воспроизведение

```bash
python -m pytest -q tests/test_observations.py
python -m space_env.eval.ci_profiles reference
python examples/observations/demo.py /tmp/observations-revisit revisit
python examples/observations/demo.py /tmp/observations-fleet stereo-fleet
```

Первый пример снимает два раза одним аппаратом и выполняет fan-in processing.
Второй снимает двумя аппаратами, учитывает закрытие станции и последовательно
доставляет обе части пакета. Оба восстанавливаются посреди acquisition и partial
radio. Scenario/plan/resume пригодны для CLI mission-validate/run/resume/replay.
Installed-wheel проверки запускаются с Python -I вне checkout.

Analytic checks сравнивают convergence с `2*atan(baseline/(2*height))`, независимую
многократность и границы интервалов; отдельно проверены partial/cloud overlap,
duplicate lineage, невозможные пары, station ACK, TTL/cancel/deadline, budgets
и snapshot restore. Это инженерные проверки, не внешняя flight qualification.
[Артефакт приемки](../acceptance/B04_OBSERVATIONS.json).

B04-C добавил [parallel managed resource lanes](RESOURCE_LANES_CONTRACT.md),
memory bandwidth и causal faults. Следующий B04-D — actuator/thrust coupling. Limb clipping/другие сенсоры, actuator/thrust coupling,
подробная thermal модель и независимая калибровка остаются открыты.
