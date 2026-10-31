# B04-C: параллельные CPU/payload/radio и общие ресурсы

2026-10-31. Opt-in `definition.execution.mode="resource-lanes/1"` в
`reference-spacecraft-bus/1`, Scenario v3. Отсутствие execution сохраняет прежний
serial contract. Это инженерная модель управляемых каналов, не hardware-qualified
RTOS, не оптимальный scheduler и не подтверждение больших mission SLA.

## Явная конфигурация

```json
{
  "execution": {
    "mode": "resource-lanes/1",
    "actors": {
      "a": {
        "cpu": true,
        "payload": true,
        "radio": true,
        "memory_bandwidth_bps": 1000000.0,
        "priority": ["payload", "cpu", "radio"]
      }
    }
  }
}
```

Configuration обязана покрывать всех actors. У каждого допускается по одному
CPU, payload и radio lane; boolean отключает канал. Memory bandwidth задается
в **битах/секунду**, priority — перестановка всех трех имен. Неизвестный профиль,
отсутствующая емкость или неверная очередь отвергаются без serial fallback.
Требуется coupled spacecraft bus: legacy/fixed/lifecycle-only профили эту
конфигурацию не принимают.

CPU lane исполняет **managed transform DAG B03**. Команда processing для native
in-place предзагруженных продуктов здесь явно unsupported и не рекламируется
в capabilities. Ее serial профили сохранены. Мультикор, несколько payload/radio
цепей, parallel scientific kernels и GPU этой версией не моделируются.

## Владение командами

Submit допускает несколько команд одного actor в одном batch или последовательно,
если нет конфликта claims:

- transform: CPU;
- imaging: payload + attitude;
- downlink: radio; при radio_tolerance_deg < 180° также attitude;
- slew: attitude;
- wait: все четыре claims, явное exclusive ожидание.

Отключенный канал не допускается. Две CPU-команды, две съемки или две разные
команды ориентации не получают один ресурс. CPU и omni-radio могут выполняться
во время slew; imaging и направленная radio не получают двух владельцев attitude.
Это консервативное ограничение одного общего корпуса, без модели подвеса антенны.
Оно не запрещает геометрическую линию связи другим actor.

Проверка batch атомарна; конфликт дает resource_busy, старый serial профиль —
actor_busy. Дополнительно исключены дубли active request/job между **разными**
actors: два контроллера не владеют одним будущим output ID. Этот guard исправляет
также общую lifecycle admission, не меняя корректные прежние сценарии.

Control window не означает постоянную нагрузку. CPU/payload прекращают работу
после фактического завершения; радио с пустым/полностью отправленным продуктом
не расходует TX energy. Радиокоманда может сохранять claim до конца своего окна
или cancel, даже если данные уже отправлены. Команда с истекшим окном может быть
completed, в то время как CPU job paused с оставшейся работой: B03 различает эти
состояния, fulfillment требует реальных продуктов и ACK.

## Арбитраж по общей шкале времени

На каждом integration interval candidates готовятся в объявленном порядке
actor_order и priority, независимо от порядка команд в submit. Existing geometry,
health, safe mode, memory и product readiness проверки сохраняются.
Существующий native station selector выделен в readonly helper и повторно
используется для visibility/outage/bookings/terminal/rate/SOC admission.
Сортировка station contention остается прежней; это не оптимизация покрытия.

Затем first-fit priority одновременно ограничивает полезную среднюю нагрузку,
совместную peak demand, SOC reserve и bandwidth каждого аппарата. Для SAR
учитывается peak DC admission отдельно от mean pulse DC. Низкоприоритетный канал
не обходит лимит только потому, что каждый потребитель отдельно укладывается.
Общая SOC граница — максимум reserves реально выбранных каналов; при ее
достижении дорогой CPU может остановиться, а работа с меньшим reserve продолжиться.

CPU и радио допускают pause/throttle. Imaging до первого прогресса может ждать
ресурс, сохраняя control/attitude claim и резерв памяти. После начала continuous
acquisition вытеснение ресурса завершает попытку с resource_preempted, без raw;
разрозненные куски не выдаются за непрерывную экспозицию. На исчерпании command
window незаконченная съемка не материализует продукт.

Интервалы разделяются по resource/geometry/event grids, command ends, native
outages/bookings, radio completion/quota, CPU completion и shared SOC boundary.
При подготовке geometry-invalid capture или при изменении eligibility некоторые
резервы могут оставаться консервативными до следующего подшага. Scheduler не
обещает мгновенный backfill, starvation freedom или лучшую makespan.

## Пропускная способность памяти

В едином bus считаются:

- camera stream write = raw_size_bits / acquisition_sec;
- CPU read+write = (сумма declared input bits + output bits) / work_sec,
  умноженная на нормированную скорость исполнения;
- radio read = фактически переданные bits.

При недостатке bandwidth CPU получает скорость в (0,1]; useful work уменьшается
на elapsed_sec * speed, а не на полное wall time. CPU switching использует
электрическую мощность и время, но не создает useful work и в этой модели не
использует data I/O. Радио rate временно ограничивается оставшейся bandwidth
в пределах native link/station upper bounds; постоянные configured rates после
шага восстанавливаются. Для выбранного payload нужен полный sensor stream rate.

Это объявленная deterministic I/O модель, не замер реального DRAM/cache/DMA
трафика. Нет memory latency, contention отдельных банков, workspace random I/O,
компрессионного bitstream или физического JSON-размера metadata. Raw stream до
materialization закрыт от других jobs и занимает свой peak memory reservation.

Persistent workspace и outputs CPU резервируются одновременно с raw capture
buffer. Outputs размещаются до consumption inputs, как B03. Нельзя использовать
одни свободные bits дважды. Уже оплаченные паузой/отменой CPU work и workspace
не исчезают; освобождение persistent job требует существующего job policy/event.
Такой резерв может законно блокировать новую работу, без скрытой эвакуации данных.

## Энергия, тепло и ошибки

Managed powers суммируются по actor, radio power добавляет сам native transport.
Base/ADCS учитываются один раз; solar generation, converter losses и battery
интегрируются один раз на wall interval. CPU при throttling сохраняет declared
active power: меньшая скорость может потребовать больше энергии на тот же job.
Radio также сохраняет active TX power при меньшей скорости передачи.

SAR DC/RF/amplifier heat и session duty из B04-B2 не дублируются при работающем
CPU/radio. Остается один synthetic RC node. Неправильное pointing и отказы одного
подсистемного канала не стирают мощность других выбранных каналов. Safe mode,
floor и общий fault могут остановить несколько потребителей причинно.

Cancel одной команды освобождает ее claims, не отменяя остальные. Повторный CPU
window продолжает сохраненную работу без бесплатного завершения или повторной
оплаты уже сделанного setup. Unexpected exception откатывает весь текущий advance:
native inventory/receipts, battery/thermal/attitude, jobs, scheduler и views.
Snapshot сохраняет общий bus/resource object, а disk restore по-прежнему prefix
replay, без заявления O(1).

## Наблюдение и приемка

Capabilities явно описывают concurrency и execution.resource_lanes. Local view
содержит configuration и последний интервал своего actor, собственные managed
grants/phase/work_rate, фактические radio bits, I/O, memory reservations и причины
блокировки. Peer command IDs/poses туда не попадают. Operator resource_lanes report
содержит все интервалы и configuration. Изменение возвращенного view не меняет
исполнение. Reserved peak/bandwidth — admission bounds, не billed consumption.

```bash
python -m pytest -q tests/test_resource_lanes.py
python -m space_env.eval.ci_profiles reference
python examples/resource_lanes/demo.py /tmp/lanes-wide wide
python examples/resource_lanes/demo.py /tmp/lanes-limited constrained
```

В примере нет preloaded products: сначала снимаются archive и CPU input, затем
одновременно исполняются новая съемка, обработка и downlink архива. Учтено
существующее закрытие станции. После доставки outputs выполнены три заявки.
Режим constrained имеет 1 Мбит/с общего I/O и более длинное окно для задачи.
Количество overlap intervals включает активный CPU switching; work_rate и phase
позволяют отличить его от полезного вычисления. Это не сравнительный benchmark
планировщиков: для него нужны одинаковые workloads и objective B07.

Тесты включают независимый интеграл суммарной нагрузки, 3.5 Mbit I/O balance
параллельного этапа, normalized CPU work, memory/power/SOC bottlenecks,
station contention, faults/cancel/preemption, global output ownership,
rollback, SAR accounting и mid-three-lanes replay. Installed wheel проверяется
через Python -I и CLI validate/run/resume/replay вне source checkout.
[Артефакт](../acceptance/B04_RESOURCE_LANES.json).

Следующий B04-D — actuator/thrust/attitude/health coupling, чтобы вернуть
управляемые маневры в coupled spacecraft profile с явным физическим контрактом.
Подробный thermal model, независимая calibration, RF fidelity B05 и mission-scale
qualification B09 остаются открытыми.
