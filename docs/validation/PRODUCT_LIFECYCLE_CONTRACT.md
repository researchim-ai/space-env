# B03: исполняемый request → acquisition → transform → delivery

Дата: 2026-10-31. Новый явный профиль **reference-lifecycle/1**, backend
reference-lifecycle, objective fulfilled_requests. Инженерная приемка ограничена
описанным point/dwell/resource surrogate; промышленная квалификация B02/B12 открыта.

## Что изменилось относительно downlink-профиля

Раньше OnboardProduct уже существовал, а processing_sec менял готовность тех же
байтов. Этот режим сохранен для fixed/dynamic downlink. Новый optional lifecycle
нельзя включить молча: требуется lifecycle_scenario(...) / отдельный backend и
profile. Существующие reference/legacy scenarios не превращаются в lifecycle.

CustomerRequest описывает спрос независимо от геометрических окон: target,
eligible_actors, release/deadline/priority, требуемый dwell, raw ID/размер,
delivery product/destination, optional TTL и max_attempts. Идентификаторы заявки,
команды, raw/intermediate/final артефактов и transformation jobs различаются.
Ни один из этих объектов не означает автоматического успеха другого.

Demand и jobs можно задать заранее, включая заведомо невыполнимые заявки.
Unreleased заявки/зависимые jobs не открываются в agent observation. В denominator
попадают все объявленные/добавленные заявки, включая failed/cancelled/expired.
Это сырой total_requests, а не изменяемый отбор удобных окон.

## Команды, состояния и события

```python
from space_env.runtime import AccessPolicy, MissionRuntime, lifecycle_scenario

mission = MissionRuntime(lifecycle_scenario(definition, id="eo", lineage="pilot"),
                         access=AccessPolicy(truth=True, forecast=True))
mission.submit([{"command_id":"image","actor_id":"a","kind":"imaging",
                 "duration_sec":2.,"request_id":"customer-a"}],
               expected_revision=0, idempotency_key="image")
```

imaging требует released заявку и eligible actor, достаточную длительность
управляющего окна и declared payload. transform требует job_id; существующее
processing для preloaded/in-place CPU остается отдельным режимом. Все мутации
подчиняются общим revision/idempotency/rollback/forecast-binding правилам.
Раннее физическое завершение возвращает completed и освобождает command window;
завершение короткого transform window может означать pause работы, а не final.

Request state machine: pending → released → accepted → scheduled → running →
completed, с явными failed/cancelled/expired и ограниченным retry → released.
accepted/scheduled фиксируются при admission; running — при исполнении.
После acquisition request остается running вплоть до конечной delivery.
Нельзя получить fulfilled_request просто из raw/CPU/opportunity/partial receipt.

Через inject доступны request_added (request + jobs), request_cancelled,
request_priority_changed, request_deadline_changed, job_cancelled и
artifact_discarded. Envelope/ID/time-at-now и бюджеты проверяются; запрещены
backdating/reuse и коллизии product/job IDs. Батч атомарен на уровне runtime,
включая ранее принятые внутри батча updates. Demand не создается из найденного
удобного прохода. Внешняя onboard_product_added остается отдельной инъекцией:
она не может подделать managed raw/final или обойти резерв памяти и не дает reward.

## Исполнение acquisition

Данные появляются только после накопления полного **непрерывного sampled dwell**.
Исполнитель проверяет текущий provider/state и Q2 geometry, а не верит прогнозу.
Учитываются WGS84 target, off-nadir/elevation/range/solar constraints и availability.
На каждом шаге проверяются начало, середина и левый предел конца интервала.
Default execution_step_sec=0.5, допустимо 0.01–5 sec; сетка абсолютная, дополнительно
разбивается по releases/deadlines/TTL, completion и power floor. Для кратких/grazing
изменений между проверками полнота не доказана; это не непрерывный flight detector.

Дополнительная payload/CPU мощность входит в реальную energy integration того же
аппарата. Generation использует текущую absolute-grid solar модель; floor ограничивает
полезную работу. Полный raw size резервируется перед началом acquisition. Частичная
работа потребляет энергию, но не создает успешный raw. Потеря geometry/power,
отмена или исчерпание attempts фиксируют reason. Более короткая, чем требуемый dwell,
команда отвергается. В примере infeasible target действительно за Землей.

Raw metadata хранит время/validated dwell, последний geometry sample, actor,
request lineage, quality model и content hash. Coverage — точка/dwell; площади AOI,
снимка и реальных пикселей нет. Идеальные steering и независимое solar tracking
сохраняются; coupling attitude/payload/thermal требует B04.

## Transform DAG, память и удержание

TransformationJob содержит inputs/outputs со стабильными IDs, declared work_sec,
workspace_bits, switching_sec, power/min_soc, consume_inputs и transformer version.
Graph валидируется топологически, без рекурсивного обхода глубокой цепочки;
unknown input/duplicate output/cycle отвергаются. Продукт доставки обязан иметь
lineage своей заявки; общий output для нескольких заявок требует одной destination.

Fan-in/out поддержан при целых входах **на одном аппарате**. Cross-actor transfer
не подменяется чтением чужого artifact. Pending job без входов явно blocked;
нет скрытого CPU scheduler. После паузы сохраняются work remaining и неоплаченный
switching interval. Persistent workspace и выходной memory reserve защищаются от
новых acquisition/injection и учитываются при admission native CPU. Полный peak
inputs + workspace + outputs должен помещаться до удаления inputs.

Плагин запускается атомарно в конце ресурсного work interval. Все outputs
материализуются одним native inventory batch, затем освобождаются inputs. Shared
input удерживается до последнего потребителя; физические consumed bits списываются
один раз, а размеры outputs могут отличаться. Нет ложного сохранения общего bit
count при преобразовании. retain-only consume_inputs=False сохраняет входы.

Declared MaterializationError фиксирует job failure и уже потраченные work/energy,
не создает частичных outputs. Неожиданное исключение откатывает весь advance,
включая энергию/время/ledger. Для единственного недоступного producer/потерянного
обязательного artifact заявка получает failed, а не скрывается из denominator.

## Плагины и что является данными

Публичный ProductTransformer отделяет materialize от модели work/energy/memory.
Встроены metadata-surrogate/1 и geometry-features/1. Второй создает настоящий
JSON payload с acquisition geometry/lineage, **не изображение или научное сжатие**.
Размер size_bits задает ресурсный объем; он не равен размеру JSON сериализации.
Default raw также synthetic-resource-metadata. Pixel/optical/SAR quality не
появляются из одной строки compression ratio или поля quality.

register_product_transformer(factory) — trusted SDK extension. Плагин должен быть
детерминированным, deepcopyable, JSON-producing и не иметь внешних side effects:
rollback симулятора не отменяет внешний I/O. Version и source_sha256 обязательны;
внешний fingerprint надо явно объявить в lifecycle.transformer_manifests.
Это caller-declared provenance (должно включать code/model/data), не attestation.
В effective scenario сохраняются все используемые manifests. Без нужной регистрации
или с другим fingerprint checkpoint не загружается; fallback к surrogate отсутствует.
CLI не импортирует произвольный Python из сценария. Для внешних плагинов загрузку
выполняет доверенный SDK bootstrap, register → MissionRuntime.load.

Плагин выполняется целиком после modeled CPU work: checkpoint хранит **ресурсный
progress**, не приостановленный процесс реального алгоритма. Это предсказуемая
среда планирования; scientific kernels/image data должны отдельно квалифицироваться.

## TTL, delivery и учет

Unpinned TTL истекает по явной политике. Started/paused jobs pin inputs и удерживают
workspace, поэтому TTL откладывается до освобождения/cancel. Pending consumers
блокируют преждевременный downlink, но сами не продлевают TTL навсегда. Eviction
явный artifact_discarded; unconfirmed final защищен default allow_final_discard=False.
Конец mission выполняет terminal housekeeping, а не backdated operator event.

Transport использует прежний **drain-on-send** storage model и fixed-rate station
service; full ACK — instant-on-complete-station-receipt/1. Это не CCSDS packets,
retransmission или ground processing. Partial receipts освобождают отправленную
часть памяти, но не подтверждают весь output и не выполняют заявку. Receipt содержит
product/request/destination связь; shared final ACK считает физические bits один раз.
Customer deadline проверяется по фактическому full-receipt time (только ULP snap),
а не времени выдачи команды. Late/cancelled заявки не становятся completed.

Actor ledger закрывает:
initial + external injected + acquired + transformed_out = transformed_in +
transmitted + expired + discarded + remaining.
acked — **подмножество** transmitted, не дополнительный расход в равенстве.
Показан balance_error_bits; fixed-rate resource transport допускает дробные bits.
Native transport ledger использует внутренние onboard registrations для хранения
managed продуктов; его нельзя складывать с lifecycle ledger как второй источник.

## Наблюдения, checkpoint и ограничения

Agent получает только released/eligible request/job definitions и локальные
artifacts; после выбора владельца входных bytes job скрывается от чужого актера.
Полный operator результат доступен через
mission.observe(view="truth")["backend"]["report"]["lifecycle"].
В CLI artifact это backend_report.lifecycle. Операторские future demand и truth
не добавляются в actor view.

Self-contained prefix checkpoint сохраняет initial/effective scenario и журнал;
проверены mid-acquisition и mid-transform restore, reservations, TTL, events,
plugin fingerprints, lineage и idempotency. Восстановление повторяет префикс,
не является O(1) snapshot. max_execution_steps ограничивает advance; dense workloads,
bounded-memory streaming и SLA группировки остаются B08/B09.

Исправлен также run-manifest frame_contract: все reference-* backend, включая
динамический и lifecycle, честно имеют GCRS/ITRS; legacy tag сохраняется только
legacy backend. Source-version gating старых artifacts остается явным.

## Проверки и пример

31 новых tests, reference 388 passed, fast 110 passed, 818 collected. Полный
regression пакета не запускался. Проверены two-satellite competition, fan-in/out,
partial/failed materialization, causal events, deadlock/power/memory, ownership,
TTL/pins, terminal housekeeping, partial/full ACK, registry mismatch и rollback.

```sh
python examples/end_to_end_eo/demo.py /tmp/eo
space-env mission-validate /tmp/eo/scenario.json
space-env mission-run /tmp/eo/scenario.json /tmp/eo/plan.json /tmp/eo/cli-prefix.json
space-env mission-resume /tmp/eo/cli-prefix.json /tmp/eo/resume.json /tmp/eo/cli-final.json
space-env mission-replay /tmp/eo/cli-final.json
```

Пример начинает с нулевым inventory; mid-acquisition и mid-processing checkpoints
восстанавливаются. Из двух заявок одна доставляет 500000 bits после приобретения
1000000 raw bits, вторая failed по geometry. Метрика 1 fulfilled / 2 total,
нулевой inventory balance residual. [Приемка](../acceptance/B03_PRODUCT_LIFECYCLE.json).
Следующий B04 связывает spacecraft attitude/payload/энергию/здоровье. Промышленная
квалификация физики и пилотные данные остаются отдельными gates, не выводятся
из успешной работы этого инженерного surrogate.
