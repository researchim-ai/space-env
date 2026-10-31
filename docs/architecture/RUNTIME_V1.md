# Runtime v1: единый путь исполнения миссии (B01)

Дата: 2026-10-31. Scenario schema 3, checkpoint `space-env.runtime-checkpoint/1`.
Публичный API: `space_env.runtime`. Это интеграционный контракт двух явно
различающихся физических профилей, а не утверждение их одинаковой точности.

## Границы и единицы

`ScenarioSpec` содержит backend/profile, обязательный уникальный `actor_order`,
self-contained `definition`, lineage, fidelity, KPI, источники и лимиты.
Reference definition содержит эпhemerиды, аппараты, станции, продукты, брони,
DAG, ресурсы и горизонт; legacy definition — portable MissionWorld snapshot.
Неизвестные поля оболочки запрещены; внутреннюю definition проверяет backend.
Исходная и эффективная конфигурации сохраняются отдельно, defaults материализуются.
JSON schema экспортируется `space-env mission-schema schema.json`.

Время команд — секунды от начала миссии; геометрия reference — GCRS, метры и
метры/секунду, абсолютная эпоха задается эпhemerидой. Legacy сохраняет исходную
численную frame convention: его векторы не становятся GCRS при адаптации.
Wh и bits сохраняют явно названные единицы моделей ресурсов. Seed оболочки —
метаданные эксперимента; RNG legacy восстанавливается из snapshot, seed не
пересевает его молча. SourceRecord содержит checksum/URI/license декларацию;
URI не загружается, декларация не заменяет независимую проверку источника.

## Команды, доступ и порядок

`RuntimeCommand`: уникальный command_id, actor_id, kind, duration_sec и только
совместимые destination/product/request поля. На актера допускается одно активное
окно. Reference поддерживает wait/downlink/processing, legacy — wait/downlink/imaging.
Отсутствующий capability или зависимость дает ошибку без смены backend.
В reference CPU работает только по явной processing-команде; idle не включает
скрытый планировщик. Завершение окна означает завершение команды управления,
а не обязательную доставку продукта: результат проверяют KPI и backend report.

Каждая мутация требует expected_revision и idempotency_key. Revision увеличивается
на один только после успеха. Повтор ключа с тем же op/body возвращает сохраненный
ответ даже после других операций; другой body конфликтует. Невалидный batch и
ошибка в середине advance откатывают весь вызов. RLock сериализует вызовы одного
runtime внутри процесса; распределенной блокировки и hosted authentication нет.

Операции при одном времени следуют порядку принятых вызовов/revision. Advance
разбивается на границах command windows: backend исполняет интервал, затем runtime
фиксирует завершения; последующие cancel/inject/submit видят это состояние.
Внутренние availability/fault/booking/thermal границы обрабатывает соответствующий
backend с его объявленной точностью. Это не новый глобальный физический solver.

Agent observe содержит текущее локальное состояние и доступные продукты.
Truth, геометрический forecast и журнал доступны только при явном AccessPolicy.
Forecast дополнительно требует capability и не продвигает время. AgentView
ограничивает командование своим актером; Python facade не является sandbox для
недоверенного кода. Артефакты содержат truth и требуют операторского доступа.

## Журнал, checkpoint и восстановление

RunManifest фиксирует версии среды/profile/data; checksum защищает целостность,
а не доказывает физическую достоверность. Журнал хранит запрос, revision,
наблюдения, ответы и backend outcomes. `iter_events(cursor, limit)` возвращает
копии страниц журнала мутаций, а не поток всех внутренних физических событий.

Checkpoint самодостаточен: original/effective scenario, manifest, journal и final
reports. Load заново исполняет префикс и сверяет каждую запись и результат;
тем самым восстанавливает активные окна, cache идемпотентности, CPU/DAG, ресурсное
состояние, брони и thermal hysteresis без первоначального YAML. Дискретные исходы
сверяются точно, числа — rel=1e-12, abs=1e-8. Несовместимый schema/manifest или
измененный результат отвергается. Сохраненные права не предоставляют новых прав
загрузившему процессу: AccessPolicy задает вызывающий код.

Это verified-prefix checkpoint, а не быстрый O(1) snapshot: стоимость load растет
с длиной журнала. Лимиты ограничивают операции/команды/advance, но не являются
доказательством ограниченной памяти. Streaming data factory остается B08.
In-place restore разрешен только в свежий runtime (revision 0): это исключает
повторное использование старых revision после rewind; для ветвления используют load.

## SDK и CLI

См. `examples/unified_runtime/demo.py`. Команды:

```sh
python examples/unified_runtime/demo.py /tmp/unified-runtime
space-env mission-validate /tmp/unified-runtime/scenario.json
space-env mission-run /tmp/unified-runtime/scenario.json /tmp/unified-runtime/plan.json /tmp/unified-runtime/cli-prefix.json
space-env mission-resume /tmp/unified-runtime/cli-prefix.json /tmp/unified-runtime/resume.json /tmp/unified-runtime/cli-final.json
space-env mission-replay /tmp/unified-runtime/cli-final.json
```

CLI не перезаписывает входные файлы; результат записывается атомарно после всего
плана. Каждый runtime-вызов транзакционен; весь план не является общей транзакцией
для существующего Python runtime. Общие Gym/LLM wrappers и физические маневры — B06/B02.


## Расширение B02 (2026-10-31)

Добавлен явный backend `reference-dynamic` / profile `reference-dynamic-downlink/1`,
helper `dynamic_reference_scenario`, capabilities `commands.impulse` и
`commands.finite_burn`. Impulse имеет duration=0 и выполняется атомарно при submit;
другие команды требуют duration>1ns. Геометрический forecast содержит отдельную
geometry_revision. Command schema получила optional impulse/burn specifications;
у старых backend эти capabilities отсутствуют. Старые checkpoints не мигрируются
неявно: несовпадение source/runtime manifest по-прежнему явно отвергается.
[Физические ограничения и квалификация](../validation/PHYSICS_QUALIFICATION_MATRIX.md).


## Расширение B02-Q1

`definition.orbits.<actor>.navigation` опционально включает Cartesian covariance
prediction. Capabilities `state.covariance_propagation`/`state.stm` объявляются,
если navigation задана всем актерам; `state.covariance.partial_actors` и список
navigation_actors описывают частичную конфигурацию. Agent видит текущее P,
privileged truth — P и глобальный STM. Журнал/rollback/prefix replay сохраняют
тот же контракт при маневрах и cancel.
[Единицы, noise semantics и ограничения](../validation/NAVIGATION_COVARIANCE_CONTRACT.md).


## Расширение B02-Q2

Capabilities forecast.eclipse (оба reference) и forecast.acquisition.geometry
(configured dynamic sensors/targets); отдельные readonly методы с AccessPolicy.
Forecasts включают binding по scenario/actor/trajectory/revision. Optional
RuntimeCommand.forecast проверяется под тем же mutation lock до действий batch;
повтор idempotency request по-прежнему возвращает cached response до admission.
Это stale-plan guard, не reservation/feasibility certificate. Новые поля входят
в schema v3 и source-versioned artifacts, старые snapshots не мигрируются молча.
[Единицы, sampling limits и семантика](../validation/GEOMETRY_EVENTS_CONTRACT.md).


## Расширение B03

`reference-lifecycle` / `reference-lifecycle/1`, helper lifecycle_scenario,
objective fulfilled_requests. Optional lifecycle definition принадлежит только
этому профилю; старые downlink profiles сохраняют прежнюю семантику. Добавлены
transform(job_id), imaging(request_id) и causal lifecycle event types. Backend
может завершить control window раньше declared end, когда операция фактически
закончилась. Transform window completion не означает job completion при pause.
Плагины отделены от resource work, fingerprint проверяется при load. Все
reference-* run manifests теперь корректно помечены GCRS/ITRS; source-version
migration остается явной. [Контракт](../validation/PRODUCT_LIFECYCLE_CONTRACT.md).
