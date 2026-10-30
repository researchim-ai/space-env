# Проверка space-env 0.134.0

Дата: 30 октября 2026. HEAD `1e5a9f603dcd2af36beb25920f3e0b1ce1e983cc`.
Рабочее дерево до аудита чистое. Проверены актуальные исходники, не старые планы.
Изменены только документы и read-only diagnostics; продуктовый код не исправлялся.

## Вывод

Это существенно расширенный исследовательский симулятор операций, пригодный для
дальнейшей инженерной разработки. Функций стало больше, но согласованность
исполнения, кадров, обучения и длинных миссий пока не подтверждена промышленными
критериями. Есть реальные воспроизводимые дефекты и падающий сценарный тест.
Заявления о превосходстве над другими симуляторами пока не имеют сравнительных данных.

Актуальное задание: [INDUSTRIAL_ROADMAP_V0134.md](../INDUSTRIAL_ROADMAP_V0134.md).

## Объём

- Diff относительно `9a95032`: 113 файлов, около9тыс добавленных строк.
- AST-инвентаризация144 модулей пакета,72 тестовых файлов,37 зарегистрированных сред.
- Сбор тестов:358. Полный suite в этом аудите не запускался.
- Детальное чтение текущих world/commands/snapshot/session, frame/time/Sun/catalog,
  fidelity report, schema, sensors/tools, Gym/parallel, load/CI, replay/datasets.
- Дополнительно просмотрены engine capture/restore, geometry/SAR/knowledge helpers
  и изменённые тесты/границы. Это не построчная квалификация каждого модуля.
- Контрольные hashes и пути: [inventory](v0134_inventory.json).

## Прогресс относительно 0.98

Уже есть исправления batch/request/acquisition/stations старого аудита, running/cancel
и operation duration, блокировка advance, интервалы станций и резервирование.
Добавлены stage epochs RK4, ERFA utilities, finite burn, память/processing/compression,
slew/settling, safe hysteresis, подготовка антенны и частотные/терминальные ограничения.
Появились independent demand, cloud quality, SAR helpers, расширенные типы заявок,
измерения и FDIR, DTN, orbit operations, Dict/parallel поверхности и verified records.
Studio уже подключается к живой сессии и отдаёт её кадры/трассу.

Поэтому прежние выводы «нет сессий/IAU utilities/конечного включения/независимого
спроса» устарели. Следует оценивать глубину интеграции и пригодность конкретного профиля.

## Выполненные проверки

Первый набор — m0/execution/frame_grade/agent_surface/plan_three/resume_bundle/
finale/live_chain/session_dataset/mission_benchmark: **33 passed,16.86с**.

Второй набор — все файлы нынешних CI-профилей: version/cards/agent_surface/dynamics/
physics_rest/mission_benchmark/session_dataset: **29 passed,1 warning,15.34с**.
Наборы пересекаются: нельзя сложить их и назвать62 уникальными пройденными тестами.

Отдельно: `test_day_mission_names_every_request` — **failed,43.43с**, assertion
`len(on_time)>=30`, фактическое значение18. Это не sandbox error. Статус проекта
уже сообщает18/32, но тестовая приёмка не согласована. Причину ухудшения/ожидания
этот аудит не устанавливает; нельзя просто снизить порог без исследования.

HTTP/UI browser, полный358-test suite,72h/endurance и установка wheel0.134 в новом
окружении не проверялись. Installed-wheel evidence в текущем статусе относится
к предыдущим версиям и не подтверждает автоматически нынешний пакет.

## Новые диагностические наблюдения

Воспроизведение: `python docs/audits/v0134_probes.py`.
Результат: [v0134_probe_results.json](v0134_probe_results.json).
Примеры синтетические; script сообщает observations, не acceptance pass.

### 1. Длительность команды не ограничивает исполнение внутри шага

Команда comms duration2с, world.step10с, station Shared над аппаратом.
CommandBook: started0/until2/completed. Реальная квитанция заканчивается t10;
передано10Мбит. `settle` выключает радио только после whole step.
Нужно деление общего шага на границу operation end и проверка всего пути расхода
ресурса. Наличие until_sec в книге не доказывает соблюдение времени двигателем.

### 2. ParallelFleet.reset не сбрасывает мир

После step1с и reset(seed=1) world.now остаётся1с. Reset только переустанавливает
список agents; seed игнорируется. На обучении следующие эпизоды продолжают старый
мир. Также нет очищения agents после окончания, обработки options и обрезки dt
по оставшемуся горизонту. Это частичная обёртка, не verified PettingZoo implementation.

### 3. Snapshot не возвращает ленту измерений

note_truth→snapshot→step1→restore оставляет now0, но truth log содержит2 записи,
последняя t1. `_truth_log` не входит в snapshot. История, по которой SensorStream
строит delayed readings, больше не соответствует восстановленному миру. Даже если
stream фильтрует future samples, последующие ветвления получают загрязнённую историю.

### 4. DictEnv не проходит стандартный rollout writer

Вызов `rollout_episode` на MissionDict с wait policy падает TypeError в numpy
преобразовании dict→float. Новая observation surface не соединена со старым
путём генерации RL данных. Нужны recursive schema/serializer и end-to-end tests.

### 5. Численный кадр отличается от полноценного GCRS

Для r=[7000000,0,0] на2026-10-20T00Z, одинаковых EOP/UT1/TT сравниваются
`eci_to_ecef(r)` и ERFA c2t06a@r. Различие46011.24м.
Код frames.py прямо оговаривает legacy numerical frame под строкой GCRF.
Это не доказательство, что вся ошибка каждой дуги46км, и не требование вращать
любой существующий вектор как GCRS без миграции. Но импорт стандартных состояний,
смешение эпhemerides и claims о точности требуют явного frame contract.

## Ограничения, установленные чтением

- Default physics зависит от доступности ERFA: Sun/gast молча используют fallback.
  Фактический backend должен быть частью fidelity manifest. ERFA не указан отдельно
  как обязательная runtime dependency в pyproject; транзитивное наличие не контракт.
- Узел каталога сознательно вычисляется по Meeus для сохранения старых целей;
  силы/тень используют epv00. Требуется единый declared profile и versioned rebaseline.
- fidelity profile frame string до сих пор содержит iau1980, тогда как текущая
  рабочая функция использует2000A при наличии ERFA. Metadata требует сверки с кодом.
- Snapshot JSON всё ещё хранит внешний путь YAML. В нём нет полностью переносимого
  набора сценария, task registry, sensor streams и session continuation state.
- Geometry polygon проверяет размещение вершин в круге; strip — длину относительно
  diameter; stereo — разность скалярных off-nadir. Это упрощённые helpers, не полная
  модель многокадрового покрытия и базиса стереопары.
- MissionDict наблюдает clock/soc/contact/open_requests первого аппарата. Success
  даёт награду повторно на последующих шагах; действия не адресуют заявки.
- Load report принимает requests, но не моделирует их workload: создаёт идентификаторы,
  делает propagation-only и served_requests0. Длинный simulate_load также без станций.
- CI full —2 файла, всего union profiles7 из72. Наличие строки full не означает
  полный pytest; выявленный day-test отсутствует во всех профилях.
- SFT example — сокращённый список команд/on_time/unmet. Нет исходного observation
  context, полного времени/receipt identity и ready training manifest.
- Session report возвращает весь track/episode, frames растут без retention;
  episode epoch берётся из текущего orbit.epoch, summary события для всех заявок
  назначены первому actor и интервалу0→now. Это отображение статусов, не журнал
  фактически выполненных операций с точным actor/time.
- В physics tests Astropy transforms.get_polar_motion меняется прямым присваиванием;
  без fixture cleanup это может делать результат зависимым от порядка тестов.

## Пределы конкурентного вывода

Официальные страницы на дату аудита:
[BSK-RL API](https://avslab.github.io/bsk_rl/api_reference/index.html),
[Basilisk](https://avslab.github.io/basilisk/),
[STK](https://ansys.synopsys.com/products/missions/ansys-stk).
Они уже документируют развитые профильные возможности. Собственный replay не
заменяет независимый benchmark. Наша потенциальная специализация — надёжный
сквозной процесс mission→agent→verified data→evaluation с объяснимыми результатами.
Это гипотеза для сравнительных испытаний и клиентского пилота, не подтверждённая победа.
