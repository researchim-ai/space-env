# Space-Env: план развития платформы моделирования и обучения агентов

Дата аудита: 25 октября 2026. Версия рабочего дерева: 0.69.0.
Базовый HEAD: `b6d509e5943b51aacfa2d7921a4c43c164bee6ec`; дерево содержит изменения пользователя, включая незакоммиченные API и Studio.

## 1. Назначение документа и правила исполнения

Это актуальный порядок разработки для следующего агента. Основание — чтение текущего кода; подробности в [аудите](audits/2026-10-25_READINESS_AUDIT.md). Старый [master plan](../SPACE_ENV_AGENT_MASTER_PLAN.md) остаётся каталогом направлений, но его предпосылка о версии 0.45.0 устарела. При конфликте порядка работ следовать этому документу. Существующий [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) описывает историю реализации и не подтверждает готовность продукта.

Обязательные ограничения:

- **Не выполнять git push, не публиковать релизы, не создавать PR и не отправлять изменения во внешние системы без отдельного поручения владельца.** Пушит владелец. Не переключать ветку, не сбрасывать и не перезаписывать его изменения ради исполнения плана.
- Сначала проверить локальные инструкции, статус дерева и актуальность описанных пробелов. Не переписывать существующее только ради новой архитектуры.
- Выполнять пакеты по зависимостям; не пытаться одним изменением реализовать весь космос. Допустимы независимые параллельные задачи.
- Отмечать выполненным только при наличии кода, проверок, примера и воспроизводимых доказательств. Число тестов, число env ID и номер версии не являются доказательством реализма.
- Сохранять старые импорты и учебные среды; при смене семантики наблюдений/наград выпускать новую версию среды, а не молча менять benchmark.
- По умолчанию всё работает офлайн, без платных сервисов и реальных команд космическим аппаратам.
- Не называть модель flight-grade, сертифицированной или подтверждённым цифровым двойником без независимых оснований.

## 2. Какой продукт строим

Общая настраиваемая платформа, в которой пользователь описывает аппараты, орбиты, наземную сеть, заявки, неопределённости и правила миссии; затем человек, оптимизатор или обучаемый агент управляет одной и той же симуляцией. Результат — исполненные операции, телеметрия, проверяемые ограничения, mission KPI и воспроизводимый набор данных.

Первый продуктовый вертикальный срез: **планирование и перепланирование гетерогенной группировки ДЗЗ с общей наземной сетью**. Он связывает реальные для этой постановки ограничения: видимость целей, переориентацию, энергию, память, пропускную способность, сроки доставки, приоритеты, отказы и новые заявки. Это выбранная продуктовая гипотеза, а не утверждение о подтверждённом спросе.

Целевые пользователи: исследователь RL/MARL, разработчик агентного планировщика, инженер наземного сегмента, команда оператора для обучения и проверки процедур. Начальный коммерческий результат: воспроизводимый пилот на согласованной с партнёром миссии, где можно сравнить подходы и объяснить ограничения модели.

Четыре уровня зрелости, которые нельзя смешивать:

1. **Учебная модель:** понятные допущения, корректная логика, небольшие примеры.
2. **Исследовательская платформа:** разнообразие сценариев, честный benchmark, reproducibility и обобщение.
3. **Пилотный продукт:** выбранная миссия валидирована в оговорённом диапазоне, надёжные SDK/API/Studio, измеримые KPI.
4. **Эксплуатационная интеграция:** клиентские данные, безопасность, процессы и независимая квалификация конкретного применения. Это отдельный проект, а не автоматическое следствие этапа 3.

## 3. Точка старта

Уже есть orbital/subsystem core, Fleet/Walker, миссионные планы, Gymnasium-задачи, простые эксперты и REINFORCE, JSONL/SFT-генераторы, интеграционные обёртки, карта/глобус и операторский пульт. Эти компоненты нужно сохранить и укрепить.

Критический путь: **корректность ресурсов и времени → независимая валидация → сценарии и честные наблюдения → полная миссия → benchmark/данные → сервис и пилот**. Улучшение 3D-моделей и увеличение каталога не заменяют этот путь.

## 4. Как вести реализацию

Создать `docs/IMPLEMENTATION_STATUS.md`: ID пакета, статус `todo/in_progress/blocked/verified`, зависимости, изменённые контракты, команды проверок, артефакты, ограничения. Начать со всех пакетов в `todo`; этот план ничего не объявляет реализованным.

Для каждого пакета обязательны:

- Публичный контракт и пример запуска на небольшом сценарии.
- Проверки важных инвариантов, интеграционный сценарий и сценарий отказа.
- Документация единиц, систем координат, времени, источников и ограничений.
- Совместимость или явная миграция; обновление evidence/status.
- Измеренная стоимость выполнения, если меняется горячий путь симуляции.

Предлагаемые артефакты: `validation/` для эталонов и отчётов, `benchmarks/` для неизменяемых спецификаций, `artifacts/` для игнорируемых больших результатов, `examples/` для коротких запусков. Итоговый manifest хранит commit, признак dirty tree и хеш diff/исходников, config/scenario/backend versions, seed streams, версии зависимостей, входные хеши и единицы. Один HEAD недостаточен для текущего незакоммиченного дерева.

## 5. Этап A — исправить фундамент

### A0. Воспроизводимый baseline и карта возможностей — P0

Работы:

- Инвентаризировать все env ID, action/observation spaces, defaults, seed semantics и termination; сопоставить README с кодом.
- Для каждой возможности указать `present/partial/missing`, уровень валидации и ограничения. Не ставить один общий процент готовности.
- Зафиксировать полный pytest, Gymnasium checker, запуск основных скриптов и Studio smoke. Проверить wheel из чистого каталога без корневого `viz/`, `configs/`, `data/`.
- Измерить время построения кампании, step, память и размер событий на фиксированном оборудовании.
- Отдельно перечислить optional backend/data, без которых тесты пропускаются.

Приёмка: повторяемый baseline с exit codes и machine-readable отчётом; нет потери пользовательских изменений. Зависимости: нет.

### A1. Единый учёт данных и ресурсов — P0

Точки изменения: `simulation/engine.py`, `simulation/fleet.py`, `mission/workplan.py`, `envs/campaign_env.py`, `simulation/state.py`.

Работы:

- Ввести единое хранилище продуктов/пакетов: ID, источник, размер в битах, время создания, deadline, приоритет, текущий владелец, delivery/drop status.
- Хранить объём в одной канонической единице; явно различать bit, byte, Mbit и MB. Мигрировать неоднозначный `buffer_mb`.
- Съёмка создаёт объём по длительности/режиму прибора только при выполненных условиях; память имеет ёмкость, политику переполнения и явные потери.
- Downlink ограничен доступным объёмом, интегралом скорости линии и длительностью реально открытой связи. Пустой передатчик не создаёт доставленные данные.
- Relay списывает данные отправителя и добавляет получателю атомарно; учитывать ёмкость, half/full duplex, потери и подтверждения в выбранном уровне модели.
- Разделить логический продукт и физические копии при репликации: доставка уникальных данных не считается дважды.
- Увязать питание/тепло/режим/длительность payload и radio с теми же операциями; исключить двойной учёт нагрузки.
- Δv-budget, масса топлива и модель тяги согласованы с профилем; для импульсного приближения явно описать, что конечная длительность и расход массы не моделируются.

Приёмка: инвариант `initial + generated = buffered + uniquely_delivered + explicitly_lost` для режима без реплик; отдельный accounting для реплик. Тесты пустой памяти, полного буфера, обрыва LOS посередине операции, цепочки relay→downlink, повторной команды и восстановления. План и исполнение используют одну семантику объёма.

### A2. Общий мир, время и жизненный цикл операций — P0

Работы:

- Ввести World/SimulationSession над существующими engines: epoch, monotonic simulation time, spacecraft registry, ground resources, event queue.
- Общий барьер времени для наблюдений/арбитража. Разные внутренние integration steps разрешены; видимые состояния должны соответствовать одному timestamp.
- Не останавливать индивидуальное время борта на его будущей задаче. Незавершённую задачу хранить как операцию в общем мире.
- Жизненный цикл команды: submitted→validated→queued→started→completed/failed/cancelled; ID, длительность, prerequisites, причина отказа.
- Конкурирующие команды и одинаковые timestamps разрешать детерминированно; очереди антенн, разворотов и payload — явные ресурсы.
- Разделить scheduler, command executor, physics и observers. Отказ валидатора не должен частично менять мир.
- Snapshot/restore/clone включает RNG, pending events, операции, данные, ресурсы и версии. Replay из событий; ветвление альтернативных планов.

Приёмка: одинаковые часы наблюдений всех бортов после шага; отсутствие пропуска событий на границе шага; snapshot→restore→continue совпадает с непрерывным прогоном по задекларированным допускам; одновременные image/downlink/burn дают объяснимый результат.

### A3. Физические контракты и исправление контекста — P0

Работы:

- Явно маркировать state frame, epoch/time scale, units; не называть любой вектор просто ECI.
- Сохранить TEME при импорте SGP4 и выполнить проверенное преобразование при переходе в другую frame; преобразовывать также скорости и covariance.
- Время-зависимые силы вычислять на времени каждой стадии интегратора. Для MSIS использовать геодезические координаты аппарата, а не выбранной станции.
- Политики для отсутствующих EOP, space weather, SPICE kernels: error/explicit approximation; никаких незаметных fallback для режима validation.
- Проверять конечность, диапазоны, положительность массы/ёмкости, валидность epoch и параметров интеграции.
- Отделить упрощённый профиль от reference backend; задокументировать форму Земли, вращение атмосферы и границы высот/режимов.

Приёмка: тесты смены станции не меняют аэродинамику того же аппарата; зависимости сил от времени корректны; reference cases TEME/ITRF и преобразования скоростей; неподдерживаемая область даёт явную диагностику.

### A4. Независимая физическая валидация — P0/P1

Работы:

- Двухтельная задача: энергия/момент, несколько эксцентриситетов, сходимость шага и multi-day drift.
- J2: secular rates и cross-backend траектории при совпадающих force models.
- Drag/SRP/third-body: независимые cases, единицы и отключение каждого эффекта; не сравнивать разные force models как будто это один эталон.
- Eclipse/penumbra, ground access AOS/LOS, off-nadir, slew, SoC и thermal: аналитические и reference fixtures.
- TCA: поиск корней/локальное уточнение сходимости, несколько минимумов, а не только ближайший sample на минутной сетке.
- Pc: covariance frames/PSD, hard-body radius, projection, применимость линейной encounter model, численная устойчивость и случаи, когда оценке доверять нельзя.
- Точность задавать по применению: требования для планирования ДЗЗ отдельно от требований оценки сближений.

Приёмка: `validation/report` с источником, версиями, конфигом, допуском, измеренной ошибкой и областью применимости. Допуски выбрать и обосновать до финального сравнения; не подгонять их под текущий ответ. Непрошедший профиль остаётся experimental. Сравнение двух обёрток одного алгоритма не считается полностью независимой проверкой.

## 6. Этап B — настоящий mission environment

### B1. Строгий Scenario DSL и реестр моделей — P1

Работы:

- Версионируемые Pydantic-схемы сценария с запретом неизвестных полей, cross-field validation и migration.
- Отдельно: bus, payload, propulsion, orbit, ground network, demands, disturbances, observation model, objective, fidelity profile.
- Уникальные instance ID: несколько Sentinel-like или одинаковых 6U в одной миссии; model ID не должен быть идентификатором борта.
- Детерминированный scenario generator с независимыми потоками seed для орбит, спроса, погоды, отказов, сенсоров и политики.
- Лицензии и provenance каждого параметра: published/derived/assumed/calibrated, источник, дата, uncertainty, unit. Отсутствующий параметр не выдавать за реальный паспорт.
- Канонический config hash, переносимые packaged resources и один источник конфигов вместо расходящихся копий.

Приёмка: сценарий без изменения Python-кода создаёт новую миссию; одинаковый scenario seed воспроизводится, разные seeds действительно меняют параметры; invalid YAML/NaN/дубликаты/отрицательные бюджеты отвергаются.

### B2. Наблюдения, неопределённость и ограничения действий — P0/P1

Работы:

- Отделить simulator truth, onboard measurements, наземную оценку и прогноз. Добавить noise, bias, latency, dropouts, timestamps и stale indicators.
- Убрать зависимость признаков от `required_items()` и `job.kind`, если они подсказывают правильное действие. Заявки/план можно показывать, но только как доступные оператору входные данные.
- Action masks строить из доступной информации/контрактов, а не из скрытого оптимального решения; рискованные и неизвестные исходы не скрывать автоматически.
- Награда оценивает результат миссии и стоимость, а не совпадение с keeper эвристики. Разные допустимые планы могут быть успешными.
- Разделить invalid command, rejected unsafe command, operational failure и catastrophic terminal state. Обычная опечатка оператора не всегда завершает миссию.
- Стабильные observation schemas, masked variable-size fleet, graph/set representations; time remaining для конечного горизонта.
- Правильные terminated/truncated, seed action spaces и воспроизводимость stochastic policies.

Приёмка: контрольный агент не получает скрытый ответ через obs/info/tool API; permutation tests на названиях и порядке бортов; несколько разных планов проходят по KPI; тесты stale telemetry и команды по устаревшему revision.

### B3. End-to-end ДЗЗ и rolling-horizon planning — P1

Работы:

- Заявки: point/polygon AOI, revisit, resolution/off-nadir, priority, release/deadline, delivery destination, quality, weather/cloud model с честной provenance.
- Аппараты: optical/SAR профили, геометрия доступа, duration, slew/settle, duty cycle, память, power/thermal и ограничения одновременных режимов.
- Станции: расписания, несколько антенн, полосы/скорости, setup/handover, outages, реальные интервальные reservations.
- Planner строит план и передаёт его executor; execution не получает награду только за принятие готового плана.
- Новые заявки, отказ станции, потеря связи и ошибка прогноза вызывают перепланирование с минимизацией изменений committed schedule.
- Базовые алгоритмы: FIFO/priority/earliest-deadline, greedy, rolling horizon; CP-SAT/MILP для малых экземпляров с timeout и feasibility report.
- Метрики: уникальный доставленный полезный объём, доля заявок в срок, задержка, качество/покрытие, стоимость связи, energy reserve, fuel, число нарушений.

Приёмка: одна 72-часовая миссия связывает заявку→съёмку→память→сеанс→доставку; 4–16 гетерогенных аппаратов, несколько станций, общая конкуренция и хотя бы одно возмущение. Физически невозможная заявка возвращает объяснимый unmet status. Масштабы здесь — целевые тесты, не уже достигнутые показатели.

### B4. Настоящий multiagent слой — P1

Работы:

- Сохранить single-agent централизованного диспетчера; добавить PettingZoo Parallel для одновременных решений и AEC только для действительно последовательного протокола.
- Агенты: борт, наземная сеть или операторская роль. Local observations, communication budget, задержки и ограниченное знание соседей.
- Явный арбитраж конфликтов, shared/team rewards и индивидуальные затраты; CTDE-state только для обучения/critic.
- Динамический состав флота, disabled agents, padding/masks, неоднородные capabilities.

Приёмка: API conformance tests, единое обновление мира на совместный шаг, инвариант к порядку словаря агентов, оценка новых размеров флота и отключения одного борта. Один Fleet-объект сам по себе не является MARL environment.

## 7. Этап C — benchmark, RL, SFT и LLM-агенты

### C1. Версионированный Space Mission Benchmark — P1

- Разделять train/validation/test по семействам миссий, географии, эпохам, моделям payload/сети и возмущениям; seed alone недостаточен.
- Хранить scenario fingerprints, split manifest и проверку пересечений, включая близкие дубликаты.
- Уровни: unit skill → coupled operations → multi-day mission → fleet scale → OOD/robustness.
- Сравнивать random, rule-based, classical optimizer, learned policy; oracle отдельно и явно с privileged state.
- Отчёты по mission KPI, safety, infeasibility, cost/runtime/memory, confidence intervals и вариативности по scenario/policy seeds.
- Показать failure cases; не агрегировать разные физические задачи в один непрозрачный success score.

Приёмка: фиксированная suite v1, reproducible baseline report, независимый replay результатов; улучшение подтверждено на отложенных семействах и не покупается ухудшением критичных ограничений.

### C2. Обучение RL/MARL — P1/P2

- Подключить устойчивую внешнюю training library через optional adapter; текущий REINFORCE оставить учебным baseline.
- Vector env, normalization, checkpoints, resume с RNG, evaluation-only runner, config/hyperparameter manifests.
- Masked discrete/parameterized actions, recurrent policies для частичной наблюдаемости, permutation-aware fleet policy; continuous control — только для соответствующих задач.
- Curriculum по сложности/неопределённости, а не по механическому увеличению списка env ID.
- Контроль reward hacking, comparison с оптимизатором на малых экземплярах, перенос на новые размеры/условия.

Приёмка: воспроизводимый train→checkpoint→resume→held-out evaluate, наблюдаемый learning curve и превосходство над объявленным baseline по выбранному критерию. Не обещать заранее обязательное превосходство RL над оптимизатором.

### C3. Dataset factory — P1

- Единая запись перехода: global episode/scenario IDs, time, obs schema, action+parameters, command result, reward components, constraints, terminal reason, provenance.
- SFT: state/task→structured tool call; план→выполнение→исход; объяснение только проверяемыми фактами и ссылками на события.
- Генерировать демонстрации оптимизатора/эксперта и человека, ошибки, восстановления, невозможные запросы, безопасные отказы и preference pairs с явным критерием выбора.
- Не считать шаблонные QA физической ground truth; отделить pedagogical text от simulation-derived и externally validated samples.
- JSONL для обмена, columnar формат для больших массивов, manifests/checksums, schema validation, dedup, split isolation, resume/atomic writes.
- Offline RL хранит behaviour policy и известные probabilities, если доступны; оценка вне поддержки данных имеет явные ограничения.

Приёмка: dataset build воспроизводится по manifest; каждая демонстрация проигрывается и достигает заявленного результата; нет test leakage и скрытой truth в agent inputs. Объём данных сам по себе не считается качеством.

### C4. LLM tool environment — P2

- Типизированные инструменты: inspect telemetry, query opportunities, propose plan, validate, simulate branch, reserve resources, submit/cancel command.
- Те же session/command contracts, что у человека и RL; LLM не получает специальные прямые изменения state.
- Structured outputs, bounded retries, budget/timeouts, malformed arguments, delayed results и audit trail.
- Разделять read, propose и execute; права сценария и режим simulation-only. Не сохранять скрытые рассуждения модели; сохранять действия и проверяемые объяснения.
- Метрики: выполнимость планов, factual consistency, tool correctness, mission outcome, cost/latency, recovery.

Приёмка: воспроизводимый agent harness с mock model для CI и optional внешней моделью; неизвестные параметры и небезопасные действия обрабатываются без изменения мира.

## 8. Этап D — новые классы миссий

Добавлять семейства после A1–A4 и B1/B2. У каждого: минимальная physics fidelity, observation/action contract, baseline, KPI, failure cases, источники и отдельный split. Перечень — backlog, не обещание одновременной реализации.

### D1. Наземная сеть и delivery SLA — приоритет 1

Распределение антенн между операторами, maintenance, стоимость, handover, weather-dependent availability и аварийная станция. Действия: booking, preemption, rate/mode, reroute. KPI: своевременная доставка, загрузка, справедливость и стоимость. Зависимости: A1/A2/B3.

### D2. FDIR и автономность борта — приоритет 1

Частично наблюдаемые отказы power/thermal/payload/radio/actuators, диагностика, safe mode, recovery и degraded mission. Faults должны иметь причинную модель, а не только произвольную метку. KPI: время восстановления, false alarms, потеря миссии и резерв энергии. Baseline: проверяемые rule/state-machine процедуры.

### D3. ISL/DTN и коммуникационные группировки — приоритет 2

Store-and-forward, contact graph, очереди, expiration, routing, ограниченные терминалы/pointing, outages и offered traffic. Отделить RF/optical link model от routing. KPI: доставленный goodput, latency/loss, fairness, energy. Геометрический маршрут без бюджета линии маркировать как geometry-only.

### D4. SSA, сопровождение и collision avoidance — приоритет 2, повышенные требования к валидации

Наблюдения сети сенсоров→оценка состояния/covariance→прогноз TCA/Pc→решение о манёвре→перескрининг. Несколько объектов, обновления CDM, uncertainty и ограничения координации. KPI: остаточный риск в заявленной модели, расход топлива, нарушение миссии и число ложных тревог. До A4 не позиционировать текущие учебные Pc как основание эксплуатационного решения.

### D5. Orbit maintenance, phasing и deployment — приоритет 2

LEO drag compensation/differential drag, сохранение фаз/плоскостей, ввод и замена аппаратов; отдельно GEO station keeping с подходящими возмущениями. Finite-burn, ограничения thrust/power и ошибки исполнения вводить по необходимости. KPI: slot error, fuel/lifetime, time-to-slot, safety margins.

### D6. Coverage/revisit и responsive tasking — приоритет 2

Обслуживание регионов, обнаружение событий, новые высокоприоритетные заявки, stereo/paired observations, heterogeneous payload cooperation. KPI: coverage, revisit distribution, information value, response/delivery time. Не подменять качество снимка одной геометрической видимостью.

### D7. Proximity operations, rendezvous и servicing — приоритет 3

Относительная динамика, sensors/navigation uncertainty, approach corridors, keep-out zones, abort, finite thrust, связь и 6-DoF там, где это требуется. KPI: docking/hold precision, fuel, violation rates. Нужен отдельный валидированный профиль; текущая учебная задача удержания не является симуляцией стыковки.

### D8. Disposal и end-of-life — приоритет 3

Пассивирование, deorbit/reorbit, оставшееся топливо, долгосрочная uncertainty атмосферы и соблюдение заданных сценарных правил. KPI: вероятность выполнения в модели, fuel reserve, time distribution. Нормативные требования подключать как версионируемые профили с проверенными источниками, не зашивать универсальные числа.

### D9. Дальнейшие направления — после пилота

GNSS/PNT, RF interference, hosted payloads, lunar/cislunar missions, interplanetary trajectories. Для каждого сначала отдельная спецификация и reference backend. Это не бесплатное расширение LEO-ядра: нужны другие силы, сигналы, навигационные измерения, задержки и benchmark.

## 9. Каталог моделей спутников

Расширять не количеством названий, а разнообразием механики:

- 3U/6U/12U CubeSat: ограниченные power, memory, attitude, propulsion optional.
- Optical EO: agility, off-nadir/quality, illumination/clouds, recorder/downlink.
- SAR EO: eclipse operation, высокие peak power/thermal/duty-cycle требования.
- LEO telecom/relay: терминалы, routing, storage, feeder/ISL contracts.
- GEO telecom: station keeping, pointing, eclipse seasons и lifetime profile.
- Servicer/target/debris: разные управляемость, observability и uncertainty.

Для каждой карточки: опубликованные факты отдельно от proxy assumptions, версия источников, uncertainty bounds, разрешённые режимы, example mission и validation status. Реальное имя допускается как reference profile; неизвестные характеристики не дорисовывать как факты. Добавление экземпляра не требует нового env класса.

## 10. Этап E — SDK/API/Studio, масштаб и эксплуатация

### E1. Надёжный session API — P1 после A2

- Public SimulationSession SDK; убрать чтение приватных `_jobs`, `_engines`, `_campaign` из UI/API.
- Session IDs, isolated state/RNG/resources, revision/optimistic concurrency, command idempotency keys и structured errors.
- Версионированные schemas/OpenAPI, request size/type/range limits, quotas/cancellation/timeouts для тяжёлых прогонов.
- Persistence, snapshot/recovery, background jobs, event stream с reconnect и backpressure.
- Auth/RBAC/tenant isolation до совместного доступа или deployment; локальный одиночный режим оставить простым.

Приёмка: две независимые сессии не влияют друг на друга; повторный запрос не дублирует импульс/снимок; рестарт не теряет committed state; concurrent stale command получает конфликт.

### E2. Studio как интерфейс объяснения — P1/P2

- На одной странице: 2D ground track, 3D, UTC timeline, queue, resources, mission KPI и command/event journal.
- Явно разделять выполненную траекторию, прогноз и альтернативную ветку; timestamp/frame/source на каждом слое.
- Карточка аппарата: modes, telemetry age, memory, SoC, thermal, fuel, current operation и ограничения.
- Scenario editor, pause/step/speed, branch/replay, manual command→validation→result, сравнение human/solver/agent.
- Безопасные масштабы: decimation/LOD, выбор объектов, uncertainty overlays, accessible colors/keyboard, empty/loading/error states.
- Профессиональная 2D-проекция и пересечение ±180°, проверка согласованности координат с backend; визуальное качество не означает точную геодезию.

Приёмка: оператор может понять, почему заявка не исполнена и куда ушли ресурсы; после команды UI показывает то же состояние и revision, что SDK; UI e2e и visual regression ключевых сценариев.

### E3. Производительность и масштабирование — P2

- Сначала profiler и complexity budgets; отдельно physics, access, screening, planning и rendering.
- Vectorized/batched propagation, spatial screening broad phase + validated narrow phase, кэш по полному model/config/input hash.
- Event-driven coarse steps вдали от событий с уточнением у границ, без потери fidelity guarantees.
- Независимые rollout workers, deterministic seed splitting, streaming datasets/logs, bounded caches.
- Целевые ступени испытаний: 4/16/64/256 аппаратов, 1/7/30 суток; 1000+ объектов сначала geometry/screening, затем по результатам измерений.
- Не считать число отрисованных маркеров масштабом полной физической симуляции. Публиковать hardware, fidelity, wall time, throughput, peak RSS, error и failure rate.

Приёмка: согласованные измеримые performance budgets и regression thresholds; оптимизированный режим сверяется с reference на контрольных случаях. Значения SLO утвердить по baseline и потребностям пилота, а не придумать в документации.

### E4. Engineering/release readiness — P1/P2

- Ruff/type checks постепенно с baseline существующих нарушений, unit/integration/property/metamorphic tests.
- CI: Python matrix, package build/wheel install out-of-tree, optional backend jobs, contracts, short e2e, benchmark regression; expensive validation отдельно.
- Dependency constraints/lock strategy, SBOM/license attribution, vulnerability scans, third-party data rights и reproducible offline fixtures.
- Structured logs/metrics/traces, run IDs, health/readiness, error budgets, resource limits и operational runbooks.
- Container для локального воспроизводимого запуска; production topology выбирать после реальной нагрузки, без преждевременного Kubernetes.
- Backups/restore, migrations, rollback, compatibility policy, versioned changelog и troubleshooting.

Приёмка: новый пользователь устанавливает wheel и выполняет пример без репозитория; задокументированный restore проверен; release checklist с фактическими evidence и открытыми ограничениями.

## 11. Этап F — довести до стартап-продукта

### F1. Подтвердить продуктовую гипотезу

С владельцем выбрать один ICP и пилот: кто пользователь, какая текущая боль, какие входные данные доступны, чем измеряется результат, с чем сравнивают и за что готовы платить. Интервью и доступ к данным требуют участия владельца; агент не должен выдумывать договорённости и отправлять письма самостоятельно.

Кандидаты монетизации: локальный SDK/enterprise license, закрытый benchmark evaluation, mission-specific validation/adaptation, dataset generation. Выбор — после customer discovery, не все модели одновременно.

### F2. Пилот и evidence package

- Согласовать scope, allowed assumptions, data rights, baseline и acceptance metrics до настройки модели под пилот.
- Выполнить ретроспективный replay или согласованный synthetic proxy; отделить calibration от независимого test.
- Показать качество решений, устойчивость, runtime, объяснимость и воспроизводимость; описать неудачи.
- Доставить runnable demo, SDK tutorial, model/benchmark cards, dataset sample, deployment guide и ограничения.
- Учесть onboarding/support, стоимость вычислений на сценарий и сопровождение модели.

Приёмка стартап-пилота: внешний пользователь самостоятельно запускает релевантную задачу, получает проверяемый результат и подтверждает полезность по согласованным критериям. Это не может быть доказано только кодом или внутренними тестами.

## 12. Порядок ближайших итераций для агента

1. **Первый пакет:** A0 + A1. Сначала regression tests на downlink из пустого буфера, conservation и relay; затем исправление модели и интеграция planner/executor. Не начинать с нового env ID.
2. **Второй пакет:** A2 + A3. Общий event clock, typed operations, snapshots, frame/time contracts и исправление atmosphere context.
3. **Третий пакет:** A4 + B1/B2. Validation report, scenario generator, observation integrity и KPI-based success.
4. **Четвёртый пакет:** B3 + C1. Сквозная ДЗЗ-миссия, оптимизатор и настоящие независимые splits.
5. **Пятый пакет:** B4 + C2/C3. Multiagent, воспроизводимое обучение и dataset factory.
6. **Шестой пакет:** E1/E2/E4 + F1/F2. SDK/API/Studio, упаковка и пилот; подготовка UI/CI может идти параллельно после стабилизации контрактов.
7. **Далее:** D1/D2, затем D3–D6; C4 и E3 по потребностям пользователей; D7–D9 только отдельными подтверждёнными пакетами.

## 13. Финальные ворота приёмки

**Research-ready:** resource/time invariants, воспроизводимые сценарии, честные наблюдения, reference validation для заявленных профилей, независимые splits, classical baselines, dataset provenance и стабильные APIs.

**Pilot-ready:** research-ready + одна сквозная миссия в согласованном диапазоне, isolated/recoverable sessions, переносимая установка, понятная Studio, measured runtime, customer acceptance specification и эксплуатационные инструкции.

**Production-ready для конкретного применения:** pilot-ready + подтверждённые требования клиента, независимая проверка применимости моделей, безопасность и процессы deployment/support, восстановление/наблюдаемость, договорённости о данных и ответственности. Универсального сертификата «симулятор всего космоса готов» нет.

## 14. Первичные источники для реализации

Ссылки являются отправными точками; версии и эталоны фиксировать при реализации.

- Gymnasium: [TimeLimit и различие termination/truncation](https://gymnasium.farama.org/api/wrappers/misc_wrappers/).
- PettingZoo: [официальная документация MARL API](https://pettingzoo.farama.org/).
- CelesTrak: [Revisiting Spacetrack Report #3, SGP4 reference](https://celestrak.org/publications/AIAA/2006-6753/).
- Orekit: [официальная документация](https://www.orekit.org/documentation.html) для независимых reference cases; конкретный backend не навязывается.
- CCSDS: [каталог стандартов](https://public.ccsds.org/Publications/BlueBooks.aspx); conformance декларировать только для действительно реализованного профиля.
