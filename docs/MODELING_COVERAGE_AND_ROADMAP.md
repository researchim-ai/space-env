# Покрытие моделера: возможности, глубина и программа сравнительной проверки

Текущее расширение: [M02-B2b finite thrust и matched forces](NAVIGATION_FINITE_FORCES.md)
добавляет opt-in inertial finite-burn estimator и declared J2/drag/SRP/Sun/Moon.
Прежние запреты finite/perturbed profiles ниже исторические; неизвестные
execution errors, body-frame thrust и остальные ограничения не сняты.

Текущее расширение: [M02-B2a known impulses и native sensor power](NAVIGATION_CONTROL_POWER.md).
Пассивные ограничения и цены измерений в исторических разделах ниже относятся
к прежнему этапу; текущий профиль поддерживает ideal inertial impulses и
continuous native power, но не finite burns/perturbed estimator или sensor CPU scheduling.

Текущее расширение: [M02-B1 sensor-aware tasks](SENSOR_AWARE_TASKS.md) уже
подключает passive measurements/EKF к TaskSession, Gym/PettingZoo и LLM.
M01 navigation features теперь enabled только при валидном sensor task recipe.
Исторические статусы/приёмки ниже относятся к своим исходникам; полная M02
и остальные комбинации моделей остаются открытыми.

Обновление M02-A: [passive navigation baseline](NAVIGATION_MEASUREMENTS.md)
добавляет отдельные measurements/EKF. Историческое описание пробелов ниже
не означает отсутствия нового standalone API; общая task integration ещё открыта.

Обновление: [M01-A modeling requirements](MODELING_REQUIREMENTS.md) реализует
первую часть registry/compatibility для восьми физических признаков. Остальные
задачи M01–M10 ниже остаются открытыми. Снимок аудита от начала дня исторический.

Дата исследования: **9 ноября 2026**, пакет 0.134.0. Название версии само по себе
не определяет состояние: [структурный снимок](audits/MODELING_COVERAGE_20261109.json)
фиксирует SHA-256 всех 321 Python-модуля и 17 проверяемых групп файлов.

## Вывод для проекта

**Наибольшая широта или превосходство над всеми системами пока не доказаны.**
Код уже охватывает много этапов спутниковой миссии; последние 19 scenarios —
только одна регрессионная матрица, не перечень всего продукта. Проблема —
не только отсутствующие модели, но и различная глубина, разные executors,
ограниченная совместная квалификация и отсутствие matched external comparison.

Цель: обеспечить широкую библиотеку **исполняемых, конфигурируемых и проверенных
миссий для AI**, в которой каждое физическое допущение явно, а агент действует
на доступных ему измерениях. Требование «больше всех» нужно разбить на проверяемые
утверждения по отдельным классам задач, физической точности, масштабу и данным.
Универсального корректного суммарного балла для этих свойств здесь не вводим.

## Метод аудита и смысл статусов

Проверены реализация ключевых kernels/schema/goal/adapter, контракты подсистем,
текущая source identity и официальные источники семи систем. AST-инвентаризация
охватила структуру всех Python-модулей, но **не является построчным семантическим
аудитом всех 321 файлов**, аудитом всех комбинаций или новым прогоном тестов.
Сравнение с другими системами — документальное, не экспериментальное.

У каждой возможности различаем четыре независимых вопроса:

1. Есть ли реализация и исполнимый пример?
2. Соединена ли она с выбранным native executor и agent interface?
3. Как проверена точность: analytic, numerical refinement, внешний движок,
   измерения? Что у сравнения общее с нашим кодом?
4. В каких пределах аппаратуры, орбит, длительности и нагрузки это проверено?

«Есть kernel», «есть Gym class», «адаптер импортируется» и «общая миссия
проверена» — разные статусы. `source_status=calibrated`, выставленный автором
recipe, сам по себе не доказывает калибровку. Совпадение replay подтверждает
исполнение выбранной модели, но не её сходство с реальным спутником.

## Фактическое покрытие

### 1. Орбиты, время и системы координат — native reference, ограниченный профиль

[orbit.py](../src/space_env/reference/orbit.py),
[dynamic.py](../src/space_env/reference/dynamic.py),
[frames.py](../src/space_env/reference/frames.py): GCRS/terrestrial transforms,
центральная гравитация, J2, сопротивление, SRP и возмущение Sun/Moon,
фиксированные и динамические эфемериды, managed maneuver arcs.
Drag reference profile явно ограничен 150–800 км. Источники атмосферы выбираются
явно; optional NRLMSISE требует установленной зависимости.

**Пробел:** native ForceModel не является произвольным harmonics/n-body solver:
нет конфигурации degree/order поля, самостоятельного универсального multi-body
центра притяжения и подтверждённых cislunar/interplanetary миссий. Sun/Moon как
возмущающие тела не равны полноценной лунной программе.

### 2. Двигатели и манёвры — native, с областью применимости

[thrust.py](../src/space_env/reference/thrust.py),
[actuation.py](../src/space_env/reference/actuation.py),
[unloading.py](../src/space_env/reference/unloading.py): импульсы, finite burn,
chemical/electric profiles, gimbal/ignition, расход топлива, power/heat,
rotating off-center thrust и оплачиваемая разгрузка маховиков.

**Пробел:** constant-inertia coupling допускает только заявленную малую потерю
массы; нужны ошибки исполнения, минимальный импульс, калибровочные кривые,
переменная инерция и изменение центра масс. Не выдавать surrogate двигателя
за модель всей силовой установки с баками, slosh и термохимией.

### 3. Ориентация — rigid kernel и native mission coupling

[rigid_attitude.py](../src/space_env/reference/rigid_attitude.py),
[rigid_bus.py](../src/space_env/reference/rigid_bus.py): quaternion dynamics,
полный тензор инерции, три ортогональных ideal wheels, saturation, torque,
angular-momentum witnesses, sampled controller и связь с миссией.

**Пробел:** произвольная геометрия/резервирование маховиков, magnetorquers,
подробный RCS, gyro/star-tracker errors, flexible appendages, jitter и
переменная сборка аппарата не доказаны этим профилем.

### 4. Относительное движение — новый native goal, не полная стыковка

[Контракт](RELATIVE_MISSION_GOALS.md): две реальные орбиты, timed relative
position/velocity, этапы, sampled keepout, правильная RTN-скорость. Для truth-relative
цели требуется роль, владеющая обоими аппаратами.

**Пробел:** измерения, потеря связи/навигации, непрерывная безопасная траектория,
approach corridor, docking/contact dynamics. Новые контрольные cases короткие,
5s; это не квалификация длительного удержания строя.

### 5. Навигация — covariance prediction уже есть

[covariance.py](../src/space_env/reference/covariance.py),
[контракт](validation/NAVIGATION_COVARIANCE_CONTRACT.md): STM, P propagation,
process-noise integration, maneuver covariance и frame transforms.

**Пробел:** генерация tracking measurements, EKF/UKF updates, clock/bias states,
observability, межаппаратная корреляция и статистическая calibration.
Covariance prediction не создаёт ошибки истинной траектории автоматически.
Называть всю навигацию отсутствующей было бы неверно; отсутствует замкнутый
measurement/estimation workflow.

### 6. Опасные сближения — есть legacy модели, нет общей гарантии native goals

[encounter.py](../src/space_env/core/encounter.py),
[probability.py](../src/space_env/core/probability.py),
[conjunction_env.py](../src/space_env/envs/conjunction_env.py): encounter/CW
propagation, B-plane Gaussian hard-body probability и отдельная assessment env.

**Пробел:** эти kernels не означают, что `relative_orbit` уже использует continuous
collision detection или Pc как критерий общего episode. Нужен мост с общей
ковариацией, frame/epoch semantics, safety events и проверкой missed crossings.
Не реализовывать Pc заново, не разобрав текущие предположения и ограничения.

### 7. Энергетика — часть исполнения миссии

[power.py](../src/space_env/core/power.py),
[operations.py](../src/space_env/reference/operations.py): батарея, потребление,
solar assumptions, native ledger; budgets связаны с goal success.

**Пробел:** подтверждённые IV/MPPT curves, charge/discharge efficiency по условиям,
старение батарей/панелей, температурная обратная связь и fault modes требуют
отдельных моделей/данных. Идеальное sun tracking нельзя смешивать с платной
реальной ориентацией под одним названием профиля.

### 8. Тепло — multi-node reduced model, не отсутствие тепловой физики

[thermal_network.py](../src/space_env/reference/thermal_network.py),
[materials.py](../src/space_env/reference/materials.py),
[контракт](validation/THERMAL_NETWORK_CONTRACT.md): lumped thermal network,
проводимость, radiation, heaters, temperature-dependent material assumptions,
sampled solar/IR/albedo inputs и latched excursions.

**Пробел:** CAD/FEA, spatially varying Earth radiance, полная взаимная radiation
геометрия и аппаратная thermal-vacuum calibration. Несколько unresolved crossings
в одном интервале не гарантируются. Повышать fidelity нужно там, где меняется
допустимое решение агента, а не только число узлов.

### 9. Наблюдение Земли — геометрия и происхождение продукта связаны

[acquisition.py](../src/space_env/reference/acquisition.py),
[area_coverage.py](../src/space_env/reference/area_coverage.py),
[observations.py](../src/space_env/reference/observations.py): footprints на WGS84,
AOI coverage, revisit/stereo evidence и привязка fulfillment к доставленному
продукту. `observations.py` здесь означает EO observations, **не навигационный
сенсорный стек**.

**Пробел:** terrain/occlusion, actual image formation/reconstruction,
registration и quality losses при processing. Геометрическое stereo evidence
не является фотограмметрической реконструкцией.

### 10. Optical/SAR — инженерные quality surrogates

[payload_quality.py](../src/space_env/reference/payload_quality.py),
[контракт](validation/PAYLOAD_QUALITY_CONTRACT.md): GSD/illumination/cloud policy,
SAR resolution/geometry/power applicability, пространственное quality evidence.

**Пробел:** calibrated SNR/MTF/PSF и raw images; SAR complex signal, speckle,
interferometry, pulse-energy storage и погодная RF-модель этим не реализованы.
Слово SAR в классе не подтверждает signal-level radar simulator.

### 11. Радиосвязь и наземный сегмент — native sampled engineering model

[radio.py](../src/space_env/reference/radio.py),
[radio_network.py](../src/space_env/reference/radio_network.py),
[booking.py](../src/space_env/reference/booking.py): link budget, discrete modes,
Doppler, occupied band, terminals, acquisition, storage, DC/thermal accounting,
receipts/ACK. Текущие RF-config bounds — до 64 actors и 128 stations.

**Пробел:** measured antenna maps, interference, rain/atmospheric models,
waveforms/BER/PER, protocol behaviour и shared-band contention с независимой
проверкой. Config bound не является подтверждённой производительностью.

### 12. ISL, custody и обработка — миссионная цепочка

[dtn.py](../src/space_env/reference/dtn.py),
[lifecycle.py](../src/space_env/reference/lifecycle.py),
[dependencies.py](../src/space_env/reference/dependencies.py): custody,
TTL/storage/relay, product lineage, processing dependencies и конечная доставка.

**Пробел:** calibration оптического/радио ISL, realistic processor workload и
аппаратные constraints. Передача application bits не равна моделированию
полного сетевого стека и waveform. Coupled миссии должны проверять competition
за память, power, pointing и terminals одновременно.

### 13. Изменяемые миссии и планировщики — есть интеграция, область ограничена

[sequence_lanes.py](../src/space_env/planning/sequence_lanes.py),
[sequence_recovery.py](../src/space_env/planning/sequence_recovery.py),
[workspace_dynamic.py](../src/space_env/api/workspace_dynamic.py): новые заявки,
recovery/prefix preservation, последовательности работ и операторские workflow.

**Пробел:** совместная политика для полного набора физики, неопределённости,
частичных продуктов и больших масштабов. Наличие scheduler не доказывает
оптимальность или устойчивость неизвестным отказам.

### 14. Агентные задачи, данные и оценка — наиболее перспективная специализация

[tasks](../src/space_env/tasks), [llm](../src/space_env/llm): общий физический
goal/reward, actor projection, replay, Gym/PettingZoo, RL и положительный native
SFT. Последний пакет: 19 cases, 38 runs, 238 local tests; receipt отдельно
фиксирует source identity и installed-wheel replay.

**Пробел:** learned-agent improvement, независимые геометрии, long-horizon/OOD,
согласованные external baselines, leakage-resistant семантические split и
единая оценка task diversity. Скриптированный teacher — не доказательство
оптимальности и не результат обучения LLM.

### 15. Масштабирование — отдельные профили нельзя суммировать

[scaling](../src/space_env/scaling), [sequence ladder](SEQUENCE_SCALE.md): process
workers, ensemble, restart/history/RSS tooling. Конкретный sequence ladder —
1–4 actors, 1–16 requests, 60–1200s, с сохранёнными неуспехами.

**Пробел:** объединённое исполнение полной физики на 32/128/512 аппаратах,
длительные миссии, throughput/memory curves, crash recovery всех workflow.
Числа в registry и максимально допустимые размеры не доказывают эти масштабы.

### 16. Внешние движки и каталоги — есть, но не универсальная интеграция

[integrations](../src/space_env/integrations), [catalog](../src/space_env/catalog):
Orekit/SPICE/SGP4/CCSDS adapters, TLE/profile sources и optional dependencies.

**Пробел:** одинаковый полный mission contract для произвольного внешнего
executor и native physics. Импорт адаптера не означает, что все native bus,
RF/thermal/maneuver/goal функции уже доступны на его основе. TLE реального
спутника не задаёт автоматически его реальные hardware характеристики.

## Что обнаружено у других систем

Изучены официальные страницы; это наличие документированных функций, не
независимая проверка качества, скорости или исчерпывающий аудит их кода.

- **Basilisk:** модульные dynamics/FSW, orbit/attitude, Monte Carlo, HIL;
  examples включают formation, sensors, optical navigation и multi-body.
  Серьёзный ориентир для coupled spacecraft physics и actuator/sensor fidelity.
  [Основная документация](https://avslab.github.io/basilisk/),
  [каталог примеров](https://avslab.github.io/basilisk/Learn.html).
- **BSK-RL:** уже предоставляет Gymnasium satellite tasking и PettingZoo
  constellation training. RL/MARL само по себе не наше уникальное преимущество.
  [Официальный API](https://avslab.github.io/bsk_rl/api_reference/index.html).
- **Orekit:** широкий flight-dynamics toolbox: frames/time, разные propagators,
  maneuvers, covariance и estimation/measurements. Проверять против конкретного
  релиза и набора assumptions. Основной сайт при исследовании отклонил запрос;
  доступен официальный API на `test.orekit.org`, его версия записывается при
  будущем эксперименте. [API overview](https://test.orekit.org/site-orekit-latest/apidocs/index.html).
- **Tudat:** multi-type/multi-body propagation, variational equations, observation
  models, estimation и mission-design tooling. Ориентир для расширения physics
  и отдельный потенциальный comparator.
  [User guide](https://docs.tudat.space/en/latest/index-user-guide.html).
- **GMAT:** mission/orbit/maneuver planning, optimization, navigation data,
  orbit determination и operations support. Не сводить к простому propagator.
  [NASA overview](https://etd.gsfc.nasa.gov/capabilities/capabilities-listing/general-mission-analysis-tool-gmat/).
- **STK:** reference для coverage, RF/environment и связанных наземно-космических
  сценариев; коммерческий продукт, этот аудит его не устанавливал.
  [Communications](https://help.agi.com/stk/Content/comm/CommRadar.htm),
  [Coverage](https://help.agi.com/stk/content/cov/coverage.htm).
- **FreeFlyer:** mission analysis и orbit determination с tracking/estimation;
  полезный reference для профессионального workflow.
  [Features](https://ai-solutions.com/freeflyer-features/),
  [Orbit determination](https://ai-solutions.com/_help_Files/orbit_determination.htm).

**Вывод:** открытые системы уже сочетают богатую физику и агентные интерфейсы.
Потенциальная сильная сторона нашего проекта — связать разнообразные операции,
физические цели, права наблюдения и проверяемые LLM-данные в одном workflow.
Уникальность этой комбинации и её полезность требуют проверки, а не заявления.

## Единая программа разработки: крупные исполнимые пакеты

Все пункты ниже — **будущая работа**. Не закрывать пункт только документацией,
новым env ID, smoke test или очередным примером на идеальных входах. Поддерживать
один mission runtime/goal/data contract и явные профили fidelity.

### M01. Контракт покрытия и матрица совместимости — первым

Развить текущий структурный inventory в versioned capability registry:
backend × orbit profile × hardware × sensor × mission × actor access × goal.
Разделить `supported`, `unsupported`, `experimental`, `qualified_in_envelope`.
Unsupported комбинации отклонять до запуска, не заменять ideal defaults.
Связать каждую клетку с recipe, диапазоном параметров, examples, tests и receipt.

Приёмка: не менее одного положительного и отрицательного native исполнения
каждой объявленной связки; проверка установленного wheel; явное отличие kernel
от интегрированной подсистемы. Генерация документации из registry без второго
несогласованного каталога. Current AST inventory этот пункт ещё не закрывает.

### M02. Измерения и оценка состояния — приоритет P0

Добавить range/range-rate, угловые и GNSS-like measurements с кадрами, временем
эмиссии/приёма, cadence, latency, noise/bias, dropout и seed. Truth остаётся
скрытым; будущие измерения не доступны агенту. Начать с documented reduced models,
а не обещания полного GNSS receiver. Подключить EKF и существующие STM/covariance;
разделить uncertainty оценки и stochastic truth/исполнения.

Приёмка: аналитические measurement/Jacobian tests, PSD/singular handling,
fixed-seed replay, checkpoint RNG/filter state, delayed/out-of-sequence policy,
NEES/NIS/coverage с заранее заданным статистическим протоколом и несколькими
независимыми seeds. Номинал, bias, dropout, ошибочная начальная оценка и плохая
геометрия обязаны оставаться в знаменателе. SFT не содержит privileged truth.

### M03. Непрерывная безопасность сближений — приоритет P0

Соединить текущие encounter/Pc kernels с dynamic mission. Добавить поиск
событий внутри committed дуг, конечные размеры, подход/уход/abort corridor,
uncertainty semantics и joint covariance при наличии correlation.
Выбрать и документировать event localization или conservative interval bound;
если solver не разрешил интервал, возвращать unresolved/failure, не safe=true.

Приёмка: пересечение keepout при безопасных концах шага, касание, multiple
crossings, манёвр внутри дуги, осевой вырожденный B-plane и correlated cases.
Сходимость по шагу/точности, сравнение с отдельно реализованным oracle,
разные safe/unsafe профили и native replay. Probability model не объявлять
абсолютной гарантией или полноценным международным operational assessment.

### M04. Расширенные орбиты и длительные миссии — P1

Явные levels force fidelity, higher-degree gravity с provenance coefficients,
body/ephemeris abstractions и frame consistency. Earth profiles улучшать до
переключения на лунные/межпланетные. Внешние движки использовать как явно
обозначенные reference backends/поставщики данных, а не скрытый заменитель модели.

Приёмка: two-body invariants, J2 secular effects, convergence, независимые
force vectors; сравнительные orbit/energy/mass/event residuals на одном витке,
сутках и неделе там, где профиль это поддерживает. Cases за областью действия
отклонять. Cislunar/n-body задачи выпускать отдельной проверяемой областью,
не выводить автоматически из наличия Sun/Moon terms.

### M05. ADCS, propulsion и hardware diversity — P1

Произвольная wheel allocation, gyro/star tracker/magnetometer reduced models,
magnetorquer/RCS по приоритету миссий, исполнение с bias/delay/failure.
Для значимой потери массы — переменная масса/инерция/центр масс, а не снятие
старого small-mass guard. Flexible/slosh dynamics — последующий отдельный profile.

Приёмка: momentum/energy balances, saturation/reconfiguration, noise/bias
replay, detumbling, pointing, burn alignment и safe-mode recovery. Published
точки, synthetic assumptions и measured calibration разделены. Проверять
применимость профиля ко всему аппарату, не только к купленному компоненту.

### M06. Полезная нагрузка и продукты — P1

Два явных режима: быстрый engineering surrogate и более дорогой sensor model.
Оптика: SNR/MTF/PSF/smear, terrain/occlusion по необходимости. SAR: согласованная
геометрия/энергия/resolution; signal-level radar только отдельной заявленной
реализацией. Продукты processing действительно меняют quality/coverage/lineage.

Приёмка: независимые footprints/rays/quality vectors, видимые эффекты jitter,
ночь/облака/ошибка ориентации, missed scene, невозможный продукт; каждый результат
агент получает только после native acquisition/processing/delivery. Synthetic
картинки маркируются и не подменяют измеренные данные.

### M07. Связь, сеть, энергия и тепло в одной миссии — P1

Расширить RF patterns/environment/interference и network congestion без потери
custody/ACK. Согласовать band/time/power/pointing/thermal/CPU/memory competition;
деградацию и MPPT вводить с источниками параметров. Устранить комбинации,
которые сейчас возможны только в отдельных demonstrations.

Приёмка: одновременно imaging, processing, ISL, ground TX и maneuver;
сохранение bits/products/energy/mass, отсутствие двойного расхода ресурсов,
отказы/перепланирование без потери частичных продуктов. Отдельные physically
infeasible сценарии и refinement cadence study. Waveform accuracy не заявлять
по одному link-budget сходству.

### M08. Библиотека больших миссий и масштаб — P1/P2

Общий генератор operational recipes: EO/revisit/stereo, relay/communications,
formation/inspection, conjunction/avoidance, maintenance, disposal, anomaly
recovery. Все используют согласованные профили из M01, а не отдельные мини-env
с другими упрощёнными правилами успеха.

Scale ladder как **цель будущих измерений**: 1/8/32/128/512 actors, разные stations,
requests и mission durations. Нынешние лимиты не снимаются до изменения моделей
и измерения ресурсов; неподдерживаемые клетки помечаются явно. Изолированные
ensemble workers не выдавать за одну взаимодействующую группировку.

Приёмка: paired workloads, wall time, p50/p95 decision latency, peak RSS,
throughput, simulation/real-time ratio и completeness. Сохранять OOM/timeout/
unsatisfied задачи. Restart после падения процесса, bounded history/storage,
валидный source-bound installed-wheel replay. Не обещать preset размер по одному
быстрому ideal-orbit case.

### M09. Реальный сравнительный benchmark — начать спецификацию сразу

Три дорожки без смешивания критериев:

1. **Физика:** matched gravity/ADCS/actuator cases против применимых Basilisk,
   Orekit/Tudat/GMAT reference; external executable должен действительно работать.
2. **Миссии:** одинаковые initial state, hardware constraints, demand, horizons,
   observability, deadlines и budgets; фиксированные baseline policies и solver.
3. **Обучение агентов:** единый data/token/compute budget, train/test geometries,
   seeds, blind evaluator; качество после SFT/RL, generalization и safety.

Для каждого эксперимента фиксировать release/commit/options и licences данных,
constants/frame/EOP/ephemerides, shared dependencies, hardware, tolerance,
planned denominator и что невозможно сопоставить. Модель с дополнительной
физикой не считать менее точной за отличие от упрощённой reference. Дорогой
simulator нельзя объявить медленнее без matched fidelity.

Приёмка: один воспроизводимый публичный comparison bundle на дорожку, включая
все ошибки/неуспехи, raw outputs и инструкции запуска человеком; независимое
повторение. Где доступа/лицензии нет, писать `not measured`, не оценку «хуже».
Превосходство заявлять только для измеренного показателя и области эксперимента.

### M10. Публичное доказательство и зрелость — после соответствующих измерений

Документация fidelity/profile matrix, turnkey examples и локальный acceptance
runner; packaged data и reproducible install. Поддерживаемые платформы проверять
реальными запусками. Published datasets с provenance/licensing/splits и model
cards; внешний пользователь должен повторить задачи и результаты.

Приёмка: независимый человек запускает полный путь scenario → native mission →
agent episode → SFT/RL → benchmark без ручной правки исходников. Ни один блок
не требует CI: согласно решению проекта используются локальные тесты/аудиты.
Commit/push/publishing остаются действиями пользователя.

## Порядок выполнения и критерий «достаточно широко и глубоко»

Сначала M01 и спецификация M09, затем M02/M03; M04–M07 развивают физические
профили; M08 проверяет объединённую систему, M09/M10 подтверждают результат.
Навигация/безопасность нужны для достоверных close-proximity задач раньше
расширения каталога ещё десятками env IDs.

Функция готова к публичному strong claim только при наличии одновременно:
recipe с limits, native integration, positive/negative cases, независимого
physical comparison по необходимости, agent/data observability tests и
доказанной рабочей нагрузки. «Самые широкие» требует заранее зафиксированной
таксономии задач и одинаковых условий для всего выбранного набора конкурентов.
Этот аудит такой claim не подтверждает и не подменяет будущий эксперимент.

## Как обновлять аудит

```bash
python -m examples.validation.modeling_inventory \
  --output docs/audits/MODELING_COVERAGE_20261109.json
```

Скрипт проверяет наличие reviewed modules, индексирует AST и сравнивает hashes
двух goal receipts с текущим кодом. Он не запускает физику и не аттестует
подсистемы. В текущем снимке relative receipt совпадает с исходниками;
предыдущий physical-goals receipt исторический и имеет другой source manifest.
Это ожидаемое различие, не основание задним числом менять старую приёмку.
