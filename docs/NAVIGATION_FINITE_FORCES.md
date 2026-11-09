# M02-B2b: конечная тяга и согласованные возмущения в estimator

9 ноября 2026. Расширение [known-control навигации](NAVIGATION_CONTROL_POWER.md).
Фильтр теперь прогнозирует **своё оценённое состояние** при постоянной инерциальной
тяге с номинальным расходом массы и opt-in возмущающих силах. Он не копирует
native trajectory/mass вместо прогноза. Это ограниченный matched-model профиль,
не независимая физическая калибровка и не завершение всей M02.

## Явное включение

```python
task["navigation_sensors"]["a"].update(
    control_profile="inertial_finite_burn",
    force_profile="matched_native",
)
task["model_requirements"] += [
    {"feature": "finite_burn_estimation", "actors": ["a"]},
    {"feature": "matched_force_estimation", "actors": ["a"]},
]
```

Default `control_profile="impulse"` сохраняет запрет finite burn;
`force_profile="central"` отклоняет perturbed truth у соответствующего sensor
actor. Модели не подменяются автоматически. `matched_native` использует declared
force config, EOP, epoch, propulsion constants и известные команды. Новые M01
features имеют `enabled` только при соответствующем actor profile; нет promotion
по модели другого аппарата. Registry теперь содержит 12 implemented признаков
и два explicit not-implemented: continuous encounter safety и docking contact.

Native constructor сохраняет проверки common SI epoch, weather/EOP validity,
propellant/thrust/horizon и frame/unit input. Sensor profile по-прежнему имеет
reference-dynamic backend, до восьми sensors, до 3600s horizon и 512 acquisitions
на actor / 512 task decisions. Эти contract bounds не являются измеренной
квалификацией больших группировок или часовых миссий.

## Уравнения и переменные

\[
\dot{\hat r}=\hat v,\quad
\dot{\hat v}=a_{declared}(\hat r,\hat v,t;m_{nom}(t))+
\frac{F(t)}{m_{nom}(t)}u_{GCRS},\qquad
\dot P=AP+PA^T+GQG^T.
\]

В каждом finite interval F и direction заданы командой. Направление фиксировано
в GCRS, ADCS считается идеальным в этом backend. Номинальный mass flow:

\[
\dot m_{nom}=-F/(I_{sp}g_0),\qquad
m^+_{nom}=m^-_{nom}\exp(-\|\Delta v\|/(I_{sp}g_0))
\]

для burn и impulse соответственно. Mass history рассчитывается **хронологически**:
потеря на burn до impulse тоже масштабируется последующим rocket equation.
Нельзя независимо вычитать суммарное топливо из исходной массы после всех
экспонент. Модель учитывает initial dry/propellant из recipe, known start/end,
completed inertial impulses и known cancellation; не читает actual mass provider.
Превышение nominal propellant envelope даёт ошибку, а не clamp к идеальной массе.

Mass — deterministic declared parameter, **не седьмое оцениваемое состояние**.
Её uncertainty/correlation не переносится в P, случайная ошибка исполнения
двигателя не создаётся. При фиксированных инерциальных F и u derivative thrust
по [r,v] нулевая. Jacobian A содержит partials заявленных gravitational/drag/SRP
сил на estimated trajectory. На известных control cutoffs solver разбивается,
чтобы не интегрировать одну smooth RHS через скачок тяги.

Default central/impulse путь продолжает использовать прежний fast predictor.
При explicit matched profile доступны native selectors J2, drag, SRP и
Sun/Moon third-body. Drag сохраняет native высотную область 150–800km и explicit
weather provenance. Квалификационный all-forces case использует table atmosphere;
NRLMSISE optional dependency и другие параметры не квалифицированы им автоматически.

Для matched profiles повторно используются `reference.orbit.acceleration`,
`state_jacobian`, Earth constants, EOP/frame и Sun/Moon inputs. Поэтому сравнение
с native — **model/numerical consistency**, а не независимая проверка этих
уравнений или исходных астрономических данных. Central finite-thrust mean
дополнительно сравнивается с отдельным fixed-step RK4 oracle из modeling suite.

## Управление, отмена и delayed measurements

В profile разрешены inertial `finite_burn` commands наряду с импульсами,
wait/downlink/processing/cancellation. Native продолжает контролировать
actor resource windows, thrust и fuel; estimator не разрешает отклонённую команду.
Body-frame/rotating thrust, slew, faults/injection и unknown execution не добавлены.

Native cancel status не хранит stop epoch отдельным полем, поэтому navigation
journal фиксирует **известное время собственной отмены** до следующего advance.
На будущих шагах оно не удлиняется до planned end. Natural completion использует
известный planned cutoff. Неизвестные failed/rejected execution states finite
burn не угадываются как полное исполнение: unsupported ambiguity даёт ошибку.
Электрический/тепловой actuator model самого finite-thrust engine здесь ideal;
расход топлива не означает оплату его реальных ignition/ADCS/thermal phases.

Момент состояния в начале/конце finite burn непрерывен. Delayed packets всё так
же фиксируются при acquisition и не вычисляются повторно после cancellation.
Own impulses сохраняют pre/post acquisition ordering и applied ID cursor.
Фильтр использует только произошедшие controls; unknown future policy не
подставляется в прогноз. Внешний пользователь не получает native future ephemeris.

Checkpoint внутри burn восстанавливает принятые команды, pending packets,
posterior и native power state через холодный replay. Task transition остаётся
atomic вместе с native executor. Попытки неподдерживаемых моделей отклоняются
до первого решения. Никакой arbitrary Python callback из task JSON не запускается.

## Публичные данные и цена измерений

Сохраняется allowlisted sensor projection: estimated own r/v, covariance,
received packets, own products и ideal operational counters. В public navigation
metadata указаны выбранные force/control profiles и deterministic nominal mass
assumption, **не actual mass/true state**. Numeric navigation encoding остаётся
47-column; профильные descriptors доступны в structured SDK/LLM views.

Geometric native goal errors и truth-derived masks по-прежнему запрещены;
поддерживается только safe product_delivery criterion. Known ideal beacon
сохраняет явно заявленное допущение предыдущего профиля. Clock/bias state,
неизвестная beacon uncertainty и произвольные OOS arrivals остаются открытыми.

Continuous sensor power из M02-B2a остаётся частью native base load. Этот пакет
не добавляет duty-cycle switching, цену отдельного fix, native sensor CPU queue
или реалистичный RF transport. Filter covariance не является calibrated sensor
uncertainty, особенно около non-smooth shadow/atmosphere boundaries или при
больших нелинейных ошибках. Длительные и сильно нелинейные profiles требуют
отдельной проверки, а не только уменьшенного solver tolerance.

## Проверки и воспроизведение

```bash
python -m pytest tests/test_navigation_finite_forces.py -q
python -m space_env.navigation.finite_suite --output artifacts/new-finite-navigation
python -m space_env.navigation.finite_suite --audit artifacts/new-finite-navigation
python -m examples.modeling.controlled_navigation_consistency --mode finite-j2 \
  --output artifacts/new-finite-consistency.json
python -m examples.modeling.controlled_navigation_consistency \
  --audit artifacts/new-finite-consistency.json
```

Unit/integration checks включают:

- Mean/P propagation без received measurements, все четыре force switches и
  отдельные комбинации, mean budget 1mm / 1e-6m/s против native на короткой дуге.
  Это **численный budget**, не operational accuracy реального аппарата.
- Central burn против отдельного RK4 oracle; J2 finite covariance против
  симметрично возмущённой mean map, отдельно Prr/Pvv, relative budget5e-6.
- Impulse→burn и burn→impulse mass order, cancellation, cutoff partition,
  mid-burn checkpoint и скрытый actual provider state/mass, недоступный predictor.
- Safe opt-in guards, invalid force fields, callbacks, common-epoch input guard,
  actor-specific model requirements.

Mission matrix: 6 fixed cases × reference/wait = 12 native rollouts. Central,
J2 и table-drag/SRP/Sun/Moon комбинация; cancellation; rejected over-thrust и
недостаток топлива. Четыре дополнительные preflight rejections проверяются
отдельно, **не считаются четырьмя выполненными физическими миссиями**.

Frozen mission inputs содержат известный Ground outage [2,4)s. Пауза передачи
в этот период — operational outage, не внезапная потеря геометрической видимости.
Сценарии наследуют одну orbital geometry; teacher заранее задаёт операции и
не считается оптимальным. Invalid command может сопровождаться последующей
успешной доставкой, но rejected transition исключает positive SFT. Все попытки
остаются в RL. Episode records/witnesses — trainer artifacts, messages не содержат
initial truth или hidden sensor bias. Shared corpus registry/OOD split нет.

Audit повторяет native traces, energy/mass witnesses и реконструирует training
exports. Nominal и actual mass сравниваются только в privileged witness; actual
mass не подаётся обратно estimator. Исторические receipts не переписываются.

Отдельный 64-seed study проверяет approximate NEES/first-NIS consistency при
matched J2 + finite inertial 0.5N/1s burn, latency0.5s и sensor3W за 5s.
Prior errors и sensor noise независимы; engine execution ideal, mass known,
Q/impulse uncertainty нулевые. Chi-square bounds и denominator те же, что в
предыдущем study: alpha .01 на каждую метрику, без отбора seeds. Повтор в wheel
не считается новой независимой геометрией. Это не uncertainty calibration,
flight qualification, competitor benchmark или learned-agent improvement.

[Source-bound приёмка](acceptance/NAVIGATION_FINITE_FORCES_V1.json).

## Следующие блоки

Управляемый sensor duty cycle/native CPU, sensor-safe orbit/relative goals,
неизвестные параметры двигателя и массы, body-frame thrust/ADCS coupling,
более длинные миссии и независимые орбиты. Затем M03 — continuous encounter events
и matched comparisons с внешними движками. Эти пункты не закрываются наличием
matched force flag или успешной пятисекундной дуги.

## Результаты локальной приёмки

**577 различных тестов прошли**: 287 core/navigation (23 новых), 87 LLM/data
и 203 native dynamics/operations. Source и extracted wheel отдельно выполнили
и cold-replayed по 12 physical rollouts и 4 preflight rejections. В каждой
среде: 6 delivery-goal success, но только 4 SFT-eligible trajectory; остальные
rejected-command successes сохранены только в RL. Всего 12 RL и 4 native SFT
exports на среду, wire-SFT 0. Наличие успешной доставки после invalid command
не маскирует эту ошибку политики.

Все native mass witnesses уложились в 1e-9kg nominal/actual error и energy
balance в 1e-12Wh. Это self-consistency ideal declared controls, не измеренные
погрешности реального двигателя. Сообщения, события, метрики и witnesses
совпали source/wheel; export rows реконструированы audit.

64-run finite-J2 consistency: mean NEES=6.1611246528 при bounds
[4.9433549510,7.1739733209]; mean first NIS=3.6038092310 при bounds
[2.2700275581,3.8472351105]. Все 64 runs завершились; native 65J/balance gates
пройдены. Source и installed consistency reports совпадают, оба audit повторили
64 seeds. Повтор не увеличивает число независимых орбит/профилей.

344 Python/JSON файлов wheel совпали с checkout; receipt фиксирует 331 Python
module hashes и generator SHA. Source distribution metadata по-прежнему
`space-env=0.24.0`, extracted wheel `0.134.0`: полные checkpoint/episode hashes
различаются provenance. Replay проверен внутри каждой dependency identity;
межсредовая миграция checkpoint и clean dependency install не заявлены.

Артефакты: `artifacts/navigation-finite-forces-20261109/`. Полный repository
suite, внешний comparator, real-model training, CI и commit/push не запускались.
