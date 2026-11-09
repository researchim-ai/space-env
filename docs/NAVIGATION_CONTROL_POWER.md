# M02-B2a: известные импульсы, причинные пакеты и native питание сенсора

Текущее расширение: [M02-B2b finite thrust и matched forces](NAVIGATION_FINITE_FORCES.md)
добавляет opt-in inertial finite-burn estimator и declared J2/drag/SRP/Sun/Moon.
Прежние запреты finite/perturbed profiles ниже исторические; неизвестные
execution errors, body-frame thrust и остальные ограничения не сняты.

9 ноября 2026. Расширение [sensor-aware задач](SENSOR_AWARE_TASKS.md): estimator
учитывает **выполненные собственные GCRS impulse commands**, delayed measurements
сохраняют исходное значение, а постоянно включённый сенсор потребляет энергию
в общем native балансе. Это следующий ограниченный профиль, не завершение M02,
не detailed thruster simulation и не доказательство превосходства над другими системами.

## Что теперь поддерживается

- `impulse` вместе с wait/downlink/processing/cancellation в sensor tasks.
- Central-gravity prediction оценённого состояния с known control history.
- Fixed latency/dropout/bias, в том числе пакет, приобретённый до манёвра,
  но полученный после него.
- Опциональная covariance jump для неопределённости исполнения команды.
- Continuous sensor electronics power и доступность fix по native battery SOC.
- Atomic advance, acquisition/reception boundaries, checkpoint replay и
  сохранение RL/SFT gates.

Ограничения сохраняются: `reference-dynamic`, central gravity, до восьми sensor
actors, horizon <=3600s, до 512 acquisitions на actor и 512 task decisions.
Это **config bounds**, а не измеренная работоспособность большой группировки
на часовой миссии. Квалификация текущего пакета — короткие fixed 5s scenarios.

## Прогноз фильтра использует оценку и команду, а не truth trajectory

Для выполненного собственного импульса на tc:

\[
\hat r^+=\hat r^-,\qquad \hat v^+=\hat v^-+\Delta v_{commanded},
\quad P^+=P^-+\begin{bmatrix}0&0\\0&\Sigma_{\Delta v}\end{bmatrix}.
\]

В GCRS fixed-vector случае Jacobian перехода по [r,v] равен I. Между импульсами
mean/covariance интегрируются по центральной гравитации на estimated state.
Из native command status берутся только actor, completed status, epoch и
заявленная delta-v. Истинные r/v и интегральная actual impulse vector не
подставляются в estimator. Отклонённая команда не применяется; при rejected
batch candidate полностью отбрасывается вместе с его возможными командами.

`SensorSpec.impulse_velocity_cov_m2_s2` — explicit finite symmetric PSD 3×3,
по умолчанию нулевая. Covariance добавляется один раз на control ID.
Это **допущение фильтра**: само поле не добавляет случайную ошибку тяги в truth.
Native ideal impulse исполняет принятую delta-v с расходом топлива по своему
контракту. Электрический budget engine/ADCS, finite thrust, mount/gimbal и
реальная ошибка исполнения этим не моделируются.

Filter posterior хранится на acquisition epoch последнего received measurement.
Applied control IDs у этого posterior могут отставать от текущей команды;
public current estimate прогнозируется вперёд со всеми уже известными controls.
При следующем измерении контролы применяются к posterior до нужной эпохи и
фиксируются в его cursor. Repeated observe/idempotent retry не добавляет impulse
или covariance повторно. Не трактовать `filter_controls=0` при задержке как
отсутствие импульса в текущем public prediction.

## Исправлена причинность delayed measurements

Ранее пассивный профиль мог запрашивать provider state в acquisition epoch
только при получении пакета. При манёвре граница орбитальной дуги стала бы
правосторонней: поздний запрос того же времени мог вернуть уже post-impulse
state. Для range/rate это изменяло бы физический смысл старого измерения.

Теперь noise, measurement value и known beacon state фиксируются **при acquisition**
и хранятся в private pending queue. Delivery только открывает сохранённый packet
и обновляет фильтр. Нет повторного вычисления старого noisy value по новой дуге.
Если acquisition совпадает с decision boundary, она обрабатывается при достижении
этой границы **до следующей внешней команды**. Нулевой импульс нового решения
происходит после уже зафиксированного measurement этой эпохи.

Pending packet также хранит snapshot известных этому измерению own control IDs.
При delayed update не используются команды, исполненные после acquisition,
даже если к delivery они уже известны. Затем current prediction применяет их
в правильном хронологическом порядке. Проверка включает range/rate packet
эпохи t=0, delivered в t=0.5 после импульса в t=0: значение совпадает с pre-burn
measurement и не заменяется новым range rate.

До `received_sec` packet не попадает в `observe`, Gym/PZ или LLM wire. Checkpoint
по-прежнему trainer-only. Cold replay восстанавливает acquisition queue из
рецепта и исполненных native переходов. Фиксированная latency гарантирует порядок
поступления; неизвестная/случайная latency и произвольные OOS updates не добавлены.

Advance делится на acquisition/reception events. Это позволяет проверять SOC
именно при acquisition и не переосмысливать прошлый ресурсный state на конце
большого шага. Пользовательский task revision остаётся одним atomic решением,
хотя внутри выполняется несколько native advances. Сохраняется численный допуск
временных границ 1e-10s. No-sensor tasks не получают этих дополнительных разбиений.

## Реальная цена постоянно включённого сенсора

```python
task["navigation_sensors"]["a"].update(
    continuous_power_w=3.0,
    minimum_soc=0.05,
    impulse_velocity_cov_m2_s2=[
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ],
)
```

При создании runtime из **копии** исходного recipe выводится native load:
`base_power_w_native = base_power_w_original + continuous_power_w`.
Default original base — тот же, что в VehicleResources. Пользовательские input
словарь и baseline не меняются; task checkpoint хранит original scenario +
sensor recipe, поэтому cold restore добавляет нагрузку ровно один раз.
Не добавляйте sensor power вручную в baseline второй раз.

Это не отдельный условный sensor-energy счётчик: нагрузка реально входит в
`energy_wh`, `consumed_wh`, `unserved_base_wh` и native balance error, конкурирует
с TX/processing и влияет на доступность передачи. Например, base=10W и sensor=3W
дают 65J consumed за 5s при отсутствии других нагрузок и достаточной энергии.
Разность с zero-sensor baseline за 2s составляет 6J.

Когда enabled power/SOC profile имеет SOC <= minimum_soc (с numerical tolerance),
новый fix помечается `dropped=true, unavailable="power"`, без value. Default
continuous_power_w=0 и minimum_soc=0 сохраняют ideal-free availability.
Power check действует на acquisition, не ретроспективно на receipt: пакет,
снятый до разряда, всё ещё может прийти позже по заданной ideal latency модели.

Это **постоянная электроника**, а не динамически выключаемый sensor: continuous
load продолжает запрашиваться и при dropout/низком SOC; недостаток отражается
в native unserved base. Модель не описывает inrush, standby modes, энергию одного
fix, duty-cycle controls или реальные IV curves. Расход не выделяется отдельным
аппаратным meter в ledger; component provenance находится в task recipe,
а conservation проверяется по общему native балансу.

Filter computation остаётся внешней численной процедурой без native CPU cost.
Не заявляются оплаченные light-time/RF transport measurements, visibility GNSS,
реальный receiver protocol или sensor scheduling. Эти части — следующая работа.

## Что осталось запрещено

Finite/body burns и perturbed-force estimator, injection и geometric pointing
masks. По-прежнему только safe `product_delivery` goal: exact orbit/relative/ADCS
residuals не возвращаются через goal observation. Публичная projection из M02-B1
сохраняет только estimated own orbit, covariance, received packet, own products
и ideal operational resource counters. Masks декларативные; native admission
может отклонить действие. Не объявляется непрерывная безопасность сближений.

## Квалификационная матрица и данные

```bash
python -m pytest tests/test_navigation_controls.py -q
python -m space_env.navigation.controlled_suite --output artifacts/new-controlled-sensors
python -m space_env.navigation.controlled_suite --audit artifacts/new-controlled-sensors
python -m examples.modeling.controlled_navigation_consistency \
  --output artifacts/new-controlled-consistency.json
python -m examples.modeling.controlled_navigation_consistency \
  --audit artifacts/new-controlled-consistency.json
```

Six cases × reference/wait = **12 native train rollouts**:

- powered-impulse: известный impulse + оплаченный sensor + доставка;
- rejected-control: неприемлемая delta-v отклоняется, затем доставка возможна;
- no-contact: недоступная ground station;
- power-exhaustion: tiny battery, потеря fixes и возможности передачи;
- uncertain-control: declared covariance jump;
- boundary-delay: два принятых impulse, latency>cadence, затем передача во время известного Ground outage [2,4)s.

Последний case оставлен отрицательным: предписанная политика поздно начинает
передачу. Это не ошибка filter и не повод ослаблять contact constraint.
Rejected-control case имеет successful delivery, но rejected transition:
он остаётся RL и **не проходит positive SFT gate**. Teacher scripted и не
оптимален; наличие ненужного для доставки манёвра в recipe не доказывает полезность
learned navigation policy или минимальный расход топлива.

Все RL/SFT exports проверяются audit против reconstructed episode training row;
лишний SFT для неподходящей траектории отклоняется. Trainer witness отдельно
показывает native energy ledger, battery, accepted impulses, received/pending
counts и privileged errors. Он не включается в messages/SFT input.
Никакого shared split registry или blind/OOD evaluation в этой небольшой suite нет.

## Статистическая проверка

Отдельные 64 заранее заданных seeds: independent Gaussian initial errors по P0
и независимые sensor noise streams, one orbit, 5s, known 0.5m/s inertial impulse
в t=0, latency0.5s, sensor3W. Native execution не содержит случайной ошибки тяги;
filter Q и impulse covariance для этой проверки нулевые.

Mean NEES имеет 6N nominal DOF; mean **первого** NIS каждого run — 3N.
Bounds задаются chi-square quantiles .005/.995 до результатов; alpha=.01
на каждую метрику. Не смешиваются correlated NIS внутри одной траектории.
Все ошибки/неуспехи сохраняются в planned denominator 64; passed требует полного
числа успешных runs и соблюдения native 65J/balance gates.

Это approximate EKF short-arc consistency, не доказательство Gaussianity,
наблюдаемости всех геометрий, flight calibration, accuracy на долгих миссиях
или superiority относительно других программ. Повтор этих seeds в wheel не
считается ещё 64 независимыми геометриями.

[Source-bound приёмка](acceptance/NAVIGATION_CONTROL_POWER_V1.json).

## Следующие крупные блоки

- Matched estimator для finite burns/perturbed forces и execution uncertainty
  с независимой проверкой, а не отключением текущих applicability guards.
- Управляемый sensor duty cycle, native CPU/resource claims и observation latency.
- Sensor-safe navigation/orbit goals и закрытое управление по estimated state.
- Unknown beacon/clock/bias states, OOS policy и независимые train/test орбиты.
- M03 continuous encounter events и сравнение с внешним simulator на matched inputs.

## Результаты приёмки

**554 различных локальных теста прошли**: 264 core/navigation, 87 LLM/data и
203 native dynamics/operations. В том числе 20 новых тестов этого пакета.
Полный repository suite не запускался.

Source и installed wheel отдельно воспроизвели 12 mission rollouts с native
ledgers и training-export audit. В каждой среде: 3 goal success, только 2
SFT-eligible trajectory и 12 RL episodes; rejected-control delivery success
не стал SFT. Все retained contact/power failures присутствуют в denominator.

64-run consistency: mean NEES=6.1611246536 при bounds
[4.9433549510, 7.1739733209], mean first NIS=3.6038092310 при bounds
[2.2700275581, 3.8472351105]. Все runs завершены, 65J native consumption и
energy-balance gates выполнены. Source и wheel consistency reports совпадают;
оба audit заново проверили 64 seeds. Это повтор одного experiment, а не новые
независимые геометрии.

342 Python/JSON wheel files совпали с checkout; receipt фиксирует 329 module
hashes и generator SHA. Dependencies использовались из текущего окружения.
Native source environment distribution metadata остаётся `space-env=0.24.0`,
а extracted wheel — `0.134.0`: episode/checkpoint hashes различаются по
provenance. Messages, events, metrics и native witnesses совпали; cold replay
проверен внутри каждой среды, cross-dependency checkpoint migration не заявлена.

Артефакты — `artifacts/navigation-controls-20261109/`. Внешних simulator comparisons,
real-model training, CI, commit/push этот пакет не выполнял.
