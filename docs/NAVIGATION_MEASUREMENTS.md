# M02-A: синтетические измерения и оценка орбиты

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

9 ноября 2026. Добавлен отдельный `space_env.navigation.NavigationSession`:
он использует реальные пассивные траектории native MissionRuntime и выдаёт
измерения/оценку вместо ideal own-orbit telemetry. Это первый ограниченный
профиль M02, **не завершение общей навигации, не интеграция с TaskSession/Gym/LLM**.
Существующие agent tasks сохраняют прежние ideal observations. Features
`navigation_measurements`/`state_estimation` в M01 task registry поэтому пока
остаются `not_implemented` для общего task contract.

## Два типа измерений

`cartesian_position`: синтетическая координатная fix в GCRS, размерность 3.
Это engineering GNSS-like position output, **не raw GNSS pseudorange**, не
модель приёмника, созвездия GNSS, clock/ambiguity или атмосферных поправок.

`range_range_rate`: дальность и относительная радиальная скорость от известного
идеального аппарата-маяка, размерность 2. Beacon state передаётся явно как
`known_reference_state_si` на момент измерения. Его ошибки/ковариации здесь нет.
Модель мгновенная: без light time, часов, относительности, media и signal tracking.
Геометрия с расстоянием менее 1m отвергается. Это не docking sensor.

Для rho=r-r_b, dv=v-v_b, u=rho/||rho||:

\[
h(x)=\begin{bmatrix}\|\rho\|\\u^T\Delta v\end{bmatrix},\quad
H=\begin{bmatrix}u^T&0\\
(\Delta v-u(u^T\Delta v))^T/\|\rho\|&u^T\end{bmatrix}.
\]

Noise — независимая нормальная ошибка с заданными positive std по каналам.
Для координат единицы std — m; для range/rate — [m, m/s]. `true_bias` — скрытое
постоянное смещение фактических измерений, которое фильтр не знает и не оценивает.
Это механизм synthetic misspecification, не калиброванные sensor errors.
Dropout — независимый Bernoulli; пропущенные пакеты остаются в истории без value.

## Оценка и неопределённость

Шестимерная оценка [r,v] и initial covariance задаются автором явно. Нет неявной
инициализации фильтра истинным состоянием. Estimate/ковариация распространяются
в центральной гравитации; Jacobian берётся из существующего covariance kernel
на **оценённой**, а не истинной орбите. Используется continuous covariance:

\[
\dot P=AP+PA^T+GQG^T,\quad G=[0;I].
\]

`acceleration_psd_m2_s3` задаёт изотропный белый шум ускорения оценочной модели;
это m²/s³. Само по себе Q не создаёт случайную actual force в truth dynamics.
В consistency study Q=0 и truth — deterministic two-body.

Обновление EKF:

\[
\nu=z-h(\hat x^-),\ S=HP^-H^T+R,\ K=P^-H^TS^{-1},\quad
P^+=(I-KH)P^-(I-KH)^T+KRK^T.
\]

Используется Joseph form и linear solve; singular innovation covariance
отвергается. Optional `nis_gate` отклоняет выброс по nu^T S^-1 nu; отказ не
удаляет измерение или NIS из истории. Covariance проверяется на finite,
symmetry/PSD. Невырожденный prior не означает наблюдаемость любой геометрии.

Формализм фильтра: [ESA GNSS Data Processing, Vol I](https://gssc.esa.int/navipedia/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf).
Наличие этих уравнений не означает ESA qualification или GNSS receiver fidelity.

## Причинность и задержки

Acquisition times — index*cadence от начала миссии. Packet доступен только при
received=acquired+latency <= committed current time (численный допуск 1e-10s).
Seed определяется actor/seed/index, поэтому разбиение advance не меняет noise.
`observe` не рисует новый шум, не потребляет samples и не изменяет фильтр.

Latency постоянна для sensor и известна. Measurements поступают в acquisition
порядке. Фильтр хранит posterior на эпохе последнего полученного измерения,
обновляется на acquisition epoch и прогнозирует оттуда к текущему времени.
Это поддержка фиксированной задержки, **не произвольные out-of-sequence arrivals**,
не filter rewind/smoothing и не оценка неизвестной задержки. Случайная/меняющаяся
latency требует отдельного профиля.

Truth берётся из native provider только в acquisition epoch уже полученного
packet, не из будущего snapshot. Native initialization может заранее вычислять
орбитальные дуги; они остаются внутри модели и не выводятся агенту.

## Публичный интерфейс и контрольные артефакты

```python
from space_env.navigation import NavigationSession

session = NavigationSession(scenario, {"a": sensor_spec})
view = session.observe("a")
result = session.advance(1.0, expected_revision=0, idempotency_key="sample-1")
packets = session.measurement_history("a")
```

`observe`/`advance` возвращают estimate, covariance, epoch/age и последний
полученный packet; `measurement_history` — все уже полученные packets, включая
пропуски. В public view нет own truth, true bias, фактической ошибки или NEES.
Истинный beacon state — явное допущение выбранного range profile, не скрытая
утечка состояния наблюдаемого аппарата. Доступны только actors с sensor config;
метод не возвращает произвольную телеметрию runtime.

`diagnostics(actor)` с error/NEES и `checkpoint()` **trainer-only**. Checkpoint
содержит scenario initial truth и sensor true_bias; его нельзя включать в prompt
агента. Session не является sandbox против Python-кода, читающего private поля.
Это явный интерфейс и политика разделения данных, не security boundary.

Advance транзакционный: native fork и копия фильтров коммитятся после получения
всех измерений и сериализации. Ошибка не меняет время, RNG sequence или posterior.
Idempotent retry возвращает прежний результат; конфликт key/request отвергается.
Cold restore проверяет source manifest, seal и повторяет переходы с самого начала.
Resealed подмена результата не заменяет этот replay.

## Область применимости

- Только passive `reference-dynamic`, central gravity у всех аппаратов.
- Только advance; manoeuvre, injection, mutable runtime и control loop здесь не exposed.
- До 8 sensor actors, горизонт до 3600s; это предел контракта, не квалифицированная
  производительность на восьми аппаратах/часовой миссии.
- До 512 samples на actor и 512 advance calls; cadence >=0.1s; budget проверяется
  заранее по native horizon/latency, без silent history truncation.
- Измерения не оплачивают power/CPU/RF, acquisition/visibility не ограничены.
  Это синтетический sensor/estimator baseline, не ресурсная модель GNSS/RF hardware.
- Нельзя объявлять произвольные perturbed trajectories/finite burns поддержанными:
  они отклоняются, чтобы центральный estimator не выдавался за согласованную модель.

## Проверка и данные

```bash
python -m pytest tests/test_navigation_measurements.py -q
python -m examples.modeling.navigation --output artifacts/new-navigation-study
python -m examples.modeling.navigation --audit artifacts/new-navigation-study
```

Нужна reference dependency group с SciPy. Examples запускаются из checkout;
сам navigation package и frozen inputs доступны из wheel.

Четыре fixed synthetic cases: nominal, fixed latency+dropout, known-beacon
range/rate и bias-negative. Все сохраняются; biased case не называется успешным
просто потому, что фильтр отработал без exception. Final range/rate accuracy
нельзя напрямую сравнивать с координатными fixes без учёта observability.

Ensemble: 64 заранее объявленных seeds, independent prior errors drawn from
заявленной P0 и отдельные sensor-noise streams, интервал 5s. Никакого отбора
удачных seeds. Для mean NEES используются chi-square bounds с 6*64 DOF;
для **первого NIS каждого независимого run**, а не коррелированных samples,
с 3*64 DOF. Alpha=.01 отдельно для каждой метрики. Это approximate EKF
short-arc consistency, не universal confidence theorem или flight calibration.

Report и trainer checkpoints содержат truth/config/labels и **не являются готовым
SFT corpus**. Public measurement fields отделены внутри report, но обучающий
export должен соблюдать то же разделение. В этом пакете LLM/RL не обучаются.

[Source-bound приёмка](acceptance/NAVIGATION_MEASUREMENTS_V1.json).
Полный repository suite и внешний simulator comparison этим не объявляются.

## Что дальше в M02-B

Интегрировать sensor/estimate projection в TaskSession, Gym/PettingZoo и LLM wire;
закрыть обходы через native goal errors, exact geometry/link features и masks.
Покрыть native controls и perturbed force models estimator prediction,
measurement power/CPU/cadence, clock/bias states и unknown beacon uncertainty.
После этого обновлять M01 feature availability для общего task contract.
Measurement truth должен оставаться trainer-only и в training exports, и в replay.

## Результаты текущей приёмки

**220 различных локальных тестов прошли**, включая 31 новый. Source и wheel
сформировали побайтово одинаковое каноническое содержимое study report:
4 fixed cases с cold replay и 64 ensemble runs. Оба audit отдельно повторили
64 seeds и проверили checkpoints. Повтор seeds в wheel не считается новыми
независимыми случаями. 338 Python/JSON файлов wheel совпали с checkout;
зависимости использовались из текущего окружения.

Mean NEES = 5.8328016188 при bounds [4.9433549510, 7.1739733209];
mean первого NIS = 3.6038092310 при bounds [2.2700275581, 3.8472351105].
Оба nominal gates passed. В retained bias-negative case ошибка положения
101.7525m и NEES 6402.53: этот случай не выдаётся за достоверную оценку
только из-за отсутствия runtime exception. Range/rate case имеет ошибку
20.89m, координатный nominal — 3.46m на выбранном seed; это разные измерения,
не сравнительный рейтинг сенсоров.

Артефакты: `artifacts/navigation-measurements-20261109/`. Receipt связывает
результаты с 325 Python-модулями и SHA генератора. CI, commit/push не добавлены.
