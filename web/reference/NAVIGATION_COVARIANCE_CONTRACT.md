# B02-Q1: STM и прогноз Cartesian covariance

Дата: 2026-10-31. Область: optional navigation prediction в
`reference-dynamic-downlink/1`. Parent B02 остается `needs_external_evidence`.

## Модель и единицы

Состояние x=[r_GCRS, v_GCRS], единицы компонентов — метры и метры/секунду.
Матрица Phi(t,t0)=dx(t)/dx(t0) удовлетворяет Phi_dot=A(t) Phi, Phi(t0)=I.
A вычисляется на номинальной sealed Hermite-дуге: аналитический Jacobian для
центральной силы; двухсторонние конечные разности для полного perturbed ForceModel.
Во втором случае учитываются velocity partials вращающейся атмосферы.
Настройки шагов разностей сохраняются и проверяются на сходимость.

Covariance вычисляется как P=Phi P0 Phi^T+W; W_dot=A W+W A^T+G Q G^T.
G=[0;I] задает постоянную спектральную плотность белого ускорения в GCRS.
Q имеет единицы **m²/s³**, а не variance мгновенного ускорения. Блоки P:
Prr — m², Prv — m²/s, Pvv — m²/s². Нельзя складывать диагонали всех шести
компонентов в один физически осмысленный «общий sigma».

Формализм: [Tudat variational equations](https://docs.tudat.space/en/stable/user-guide/state-propagation/propagating-variational-simulation.html),
[NASA GEONS, state error covariance propagation](https://ntrs.nasa.gov/api/citations/20250006484/downloads/GEONSMSR3_1.pdf).
Использование этих уравнений не означает прохождения квалификации Tudat/NASA.

## Включение через сценарий

В каждом `definition.orbits.<actor>` опционален `navigation`:

```python
import numpy as np
from space_env.reference.covariance import NavigationSettings

navigation = NavigationSettings(
    frame="GCRS",
    initial_covariance_si=np.diag([100., 100., 100., .01, .01, .01]).tolist(),
    acceleration_psd_m2_s3=(np.eye(3)*1e-6).tolist(),
    impulse_velocity_cov_m2_s2=(np.eye(3)*.0025).tolist(),
).model_dump(mode="json")
```

Все числа здесь — пример синтетических допущений, не калиброванные параметры
аппарата. Нужны согласованные frame, symmetric finite PSD 6x6 и 3x3 матрицы;
неизвестные поля, bool вместо чисел и mixed-frame ввод отвергаются. Численный
PSD-допуск равен 1e-12 от max(1, spectral norm). Малые отрицательные roundoff
значения не обрезаются молча; существенно не-PSD результат вызывает ошибку.
Нулевые/сингулярные covariance допустимы. Матрица с положительной диагональю,
но недопустимыми cross terms также отвергается.

Если navigation задана всем актерам, runtime объявляет
`state.covariance_propagation` и `state.stm`. Для части актеров доступен
`state.covariance.partial_actors` и явный список navigation_actors. Требование
общей capability при частичной конфигурации отвергается, без неявного P=0.
Без navigation поведение предыдущего профиля сохраняется.

## Маневры, ресурсные дуги и доступ

Фиксированный GCRS impulse имеет state jump Jacobian I. В его эпоху независимо
заданная impulse_velocity_cov_m2_s2 добавляется только в Pvv. Глобальный STM
от исходной эпохи не сбрасывается. Для будущих RTN/body-pointed маневров это
правило применять нельзя; такие кадры пока не поддержаны.

Finite burn имеет известную детерминированную массу и постоянное инерциальное
направление. F/m(t) не зависит от r/v; force partials drag/SRP используют текущую
массу. Covariance/STM непрерывно переходят burn → coast. При cancel переносится
текущее P и глобальный Phi, будущая дуга пересчитывается, прошлое сохраняется.
Неопределенность массы, Isp, направления и момента включения этим не моделируется.

CovarianceArc принимает только sealed smooth GCRS arc. ForceModel и FiniteThrust
обязаны совпадать с manifest номинальной дуги; нельзя молча распространять
covariance для другой динамики. Внутри одной дуги нет impulsive discontinuity.

Agent observe содержит только **текущее** P своего доступного актера. Текущий
STM дополнительно доступен в privileged truth. Геометрический forecast не стал
неявным каналом будущей covariance/truth. Python provider API covariance(t)/stm(t)
предназначен для доверенного SDK, не является sandbox границей.

## Восстановление и ограничения вычислений

Операции продолжают использовать общие revision/idempotency/rollback правила.
Checkpoint самодостаточен: navigation settings сохраняются в original/effective
scenario, текущие covariance-наблюдения — в журнале. Prefix replay заново строит
дуги и сравнивает исходы; проверены восстановление посреди burn и последующая
отмена. Невалидный batch откатывает и covariance, и глобальный STM.

На каждой дуге отдельно интегрируются Phi и W методом DOP853 с dense output.
Применяются собственные rtol/matrix_atol/max_step_sec, шаг ограничен nominal
cadence; предварительно отклоняется более 100000 минимально необходимых шагов.
Дополнительно hard max_evaluations ограничивает число вызовов RHS, default
200000, допустимый максимум 1000000. Превышение дает явную ошибку и runtime rollback.
Это ограничения вычислений, а не RAM/latency SLA для группировки. Dense arrays,
перестройка горизонта и prefix replay остаются затратными на длинных миссиях.

## Проверки и границы применимости

24 новых теста включают:

- Phi против шести симметрично возмущенных орбит, допуски atol=5e-6/rtol=1e-6.
- Symplectic identity центральной гравитации; singular zero covariance.
- Q-discretization против аналитической constant-velocity модели: t³/3, t²/2, t.
- Рост P от PSD шума, сохранение симметрии/PSD и frame roundtrip.
- Все force switches в Jacobian и convergence шагов разностей/integration controls.
- Impulse Pvv jump, глобальная композиция STM, burn/coast и cancel continuity.
- Mid-burn portable checkpoint, rollback, capability guards, input and budget rejection.
- Отдельный vectorized central-gravity RK45 RHS с 12 симметричными sigma points.

Последняя проверка — deterministic population covariance, **не случайный Monte
Carlo**. Для синтетической 60-секундной дуги заранее задан rel Frobenius budget
1e-5 отдельно для Prr/Pvv; измерены приблизительно 5.97e-11 и 1.34e-11.
Это short-arc numerical consistency, а не калибровка uncertainty на flight data.
Численные значения и источники: `docs/acceptance/B02_Q1_NAVIGATION.json`.

Новая frozen дуга J2+Sun/Moon на другой эпохе/наклонении сравнивается с Boinor
формулами + LSODA. Общие Sun/Moon ephemerides, frames, EOP, constants перечислены;
результат не доказывает независимую достоверность этих входов.

Не реализованы измерения и EKF/UKF/filter updates, оценка состояния, clock/bias
states, mass/attitude covariance, cross-actor correlation, calibrated process
noise, негауссовская/долгая nonlinear uncertainty и collision probability.
Nominal state остается детерминированным: covariance сама по себе не создает
случайные фактические ошибки исполнения или noisy sensor observations.

## Воспроизводимый пример

```sh
python examples/unified_runtime/navigation_demo.py /tmp/navigation-demo
space-env mission-validate /tmp/navigation-demo/scenario.json
space-env mission-run /tmp/navigation-demo/scenario.json /tmp/navigation-demo/plan.json /tmp/navigation-demo/cli-prefix.json
space-env mission-resume /tmp/navigation-demo/cli-prefix.json /tmp/navigation-demo/resume.json /tmp/navigation-demo/cli-final.json
space-env mission-replay /tmp/navigation-demo/cli-final.json
space-env reference-qualify-orbits /tmp/physics-qualification.json
```

Последняя команда теперь включает covariance_prediction report. Общий
qualification_status остается needs_external_evidence, даже когда passed=true
для выполненных численных проверок. Независимые events/eclipse и калибровка
целевого промышленного профиля еще не закрыты.
