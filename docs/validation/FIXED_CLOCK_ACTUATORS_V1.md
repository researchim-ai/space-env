# D2e: joint RCS clock and rotating chemical-thrust verification

Дата:2026-11-04, source0.134.0. Synthetic verification hardware, не flight calibration.

## Новый совместный контур

При `RigidControllerSettings.clock_mode="fixed-tick/1"` momentum unloading
теперь удерживает **оба** согласованных сигнала: wheel motor torque u и external
RCS couple τ. До этого unloading пересчитывал свою обратную связь на каждом
native event boundary независимо от нового clock главного ADCS controller.
Legacy event-driven mode сохраняет прежнюю семантику и форму отчёта.

Пара вычисляется на global ticks kT, а также при начале control phase после
startup и при изменении rate-scale. Body-feedback и gyrostatic compensation
sampled одновременно с wheel unloading command. Instantaneous protection, SOC,
power, fuel/mass bounds, nozzle thrust и wheel capacity guards не отключаются.

`control_latches` хранит owned arrays отдельно от плана текущего substep.
Состояние участвует в native capture/fork/rollback и journal replay. Cancel/stop
очищает latch; завершение операции удаляет его перед следующей интеграцией.
Отдельная current telemetry `unloading.control_clocks`/`control_clock` показывает
mode `fixed-joint-wheel-rcs/1`, period, count и active-latch sample time.
Main attitude PD counter не выдаётся за counter отдельного unloading algorithm.

## Frozen native program

`qualify-coupled-actuators CONFIG OUTPUT_DIRECTORY` принимает CoupledMissionSpec
с fixed-tick flag. Конфигурация даёт геометрию; verification recipe явно заменяет
hardware/initial spin/fuel и нормализует period0.1s. Это не квалификация любой
переданной аппаратуры или подгонка её параметров.

Заранее заданы18 executions: nominal/limited RCS authority × RK maximum step
0.02/0.01/0.005s × caller partition0.1/0.025/0.01s. Designated reference каждой
mode —0.02/0.1; если она не выполнилась, другая успешная попытка её не заменяет.
Каждый member хранит scenario, prefix/final native checkpoints и report.
Ошибки остаются в progress/final denominator; result не перезаписывает output.

Программа без policy и admission oracle:

- два актора: rotating off-center chemical burn и unloading;
- startup/ignition0.137s, checkpoint/replay0.171s между joint ticks;
- controlled slew после первого burn в0.613s;
- второй burn на unload-акторе в2.137s проверяет **последовательное расходование
  общего lumped tank** двигателем и RCS;
- common samples0.171/0.337/0.613/1.137/2.137/2.537/3s.

Main engine — existing **passive rotating-body** model во время firing; после
него выполняется отдельная powered attitude command. Simultaneous active
attitude hold during engine burn этим не реализуется и не квалифицируется.
У post-burn slew нет заявленного обязательного settling success за короткое окно;
его native status сохраняется, не подменяется успешным engine completion.

Hardware assumptions: dry mass20kg + fuel0.1kg, inertia diag(2,3,4)kg m²,
wheel torque0.1Nm/h0.2Nms, ADCS supply12W. Engine0.01N, Isp200s,
lever0.2m; ideal balanced RCS, Isp60s, torque authority0.1Nm или0.02Nm.
Initial wheel momentum unload-актора0.08Nms; начальная body rate0.001rad/s
проверяет реальную feedback correction. Другой актор вращается вокруг body y
с0.02rad/s. Constant-inertia small-mass-loss limit0.001 сохраняется.
Lumped shared mass не доказывает совместимость реальных propellants/feed systems.

## Что проверяют gates

- Completion обоих engine commands и ожидаемый outcome unloading: nominal
  completed; limited failed с `unload_window_ended` и residual>0.04Nms.
  Ограничение max transfer≤0.02×(2−0.137) объясняет negative case заранее.
- Positive engine/RCS propellant expenditure, shared-tank mass bookkeeping,
  electrical/heat ledgers, rigid kinetic-work и angular-impulse balances.
- Balanced pairs имеют заявленный net force0; это ideal model property,
  не измерение nozzle misalignment.
- Один independent analytic axial check: body/GCRS y остаётся общей осью
  первого актора, поэтому τy=−0.2×0.01Nm и ∫τy dt=−0.0008Nms за0.4s.
- Native prefix and final replay; max common-grid differences при RK/partition
  variation. Bound values:1e−5deg,1e−7rad/s,1e−7Nms,0.01m,1e−5m/s,
  1e−5J DC,1e−6Wh,1e−4K,1e−8kg.

Это fixed-input numerical **envelope**, не оценка порядка/continuum error и не
полностью независимый physical oracle. Не заявляется conservation полной
химической/орбитальной энергии: проверяются конкретные declared ledgers.
Electric thrusters, реальная топливная система, flexible-body dynamics,
misalignment, plume/combustion и измерительная калибровка остаются отдельными.

## Extended service regression

`examples/coupled/extended.py` фиксирует600s,2 actors,24 requests, прежнюю
независимо заданную grid geometry и fixed clock. Это extended regression,
**не полный орбитальный цикл и не реальная наземная сеть**.

Nominal minimum1/24, minimum acquired1; all-outage minimum delivery0/24,
minimum acquired1. Оба режима исполняются direct/restored; prefix160s проверяет
continuation после первого acquisition/CPU/delivery участка. Все недоступные
запросы остаются в denominator. Source/data/replay/resource gates обязательны;
неудачу нельзя исправить уменьшением service minima после запуска.

```bash
# CONFIG — CoupledMissionSpec JSON, а не полный mission-study spec:
space-env qualify-coupled-actuators /tmp/actuator-config.json /tmp/actuator-check
space-env verify-coupled-actuators /tmp/actuator-check
# Здесь SPEC — полный study spec с development:
python examples/coupled/extended.py /tmp/fixed-clock-spec.json /tmp/extended-600s
```

## Офлайн-аудит пакета

`verify-coupled-actuators DIRECTORY` проверяет root digest, точный состав18
members, неизменённые budgets/recipe и полный inventory файлов. Для каждого
member заново компилируется frozen scenario; из native report пересчитываются
ledger/impulse/outcome witnesses, из common samples — численные сравнения.
Проверяются hashes checkpoint envelopes, согласованность progress/result,
отсутствие symlinks, duplicate JSON keys и NaN/Infinity. Чтение ограничено32MB
на JSON и256MB суммарно. Контрольная сетка должна совпадать с declared TIMES,
а не просто быть одинаковой у двух сравниваемых записей.

`valid=true, passed=false` означает целый пакет с сохранёнными неудачами.
CLI возвращает0 только при полной успешной проверке,1 при retained failed
qualification и2 при повреждённом/некорректном пакете. Hash не является цифровой
подписью; офлайн-аудит не повторяет native replay и не удостоверяет автора.
`native_execution_rerun=false` явно отражает это ограничение.

В CI добавлен отдельный weekly/manual installed-wheel job с сохранением всех
артефактов, в том числе failed attempts. Он не увеличивает обычную PR smoke
проверку и не считается выполненным локальным запуском.

## Следующий пакет

Full orbital contact cycles и positive-service benchmark на realistic independent
station deployment; explicit active engine-attitude hold/power interlocks;
controller period/gain model cards и measured calibration; независимые physical
oracles, Q6 ML benchmark и Q7 OS/job isolation. Эти задачи не закрываются короткой
actuator программой или zero-minimum outage case.

## Локальная приёмка — 2026-11-04

[B11_ACTUATORS](../acceptance/B11_ACTUATORS.json) связывает15 артефактов и
точный UTF-8 JSON archive со133 файлами. Wheel SHA
`87228be247b2abc411f2424f186187080d076a2f3dc27931ac1cfd95f19b224b`:
259 Python modules/292 source+data files и README byte-matched.
Установка вне checkout, Python-I; venv использует system-site-packages,
поэтому dependency-clean/remote CI не заявляются.

Все18 frozen members прошли conservation/outcome/replay и numerical gates.
Nine limited-authority cases сохранили ожидаемый `unload_window_ended`.
Max differences: attitude1.7121e−14deg, position7.581e−9m,
velocity1.782e−11m/s, wheel4.025e−16Nms, body rate2.320e−16rad/s,
DC9.238e−14J, battery1.024e−12Wh, temperature1.021e−8K, mass2.843e−14kg.
Офлайн-аудит прошёл без повторного native execution.

Установленный wheel проверен external six-process harness с тем же frozen recipe;
3 уже опубликованных serial members переиспользованы без изменения входов.
Избыточный serial run остановлен после завершения полной параллельной матрицы;
его partial artifacts сохранены отдельно и не считаются полной квалификацией.
Код harness сохранён в acceptance bundle. Это способ исполнения проверки,
не заявленная новая parallel API библиотеки.

Midphase direct/fresh-process pair прошла: prefix0.171s, engine phase firing,
joint sample0.137s, разные producer/consumer PID; final3.42s совпал.
Четыре600s исполнения прошли: nominal2/24 delivered при2 captures/CPU outputs,
all-outage0/24 при2 captures; direct/restored semantic signatures совпали.
Всего24 logical final executions и3 separate-process restores.

Reference restricted run:1705 passed,1 localhost PermissionError; этот HTTP test
отдельно прошёл с разрешением localhost. Fixed-actuator tests21 passed плюс
добавленный CLI contract1 passed; fast110 passed. Наборы пересекаются.
Это synthetic regression и fixed-input verification;2/24 — наблюдаемый service
данного сценария, не SLA, generalization score или готовность всех миссий.
