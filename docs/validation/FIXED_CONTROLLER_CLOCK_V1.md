# D2d: latched controller clock

Дата:2026-11-04, source0.134.0. Explicit opt-in digital sample-and-hold model;
не новая измерительная калибровка flight controller.

## Контракт

`RigidControllerSettings.clock_mode="fixed-tick/1"` включает удержание motor
command между тактами. `event-driven/1` остаётся legacy default и не добавляет
новое поле в прежнюю сериализацию. Coupled compiler принимает
`controller_clock="fixed-tick/1"`; остальные scientific settings не меняются.

Для периода T global ticks равны kT. Первый положительный advance после новой
команды немедленно вычисляет момент по текущим q/ω и доступной цели. Далее
момент удерживается до следующего tick. Новая команда/смена режима или
rate-scale прерывает удержание. Численное совпадение с tick использует существующий
1ns boundary tolerance; произвольная точность wall-clock не заявляется.

Protection проверяется на каждом native prepare независимо от clock: safe mode,
fault, SOC и power guards могут немедленно отключить управление, обнулить момент
и очистить latch. Отмена также прекращает питание при следующем положительном
advance. Готовая ориентация при admission не присваивается; управление остаётся
оплачиваемой динамикой. Motor limits и momentum saturation применяются existing
rigid plant внутри integration, даже когда command held. Unused supply становится
heat по существующему conservative electrical envelope.

Внутренний latch содержит tick, command/mode/rate token и owned torque. Current
telemetry показывает samples count, last sample time и trigger. Это ideal current
actuator telemetry, не прогноз или hidden mission oracle. Latch и счётчик входят
в native capture/rollback/fork; checkpoint восстанавливает их audited journal replay,
а не десериализацией произвольной памяти.

External torque обновляется независимо от held motor command. Flight-valid
rotating-engine/unloading missions на новом clock не квалифицируются только тем,
что legacy regression проходит; для них нужны отдельные profiles/receipts.

## Две разные проверки

1. `qualify-controller-clock SCENARIO OUTPUT_DIRECTORY` выполняет одинаковую
   native программу при caller partitions0.1/0.025/0.01s. Commands не выбирает
   policy: два актора получают static inertial slew и wheel torque, затем
   cancel/recommand в0.137/0.263s, checkpoint/replay в0.211s и safe-mode interrupt
   в0.41s, финал0.6s. Для каждого partition сохраняется final native checkpoint,
   replay и conservation witness. Ошибки каждой попытки остаются в denominator.
   Это partition invariance, не mission SLA, не independent numerical truth.
2. `examples/coupled/study.py --fixed-clock` фиксирует13 mission executions:
   kernel resolution, различающиеся digital periods, event-check resolution и
   direct/restored regression geometry cases. Здесь `require_event_stability=True`
   делает budget событий обязательным, без повышения прежних tolerances.

Разные digital periods0.2/0.1/0.05s — разные sampled controllers: invariance между
ними не обещается. Общий прежний strict temporal release gate требует обеих
sensitivity families и не подменяется новым отдельным event gate.

```bash
python examples/coupled/study.py /tmp/fixed-clock-spec.json --write-spec --fixed-clock
space-env mission-study /tmp/fixed-clock-spec.json /tmp/fixed-clock-study
space-env verify-mission-study /tmp/fixed-clock-study
# Передать coupled fixed-tick scenario, например из frozen execution manifest:
space-env qualify-controller-clock /tmp/fixed-clock-scenario.json /tmp/clock-program
```

Новый профиль использует прежние geometric regression cases, а не новый untouched
holdout. Их zero minimum service не превращает successful execution в SLA.
Дальнейшие independent feasible long-horizon service и C2/Q6/Q7 открыты.

## Приёмка 2026-11-04

[Sealed receipt](../acceptance/B11_FIXED_CLOCK.json), финальный wheel SHA256
`062b7b4d9bd057d98b8e7fe0f579c4f612bcaadfa2daeacbbff2728b01718783`.
258 modules/291 source-data files и README совпали с wheel. SDK verifier владеет
копией входного сценария и возвращаемым результатом; изменение report budgets
вызывающим кодом не меняет допуски будущих проверок.

**1685 reference passed** одним запуском (441.36s,15 upstream warnings),154
профильных и110 fast passed; наборы пересекаются.14 новых tests покрывают hold,
command/protection interrupts, sample timing, ownership, rollback/fork/replay,
saturation, failed-attempt denominator и mandatory same-schedule event gate.
Expanded full suite и remote CI не выполнялись. Venv использует system site
packages/--no-deps; dependency-clean release этим не подтверждён.

18 финальных installed logical attempts:

- 13 fixed-clock mission runs прошли kernel/self-refinement, conservation/lineage,
 event sensitivity, declared service minima и3 separate-process restore pairs.
- 3 fixed-input native programs прошли caller partition comparison, mid-tick
 same-process restore и final checkpoint replay. Max q difference≤1.42e−14deg,
 DC difference≤2.67e−15J — numerical roundoff, не точность относительно полёта.
- 2 дополнительных direct/fresh-process runs: decision1.03s, prefix после5 решений
 в5.15s, period0.1s. В обоих prefix actor states last sample5.1000000000000005s,
 count52. Producer/consumer PIDs различны; final native/trace signatures совпали.
 Это восстановление между ticks5.1/5.2s, а не только на границе tick.

До hardening владения данными SDK были ещё18 successful attempts на предыдущем
wheel; их raw artifacts и inventory сохранены отдельно. Изменился только
`validation/controller_clock.py` (detached inputs/results); frozen mission
manifest и все13 numerical sample hashes между сборками совпали. Эти повторы
не считаются дополнительной независимой статистической выборкой.

### Что именно улучшилось

При неизменном digital period0.1s и тех же budgets event-check resolutions
0.5/0.25/0.125s теперь дают max attitude difference1.681e−12 и2.431e−12deg
на91-point decision grid. Ранее event-driven модель давала0.4317/0.6558deg.
Max temperature difference≤3.52e−8K. Commands, final outcomes и delivery times
совпали. Mandatory `event_clock_stability` прошёл без увеличения tolerances.
Это проверка на объявленной сетке и коротком интервале, не глобальная error bound.

Разные digital periods остаются разными sampled controllers.0.2↔0.1s даёт max
q difference5.1738deg,0.1↔0.05s —2.2270deg. Прежний strict **multi-period
robustness** gate не пройден; он не подменяется успешным event gate. Эти отличия
нельзя называть погрешностью RK или обещать invariance между физически разными
контроллерами. Реальный flight software period/gains требуют отдельной model card.

Development outcomes2/24 на всех семи resolution configs. Regression геометрии:
0/24,0/24 и0/2 deliveries, с explicit minimum0. Они не подтверждают SLA или
статистическое обобщение. Иные виды mission fidelity/калибровки не повышаются
по результатам clock verification.

CI weekly/manual job запускает fixed-clock study, mid-tick process pair и
frozen-input program. PR reference profile содержит новые тесты. Checksum audit
не является authentication или повторным native replay; прежние receipts
сохранены с прежними source/wheel hashes.

## Следующий пакет D2e

1. Зафиксировать controller period/gains/phase assumptions в model cards. Проверить
   новый clock с внешними disturbances, rotating thrust и RCS unloading;
   не переносить legacy qualification автоматически.
2. Расширить frozen-input программу до явно меняемой RK mesh, независимого
   численного oracle и комбинаций saturation/SOC/thermal protection.
3. Создать independent feasible positive-service миссии на полном contact cycle:
   horizon, station deployment, deadlines и resource budgets фиксировать заранее,
   сохранять unreachable requests, но потребовать nonzero service в положительных
   случаях. Новые sampling/substep budgets обосновать и проверить отдельно.
4. Сравнить baseline/planner на таких миссиях с feasibility/capacity diagnostics;
   не выдавать zero-minimum regression или verified execution за экспертный teacher.
5. Продолжить C2 open measurement calibration, Q6 independent AI benchmark и
   Q7 durable jobs/OS isolation. Clock correctness не закрывает эти направления.
