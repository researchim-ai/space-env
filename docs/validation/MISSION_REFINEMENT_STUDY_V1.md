# Mission refinement and unseen-geometry engineering study

Дата: 2026-11-04, source version0.134.0. Дальнейшая проверка ограниченного
coupled metadata profile; не универсальная квалификация космического симулятора.

## Что сравнивается

`MissionStudySpec` фиксирует development configuration, 1–4 геометрически
отличающихся контрольных миссии, допуски и требования до первого native execution.
`run_mission_study` запускает bounded fresh-process workers через `run_scale`.
Ни неудачные попытки, ни невыполнимые запросы не удаляются из знаменателя.

Одна study строит7 разных development workloads с переиспользованием одинаковых
центральных конфигураций, затем direct/restored пару каждого контрольного случая:

- **Rigid kernel refinement:** RK integration maximum step0.02/0.01/0.005s,
  controller cadence0.1s и lifecycle check cadence0.5s фиксированы.
- **Controller cadence sensitivity:**0.2/0.1/0.05s при kernel step0.01s.
  Меняется дискретная модель контроллера, а не только ошибка интегрирования.
- **Lifecycle event-check sensitivity:**0.5/0.25/0.125s при controller0.1s и
  kernel0.01s. Меняется разрешение native guard/dwell/resource checks.

Это **closed-loop comparisons**, не replay одной замороженной native command
программы на всех сетках. Политика получает только public observations и может
изменить действие после малой численной ошибки. Поэтому kernel qualification
требует одинаковые action IDs, длительности и admission outcomes на общей сетке.
Изменение команд блокирует такой вывод о self-refinement. Внутренние hybrid
branches этим не объявляются доказанно одинаковыми, истинный continuum error
и порядок сходимости не оцениваются.

## Evaluator-only sampling

`Workload(sample_numerics=True)` разрешён только для coupled gates, ≤4 actors,
≤601 решений. Значение false не сериализуется, сохраняя legacy identities.
Worker снимает owned state после решений и начальную точку, не создавая native
команд и не добавляя features в public observations, policy input или RL/SFT.

Снимок содержит текущее GCRS position/velocity, quaternion wxyz, body rate,
wheel momentum, battery Wh, temperature K, ADCS DC energy J, request outcomes
и completed public control. Sampling hash, finite/unit/vector checks, actor/
request identity и strictly increasing grid проверяются отдельно. Общая сетка
refinement должна покрывать весь заявленный horizon с decision cadence.

При process restore consumer сохраняет **только continuation** начиная с prefix
boundary. Эти samples не выдаются за полный trajectory от t=0. Полная native
replay/pair проверка остаётся existing D1b protocol. Для refinement используются
прямые исполнения, а не дополнение отсутствующего prefix synthetic samples.

## Численные метрики и допуски

Для каждой пары сеток по всем общим временам и акторам сохраняются max и RMS:

`position error = ||r_a − r_b||₂`, `velocity error = ||v_a − v_b||₂`;
`attitude error = 4 atan2(min(||q_a−q_b||,||q_a+q_b||),
max(||q_a−q_b||,||q_a+q_b||))` с переводом radians→degrees.

Так учитывается эквивалентность q и −q. Для body rate и wheel momentum
используется Euclidean norm, для battery/temperature/DC energy — absolute error.
Отдельно сравниваются final request statuses, наличие ground delivery и
максимальное расхождение времени доставки. Отсутствие доставки у одной версии
не трактуется как нулевая ошибка времени.

Default thresholds, сохранённые во frozen input:1m,0.01m/s,0.1deg,0.001rad/s,
0.005Nms,0.01Wh,0.05K,10J,0.5s соответственно. Это **заявленный engineering
numerical budget**, не измеренные погрешности реальной аппаратуры.

Обе соседние kernel пары должны пройти эти допуски, совпадение расписания и
outcomes; finer difference должен быть ≤1.25×coarser difference либо ниже1% от
соответствующего допуска. Это условная empirical self-refinement проверка с
absolute floor, не доказательство монотонной асимптотики или порядка метода.

Controller/event comparisons отдельно показывают ошибки, изменённые controls
и outcomes. `require_temporal_stability=False` разрешает **сохранить неустойчивость
как результат исследования**, не маскируя её под пройденную проверку. Report
содержит `temporal_stable` и `temporal_stability_required` раздельно. Значение true
делает эту стабильность обязательным gate общего результата.

## Контрольные геометрии

Frozen example содержит3 случая, определённых до запуска без поиска координат:
северные цели30° с RAAN+35°/phase40°; южные цели−30° с phase220° и наземными
станциями45°; полярная цель89° с исходными орбитами. Все actor×target запросы
сохраняются. Меняется только объявленная конфигурация, нет outcome-based rejection
или пересэмплирования. IDs или изменения numerical steps сами по себе не дают
геометрически нового holdout — schema отклоняет такие случаи.

Для этих исследовательских контрольных случаев min service явно равен0:
измеряются переносимость исполнения и native outcomes, **не заявляется SLA**.
Это несколько фиксированных unseen engineering cases, не статистический
holdout обученной модели, не независимые данные оператора и не оценка ML
обобщения. Случаи нельзя затем использовать для настройки и продолжать называть
нетронутым holdout. Для такой настройки нужен новый отдельный набор.

## SDK, CLI и offline audit

```bash
python examples/coupled/study.py /tmp/mission-study.json --write-spec
space-env mission-study /tmp/mission-study.json /tmp/mission-study
space-env verify-mission-study /tmp/mission-study
```

Один spec создаёт13 logical attempts в default example:7 development и6 holdout.
Timeout, memory/checkpoint budgets, native replay, conservation/lineage witnesses,
verified exports и pair restoration проверяются scale workers. Ошибка сохраняется
с reason в общем denominator. `mission-study` возвращает exit1 при невыполненном
qualification gate, exit2 при contract/I/O error. Непустая output directory
отклоняется. Результаты не перезаписывают предыдущую приёмку.

Offline audit проверяет bounded artifacts, SHA identities и пересчитывает
comparisons/aggregation. Это **не fresh native replay и не authentication**.
Совпадение текущего source/build выводится отдельно; checksum audit не
переносит физическую квалификацию на новую сборку. Historical receipts и их
wheel/source hashes сохраняются.

## Что остаётся открытым

Independent numerical oracle, fixed-input native replay refinement, orbital/
contact-search convergence, long-horizon resource stress, mission/controller
statistical holdout, измерительная калибровка, sensor imagery fidelity, навигация
с ошибками измерений и эксплуатационный production isolation остаются отдельными
пакетами. Устойчивый self-refinement текущей модели не доказывает её физическую
достоверность и не заменяет эти проверки.

## Обнаруженные дефекты и неизменность science inputs

Первое установленное исследование сохранено отдельно как
[B11_STUDY_FINDING](../acceptance/B11_STUDY_FINDING.json):13 attempts,11 passed,
2 rejected witness cases. Полярные no-op миссии выявили некорректное требование
`DC energy > 0` даже при отсутствии actuation. Исправление не добавляет манёвров
и не меняет геометрию, policy, numerical budgets или service thresholds.
Теперь проверяются обе связи: `DC_J = integrated_ADCS_Wh × 3600` и
`DC_J = powered_duration_s × configured_supply_W`, плюс прежние conservation
residuals. Законный пассивный полёт даёт нулевые значения с обеих сторон;
неоплаченные активные операции и выдуманное питание не проходят witness.

В первом общем reference-прогоне также обнаружен existing intermittent CLI bug:
URL-safe generated operator session ID может начинаться с `-`/`--`, и argparse
считал его флагом. Local recovery теперь распознаёт ровно такой24-character ID
в существующем трёхпозиционном вызове как literal; прочие флаги сохраняются.
Private output остаётся O_EXCL/0600; access checks и capability revocation не
ослаблены. Добавлены deterministic regression tests для обоих префиксов.

Исходный неуспешный study, failed reference log и wheel/source inventory
сохранены. Повторная приёмка использует тот же frozen spec, отдельную сборку и
новую output directory. Это исправление witness/parser defects, не поиск новых
удобных целей и не повышение допусков после наблюдения результатов.

Тяжёлый13-attempt installed study добавлен в weekly/manual CI; pull requests
выполняют reference tests. Remote CI в этой сессии не запускался.

## Итог новой сборки и честная граница приёмки

[Sealed receipt](../acceptance/B11_MISSION_STUDY.json), wheel SHA256
`89bf4fc8cb1e09a3369dfc7b1bbe59a068d016ca7c5a4acce3d2eed17bce693b`.
257 modules/290 source+data files byte-matched. Установленный SDK/CLI проверены
через Python -I вне checkout. Venv использует system site packages/--no-deps,
поэтому это не dependency-clean release CI.

**1671 reference passed** одним прогоном (467.64s,15 upstream warnings),
198 targeted и110 fast passed; наборы пересекаются. Добавлены28 study tests и2
local recovery cases. Expanded full suite и remote CI не выполнялись.

13 corrected installed attempts прошли native replay, quality/resource/lineage,
service gates с явно объявленными minima и3 separate-process restore pairs.
Первоначальные13 attempts сохранены отдельно. Итого26 попыток двух сборок;
их qualification не смешивается. Frozen specs/manifests и **все13 numerical
sample hashes между сборками совпали**: дефект witness исправлен без изменения
scientific outcomes. Offline audit и bounded/invalid CLI contracts проверены.

- Kernel self-refinement прошёл:91 common-grid samples,2 actors, одинаковые
  commands и2/24 deliveries на всех development уровнях. Максимальные различия
  ориентации соседних RK сеток порядка7.4e−12 degrees находятся на уровне
  numerical roundoff. Это не заявленная точность симулятора относительно полёта.
- Controller interval0.2↔0.1s даёт max attitude difference4.1571°,0.1↔0.05s —
  2.2270°. Доставки и верхнеуровневые commands совпадают, но budgets ориентации,
  body rate и wheel momentum не проходят.
- Event resolution0.5↔0.25s даёт0.4317°,0.25↔0.125s —0.6558° max attitude
  difference. Это также выше0.1° declared budget.
- Северный unseen case: acquisition1/CPU1/delivery0 из24. Южный:2/2/0 из24.
  Полярный:0/0/0 из2. Для каждого direct/restored signatures совпали. Нулевой
  declared minimum этих случаев не превращает результаты в SLA qualification.

`control_step_sec` текущей модели ограничивает **максимальный control interval**;
другие native event boundaries могут вызывать дополнительные PD updates.
Поэтому event-resolution experiment меняет effective resampling контроллера:
это не независимо изолированная погрешность guard поиска. Эту семантику нельзя
описывать как точную независимую частоту цифрового flight controller.

`temporal_stable=False` сохранено. Общий study gate зелёный только для заранее
объявленного инженерного scope с `require_temporal_stability=False`.
[Дополнительный строгий release gate](../acceptance/B11_MISSION_STUDY_RELEASE_GATE.json)
на тех же executions, с теми же допусками и обязательной temporal stability,
**НЕ пройден**. Это recomputation критериев, не новый mission run. Промышленная
готовность, полётная точность и качество обслуживания unseen scenarios не заявляются.

## Следующий большой пакет D2d

1. Ввести explicit controller-clock contract: latched motor command и fixed
   ticks, отдельно разрешённые command/fault safety interrupts. Разделить
   physics integration substeps, event checks и controller updates.
2. Проверить paired executions с разным event cadence при неизменном clock:
   transient q/ω/h, settling, momentum saturation, energy and thermal ledgers.
   Не поднимать budgets ради зелёной проверки; сравнивать с прежним event-driven
   профилем отдельно, с новым model identity.
3. Добавить fixed-input native command replay refinement; повторный closed-loop
   запуск — отдельная проверка чувствительности политики, а не его замена.
4. Расширить independent mission horizon до настоящих contact/delivery cycles,
   установить feasibility/capacity diagnostics и заранее заданный nonzero service
   gate для положительных evaluation cases. Нынешние min0 cases сохранить как
   отрицательные/исследовательские regression cases.
5. Создать новый untouched controller/mission evaluation набор до настройки
   нового контроллера. Нынешние случаи уже являются regression suite и не должны
   выдавать себя за свежий statistical holdout.
