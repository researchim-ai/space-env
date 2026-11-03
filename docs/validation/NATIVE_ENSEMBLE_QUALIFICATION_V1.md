# B11-Q5-D1: native-исполнение всего ансамбля параметров

Дата: 2026-11-03. Исходники 0.134.0; точная сборка привязана к source/data и wheel SHA.
Этот пакет связывает C1 sensitivity compiler с изолированным scale harness.
Он не закрывает аппаратную калибровку C2 и всю mission qualification Q5-D.

## Что теперь можно сделать

Зафиксировать базовую миссию, задачу, политику и допущения о параметрах;
запустить baseline и каждый допустимый вариант в новых процессах;
проверить native replay, quality, ресурсы, completion и восстановление из
checkpoint. В итог входят все попытки, включая structural rejection и timeout.
Один успешный baseline не делает весь ансамбль успешным.

```bash
space-env qualify-ensemble /path/to/spec.json /tmp/ensemble-run
space-env verify-ensemble /tmp/ensemble-run
python examples/calibration/qualification.py /tmp/ensemble-example
python examples/calibration/qualification.py /tmp/ensemble-rejections --rejected
```

Python API: `prepare_ensemble_qualification`, `run_ensemble_qualification`,
`summarize_ensemble_qualification`, `verify_ensemble_qualification` из
`space_env.scaling.ensemble`. В runtime нет загрузки внешних данных.

## Входной контракт

`EnsembleQualificationSpec` содержит:

- `id`: безопасный идентификатор пакета.
- `workload`: существующий `scaling.benchmark.Workload`: scenario, task, policy,
  repeats, timeout, resource budgets и mission gates.
- `sensitivity`: существующий `calibration.EnsembleSpec` с явно assumed distributions.
- `workers`: 1–4, `resume_after_decisions`: по умолчанию1, либо null без cold pair.

`min_completion_fraction` должен быть **явно** задан, в том числе 0 для
integrity-only probe. Значение 0 не доказывает выполнение mission service.
Для ожидаемого отрицательного mission outcome можно явно задать upper bound;
passing gates тогда означает соответствие этому ожиданию, а не успешную доставку.

Роли, catalogue, policy и seeds не подстраиваются после результатов. Base workload
не может задавать comparison_group/resume: парные привязки создаёт пакет.
Task проходит TaskSpec validation. Если выбранная точка восстановления оказалась
уже после фактического завершения эпизода, worker сохраняет ошибку; точка не
перемещается автоматически ради прохождения проверки.

Потенциальный бюджет до compilation:

$$W=(N+1)M,\quad E=W R,\qquad W\le64,\quad E\le128,$$

где N — число sensitivity attempts, +1 — baseline, M=2 с warm/cold или1 без,
R — repeats. Отвергнутые compiler attempts не освобождают декларируемый бюджет.
Например, с cold pair допустимы максимум31 samples при одном повторе.

Raw spec ограничен2MB, base scenario —1MB compiler budget, prepared JSON —64MB,
финальный receipt —32MB. Хешируемый артефакт ограничен64MB, весь inventory —256MB
и400 файлов в текущем D1b (историческая D1 приёмка имела лимит140). Worker timeout — прежний wall-clock subprocess budget; RSS измеряется
как lifetime high-water и проверяется gate. Это **не** жёсткая OS memory/CPU quota.

## Порядок исполнения

1. Полностью проверить вход и скомпилировать все draws. До этого output не создаётся.
2. Создать owned `prepared.json`: исходный spec, compiler result, frozen scale
   manifest и mapping member→workload IDs. Допущения/latent draws остаются в
   диагностическом артефакте; в policy/task они не добавляются.
3. Для baseline и structural candidates создать warm/cold workloads. Хеши
   native inputs внутри каждой пары одинаковы; между разными параметрическими
   вариантами равенство результатов не требуется.
4. `run_scale` запускает fresh interpreters с thread limits и subprocess timeout.
   Worker исполняет PlanningRun, делает checkpoint/replay и verified export checks.
5. Сводка пересчитывается из raw rows. Проверяются manifest binding, уникальные
   (workload ID, repeat), обязательные boolean gates и semantic signatures.
   Cached summary/overall booleans не могут переопределить raw execution outcomes.
6. Сохраняется финальный receipt с source/data inventory и artifact hashes.

Названия pair modes `warm`/`cold` унаследованы от этого bridge: `warm` исполняет
миссию целиком, `cold` восстанавливает новый PlanningRun из checkpoint внутри
того же worker-процесса после prefix. Каждый workload запускается в отдельном
fresh interpreter, но **между prefix и continuation отдельного OS-процесса нет**.
Проверка сериализации/восстановления не доказывает независимость от process-wide
caches. Историческая D1 приёмка такого свидетельства не заявляет. В новом
[D1b process restart](PROCESS_RESTART_V1.md) добавлен opt-in
`resume_in_new_process=true`; default false сохраняет legacy режим.

Directory должен быть новым или пустым. При прерывании сохраняются prepared и
доступный native progress; успешный финальный receipt не фабрикуется. Автоматического
resume частично законченного benchmark здесь нет: новый запуск требует нового
directory, старые попытки остаются доступными для анализа.

## Как считается результат

Статусы вариантов: `structural_rejected`, `native_failed`, `passed`.
Неудачный процесс, отсутствующий repeat, проваленный gate или несовпадающие
warm/cold signatures не исчезают из результата. Для каждого admitted member
сохраняются все execution rows, worker errors, expected/observed/missing counts,
observed completion range и pair result. Отсутствующее completion не превращается в0.

$$f_{pass}=\frac{N_{passed}}{N_{attempted}}.$$

Compiler rejections входят в N_attempted; их execution slots отдельно отмечены
как not submitted, а не выдуманные runtime failures. Baseline отделён от
variant denominator и обязан пройти для общего `passed=true`. Прохождение
требуется от **каждого** варианта и всех заявленных повторов/пар. Статистическая
coverage manufacturing distribution из такой доли не выводится.

В отличие от независимых случайных миссий, environment и policy seeds сохранены
для paired common-random-number опытов. Это не гарантирует одинакового расхода
случайных чисел при изменении ветвления симуляции и не заменяет protocol/seed holdout.

## Что делает offline verifier

`verify-ensemble` проверяет receipt/body hashes, ограниченный inventory, отсутствие
unsafe relative paths/symlinks, frozen native manifest и согласованность progress.
Затем повторно компилирует исходный sensitivity spec и пересчитывает агрегацию
native rows. Произвольное reseal изменения prepared не обходит эту проверку.

Это **audit identity/aggregation**, не новая физическая симуляция, независимая
аутентификация автора или доказательство истинности raw worker measurements.
В ответе явно `native_reexecuted=false`. Исторический receipt не квалифицирует
изменённую сборку. Дополнительные файлы, не перечисленные в inventory (например,
сохранённый позже audit.json), не получают защиты от этого receipt.

CLI exit codes: 0 — qualification passed; 1 — корректный результат с проваленными
declared gates/вариантами; 2 — контракт/IO/audit error. Для verify валидный audit
отрицательной qualification возвращает1 и `verified=true` одновременно. Example
с `--rejected` ожидает отрицательную qualification и проверяет её сохранение;
успех самого example не меняет `passed=false` исходной миссии.

## Проверяемый example и границы

Example использует существующую synthetic EO/CPU/relay/ACK миссию и две assumed
battery-capacity вариации. Baseline и каждая вариация исполняются warm/cold:
шесть отдельных процессов, одинаковая политика, explicit completion lower bound1.
`--rejected` задаёт заведомо невозможную solar efficiency: оба variants должны
быть structural rejections, baseline продолжает выполняться.

Это конструктивная synthetic mission: станции и workload не становятся
независимо расположенными реальными объектами. Q5-D2 остаётся открытым: fixed-site
multi-plane workloads, контроллер slew/tracking, реальные shared windows,
отдельные resource/pointing gates, solver refinement и held-out mission scenarios.

Worker проверяет RL/SFT exports внутри isolated run, но временные corpus files
удаляются прежним scale harness. Пакет публикует evidence/metrics, **не готовый
teacher corpus**. Q6 public-view/leakage, train/holdout split и benchmark card
проверяются отдельно. C2 измерительные datasets, идентифицируемость и uncertainty
coverage также не закрыты; industrial_readiness остаётся false.

Текущая приёмка: `docs/acceptance/B11_ENSEMBLE.json`. Исторические receipts
CALIBRATION/MATERIALS/POINTING/ENVELOPES/GROUND сохраняются без переноса qualification.

## Приёмка этой сборки

[B11_ENSEMBLE.json](../acceptance/B11_ENSEMBLE.json): 39 targeted / 1540 reference /
110 fast passed, 1970 collected; наборы пересекаются. Reference прошёл одним
запуском с localhost permission. 20 installed worker executions проверили positive,
structural-rejection, failed checkpoint-budget и subprocess-timeout packages.
Все четыре offline audits прошли без изменения отрицательных qualification.
Сохранены exact UTF-8 diagnostic files в четырёх gzip JSON bundles и их hashes;
распаковка files_utf8 в отдельный directory позволяет повторить verify-ensemble.
Это восстановление артефактов для audit, не автоматический native rerun.

250 modules / 283 source-data files сверены с wheel и checkout. Wheel SHA:
`f4ba929c97778ab6b4f7ffe02beaeb7299285e5cd26afbc0f23c52722d42f7a3`.
Прежние 34 material / 14 calibration / 612 pointing / 119 envelope comparisons
на этой сборке прошли. Исторические receipts не изменены. Нет expanded full suite,
remote clean CI, dependency-clean installation или independently located large
mission qualification. C2/D2/Q6 остаются открытыми.
