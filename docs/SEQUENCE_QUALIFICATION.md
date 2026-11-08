# Sequence qualification v1 — полный регрессионный набор

8 ноября 2026. `python -m space_env.llm.qualification` проверяет прежний serial и
новый sequence-lanes на **всех** 36 frozen workload-конфигурациях Mission Families:
шесть геометрических групп, по две в train/development/test, шесть семейств на
каждой геометрии. Ожидаемая матрица — 72 исполнения, заданная до запуска.

Это не новый набор физических механик и не очередная замена удобными задачами.
Конфигурации, дедлайны, ресурсы, качество native delivery и SFT gates не меняются.
Обеим стратегиям соответствует один declared physical configuration, но свои
назначения и каталоги; исполняет один и тот же privileged reference policy.
Это сравнение planner pipelines, **не actor-only leaderboard или LLM-score**.

## Запуск, частичный прогон и возобновление

```bash
# Весь набор, новые локальные output-файлы:
python -m space_env.llm.qualification --output /tmp/new-sequence-study

# Либо намеренно выполнить только первые две ещё не завершённые задачи:
python -m space_env.llm.qualification --output /tmp/new-sequence-prefix --max-cases 2
python -m space_env.llm.qualification --audit /tmp/new-sequence-prefix --partial
python -m space_env.llm.qualification --output /tmp/new-sequence-prefix --resume

# Полный source-bound аудит:
python -m space_env.llm.qualification --audit /tmp/new-sequence-study
```

Прогон не запускает модель, не скачивает веса, не отправляет данные и не создаёт CI.
Обычному local runtime нужны зависимости reference-профиля. Входы и геометрии
публичны и ранее использовались: это regression splits, **не pristine blind test**.

## Структура и смысл статусов

```text
study.json                         # frozen suite + methods + all package source hashes
journal.json                       # sealed attempt journal
progress.json                      # fixed-denominator current state
report.json                        # только после завершения всех 36 cases
.writer.sqlite                     # локальная блокировка писателя, не dataset
cases/<case>/attempt-01/
    report.json                    # две ячейки сравнения
    progress.json                  # диагностический prefix
    <case>--serial.json             # native episode, если записан
    <case>--sequence_lanes.json
```

`complete` означает завершённость **матрицы**, а не доставку всех заявок.
`summary.replayed` считает нативно проверяемые исполнения, `full_goals` — полное
выполнение задачи. `summary.status` отдельно различает partial, полностью
проверенную native matrix и наблюдённые runtime/forecast проблемы. Даже законченный
report не называется аппаратной/полётной квалификацией.

Каждая задача фиксируется отдельно, после native replay и независимого
перестроения планировщика. Частичная матрица не создаёт финальный report;
обычный `--audit` на ней завершается ошибкой. `--partial` подтверждает только
целостность явно незавершённого состояния и возвращает complete=false.

## Crash recovery и отсутствие cherry-picking

SQLite `BEGIN IMMEDIATE` держит единственного писателя; при завершении процесса
OS/SQLite освобождают lease. Второй писатель не меняет journal и получает отказ.
Файл lease сохраняется после запуска, но не означает постоянно занятую блокировку.

Журнал записывается **до** запуска очередной задачи. При прерывании возможны два
случая:

1. Нет финального child report: попытка помечается interrupted, остаётся в журнале
   и в файловой истории; новая попытка использует другой номер/директорию.
2. Child report записан, но journal не успел зафиксировать completion: resume
   проверяет report/episodes, восстанавливает commit и не повторяет эту задачу.

Законченные child reports с runtime error cells **не перезапускаются ради более
высокого score**. Они остаются неизвестными результатами. Невалидный или подменённый
законченный report не игнорируется как удобная «прерванная попытка»: проверка
останавливается. На case допускается до трёх попыток всего, включая завершённую. Их количество
публикуется; это retries инфраструктуры, а не независимые повторы для статистики.

Изменение suite, planner settings или package source revision не разрешает resume
в той же study. Нужно новое output-имя либо точное восстановление ревизии. Child
report также привязан к source manifest всей study; смешанный source не принимается.
Результаты прежних версий остаются историческими, их проверка не ослабляется.

## Метрики с правильным знаменателем

Агрегаты сохраняются overall, по split, workload family и geometry group. Для
каждого режима знаменатель берётся из **полной предопределённой** матрицы:

- native completed episode: его фактическая delivered priority / total priority;
- runtime error или ещё не выполненная задача: неизвестный вклад [0,1];
- измеренная миссия с нулевой доставкой: известный ноль, а не неизвестный успех.

Так получаются value_lower/value_upper. Это **границы неизвестного результата**,
не статистический confidence interval. Missing cases не исключаются из знаменателя.

Парные улучшения/ухудшения считаются по фактической доставленной priority только
для пар с двумя проверенными результатами. Остальные пары явно unknown, а не
приписываются baseline как проигрыши. Отдельно сохраняются mismatch между плановым
и native количеством доставок и полнота исполнения всей матрицы.

Шесть workload variants одной геометрии не выдаются за шесть независимых орбит.
Группы — фиксированные публичные engineering fixtures, не случайная выборка
популяции всех космических миссий. Даже отсутствие регрессий во всех 36 случаях
не доказывает универсальное превосходство или OOD-обобщение.

## Связь с корпусом

После полного аудита native sequence-эпизоды можно передать в существующий
`space_env.llm.corpus`. Все train/dev/test артефакты сохраняются для аудита,
но data/ содержит только train. Partial/failed-goal native episodes идут в RL,
positive SFT требует complete-goal/error-free execution. Ячейки без валидного
native episode остаются в benchmark и не превращаются в обучающие строки.

Teacher provenance — scripted_privileged. Нет реальных provider responses, поэтому
этот прогон **не создаёт wire-SFT** и не является новым экспериментом Qwen.
Обучение/эффект SFT требует отдельного протокола и неизменных оценочных задач.

Comparison API перенесён из примера в установленную библиотеку:
`space_env.llm.comparison.run/audit`. Старый
`examples.network.pointing_comparison` остаётся совместимой точкой входа.

## Безопасная сборка полного corpus bridge

```bash
python -m examples.llm.qualified_corpus \
  --study /tmp/new-sequence-study \
  --output /tmp/new-qualified-corpus \
  --registry /tmp/shared-corpus.sqlite
```

Команда сначала выполняет полный audit study, затем включает все валидные native
sequence episodes. Она не отбирает только successful цели. Все development/test
эпизоды сохраняются в privileged audit, но исключаются из training data. Runtime
error без валидного episode публикуется как explicit missing_verified_episodes.
У каждого входа сверяется episode hash с исходным qualification report.

Сборка corpus и регистрация в SQLite — отдельные операции, а не единая
распределённая транзакция. Если процесс прервался после создания корпуса, тот же
вызов с `--resume` проверит уже готовый corpus, его принадлежность study и
идемпотентно зарегистрирует его. Другая валидная коллекция episodes под тем же
output не принимается как результат этой study. Corpus должен лежать вне дерева
qualification evidence; вывод команды — sealed link receipt двух artifact hashes.

`progress.json` можно читать для наблюдения за ходом прогона. Полный/partial audit
следует запускать после остановки писателя: он fail-closed отвергает running
attempt и несогласованный commit, а не догадывается о ещё не завершённой записи.
Для намеренного промежуточного checkpoint используйте `--max-cases`.

## Финальная локальная приёмка

На зафиксированных исходниках завершены все **36 cases × 2 метода = 72 ячейки**,
шесть геометрических групп, все train/development/test splits. После producer
проверки выполнен полный source-bound audit с повторным native replay и
независимым построением планов — он был обязательным этапом corpus bridge.

- Runtime errors: **0**.
- Planned/native count mismatches: **0**.
- По фактически доставленной priority: **23 улучшения**, 13 совпадений,
  0 ухудшений относительно serial на этих 36 случаях.
- Полные цели: serial **0/36**, sequence-lanes **16/36**.
- Средняя нормированная delivered priority: **0,4259 → 0,6852** на этом наборе.
- Sequence full goals: train **5/12**, development **5/12**, test **6/12**.

Это не статистическая оценка всей популяции миссий. Публичные геометрии, общий
EO/CPU/доставка workflow и privileged reference явно ограничивают выводы.
Другие физические профили, крупные/длительные миссии и реальные LLM отдельно не
квалифицированы этим прогоном.

Получен, зарегистрирован и нативно проверен корпус из 36 sequence episodes:
**12 train RL + 5 train SFT**, 24 evaluation episodes только в privileged audit,
0 wire-SFT, отсутствующих verified episodes нет. В corpus целом шесть geometry
fingerprints, **в training data — две геометрии**; development и test имеют ещё
по две. Нельзя объявлять training corpus шестью независимыми геометриями.
Все 36 provenance labels — scripted_privileged; реальные model responses,
fine-tuning или эффект обучения не измерялись.

Проверки: **77 integration tests + 4 bridge-contract tests = 81 distinct passed**.
Fault-injection fixtures проверяют отказоустойчивость, а не являются дополнительными
измеренными миссиями benchmark. Ruff/Python compile/diff-check passed.
[Машинный протокол с source hashes, всеми агрегатами и corpus link](acceptance/SEQUENCE_QUALIFICATION_V1.json).
