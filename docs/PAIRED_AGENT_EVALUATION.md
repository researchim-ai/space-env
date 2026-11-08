# Парное сравнение агентов на одинаковой native-миссии

`space_env.llm.paired_evaluation.evaluate_pairs` принимает frozen family suite,
список выбранных cases и полную матрицу baseline/candidate. Это статистический
слой над существующими native episodes; модель он не вызывает и не обучает.

## Что проверяется

Каждый completed record проходит `verify_record`: холодное восстановление
исходного checkpoint, повтор tools и сравнение настоящих native outcomes.
Проверяются исходная конфигурация case, group и split; исходный checkpoint также
сравнивается с повторной компиляцией frozen case выбранным планировщиком, чтобы
два одинаково подменённых каталога не считались корректной парой. Внутри пары должны
совпадать весь checkpoint (включая каталог действий), текст задачи, max_calls и
max_dialogue_bytes. Смена planner внутри пары недопустима: сравнение планировщиков
с разными каталогами имеет отдельный контракт `llm.comparison`.

У error row нет придуманных метрик: только case, arm, status и error_type.
Отсутствующая/дублированная строка отклоняет матрицу. Неизвестный результат
занимает диапазон [0,1] нормированного доставленного приоритета; реально
измеренный нулевой результат остаётся известным нулём. Delta bounds вычисляются
по всем объявленным cases. При неизвестной паре mean/CI не выдаются, поэтому
timeout не может улучшить результат удалением трудной задачи.

## Неопределённость

95% percentile bootstrap выбирает целиком geometry groups, а не отдельные
варианты workload или decision ticks. Внутри выбранных групп сохраняются все
cases и парные разности candidate−baseline. Seed локальный и воспроизводимый.
Одна геометрия даёт null вместо выдуманного доверительного интервала.

Это условная описательная оценка по небольшому публичному regression-набору.
Она не измеряет разнообразие новых орбит, stochastic model seeds, физические
ошибки или истинное OOD-обобщение. При шести геометриях интервал может быть грубым;
не трактовать его как доказательство универсального превосходства.

## Запуск

Manifest JSON содержит `selected` и `rows`. Completed row:

```json
{"case":"equatorial-relay-chain","arm":"baseline","status":"completed","episode":"base-episode.json"}
```

Для каждого case нужна также candidate row. Вместо completed допускается:

```json
{"case":"equatorial-relay-chain","arm":"candidate","status":"error","error_type":"TimeoutError"}
```

Пути episode считаются относительно manifest. Output должен быть новым:

```bash
python -m space_env.llm.paired_evaluation /tmp/pairs/manifest.json --output /tmp/paired-report.json
python -m pytest tests/test_paired_evaluation.py -q
```

Отчёт связывает episode hashes, suite hash и source manifest. Метки policy_origin
сохраняются; scripted privileged teacher не превращается в LLM. Заявления
`llm_training_gain_established` и `competitor_superiority_established` всегда false:
этот слой не проверяет provenance обученных весов, реальные provider responses,
стоимость токенов или внешний simulator. Для них нужны соответствующие audited
learning/provider/competitor bundles поверх парной native-проверки.
