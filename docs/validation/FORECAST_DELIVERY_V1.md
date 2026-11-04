# D2h-A: прогнозный планировщик доставки готовых данных

План создаётся **до native rollout** из явно разрешённых оператору прогнозов
контактов и известных outage. В отличие от recorded-policy proposal D2g,
планировщик не использует выполненный trace, admission oracle, truth report,
объекты эфемерид или RNG. Это ограниченный воспроизводимый планировщик,
а не доказательство оптимальности либо универсальный mission optimizer.

## Профиль и границы

Только `reference-fixed`: до 8 аппаратов, 8 станций, 64 заранее загруженных
продуктов, 10000 равных decision cells и полный denominator. Владение,
фиксированное назначение, ready/deadline, объём, приоритет и каталог действий
объявлены заранее. Один downlink на действие, actor-exclusive; ёмкость станции
1–4 независимых терминала. Радио — идеальная постоянная скорость каждого
аппарата. Все actor×station пары включены, включая пустые окна.

Требуется `AccessPolicy(forecast=True)`. Обычный policy/agent доступ этого
разрешения автоматически не получает. `DeliveryForecast` содержит только
значения JSON и initial revision/geometry bindings. Native audit заново получает
прогнозы и сравнивает их с сохранёнными. Checksums сами по себе не доказывают
авторство или правдивость чужого forecast.

Unsupported поля отвергаются: acquisition, processing/dependencies, rigid/body-fixed
ADCS, thermal, RF/shared bandwidth, DTN и динамические maneuver. Не моделируются
неизвестные будущие операторские события. Эти механики есть в других профилях
библиотеки, но **не интегрированы в данный scheduler**.

## Алгоритм

Для продукта i со скоростью r и размером B требуемое число ячеек
`n_i = ceil(B_i / (r_i * Δt))`. Ячейка допустима, когда продукт готов, передача
закончится до deadline и forecast-window покрывает нужный интервал. У внутренних
границ окна применяется guard 2 ms; границы 0/H обрезанного прогноза исключены.
Неполные краевые передачи длинных продуктов не используются; последний короткий
остаток может закончиться раньше конца ячейки. Это консервативное ограничение.

Детерминированный greedy порядок: минимальный запас будущих допустимых ячеек
относительно числа необходимых, затем deadline, приоритет, остаток и ID.
Продукты, уже невыполнимые на этой lattice, не забирают ячейки у остальных,
но сохраняются в знаменателе и diagnostics. Затем учитываются actor exclusivity,
терминалы и квоты. Глобальная optimality и полный поиск не заявляются.

Консервативная энергия: `E_base = P_base * H / 3600`,
`E_tx = P_tx * T_tx / 3600`; суммарные расходы не превышают
`battery_Wh * (initial_soc - min_tx_soc)`. PV credit не используется.
`max_tx_sec` проверяется отдельно. Это может отбрасывать физически допустимые
решения, но не заменяет native energy ledger.

Budget: до 640000 slot×product и 2000000 проверок окон. Переполнение отвергается.

## Исполнение и данные

`forecast_mission_plan` возвращает `forecast`, `schedule`, sealed `MissionPlan`
с `origin=declared`, без expected-native fingerprint из будущего execution.
Proposal ещё не сертификат. `PlanExecution` исполняет его через TaskSession,
`publish_plan_bundle` требует независимый cold native replay. RL сохраняется
и для failed goal; SFT создаётся только по правилам goal/quality bundle D2g.

Benchmark сохраняет исходный spec, прогнозы, планы, execution, RL/SFT bundles,
checkpoint каждого baseline, все строки ошибок, progress, teacher report,
file inventory и seal. Отдельный аудит проверяет schedule recomputation,
case/task/goal bindings, полную paired grid и метрики из native reports.
`--native` дополнительно воспроизводит execution и dataset каждой строки.
Integrity audit без флага не является новым физическим прогоном. Это локальная
инженерная проверка, не доверенная удалённая аттестация.

## Teacher gates и честное сравнение

Spec фиксируется до outcomes. Hash-based lineage split проверяется, семантически
дублированные кейсы отвергаются консервативным typed fingerprint. На семейство
требуются минимум два разных positive heldout; отдельный full-outage negative
control должен дать нулевую доставку и failed candidate goal. Ошибочные строки
не исключаются. Требуются native replay, service floor, rejection threshold,
реализация prediction и paired noninferiority/margin против каждого baseline.

**Information budgets различаются**: forecast candidate имеет явно разрешённые
будущие контакты/outages, reactive priority/earliest_deadline получают текущие
public views. Одинаковы физические сценарии, задачи и ресурсы. Это сравнение
operational configurations, не равных по информации policy architectures.

Passing gates разрешает ограниченный `eligible_for_sft_selection`; это не
автоматическое утверждение всех samples в качестве expert demonstrations.
`expert_teacher_qualified` всегда false. Нет доказательства generalization,
flight calibration, SLA или optimality. Нулевой min_margin допускает равенство.

## Запуск

```bash
python examples/planning/forecast_delivery.py --help
python examples/planning/forecast_delivery.py /tmp/forecast-spec.json --write-spec
space-env forecast-benchmark /tmp/forecast-spec.json /tmp/forecast-benchmark
space-env verify-forecast-benchmark /tmp/forecast-benchmark --native
space-env forecast-delivery-plan scenario.json task.json /tmp/forecast-proposal
```

Все output directories новые, существующие не перезаписываются. CLI proposal
является явным операторским разрешением forecast; возвращает
`native_feasibility_validated=false`. Benchmark failed gates возвращает exit 1,
ошибки ввода exit 2. Benchmark не является durable hosted job; при аварии
сохраняет завершённые строки, но автоматический resume не реализован.

## Следующий большой блок D2h-B

1. Acquisition → processing → delivery: release/dependencies и paid pointing
   включить в constraint model, без скрытого native lookahead.
2. Rolling replanning после новой telemetry/outage, versioned forecast bindings,
   frozen prefix и проверка continuation equivalence.
3. Равные information budgets для специализированных forecast baseline;
   больше независимых географий, deadline/load stress и competing-terminal cases.
4. Model-based teacher selection против stronger optimization baseline;
   отдельно измерять стоимость поиска, качество и границы generalization.
5. Long rigid execution qualification; measured calibration C2 и hosted isolation
   Q7 остаются самостоятельными незакрытыми требованиями.

## Что ещё не доказывает этот benchmark

Один train и два positive holdout — небольшой инженерный набор. Все станции
этого примера экваториальные; inclinations/RAAN/phases различны, но полярные
сети, плотные группировки и сложные deadline distributions не квалифицированы.
Невозможные продукты остаются в denominator. Порог 10% и нулевая margin —
заранее объявленный минимальный smoke gate, а не коммерческий service target.
Сгенерированные SFT файлы являются кандидатами по declared model-goal;
passing benchmark selection не делает их экспертными демонстрациями.
