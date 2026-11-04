# D2b: independent-input coupled metadata mission

Дата: 2026-11-04. Разработка 0.134.0. Это ограниченный инженерный профиль,
не flight-qualified simulator и не модель качества реальных изображений.

## Зафиксированная постановка

`space_env.scaling.coupled.CoupledMissionSpec` компилирует отдельные входы:
initial circular planes/phases, geographic targets и ground stations. Координаты
не выводятся из траекторий. До четырёх акторов, 16 целей/станций и 64 запросов.
Каждая пара actor×target создаёт отдельный запрос; недоступные пары сохраняются.
Назначение destination — детерминированное round-robin по входному списку станций,
а не подбор станции по будущему контакту. Фиксированный географический grid в demo
не является реалистичной сетью коммерческого оператора или независимым holdout.

Default demo: две плоскости 55°/75°, RAAN 0°/120°, phases=0°, radius=7000 km;
12 equatorial targets и 12 independently declared equatorial stations с шагом30°.
Всего24 запроса, 96 действий; начальная ориентация identity, готовых onboard
products нет. Интервал180s, решения2s, окна slew12s, съёмка2s, CPU1s,
raw1Mbit → metadata product0.5Mbit, geometric downlink2Mbps.

Орбиты: propagated GCRS, J2, packaged EOP, единицы SI. Ориентация: existing rigid
wheel controller, inertia diag(2,3,4) kg m², wheel inertia0.01 kg m²,
torque0.1Nm, momentum0.2Nms, paid supply12W, PD10/3, control step0.1s.
Все параметры аппаратуры — **synthetic engineering assumptions**. Body-rate
allowance0.1rad/s задаётся для metadata acquisition; это не допустимый smear
или спецификация реальной оптической камеры. Development runs с более строгим
0.02rad/s выявили native `rigid_payload_rate` failures; они не использовались
как holdout и не являются квалификацией финальной конфигурации.

Acquisition использует horizon/off-nadir/range/Sun/availability geometry и
existing native pointing/tracking/dwell checks. Min Sun elevation=-90°:
разрешена геометрическая съёмка ночью, **не моделируется ночная оптика**.
CPU transformer — metadata surrogate, не нейросеть и не сжатие реального кадра.
Radio — идеальная геометрическая видимость, constant rate, omnidirectional
attitude allowance180°, shared station terminal1. Не заявляется RF calibration,
antenna steering, DTN relay, measurement-based navigation или weather fidelity.

## Causal policy

`route_aware_rolling` получает только public task projection. В opt-in pointing
catalogue добавляется owned current pointing telemetry только для видимой,
доступной команды. Среди подходящих целей учитывается текущая угловая ошибка.
Когда текущие ошибка и body rate уже допустимы, политика не начинает новый slew:
ожидание позволяет existing settling/coast завершиться. Съёмка, CPU и доставка
используют прежние utility weights. Ни будущие окна, ни орбиты, ни outcome
manifest, ни evaluator report не передаются policy. Это эвристика, не эксперт
и не доказательство оптимальности. Текущая готовность не гарантирует future dwell.

## Отказы и восстановление

`weak_wheels` меняет только motor torque bound на1e-6Nm; `all_outage` закрывает
все станции на весь интервал. Геометрия и все24 запроса сохраняются.
Control-window failures и failed acquisitions остаются в native journal.
Demo manifest содержит direct/restored пары всех трёх режимов. Restore после
20 решений использует separate-process protocol: producer exit → disk packet →
fresh consumer → audited prefix replay. Это не O(1) загрузка native RAM.

CLI example:

```bash
python examples/coupled/demo.py /tmp/coupled-mission
```

`run_scale` сохраняет progress, native replay/quality/export checks, hashes,
paired trace/backend comparison и все failed attempts. Неудачный запуск нельзя
убирать из denominator. Historical receipt families старых wheels не меняются.

## Оставшаяся квалификация

Этот профиль не закрывает numerical convergence, independent mission holdout,
real hardware calibration, realistic station deployment, long-duration thermal
stress, relay/DTN и production OS/job isolation. Их следует выполнять отдельно
с заранее заданными допустимыми ошибками и frozen inputs. Development debugging
и отдельный короткий smoke не являются такой квалификацией.

## Installed mission outcomes

Финальная сборка wheel SHA256
`ccd258b24a00237f2b4954acf88e81b1f515d0e986a23db7e7f27b6ed6254ee2`.
255 Python modules и288 source/data files byte-matched; README совпадает с wheel.
Установка изолирована от checkout (`Python -I`, cwd `/tmp`), но venv использует
system site packages и `--no-deps`: **это не dependency-clean CI**.

Шесть frozen logical attempts прошли native replay, verified export, quality,
resource, mission и coupled lineage gates. Для каждого режима direct/restored
trace/backend signatures идентичны; три separate-process restore успешны.

- Nominal: acquisition2, CPU2, delivery2 из24 запросов; сохранены12 failed slew
  windows. Это8.33% выполнения именно объявленного denominator, не SLA.
- Weak wheels: acquisition0, CPU0, delivery0 из24; проверка ресурсоучёта пройдена,
  минимальное требование обслуживания этого отрицательного режима равно0.
- All-station outage: acquisition2, CPU2, delivery0 из24; zero-delivery gate и
  minimum-acquisition gate пройдены. Геометрия и запросы совпадают с nominal.

Предварительные developer-прогоны и ранние ошибки запуска CLI не включены в
эти шесть accepted attempts. До фиксации workload исправлен выбор rolling policy
и задан synthetic metadata payload rate0.1rad/s; дальнейшего подбора после
приёмочных результатов не было. CLI smoke исправлен: manifest пишет сам
`run_scale` в пустую директорию; неверные gates приводят к ненулевому exit code.
Success/failure exit statuses отдельно проверены stub-based CLI contract.

CI workflow теперь запускает полный шестисценарный installed example вне checkout.
Remote CI в этом сеансе не выполнялся. Исторические acceptance roots сохранены;
новые результаты не заменяют их evidence и не повышают заявленную физическую
достоверность моделей реальной аппаратуры.

## Итоговая регрессия и приёмка

[Sealed acceptance](../acceptance/B11_COUPLED.json) связывает wheel/source/data
inventory, exact UTF-8 gzip bundle native manifests/progress/restart packets,
результаты всех6 attempts и логи тестов. SHA256 проверены после сохранения.
Это integrity receipts, не подпись доверенного внешнего аудитора и не отдельный
fresh replay при чтении архива.

16 новых coupled tests и14 предшествующих pointing tests входят в общий набор.
162 targeted tests, **1641 reference passed** (один полный запуск,516.31s,
15 upstream warnings) и110 fast прошли. Наборы пересекаются. Expanded full suite
и remote clean CI не запускались. Существующий HTTP test reference-профиля
выполнен с разрешённым loopback; остальные проверки не были исключены.
