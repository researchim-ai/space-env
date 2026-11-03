# B11-Q5-D2a: независимые орбитальные плоскости и фиксированные станции

Дата: 2026-11-04. Исходники0.134.0; приёмка дополнительно связана с wheel/source/data SHA.

## Два блока реализации

1. Компилятор delivery workloads с независимо заданными круговыми initial planes,
   фазами и WGS84 sites. Станции не вычисляются из subsatellite point, целевые
   пары не фильтруются по наличию контакта или успешному результату.
2. Диагностика геометрии/ёмкости и native witnesses: окна/нулевые контакты,
   station terminal и actor TX constraints, связь receipts с окнами/outages/rate,
   product accounting, energy/inventory balances и воспроизводимое восстановление.

Это delivery-only профиль для готовых onboard products. Он не добавляет
closed-loop pointing, EO acquisition, изменяемые орбиты или новый RF link budget.
Такие компоненты есть в других профилях библиотеки; их общая независимая
qualification остаётся следующим D2b, а не объявляется закрытой этим пакетом.

## Применение

```bash
space-env independent-fleet /path/to/config.json /tmp/independent.json
python examples/scaling/independent.py /tmp/independent-small --small
python examples/scaling/independent.py /tmp/independent-four
```

CLI компилирует workload и assessment, **не исполняет native mission**.
Python API `independent_workload`, `independent_bundle`, `verify_independent_bundle`,
`assess_geometry` находится в `space_env.scaling.independent`.
Для выполнения workload используется существующий `ScaleSpec/run_scale`.
`verify_capacity_certificate` находится в `space_env.scaling.contact_capacity`.

Config: id, UTC epoch,2–4 planes,1–8 stations, horizon60–7200s,
sample cadence1–60s, decision interval5–120s, fixed product size/rate и min_delivered.
Всего максимум8 actors/64 products. Plane задаёт radius6700–7500km,
inclination/RAAN и1–4 уникальных phase angles. Совпадающие/противоположные нормали
не считаются разными геометрическими плоскостями. Sites имеют explicit lat/lon/
altitude/elevation mask и1–4 terminals. Значения по умолчанию синтетические;
никаких прав доступа к реальным антеннам или аппаратной калибровки не заявляется.

Initial states вычисляются только из plane/phase/radius:

$$r=R(\cos f\,e_1+\sin f\,e_2),\quad
v=\sqrt{\mu/R}(-\sin f\,e_1+\cos f\,e_2).$$

Далее применяется существующий J2 propagator и packaged EOP/frame context.
Все actor×station products создаются заранее. Inventory ограничен explicit
memory/battery; base/TX power оплачиваются. Устройства не становятся flight models
от выбора radius, а опубликованный hardware evidence автоматически не переносится.

## Геометрия и область достоверности

Assessment сохраняет **каждую** actor×station пару, в том числе windows=[];
эпоха, орбиты и sites не сдвигаются после анализа. Используются native
ReferenceEphemeris/Hermite и access scan30s с root tolerance1ms. Это общие
геометрические уравнения с runtime, не независимый измерительный oracle.
Узкий/grazing pass между scan nodes может быть пропущен. Tolerance корня не
равна physical event accuracy; contact completeness вне проверенного профиля
не сертифицируется. Изменение model/scan/cadence требует новой приёмки.

Диагностические windows/certificates не вставляются в task/policy/actor input.
Baseline использует существующую public catalogue/current station telemetry.
Backend имеет известную fixed ephemeris; privileged assessment отдельно от
causal agent и не является teacher answer. Public-view anti-leakage проверка
есть для этих diagnostic fields; общий Q6 benchmark этим не закрывается.

## Relaxation и точная проверка верхней границы

Контакты делятся на интервалы по AOS/LOS/outage endpoints. Native windows
расширяются наружу на1ms для оптимистичной оценки. Products готовы в t=0,
имеют fixed destination и deadline=horizon, constant rate, без обработки/DTN/RF.
Другие профили assessment отвергает. Реальные энергетические, тепловые и
целочисленные control-window ограничения в relaxation опущены: это расширяет
допустимое множество, а не доказывает существование native плана.

Для продукта p используется доля доставки f_p∈[0,1], для допустимой пары
product/interval — время TX x≥0. Связь: sum(x_p)=required_tx_sec_p·f_p.
В каждом интервале sum(x_actor)≤duration, sum(x_station)≤terminals·duration.
Дополнительный box bound x≤duration конечен. Максимизируется sum(f_p).
Построение ограничено512 intervals и20000 flow variables; матрицы sparse.

Solver — [SciPy linprog/HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html),
time_limit10s/maxiter10000. Его status и floating primal estimate сохраняются,
но **его optimality flag не используется как доказательство верхней границы**.
Для min cᵀx, Ax≤b, Ex=0,0≤x≤u, любых y≤0 и z:

$$L=b^Ty+\sum_j\min(0,u_j(c-A^Ty-E^Tz)_j),\qquad
\max\sum f_p\le -L.$$

Solver multipliers y обрезаются только по знаку до y≤0; любое конечное значение
после этого даёт valid weak-duality bound из-за явных box terms. Весь расчёт L
повторяется **точно через Fraction** для binary64 коэффициентов и multipliers.
Numerator/denominator сохраняются, upper float округляется вверх. Bound дополнительно
ограничен тривиальным числом продуктов. При solver failure сохраняются zero duals,
тривиальная граница и failure status; default require_lp_success не даёт этому
превратиться в успешный scientific gate.

`integer_upper_binary_lp_floor` — floor точной верхней границы для бинарного
числа завершений **в описанной binary-input relaxation**.
`integer_upper_conservative_ceiling` — более слабый ceil для native sanity check,
чтобы не переносить дискретную математическую границу напрямую на округлённые
native durations/receipts. При integer-adjacent bound он может быть на1 продукт
слабее. Это не доказательство optimality native controller или accuracy физики.
Обе величины различаются явно, не подменяются одной красивой оценкой.

## Native gates и артефакты

`Workload.geometry_gates` включает spec hash и explicit min_delivered.
Отсутствующий optional gate не меняет legacy serialized workloads.
Worker заново строит assessment и проверяет native report:

- delivery count и conservative capacity sanity ceiling;
- actual terminal и single-transmitter occupancy по half-open receipts;
- receipt ownership/destination, contact/outage/rate/time validity;
- product/receipt bits, inventory и battery energy accounting;
- требование успешного LP, если оно задано.

Baseline min_completion_fraction0 не заменяет min_delivered scientific gate.
Negative cases имеют explicit min_delivered0; outage case также upper completion0.
Их успешная приёмка означает правильное воспроизведение ожидаемого отсутствия
service, а не успешную доставку. Все raw outcomes сохраняются.

Example замораживает три режима: nominal, all-station outage и independently
defined polar site с89° elevation mask. Каждый выполняется direct/restored с
настоящим separate-process protocol и prefix после20 решений. Small:2 actors;
full:4 actors; две плоскости,6000s. Main sites: synthetic geographic points
(0°,0°),(45°,30°),(-30°,150°); они не выводятся из orbit samples.

Bundle audit воспроизводит compiler/геометрию/problem и проверяет exact dual
certificate. Он повторно запускает model/LP, **не native mission**; это не
аутентификация автора. Native evidence содержит assessment hash; assessments и
полные raw worker results сохраняются отдельно. Временные worker RL/SFT export
checks не равны опубликованному teacher corpus.

Приёмка: `docs/acceptance/B11_INDEPENDENT_GEOMETRY.json`. Нет обещания contact
completeness, flight calibration, multi-day scalability или independently validated
coupled pointing/EO. Следующий D2b — закрыть именно эти coupled mission gaps,
сохранив независимые входы и исходный знаменатель неудачных задач.

## Результат этой сборки

[Приёмка](../acceptance/B11_INDEPENDENT_GEOMETRY.json):30 новых tests,
110 targeted,1611 reference,110 fast passed,2041 collected; наборы пересекаются.
Small/full native suites:2/4 actors,6000s,nominal/outage/no-contact и реальные
separate-process restored pairs. Nominal доставил3/6 и6/12 соответственно;
отсутствующие контакты и отрицательные service outcomes не удалены.

Дополнительно frozen stress manifest с1000Mbit products сравнил1/2 terminals
при той же геометрии: priority baseline доставил1/12 и2/12. Min service для
stress заранее задан0; это измеренный outcome, не гарантия или proven optimum.
Manifest и assessments сохранены в stress bundle, их можно воспроизвести через
`space-env scale-benchmark`. Всего16 logical executions и8 process restores.
Все declared science/resource/replay и pair-comparison gates прошли.

253 modules/286 source-data files сверены с wheel; SHA:
`e17b266d3557c71446e48d8dc8194a3ace85fb62928b3c0c63b825c93628d66a`.
Полные assessment/certificates совпали с worker witness hashes. Exact certificates
проверены отдельно, installed CLI/bundle audit прошёл. Прежние numerical/workflow
проверки также прошли. Исторические7 receipt families сохранены.
Нет expanded full suite, clean dependency/remote CI, новой hardware calibration
или coupled EO/pointing qualification. D2b/C2/Q6/Q7 остаются открытыми.
