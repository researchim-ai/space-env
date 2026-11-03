# B11-Q5-A — проверяемые численные режимы и независимые аналитические векторы

Дата: 2026-11-03. Версия пакета: 0.134.0. Это первый законченный подпакет Q5,
а не закрытие всей независимой физической квалификации. Реализация физического
ядра и настройки миссий не меняются. `industrial_readiness=false` остаётся явным.

## Как запускать

```bash
pip install '.[reference]'
space-env reference-qualify-envelopes /tmp/envelopes.json
space-env reference-qualify-envelopes /tmp/envelopes-full.json --long --external
python -I /absolute/path/to/examples/validation/envelopes.py /tmp/installed.json --long --external
```

Обычный запуск измеряет 103 проверки для 10 случаев. `--long` включает заранее
замороженные дополнительные точки обеих эллиптических орбит до 259200 с, всего
119 проверок. `--external` дополнительно запускает уже существующие frozen
Boinor J2/J2+third-body comparison и upstream ERFA regression/circular cases.
Эти проверки имеют отдельные результаты и уровни независимости; их нельзя
складывать с analytic checks и называть числом независимых внешних эталонов.

Полный JSON записывается атомарно; CLI печатает краткую сводку. При численном
несоответствии или отказе решателя аналитического набора итог `passed=false`, CLI завершает работу с
кодом 1. Отказ решателя сохраняется как `execution_error` соответствующего
случая, остальные случаи продолжают проверяться. Некорректный fixture отвергается
до любого solver work; это ошибка входного контракта, не физическое измерение.
Ошибки установки необязательных reference-зависимостей не маскируются.

## Откуда взялись эталоны

`reference/fixtures/analytic_vectors_v1.json` хранится в установленном пакете.
Ожидаемые числа генерируются только явным offline-запуском
`scripts/generate_analytic_vectors.py`, использующим стандартную библиотеку Python.
В нём нет импорта space-env, NumPy, SciPy, ERFA или вызова производственного RHS.
В проверяющем коде нет импорта генератора и нет пересчёта ожидаемых значений.

SHA-256 тела fixture и исходника генератора входят в provenance. Дополнительно
checker фиксирует hash input/time domain каждого случая. Повторная подпись не
позволяет расширить режим, подменить константы, разредить sampling или передать
свой tolerance. Заранее объявленные budgets находятся в коде checker. Неверный,
но повторно подписанный expected vector должен давать численный fail — это
отдельно проверяется для каждой из пяти групп equations.

Уровень независимости: **independently-coded-analytic-equations/shared-parameters**.
Константы, начальные состояния, синтетические параметры и binary64 arithmetic
общие. Это независимая реализация математического решения, а не внешняя
эфемерида, измерение аппарата, сертификация стандарта или калибровка оборудования.
Проверка RF использует ту же аналитическую форму и арифметическую библиотеку;
её независимость слабее другого solver/backend и не исключает общей ошибки формулы.

## Орбитальное движение и интерполяция

Два фиксированных центральных случая: a=8000 км, e=0.1, i=0.7 рад и a=14000 км,
e=0.5, i=1.7 рад. Общие Ω=0.4, ω=0.3, M0=0.2 рад, μ=3.986004418×10¹⁴ м³/с².
Перицентры 7200 и 7000 км находятся вне Земли. Frame GCRS, UTC epoch
2026-07-18T12:00:00Z. Здесь **нет J2, drag, SRP, третьих тел или тяги**.

Генератор решает M=E−e sin E делением интервала пополам (100 итераций), затем:

- n=√(μ/a³), M=(M0+nt) mod 2π;
- x=a(cos E−e), y=a√(1−e²) sin E;
- r=a(1−e cos E), vx=−√(μa)sin E/r, vy=√(μa)√(1−e²)cos E/r;
- положение и скорость поворачиваются в инерциальные оси явной матрицей Ω,i,ω.

Native DOP853 использует Cartesian ODE. В узлах 60-секундной сетки сравниваются
его выходные r/v: бюджеты 2 м и 0.003 м/с. Между узлами проверяется именно
`ReferenceEphemeris` с cubic Hermite: отдельные бюджеты 20 м и 0.5 м/с.
Нельзя выдавать ошибку узлов ODE за точность интерполяции/контактов.

Полный набор времён: 0,30,60,330,600,3630,86400,259170,259200 с. Обычный запуск
исполняет точки ≤600 с; остальные явно отмечает как deferred. Это выборочные
проверенные точки, не доказательство accuracy во всём непрерывном диапазоне.

Уравнение эллиптического случая и ограничения классических элементов обсуждает
[NASA NAIF prop2b](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/prop2b_c.html).
Здесь SPICE не вызывается. Околопараболический режим не охвачен; universal-variable
решение и отдельные границы применимости остаются следующим расширением.

## Тепловая сеть

Два узла с C=100 Дж/К, K=0.5 Вт/К и T0=(300,280) К, q=(0,0) либо (4,0) Вт.
Bath, radiation, heaters и Earth environment выключены. Для равных ёмкостей:

- средняя температура T̄(t)=290+(q1+q2)t/(2C);
- Δ∞=(q1−q2)/(2K), Δ(t)=Δ∞+(20−Δ∞)exp(−2Kt/C);
- T1,2=T̄±Δ/2.

Один излучающий узел: C=1000 Дж/К, ε=0.8, A=0.5 м², T0=350 К, внешнего
поглощённого потока нет. Решение C dT/dt=−εσAT⁴:

T(t)=[T0⁻³+3εσAt/C]⁻¹/³.

Native radiation kernel моделирует излучение в 0K; адаптеры окружения добавляют
поглощённый поток отдельно. Этот пример не утверждает, что реальный спутник
находится в 0K пространстве без солнечной/земной нагрузки. Используется округлённое
σ=5.670374419×10⁻⁸ Вт/(м²·К⁴); полное значение приведено у
[NIST CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?sigma%7Csearch_for=stefan+boltzmann).

Точки 1,10,60,100 с проверяются двумя production solvers: DOP853 и Radau.
Температурный бюджет 10⁻⁶ К для conduction и 10⁻⁵ К для radiation. Дополнительно
проверяются ledger balance residuals: 10⁻⁶ и 0.001 Дж соответственно. Баланс —
внутренний consistency check, не ещё один независимый физический эталон.

В harness явно задано max_evaluations=50000 на вызов при max_step=0.05 с;
исходный default 20000 исчерпывается на 100-секундном DOP853 dense-output шаге.
Рабочие настройки симулятора не изменены. Точностные бюджеты после запуска не
ослаблялись. Время/температуры/источники за пределами этих случаев не квалифицированы.

## Ориентация и колесо

I=diag(2,3,4) кг·м², инерция каждого колеса 0.01 кг·м². Проверяются 1,10,60 с:
coast с ωz0=0.1 рад/с; torque колеса +0.001 Н·м; внешний torque +0.001 Н·м.
Случаи не смешиваются, насыщения нет. Начальный quaternion — identity, формат
wxyz body→inertial. Для постоянных principal-axis torques:

h_z=τ_motor t,
ω_z=ω_z0+(τ_external−τ_motor)t/I_z,
θ=ω_z0 t+(τ_external−τ_motor)t²/(2I_z),
q=(cos(θ/2),0,0,sin(θ/2)).

Бюджеты: quaternion chord min(‖q−q*‖,‖q+q*‖) ≤10⁻⁹,
‖δω‖≤10⁻¹¹ рад/с, ‖δh‖≤10⁻¹¹ Н·м·с. q и −q эквивалентны.
Это standalone kernel, не mission-integrated slew/tracking qualification.
Off-axis motion, saturation, jitter, wheel imbalance, friction и hardware truth
требуют самостоятельных эталонов и coupled-миссий.

## Радиолиния

S-band 2.2 ГГц / 700 км и X-band 8.2 ГГц / 2000 км. Сравниваются FSPL,
EIRP, C/N0 и information-bit margin при явно заданных scalar losses и gains.
FSPL=20log10(4πrf/c); EIRP=10log10(P)+G−Lfeed;
C/N0=EIRP−FSPL+G/T−Lpolarization−Lother−Lfade−10log10(k);
margin=C/N0−10log10(Rinfo)−(Eb/N0)required−Limplementation.

Все четыре численных бюджета — 10⁻¹⁰ дБ, только для вычисления данной формулы,
а не допустимая физическая погрешность линии. Free-space attenuation — область
[ITU-R P.525](https://www.itu.int/rec/R-REC-P.525/en); полный стандарт и реальные
weather/channel/modem модели этим скалярным тестом не квалифицируются.

## Что означает итоговый JSON

`evidence_registry` разделяет аналитические уравнения, external-force formulas
с общими входами, regression того же frame backend и native replay. `not_run`
означает отсутствие проверки в этом запуске, а не отсутствие реализации.
`evaluated_cases` хранит точные inputs, times, status и scope. `unqualified`
перечисляет оставшиеся свидетельства для orbit/attitude/power/thermal/payload/
RF/DTN/coupled mission. `passed=true` означает только прохождение запущенных checks.

Manifest перечисляет SHA-256 всех package .py/.json/.csv/.yaml/.yml, aggregate digest,
версии NumPy/SciPy/ERFA/Pydantic. Версия исходников пакета и версия установленного
distribution записываются отдельно; их несовпадение явно отмечено. Installed
acceptance требует совпадения и импорта из проверяемого venv. Это диагностический inventory, не wheel signature
и не идентификатор qualification native mission. Контрольная сумма отчёта защищает
от случайной порчи, но сама по себе не удостоверяет автора/достоверность данных.

Исторический `B11_GROUND.json` сохраняет свой исходный wheel SHA и результаты
40/40. Добавление validation source меняет сборку. Большой Q4 campaign на новой
сборке не повторяется и не считается заново квалифицированным; повторяется
installed small coupled smoke с replay/RL/SFT, плюс reference-регрессии.

## Следующие подпакеты Q5

1. Q5-B: независимые off-axis attitude/pointing, saturation/unloading, body-thrust
   trajectories и контакты на nonconstructive AOI; uncertainty budgets отдельно
   от solver error. Сначала фиксировать эталоны, затем менять controller.
2. Q5-C: реальные измерения/публикуемые параметры батарей, optical surfaces,
   антенн и instrument quality; declared parameter distributions и holdout.
3. Q5-D: multipane/multistation tasking с явным slew и изменяемыми shared resources;
   frozen native qualification, сбои/recovery и resource gates для новой сборки.

Q6–Q8 остаются в `docs/INDUSTRIAL_NEXT_AFTER_GROUND.md`. Этот пакет не заменяет
все эти задачи одним успешным набором аналитических тестов.


## Фактическая приёмка

Машиночитаемый итог: `docs/acceptance/B11_ENVELOPES.json`, исходный полный installed
отчёт `B11_ENVELOPES_FULL.json`, отдельные ground/wheel receipts рядом. Wheel SHA-256:
`479414a8bf156dc751d40ea9a2edb171b3837bd85f05ad89c919ab97c8b13af5`.
Все 240 package Python modules и frozen fixture совпали с исходниками побайтно;
269 записей source/data inventory совпали между installed отчётом и checkout.

119 analytic checks, три существующих external-formula arcs и backend/circular
regressions прошли. Для новых elliptic cases максимальная ошибка узлов до 72ч:
7.4841×10⁻⁵ м / 2.3287×10⁻⁸ м/с; в проверенных Hermite midpoint:
0.5491 м / 3.0199×10⁻⁴ м/с. Это максимумы только среди измеренных точек.

42 targeted tests прошли; reference — 1339, fast — 110, collected — 1775.
Counts overlap; reference включал первые 36 новых tests, а оставшиеся шесть и
provenance-only fix проверены финальным targeted запуском всех 42.
Final-wheel native small mission подтвердила paired recovery, observed ACK
cancel/resubmit, native replay и verified RL/SFT. Dependency-clean/remote CI,
expanded full suite и большой 12-actor campaign на новой сборке не выполнялись.
