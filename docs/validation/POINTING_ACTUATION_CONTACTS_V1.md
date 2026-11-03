# B11-Q5-B — off-axis ориентация, тяга и фиксированные станции

Дата: 2026-11-03. Версия исходников 0.134.0; конкретная сборка определяется также
wheel SHA-256. Это законченный пакет **численной проверки объявленных моделей**.
Он не заменяет hardware calibration, квалификацию контроллера или всей миссии.
Физические уравнения и default-параметры действующих миссий не изменены.

## Запуск и результаты

```bash
pip install '.[reference]'
space-env reference-qualify-pointing /tmp/pointing.json
python -I /absolute/path/to/examples/validation/pointing.py /tmp/pointing-installed.json
```

Один запуск исполняет все **16 случаев, 612 численных сравнений**. Выборочного
пропуска неудобных случаев нет. Report хранит значения ошибок, фиксированные
допуски, точные inputs/times, status каждого случая и execution_errors.
Некорректная схема/подпись/область входов отвергается до solver work. Численное
несоответствие даёт `passed=false`; отказ solver сохраняется как execution_error,
остальные случаи продолжают проверяться. CLI атомарно пишет полный JSON и выдаёт
код 1 при failed evidence. Programming errors и отсутствующие зависимости не
маскируются под успешные численные результаты.

`passed=true` относится только к этим сравнениям. Итоговые поля всегда сохраняют
`qualification_status=needs_external_evidence`, `industrial_readiness=false`.

## Независимость и воспроизводимость

Frozen файл: `space_env/reference/fixtures/pointing_vectors_v1.json`.
Offline generator: `scripts/generate_pointing_vectors.py`. В нём нет импорта
space-env, SciPy, production force/attitude helpers или root finders.
ERFA используется только для **явно общего** преобразования GCRS→ITRS.

Три разных уровня сравнения:

1. Симметричный gyrostat: отдельное точное решение через матрицы Родрига,
   не quaternion RK4 и не повторение production RHS.
2. Несимметричная ориентация/тяга: независимо написанные component Euler balance,
   rotation-matrix kinematics и фиксированный RK4. В production используются
   quaternion RK4, SLERP path, DOP853 и Gauss quadrature. Для forced attitude
   семейство RK4 общее, но state representation и код уравнений разные.
3. Контакты: аналитическая круговая орбита, отдельные WGS84/ENU equations,
   independent 5s scan+bisection против native orbit/Hermite/look/Brent.
   ERFA, leap table, синтетические EOP и константы общие; независимой проверки
   вращения Земли этот пакет не даёт.

SHA тела fixture и исходника генератора записаны в provenance. Checker фиксирует
hash input/time domain каждого случая, запрещает подмену констант, времён,
инерции, шага, состава случаев и вставку пользовательского tolerance.
Expected значения могут быть повторно подписаны, но тогда они всё равно должны
выдержать численное сравнение. Контрольная сумма — integrity, не авторизация
или доказательство достоверности эталона.

Генератор не вызывается verifier и не пересчитывает expected при штатной работе.
Отдельные tests воспроизводят fixture в /tmp и проверяют refinement независимого
burn oracle .005→.0025 с. Это проверка стабильности эталонного вычисления;
погрешности модели и аппаратуры она не оценивает.

## 1. Вращение вне главной оси и насыщение колеса

Пять случаев имеют I=diag(2,2,3) кг·м², ω0=(0.12,−0.08,0.23) рад/с,
начальную ориентацию на 0.8 рад вокруг нормированного (0.3,0.4,0.5), inertia
колеса 0.01 кг·м², momentum capacity ±0.13 Н·м·с. Это трёхкомпонентное вращение
с наклонённой начальной ориентацией, не principal-axis spin.

Случаи: свободное движение с h0=0.03; motor torque 0.002 Н·м; положительное
и отрицательное насыщение с torque ±0.017 Н·м; реверс из h0=0.1 через ноль.
Насыщение в последних случаях специально не совпадает с интеграционной сеткой.
Проверяются 0.01,1,7,7.06,10,20 с.

Для симметричного gyrostat с колесом на оси z и без внешнего момента:

- H0=R0·(Iω0+h0 ez) постоянно в инерциальных осях;
- h(t)=h0+u min(t,tcap), tcap=(sign(u)hmax−h0)/u;
- A=min(t,tcap), ∫h dt=h0 t+u[A²/2+A(t−A)];
- φ(t)=H3 t/I3−(∫h dt)/I3−H3 t/I⊥;
- R(t)=Rot(H0/|H0|,|H0|t/I⊥)·R0·Rot(ez,φ(t));
- Hbody=RᵀH0, ω=(Hbody,x/I⊥, Hbody,y/I⊥, (Hbody,z−h)/I3).

Матрица сравнивается по Frobenius norm; дополнительно сравниваются angular
velocity, wheel momentum, boresight chord и угол до заданного inertial target.
Quaternion wxyz — body→inertial, q и −q эквивалентны. Boresight=(0.6,0,0.8),
target=normalize(1,−2,0.5), без принятия desired pointing за actual pointing.

Бюджеты до насыщения: rotation/boresight 2×10⁻⁸, δω 10⁻⁹ рад/с,
δh 10⁻¹⁰ Н·м·с, pointing 2×10⁻⁶ градуса. После насыщения: rotation/boresight
10⁻⁶, δω 10⁻⁶ рад/с, pointing 10⁻⁴ градуса. Это инженерные numerical budgets,
не hardware accuracy.

**Saturation timing — отдельная величина.** Production ограничивает torque на
частичном подшаге, а diagnostic считает capacity-limited whole substeps.
Он не является точным event split. Для duration diagnostic допустимо .010001 с
при шаге .01 с. Физическая attitude ошибка ограничена отдельно. Ещё один gate
требует, чтобы шаг .005 с давал fine error ≤0.6×coarse error+10⁻¹⁰.
Ни точность момента насыщения, ни непрерывный диапазон инерций не объявлены
квалифицированными одним успешным случаем.

Базовые free symmetric body/Euler equations см.
[MIT classical mechanics notes](https://live.ocw.mit.edu/courses/8-09-classical-mechanics-iii-fall-2014/d9bac33f6c60b304dc0398e99b327102_MIT8_09F14_full.pdf).
Wheel terms и saturation law здесь относятся к объявленной модели gyrostat;
этот документ не является внешним hardware reference для неё.

## 2. Разгрузка импульса внешним моментом

Шестой attitude case: I=diag(2,3,4), h0=0.1, motor torque z=−0.017 Н·м,
такой же external body torque; времена .01,1,5,10 с. Wheel momentum проходит
через ноль без насыщения. External moment не выдаётся за бесплатную внутреннюю
перекачку импульса.

Oracle интегрирует матрицу ориентации Ṙ=R[ω]× и component equations:
Iω̇=τexternal−u−ω×(Iω+h), ḣ=u. Native интегрирует quaternion. Здесь сравнивается
идеальный kernel; RCS fuel/electrical/thermal ledger проверяется отдельно в
существующем native mission demo, не выводится из этого standalone эталона.

## 3. Центрированная и смещённая вращающаяся тяга

Два burn cases: начальная GCRS орбита r=(7000,0,0) км,
v=(0,√(μ/r),0), m0=100 кг, dry mass 90 кг, F=.5 Н, Isp=300 с, duration10 с.
Force direction в BODY=(.6,0,.8); начальная ориентация и spin наклонены.

Центрированный случай использует точную symmetric gyrostat attitude.
Смещённый имеет I=diag(2,3,4), h0=.01 и lever arm (.1,.2,−.05) м.
Body moment τ=rlever×Fbody=(.08,−.055,−.06) Н·м меняет ориентацию, а новая
ориентация меняет направление тяги в GCRS.

Oracle совместно интегрирует Cartesian r/v и rotation matrix с шагом .005 с:

m(t)=m0−Ft/(Isp g0),
r̈=−μr/|r|³+(F/m)R(t)b,
Ṙ=R[ω]×,
Iω̇=τ−ω×(Iω+h).

Force impulse проверяется independent composite Simpson, production использует
Gauss на SLERP intervals. В native выполняются rigid RK4 path, production
lever torque, RotatingThrust и DOP853, затем сравниваются r/v/m в 1,5,10 с,
attitude/rates/momentum в тех же точках и direction на шести off-grid временах.

Заранее заданные бюджеты: direction 3×10⁻⁶ chord, impulse 10⁻⁵ Н·с,
position .002 м, velocity 2×10⁻⁵ м/с, mass 10⁻¹⁰ кг; attitude matrix 2×10⁻⁸,
angular rate10⁻⁹, momentum10⁻¹⁰. Потеря массы мала (~1.7×10⁻⁵ от начальной),
inertia постоянна. Variable inertia, slosh, gimbal dynamics, exhaust plume,
полный спектр engines и аппаратная калибровка этим не подтверждены.

Это отдельная проверка физического пути, не доказательство native admission,
сcheduling, power/thermal или closed-loop payload quality. Для них используются
другие runtime tests/receipts.

## 4. Контакты с фиксированными станциями

Четыре заранее заданных условных site: (0°,0°), (40°,−3°), (−35°,149°),
(78°,15°), altitude100 м, minimum elevation5°. Названия equatorial/iberia/
australia/arctic — географические labels, не заявления о реальных ground terminals.
Координаты не вычисляются из ground track. Две круговые орбитальные плоскости:
radius7000 км, phase .4 рад; (i=.7, Ω=.2) и (i=1.7, Ω=1.4) рад.

Horizon12000 с от 2026-07-18T12:00Z. Явные constant synthetic EOP:
DUT1=.1 с, xp=.05″, yp=.25″. Production trajectory — central DOP853 с output
каждые5 с, затем cubic Hermite. Oracle использует analytical circular r/v,
explicit WGS84 position/up/ENU. Frame matrices ERFA общие, это не IERS validation.

Сравниваются range, range rate, azimuth, elevation, Doppler и inertial direction
до fixed target. Oracle range rate — central difference range(t±.1)/.2,
production — transformed velocity с five-point derivative Earth rotation.
Угол азимута сравнивается по кратчайшей разности, включая wrap через0/360°.
`pointing_quaternion` проверяется через actual rotated +Z, а не совпадение
формата quaternion с выбранным roll convention.

Contact roots: отдельные scan5 с+bisection до10⁻⁷ с против native scan20 с+Brent
с root tolerance .0001 с. Бюджет границ .02 с **отдельно** от root tolerance.
Range budget .1 м; range rate .02 м/с; elevation/azimuth10⁻⁵ градуса;
Doppler .2 Гц при2.2 ГГц; pointing chord10⁻⁸.

Есть семь окон в четырёх комбинациях plane/site, остальные четыре корректно
дают отсутствие контакта. Window count gate требует точного совпадения числа,
а boundary gate — совпадения границ. Потеря окна и ложный контакт проваливают
проверку; legitimate empty result не считается ошибкой сам по себе.

Ни один sampled search не получает general completeness certificate: короткие
и grazing события между узлами остаются ограничением. Не проверяются refraction,
terrain, weather, RF service или shared terminal booking.

Общий frame backend описан в
[SOFA Earth attitude cookbook](https://www.iausofa.org/cookbooks). Сам факт
использования SOFA/ERFA не является независимой валидацией frame/EOP truth.

## Приёмка и ограничения результатов

`docs/acceptance/B11_POINTING.json` связывает full numerical report, installed
wheel и native smoke receipts. Manifest перечисляет source/data hashes и версии
зависимостей; source version и installed distribution version раздельны.
Исторические Q5-A/Q4 receipts сохраняются вместе с исходными wheel SHA и не
переносятся на новую сборку. Remote clean CI и hardware measurements нельзя
подменять локальным system-site-packages venv.

52 новых tests проверяют schema/domain integrity, неверные повторно подписанные
vectors, solver failure denominators, oracle refinement, frozen-inertial-force
ошибку, неверный pointing goal, continuation после сериализации и CLI failure.
Cold/warm continuation одной и той же разбивки должна совпадать точно; сравнение
с unsplit integration использует численный допуск, не обещает bitwise equivalence
для произвольного переразбиения времени.

Следующие самостоятельные уровни:

- Q5-C: hardware parameter/uncertainty calibration и holdout измерения.
- Q5-D: nonconstructive mission tasking с actual closed-loop slew/tracking,
  несколькими плоскостями/станциями, shared resources и frozen mission gates.
- Общая Q5-B математическая база не является независимой квалификацией такого
  контроллера или customer mission; эти gaps остаются в full report явно.


## Зафиксированный результат 2026-11-03

- 1397 reference / 110 fast / 52 targeted tests, collected1827; counts overlap.
  Уточнённая проверка всех численных contact boundaries прошла отдельно.
- Final wheel: `7505302429c917b9b5cf90e988f4a9776d27f020ba11096d14b883fe9166e9a7`.
  241/241 Python modules, fixture и текущий README совпали; source/data inventory
  содержит271 запись и совпадает между checkout и установленным report.
- Installed Q5-B: 16/16 cases, 612/612 comparisons. Q5-A на той же сборке:
  119 analytic checks, три external orbit fixtures и backend/circular checks passed.
- Installed chemical/electric/off-center demos: service2/2 и completed engine.
  Nominal RCS: completed unload, service2/2. Limited RCS: **failed**
  `unload_window_ended`, service2/2; expected operation failure остаётся в receipt.
- Ground native smoke: paired recovery, observed ACK cancel/resubmit, native
  replay и verified RL/SFT. Это короткие operational tests, не 12-actor large
  campaign и не independent fixed-site tasking/control qualification.
- Wheel установлен в system-site-packages venv; dependency-clean/remote CI,
  expanded full и large native workload на новой сборке не запускались.

Все raw reports и compressed native artifacts связаны SHA-256 через
`docs/acceptance/B11_POINTING.json`. Исторические Q5-A/Q4 artifacts проверены и
сохранены без изменений.
