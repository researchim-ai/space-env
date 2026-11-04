# D2f — multi-cycle missions and conservative geometry acceleration

## Что добавлено

`FullOrbitSpec` — отдельный bounded compiler, а не расширение неявного
контракта прежнего `CoupledMissionSpec`. Короткий профиль сохраняет предел600s.
Новый профиль поддерживает12000–21600s,2–3 release epochs,2–4 actors,
не более64 expanded requests и10000 решений. Каждый epoch должен вмещать
не менее1.025 максимального номинального периода орбиты:

`T = 2π sqrt(r³/μEarth)`.

Период задаёт размер горизонта, но не заменяет проверку движения.
`full_orbit_geometry` измеряет winding на actual J2 trajectory:
`unwrap(atan2(r·e₂,r·e₁))/(2π)` на сетке≤60s, где e₁/e₂ — базис
начальной osculating plane. Это projected winding с малой J2 precession,
не универсальный orbit-element estimator для произвольных манёвров.

Все actor×station пары, в том числе без контактов, сохраняются в diagnostic.
Геометрия не оптимизируется по результату policy и не передаётся policy.
Sites synthetic fixed global grid, не operating network.

## Разные уровни ADCS

- `rate-limited-kinematic/1`: bounded slew rate, settling, paid slew/tracking,
  body-fixed PV, battery/thermal и native payload-pointing checks. Не моделирует
  момент инерции/динамику маховиков и не получает rigid conservation certificate.
- `fixed-tick-rigid/1`: существующая rigid model с явным clock и
  max_total_substeps1e6; compiler ограничивает nominal estimate отдельным
  reserve bound. Поддержка компиляции не означает, что long rigid mission уже
  квалифицирована измерениями или всеми numerical refinement studies.

`coupled_evidence` различает tiers: `kinematic_paid_slew` проверяет paid
energy/time; `rigid_paid_conservation` остаётся прежним rigid witness.
Смешанные tiers одним uniform coupled certificate не квалифицируются.

## Периодические задачи и service gates

Для каждого исходного target создаются отдельные cycle target/request/raw/job/
delivery IDs. Координаты остаются исходными. Release/deadline и availability
ограничены epoch; destination сохраняется для одного site во всех cycles,
даже когда количество targets не кратно числу stations. Preload отсутствует.

`min_delivery_epochs` проверяет не просто наличие delivery после границы epoch:
release, actual acquisition и actual delivery должны принадлежать одному epoch.
Поэтому carryover старого продукта не доказывает новую съёмку следующего цикла.
В evidence сохраняются served epochs, request IDs и времена acquisition/delivery.
Все недоступные и невыполненные запросы остаются в denominator.

Positive floor не является гарантией compiler. Он проверяется фактическим
исполнением heuristic и replay; failed gate сохраняется как failure.
Геометрический diagnostic не является executable schedule/feasibility proof.
Оптимальный planner/teacher и отдельный feasible schedule certificate открыты.

## Запуск

```bash
python examples/coupled/full_orbit.py /tmp/full-orbit-spec.json --write-spec
space-env compile-full-orbit /tmp/full-orbit-spec.json /tmp/full-orbit-workload.json
python examples/coupled/full_orbit.py /tmp/full-orbit-geometry.json --geometry-only
python examples/coupled/full_orbit.py /tmp/full-orbit-campaign
```

Example фиксирует12600s,2 actors,8 fixed equatorial sites и8 targets,
2 epochs →32 requests; nominal floor2 deliveries в2 distinct epochs,
outage floor0 deliveries при minimum2 acquisitions. Nominal/outage имеют
direct/fresh-process пары; prefix6300s. Восстановление — audited prefix replay,
а не constant-time memory restore. Все durations, resource limits и floors
фиксируются до запуска. Exact nominal floor не доказывает промышленный SLA.

## Геометрия после maneuver

`ReferenceEphemeris.access(start_sec=...)` вычисляет suffix на прежней global
30s scan lattice. При refresh backend сохраняет past windows и пересчитывает
только от floor(now/30)×30 до horizon; straddling window склеивается.
Старые boundaries остаются harmless integration splits. Forecast всей орбиты
по-прежнему доступен через default start0. Root tolerance1ms остаётся numerical
root tolerance, не physical event accuracy; grazing passes могут пропускаться.

Оптимизация mass-only balanced RCS требует отдельного dense-coast envelope:
zero net force не отменяет расход массы и её влияние на drag/SRP.
Не следует приравнивать re-propagated coarse Hermite trajectory к исходному
sealed spline: при смене узлов меняется interpolation error. Первый coarse
comparison выявил это расхождение; он не считается пройденным gate.

В `orbits[actor].integration` требуется явное
`reuse_balanced_coast: true` и `sample_sec <= 1`. По умолчанию False и поле
опускается в legacy serialization. Coarse opt-in отклоняется до запуска.
Только exact built-in BalancedCoupleThrust с zero net force, gravity-only
mass-independent model, без navigation и active/future burns переиспользует
immutable coast. Drag/SRP и иные thrust classes всегда идут через eager path.
Масса, scalar fuel flow, maneuver/revision/hash и past arcs сохраняются;
flattened coast slices не образуют цепочку глубиной в число impulses.

`examples/validation/balanced_coast.py` сравнивает20 сегментов по0.025s на
60s dense J2 arc,601 common samples. До запуска заданы bounds0.01m,
1e−5m/s и1e−10kg; отдельно измеряются propagation calls40 против0.
Timing — диагностическое измерение host, не универсальное ускорение/SLA.

Для repeated decimal durations burn endpoint/mass вычисляются по фактическому
представимому интервалу `end-clock`. Это устраняет выход за coverage на доли ulp,
когда `clock+duration-clock < duration`; ошибка воспроизведена и покрыта
отдельной регрессией20 последовательных сегментов0.025s.

## Остаётся

Measured calibration C2; long rigid actuator/ADCS refinement; independently
executable feasible schedules и Q6 dataset/baseline split; realistic RF/network
hardware; Q7 durable job/OS isolation/auth/tenancy. Полный orbital horizon и
консервация declared ledgers не означают flight qualification.

## Приёмка — 2026-11-04

[B11_FULL_ORBIT](../acceptance/B11_FULL_ORBIT.json) связывает16 артефактов,
точный26-file UTF-8 archive, отдельные findings и installed-wheel identity.
Wheel SHA `894d6d435420427d605bd152f8204ae1a62c22c77046aa721596cae171246c90`:
260 Python modules/293 source+data files и README byte-matched.
Python-I вне checkout, system-site-packages venv: dependency-clean/remote CI
не запускались локально. Новый weekly/manual CI job использует clean venv и
2 workers; локальная frozen campaign использует4 workers.

Все4 full-mission attempts прошли на12600s: nominal8/32 delivered при8 acquired
и8 transformed, outage0/32 при8 acquired. Nominal fulfilled requests охватывают
оба release epochs. Direct/fresh-process signatures совпали в обеих парах,
prefix6300s прошёл в разных producer/consumer processes. Полный native replay,
resource/quality/service/export gates прошли. Projected J2 winding:
a-00 —2.1644913503422396,b-00 —2.163412858884664 оборота.

Dense-coast program прошёл:20 segments,601 common samples, propagation calls
40→0; max position difference2.614e−8m,velocity1.379e−7m/s,fuel0kg.
Это observed numerical differences на данном recipe; не физическая точность
полёта. Два дополнительных installed native actuator programs с dense opt-in
прошли ledgers/replay: nominal completed, limited сохранил ожидаемое failure.
Это2 programs, а не повтор полной18-member матрицы D2e на новом wheel.

1728 reference,176 targeted и110 fast passed; наборы пересекаются.
Coarse interpolation mismatch, decimal endpoint failure и намеренно прерванные
superseded runs сохранены в findings; не входят в denominator4+2 final attempts.
Полные final teacher checkpoints/exports пока не сохраняются benchmark runner:
их retention и executable feasibility certificate — следующий D2g.
