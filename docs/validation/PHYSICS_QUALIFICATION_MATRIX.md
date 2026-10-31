# B02: управляемая GCRS-динамика и границы квалификации

Дата: 2026-10-31. Профиль `reference-dynamic-downlink/1`.
Статус: **needs_external_evidence**. Инженерный путь команд реализован;
полная независимая квалификация возмущенной миссии не завершена.

## 1. Что действительно связано

`space_env.runtime.dynamic_reference_scenario` создает scenario v3 с backend
`reference-dynamic`. Definition имеет schema `space-env.reference-dynamic-spec/1`:
`orbits`, `eop`, `horizon_sec`, `operations`, optional `weather`, `max_maneuvers`.
В `orbits` каждый аппарат задает GCRS initial r/v и UTC epoch, propulsion
(dry mass, propellant, Isp, max thrust, max impulsive delta-v), ForceModel switches
и integration controls. `operations` использует существующие модели станций,
продуктов и ресурсов; в ней запрещено подменять ephemerides/horizon.
Текущая масса для drag/SRP выводится из сухой массы и остатка топлива.

`StateProvider` задает общий state/look/access контракт. Фиксированный provider
не экстраполирует. DynamicStateProvider хранит отдельные sealed arcs; spline не
пересекает скачок скорости. После импульса state в его эпоху правосторонний,
а запросы прошлого возвращают прежнюю траекторию. Кадры не переименовываются:
legacy по-прежнему отдельный профиль, неподдерживаемый maneuver отвергается.

## 2. Публичные команды и физические ограничения

Импульс:

```json
{"command_id":"kick","actor_id":"a","kind":"impulse","duration_sec":0,
 "impulse":{"frame":"GCRS","delta_v_m_s":[0,0,10]}}
```

Выполняется атомарно при submit в текущую эпоху. Position непрерывна, velocity
получает delta-v; масса меняется по уравнению ракеты при заданном Isp.
Недостаток топлива/неверный frame/лимит delta-v не изменяет состояние.
Нулевая длительность разрешена только impulse. Actor должен быть свободен.

Конечное включение:

```json
{"command_id":"engine","actor_id":"a","kind":"finite_burn","duration_sec":4,
 "burn":{"frame":"GCRS","direction_gcrs":[0,1,0],"thrust_n":2}}
```

Применяется постоянная инерциальная GCRS-направленность и постоянная тяга.
Ускорение F/m(t) входит в RHS; m(t)=m0-F*t/(Isp*g0) вычисляется аналитически
через существующий `core.propulsion.mass_after_thrust`. Силы drag/SRP используют
эту массу. Направление unit-vector проверяется; RTN/body-frame не поддержаны.
Cancel останавливает тягу в текущую эпоху и перестраивает оставшийся coast.
Уже израсходованное топливо и пройденная часть дуги сохраняются.
При submit строится условный прогноз: будущее топливо не списывается в текущей
телеметрии. Все взаимодействия двигателя с ADCS/питанием/теплом еще отсутствуют.
Это модель совершенной инерциальной ориентации, не цифровой двойник двигателя.
Согласованность F, Isp и mass flow: [Tudat mass dynamics](https://docs.tudat.space/en/latest/user-guide/state-propagation/propagation-setup/mass.html),
[NASA specific impulse](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/specific-impulse/).

## 3. Геометрия, ревизии и доступности

После принятого maneuver или cancel заменяются будущие окна связи данного
аппарата, повышается geometry_revision и общая runtime revision. Forecast
содержит обе ревизии: старый план не должен исполняться без проверки revision.
Downlink использует обновленные окна и фактическое состояние ресурсов.
Траектории остальных аппаратов не меняются. Старые integration boundaries
остаются допустимыми лишними разбиениями, но не сохраняют прежнюю доступность.
Acquisition еще не является частью reference-профиля; перестройка окон съемки
закрывается вместе с B03, а не симулируется созданием заранее готовых байтов.

## 4. Предварительно заданные error budgets

- Центральная сила, круговая орбита 7000 км, 600 с / 24 ч / 72 ч:
  прежний closed-form контроль 1 м по r, 0.001 м/с по v на узлах интегратора.
  Это не допуск для всех возмущенных орбит.
- Frozen external-formula central+J2, две наклонные слабоэксцентричные дуги
  24/72 ч: 10 м и 0.01 м/с по всем часовым узлам. Измерено около
  0.00535/0.01925 м и 5.90e-6/2.06e-5 м/с соответственно.
- Constant inertial finite burn, central gravity, 50 с: сравнение production
  DOP853 с отдельно записанным RHS + RK45; допуски 0.01 м, 1e-5 м/с,
  1e-10 кг. Численное согласие не проверяет истинную тягу, Isp или pointing.
- Hermite interpolation: отдельный тест круговой 7000-км дуги при cadence 60 с,
  off-grid допуски 1 м / 0.03 м/с. Ошибка интерполяции больше solver atol;
  допуски интегратора нельзя переносить на любое off-grid state.
- Геометрический AOS/LOS root tolerance 0.001 с — настройка поиска корня,
  не доказательство точности физического контакта. Пропуски коротких/grazing
  проходов между scan nodes остаются известным ограничением.
- RK/DOP853 convergence и разные command advance разбиения проверяются отдельно.
  В runtime coarse/fine траектория согласована, поскольку прогноз не строится
  заново на каждом агентском шаге; это не независимый интеграторный oracle.

Для будущего mission profile бюджет обязан отдельно учитывать initial orbit /
navigation uncertainty, perturbation model, atmosphere/weather, EOP/leaps,
интегратор, интерполяцию и event search. Суммарный промышленный бюджет пока не
назначен: отсутствует целевая миссия с измерениями и независимыми истинами.

## 5. Происхождение эталонов

В `reference/fixtures/boinor_j2_*.json` заморожены SI r/v, epoch/frame,
EOP, constants, generator/library hashes, версия Boinor, MIT provenance и
настройки LSODA. Генератор: `scripts/generate_boinor_orbit_fixtures.py`.
Он вызывает внешнюю J2_perturbation и не использует production acceleration.
Проверяющий код DOP853 использует другие формулы и integration family.

Общие входы явно перечислены: ERFA frames, EOP, initial state, SI constants.
Следовательно, это **external-force-formula/shared-inputs**, а не полностью
независимые frames, ephemerides, атмосфера или operational orbit truth.
Fixtures не перегенерируются при тестировании/установке. Измененный checksum
отвергается; повторно подписанная неверная дуга должна провалить численный допуск.
Проверяющий API: `space_env.validation.orbit_fixtures.verify_orbit_fixture`.
Importer принимает central+J2 и явно объявленный J2+Sun/Moon third-body scope.
Дополнительная 24h дуга имеет другую эпоху/наклонение и общий provider Sun/Moon,
что явно указано в provenance.

## 6. Данные и сроки применимости

- Leap-second input: явная существующая граница 1972–2026 UTC, без расширения
  в будущее ради прохождения миссии. Leap table hash сохраняется в manifest.
- EOP: конечные ежедневные узлы, gaps/выход за покрытие запрещены; для velocity
  нужен также halo численной производной. Prediction требует boolean opt-in,
  статус остается в metadata; packaged synthetic/final/prediction не смешиваются
  с заявлением о flight-quality inputs.
- Drag: 150–800 км, без тихой экстраполяции. Static weather validity и источник
  обязательны для dynamic drag; прогноз требует explicit opt-in. F107/F107a/Ap
  берутся из конфигурации ForceModel и объявляются постоянными на этом интервале.
  Это не загрузчик реального time-varying weather или wind model.
- Trajectory: не более 72 ч, sample cadence <=60 с, production output <=100000
  intervals на дугу; scenario horizon должен покрываться всеми аппаратами.
- Max maneuvers ограничивает принятые включения; cancel остается доступен.
  Лимиты — защита от неограниченного ввода, не доказательство SLA/масштабируемости.

## 7. Силы и пока отсутствующие эффекты

Dynamic-профиль использует существующие явные ForceModel switches: central,
terrestrial J2 и вращающаяся атмосфера через GCRS/ITRS Jacobian, SRP с shadow,
Sun/Moon differential third body. Есть короткий integration test совместного
включения всех переключателей с меняющейся массой. Полная независимая проверка
совместных возмущений, SRP/eclipse и атмосферных данных еще не выполнена.

Отложены: higher-order/tesseral gravity, dX/dY, solid/ocean tides, relativity,
реальные atmospheric winds, coupled attitude/area, calibrated propulsion,
mass/pointing process noise и полноценная навигационная оценка. В B02-Q1 добавлены
optional Cartesian STM/P propagation, constant acceleration PSD и impulse velocity
noise; [контракт](NAVIGATION_COVARIANCE_CONTRACT.md). Общая capability требует
navigation на всех актерах. Это не filter/flight uncertainty qualification.
Добавление эффектов требует error
budget выбранного класса миссий, а не общего обещания «вся физика».

## 8. Воспроизводимый путь и стоимость

```sh
python examples/unified_runtime/dynamic_demo.py /tmp/dynamic-demo
space-env mission-validate /tmp/dynamic-demo/scenario.json
space-env mission-run /tmp/dynamic-demo/scenario.json /tmp/dynamic-demo/plan.json /tmp/dynamic-demo/cli-prefix.json
space-env mission-resume /tmp/dynamic-demo/cli-prefix.json /tmp/dynamic-demo/resume.json /tmp/dynamic-demo/cli-final.json
space-env mission-replay /tmp/dynamic-demo/cli-final.json
space-env reference-qualify-orbits /tmp/physics-qualification.json
```

Приемка: 18 новых тестов; оба reference/legacy runtime контракта сохранены;
reference 305 passed, fast 110 passed. SDK demo доставляет 5 Mbit после
impulse/burn и восстановления из mid-burn checkpoint. Полный regression не
перезапускался. Это результаты проверок, не заявленный performance benchmark.

На maneuver пересчитывается оставшаяся дуга и геометрия станций; prefix-checkpoint
заново исполняет историю. RAM/latency растут с horizon/числом маневров/контактов.
Промышленные нагрузки и bounded-memory streaming остаются B09/B08.

## 9. Что нужно до полной приемки B02

Независимые combined-perturbation fixtures разных эпох/орбит, калибровка optional navigation
covariance/STM/process-noise на независимых perturbed fixtures, независимые eclipse/AOS/LOS/event и
миссионные исходы; error budget целевого промышленного профиля. B02.09–B02.11
остаются незакрытыми полностью. Промышленная приемка не выводится из числа тестов.
Машинный отчет: `docs/acceptance/B02_PHYSICS.json`.


## 10. Обновление B02-Q1

Реализованы optional STM/P, белый acceleration PSD, impulse covariance jump,
глобальная композиция и checkpoint/cancel. Добавлена другая эпоха/наклонение с
J2+Sun/Moon frozen shared-input comparison. Текущие результаты и ограничения:
[контракт](NAVIGATION_COVARIANCE_CONTRACT.md), [приемка](../acceptance/B02_Q1_NAVIGATION.json).
Предыдущий B02_PHYSICS.json сохраняет исторический результат до Q1; текущие
hashes fixtures и wheel находятся в новом acceptance artifact.


## 11. Обновление B02-Q2

Добавлены bounded readonly point-acquisition/eclipses forecasts и atomic
optional forecast binding. AOS/LOS и point-target windows сравнены с analytic
orbit/независимыми уравнениями и bisection; конусы eclipse — с apparent-disc
sphere geometry, с явным model-comparison budget. Новая геометрия не меняет
существующие SRP/linear-penumbra flux или midpoint solar-resource integration.
[Контракт](GEOMETRY_EVENTS_CONTRACT.md), [приемка](../acceptance/B02_Q2_GEOMETRY.json).
B02.05 geometry gate закрыт; B02.09/.10/.11 production qualification открыта.
B03 может развивать execution на проверенном surrogate, не выдавая это за
завершенную промышленную квалификацию.
