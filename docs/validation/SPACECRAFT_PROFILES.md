# B04-A: связанный борт — инженерный контракт

Дата: 2026-10-31. Backend `reference-spacecraft`, профиль
`reference-spacecraft-bus/1`, objective `fulfilled_requests`.
Статус B04-A: verified в пределах проверенных surrogate-моделей. Родительский B04
остается in_progress; физическая квалификация B02 — needs_external_evidence.

## Что связано причинно

У каждого аппарата есть quaternion body→GCRS в порядке **w,x,y,z**, ограниченная
скорость разворота, settling, мощность ADCS, температура, защелкнутые safe reasons,
fault masks и degradation. +Z тела — ось прибора и направленной антенны.
`synthetic-agile/1` и `synthetic-constrained/1` — разные воспроизводимые конфигурации,
а не модели летавших аппаратов. Одинаковая нагрузка может выполняться первым
аппаратом и блокироваться вторым из-за ограничения мощности; время разворота
также различается.

Конфигурация `spacecraft_bus` обязательна для каждого actor; она допустима только
в этом профиле. Versioned model_id/revision, sources, parameter_status задают
происхождение параметров. Статус published/calibrated является декларацией автора
сценария, не сертификатом библиотеки. Поставляемые presets помечены synthetic.
Неподдерживаемые эффекты не имеют скрытой калибровки.

Команда `slew` принимает единичный quaternion в GCRS и ограниченное control window.
Поворот занимает время и энергию, затем оплачивается settling. Недостаточное окно
завершает команду неуспешно, сохраняя фактическую достигнутую ориентацию. Отмена
не откатывает израсходованный ресурс. Нулевое время или нулевая скорость в общем
`slew_step` сохраняют исходный quaternion вместо прежнего мгновенного разворота.

В hold ориентация фиксирована относительно GCRS. Съемка не телепортирует аппарат
к цели: необходима начальная точность pointing, после чего выполняется ограниченное
по скорости слежение. Проверяются settling, pointing и Sun keep-out в начале,
середине и у левой границы конца каждого integration interval. Это sampled
contract: узкие нарушения между samples не исключаются. Радио проверяет pointing
на начале подшага; tolerance=180° явно означает ненаправленную антенну.
Geometry forecasts остаются геометрическими возможностями, не гарантией
ресурсной выполнимости. Исполнитель проверяет актуальный борт.

## Энергия и тепло

Native VehicleResources должен объявлять `solar_mode="body_fixed_bus"`.
Нормаль панели в body-frame вращается фактическим quaternion; мощность зависит
от положительной проекции на Солнце, площади/КПД панели и существующей conical
shadow модели. Независимого идеального Sun tracking здесь нет.

Батарея имеет явную **двухпортовую** модель: весь поток PV проходит через зарядный
конвертер, вся полезная нагрузка — через разрядный. За dt секунд:

- `generated_Wh = raw_PV_W * eta_charge * dt / 3600`;
- `consumed_Wh = useful_load_W / eta_discharge * dt / 3600`;
- `E = E_initial + generated - consumed - curtailed`.

Это не net-only электрическая схема с непосредственным питанием нагрузки от PV.
Базовая нагрузка, payload, CPU, радио и активный ADCS используют согласованный
учет. Ограничение max_load относится к полезной электрической нагрузке, до потерь
разрядного конвертера. Nominal и safe base demand не могут превышать bus rating.
При насыщении аккумулятора excess charging считается curtailed. Aging,
внутреннее сопротивление, зависимости КПД и емкости от температуры не реализованы.

Тепло — **один lumped RC узел электроники** с постоянным ambient:
`C*dT/dt = heat_W - conductance_W_per_K*(T - ambient_K)`.
Heat состоит из заданной доли электрической нагрузки и потерь конвертеров;
аналитический шаг RC использует среднюю за подшаг мощность. В отчете отдельно
даны накопленное/отведенное тепло и residual. Независимый native TX thermal
запрещен, чтобы не создавать два несогласованных тепловых ledger.
Нет многослойной конструкции, нагрева Солнцем/альбедо/Earth IR, радиационных
панелей, измеренной flight temperature или доверительных границ калибровки.

Схема использует существующий integration grid плюс границы операций, завершения
разворота/settling и энергетического floor. PV удерживается на подшаге.
Тепловая и SOC защита оценивается на границах: возможен overshoot внутри подшага.
Продукт, законно завершившийся на границе, материализуется до применения новой
safe latch. Выбор execution_step влияет на дискретизационную погрешность.

## Отказы и восстановление

Через существующий атомарный lifecycle event envelope доступны
`spacecraft_safe_mode`, `spacecraft_recover`, `spacecraft_fault`,
`spacecraft_fault_cleared`, `spacecraft_degraded`. Fault subsystem:
bus/adcs/payload/cpu/radio. Payload abort не создает raw, CPU pause сохраняет
оплаченную работу, радио не выдает фиктивную receipt. Degradation уменьшает
solar/rate scale монотонно и сохраняется после recovery.

Recovery требует снятых faults, достаточного SOC и охлаждения ниже resume
threshold; команда очищает только safe reasons. Она не заряжает батарею,
не охлаждает аппарат и не исправляет деградацию. Fault clearing — явно заданное
synthetic repair event, а не доказанная процедура FDIR. Safe-mode slew может быть
разрешен конфигурацией при исправном ADCS и допустимой температуре: это позволяет
агенту ориентировать панели, но автоматического Sun-pointing recovery нет.

## Воспроизводимость и доступ

Bus состояние входит в atomic rollback и prefix checkpoint/replay. При restore
сохраняется общий объект между ресурсной миссией и bus executor. Агент получает
текущее состояние своего аппарата; operator truth report содержит модели, events,
denials и battery/PV/discharge/thermal ledgers. Доступ к truth остается явным.
Checkpoint восстанавливается повторением журнала, без заявления O(1)/SLA.

Профиль объявляет trajectory `propagated-gcrs-coast`. Импульсы и finite burns
запрещены именно в этом профиле до реализации связи тяги/топлива/ориентации/health.
Они продолжают работать в ранее существующих dynamic/lifecycle профилях.
Один control window на actor остается правилом; совместная работа ADCS и payload
не означает полноценной параллельности CPU/payload/radio.

## Проверка и запуск

```bash
python -m pytest -q tests/test_spacecraft_bus.py
python -m space_env.eval.ci_profiles reference
python -m space_env.eval.ci_profiles fast
python examples/spacecraft_bus/demo.py /tmp/space-env-bus-demo
```

Пример сохраняет scenario/plan/resume/prefix/final JSON. Он восстанавливается
посреди разворота, выполняет реальное acquisition/transform/delivery и отвергает
неисполнимую заявку без raw. Численные проверки включают независимые уравнения
постоянной электрической нагрузки и RC, геометрическое вращение и конверсионные
ledgers. Артефакт приемки: [B04_BUS.json](../acceptance/B04_BUS.json).

Ориентиры постановки: [Tudat rotational dynamics](https://docs.tudat.space/en/latest/user-guide/state-propagation/propagation-setup/rotational.html)
и [NASA thermal-modeling example](https://tfaws.nasa.gov/TFAWS08/Proceedings/Papers/TFAWS-08-1019.pdf).
Ссылки объясняют область, не являются независимой валидацией этой реализации.

## Что остается до полного B04

B04-B1 добавил sampled polygon footprints/AOI и площадь покрытия:
[контракт](AREA_COVERAGE_CONTRACT.md). Revisit пока ограничен overlap diagnostics.
B04-B2 добавил [optical/cloud/quality и SAR applicability](PAYLOAD_QUALITY_CONTRACT.md).
B04-B3 добавил [revisit/stereo requirements](OBSERVATION_REQUIREMENTS.md) на пригодной площади.
B04-C добавил [parallel resource scheduler](RESOURCE_LANES_CONTRACT.md).
Следующий B04-D — actuator/thrust/attitude coupling. Затем общий scheduler
параллельных ресурсов, memory bandwidth, более подробные thermal nodes и
actuator momentum/torque/thrust coupling. Калибровка реальных аппаратов требует
внешних измерений и declared mission error budgets. B04-A не закрывает эти задачи
и не дает оснований заявлять flight-qualified industrial simulator.
