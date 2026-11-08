# Физические цели миссий: общий контракт моделирования

9 ноября 2026. Opt-in расширение `TaskSpec.goal` связывает уже реализованную
физику с критериями успеха, наградой, окончанием эпизода и native SFT/RL.
При отсутствии goal сохраняются прежние delivery semantics и сериализация.
Это инженерные модели с явными допущениями, не flight qualification.

## Что можно моделировать через общий интерфейс

- Доставку заданного подмножества заявок с настоящим Ground ACK.
- Наведение аппарата в GCRS с допуском ориентации, скоростью вращения и
  завершённой стабилизацией; legacy kinematics и полноценный rigid wheel bus.
- Разгрузку маховиков до заданного BODY-вектора: текущий момент, native RCS
  operation, совместный расход топлива, электрическая энергия и тепло.
- Приложение заданного GCRS-вектора импульса BODY-двигателем: химическая и
  электрическая модели, конечные gimbal/settling/ignition/firing, ограничения
  ресурсов. При rigid coupling учитывается изменяющееся направление тяги,
  плечо двигателя, вращение корпуса и изменение массы.
- Достижение заданных GCRS положения и скорости в определённые моменты времени
  с конечным включением двигателя. Несколько waypoint-целей можно объединять
  с другими условиями в одной миссии.

Это расширение задач над настоящим runtime, а не новые имена для delivery.
Типы условий: `delivery`, `attitude`, `momentum`, `body_impulse`, `orbit_state`.
Callback, Python expression, неизвестный frame или исполняемый код из JSON
не допускаются. Неизвестные/неуправляемые аппараты и отсутствующая требуемая
физическая модель отклоняются при создании TaskSession.

## Схема и исполнение

```python
from space_env.tasks import MissionGoal, TaskSession

task["goal"] = {
    "schema": "space-env.mission-goal/1",
    "deadline_sec": 20.0,
    "conditions": [{
        "id": "unload", "kind": "momentum", "actor": "a",
        "frame": "BODY", "target_nms": [0.0, 0.0, 0.0],
        "tolerance_nms": 1e-5,
        "completion": "maintain",
    }],
    "limits": {"a": {
        "max_energy_j": 200.0,
        "min_propellant_kg": 1.0,
        "require_healthy": True,
    }},
}
session = TaskSession(scenario, task)
print(session.describe()["goal"])
print(session.goal_status())  # trainer/central view
print(session.goal_status("a"))  # only own-actor conditions
```

Числа примера задаёт автор миссии, исходя из её аппаратной конфигурации.
Поддерживается не более 16 условий и 64 actor resource limits; deadline должен
лежать внутри native horizon, максимум 259200 s. Наличие этих schema limits не
означает квалификацию произвольной большой миссии.

Все условия объединяются AND. `completion="maintain"` требует истинности в
момент завершения, а `completion="achieve"` сохраняет уже достигнутую веху.
Для orbit_state default — achieve. Вехи и нарушения восстанавливаются повтором
committed transitions, не принимаются из произвольного editable success flag.
Начально выполненное условие не даёт бесплатной награды.

`not_before_sec` ограничивает начало проверки условия; `command_after_sec`
ограничивает допустимую историю подтверждающих команд. По умолчанию для
actor-условий требуется настоящая завершённая команда соответствующего типа.
`require_command=False` явно разрешает цель, достижимую пассивным движением
или уже существующим состоянием. Это не scheduler зависимостей начала работ.

## Физические критерии

Ориентация задаётся unit body-to-GCRS quaternion в порядке wxyz. Ошибка:

\[
e_q=2\arccos(|q_{native}\cdot q_{target}|)\le\epsilon_q.
\]

Дополнительно проверяются текущая угловая скорость, settling и отсутствие
активных операций наведения. Для rigid bus используется настоящая BODY omega,
для legacy — публикуемая native скоростная характеристика интервала.
Один лишь status команды не доказывает достижения цели. В rigid-профиле
завершение native controller operation также должно быть успешным; его допуски
и rate tolerance должны соответствовать более строгой цели миссии.

Разгрузка проверяет текущий момент маховиков:

\[
\max_i |h_{i,native}-h_{i,target}|\le\epsilon_h.
\]

Нужна успешная native unloading operation и отсутствие продолжающейся операции,
способной менять момент. Это не мгновенное обнуление состояния.

BODY-импульс — сумма реально приложенных engine impulses после объявленной
границы истории:

\[
J_{GCRS}=\sum_k\int F_k(t)R(q(t))u_{body,k}(t)\,dt.
\]

Учитываются частичные и отменённые включения, а не только completed records.
Требуется отсутствие продолжающегося body burn и векторная ошибка не выше
`tolerance_ns`. Это импульс в N·s, не идеальная scalar delta-v. Влияние массы и
сил на реальную траекторию отдельно проверяется orbit_state.

Орбитальная цель проверяет фактический state в GCRS на native clock:

\[
\|r(t_i)-r_i^*\|_2\le\epsilon_r,\qquad
\|v(t_i)-v_i^*\|_2\le\epsilon_v.
\]

`at_sec` — секунды от epoch сценария, а не UTC timestamp. По умолчанию окно
времени ±1e-6 s. Длинный decision step обрезается на ближайшей ещё не пройденной
орбитальной вехе и deadline: нужный момент не пропускается. После пропущенного
окна возникает terminal failure. Поддерживаются дробные моменты времени.

## Ресурсы, награда и окончание

Energy budget использует накопленный native consumed Wh ×3600, fuel reserve —
настоящий остаток propellant в динамической модели. Отсчёт от начала сценария:
prefix/replay не восстанавливает потраченные ресурсы. Нарушения сохраняются.
`require_healthy` отвергает safe/fault state и записанные fault/safe events;
очистка неисправности не стирает нарушение этого явно заданного ограничения.

Проверки выполняются на committed границах и по native event history. Это не
универсальная непрерывная проверка любых safety envelopes между отсчётами.
Отдельный global deadline, исчерпание ресурсов или успех завершают episode;
TaskSpec.max_decisions по-прежнему может вызвать truncation.

Для goal-задачи положительные delivery/throughput bonuses заменяются на
изменение числа выполненных условий. Выбранный objective сохраняет penalties
за время, энергию, discard, safe entries и rejected calls:

\[
r_t=N_{t+1}-N_t-\mathbf1_{new\ failure}-cost_t.
\]

При необратимом нарушении N становится нулём: прошлые положительные достижения
не скрывают провал бюджета. Повтор idempotency key не даёт повторной награды.
Успех/провал терминальны с discount=0; прочие discounts зависят от physical dt.

## SDK, RL/MARL, LLM и данные

TaskSession, MissionTaskEnv и MissionParallelEnv используют один evaluator.
Для новых goal-задач численное observation дополнено `goals[16,34]`,
`goal_mask[16]`, `goal_feature_mask[16,34]`, `goal_resources[64,9]`,
`goal_resource_mask[64]` и `goal_resource_feature_mask[64,9]`. Имена колонок —
`tasks.goals.GOAL_FEATURES` / `GOAL_RESOURCE_FEATURES`, коды типов — `GOAL_KINDS`.
Вектор содержит тип, временное окно, допуски и связь с actor entity index,
а не только ошибки состояния. Budget-вектор даёт объявленный предел, фактический
расход и остаток; feature masks различают физический ноль и отсутствующую величину.
Legacy spaces не меняются. Actor view содержит только свои условия и бюджеты,
даже если у budget-аппарата нет отдельной goal-строки: чужие ошибки/позы и global
success/failures не добавляются в локальную телеметрию. Gym central info содержит
полный trainer goal status; team reward и terminal flags остаются общими.

LLMEpisode.finish проверяет declared goal, не требует фиктивной доставки для
разгрузки или наведения. `metrics.goal.success` и `metrics.success` определяют
успех новой задачи; прежние `fulfilled/total/weighted_fulfilled` остаются именно
delivery counters и могут быть 0/2 при успешной разгрузке. Нельзя смешивать эти
счётчики с показателями delivery benchmark.

SFT новых goal-эпизодов требует успешного native goal и отсутствия ошибок /
rejected calls. Неуспехи сохраняются для RL. Actor messages остаются causal;
native checkpoints и reference schedule — только trainer audit. Это verified
execution, не expert/optimal label. Настоящие provider wire-SFT и обучение модели
этот пакет автоматически не выполняет.

## Запуск

```bash
# 13 фиксированных cases × wait/reference, без модели:
python -m space_env.tasks.modeling_suite --output artifacts/new-modeling-suite
python -m space_env.tasks.modeling_suite --audit artifacts/new-modeling-suite

# Один новый physical control task → native SFT/RL corpus:
python -m examples.modeling.mission_goals --case momentum-nominal \
  --output artifacts/new-unload-data --split train

python -m pytest tests/test_mission_goals.py -q
```

Output должен быть новым. Пример использует локальный registry; `--registry`
позволяет разделять один ledger между всеми производителями. Разные registry DB
не обеспечивают общую защиту splits. Фиксированные geometries публичны: train
demo нельзя выдавать за новый blind test. Source-bound старые артефакты
проверяются на своей ревизии; при изменении кода нужны новые прогоны.

## Набор физических профилей и границы

Фиксированы legacy/rigid pointing, controller-precision mismatch, nominal/limited
RCS unloading, химический/электрический BODY impulse, rotating off-center thrust,
finite-burn orbit target, wrong-direction burn, два orbit waypoints и delivery.
Никакой seed search или фильтрации по успеху. Слабые/неверные случаи сохранены.

Orbital target oracle — отдельная реализация RK4 с h=0.005 s, центральной
гравитацией и аналитической массой `m0−F*t/(Isp*g0)` при горении. Она не вызывает
native derivative/propagator, но использует общие исходные константы. Tolerances:
1e-3 m и 1e-6 m/s. Это короткий идеализированный inertial finite-burn profile,
без J2/drag/SRP/ADCS; BODY-профили отдельно проверяют actuator/attitude coupling.

Rigid/coupled модели сохраняют существующие ограничения controller sampling,
wheel saturation, constant inertia и малого изменения массы. В coupled примере
тяга 0.01 N, плечо 0.2 m, допустимая относительная потеря массы 0.001, импульсный
допуск 1e-4 N·s. Это параметры synthetic engineering scenario, не калибровка
реального двигателя и не независимое внешнее сравнение.

## Локальная приёмка 9 ноября 2026

Прошли 390 различных тестов (105 интерфейсов/целей/replay, 87 данных/LLM,
198 физики). Полный repository suite этим не объявляется пройденным.
Матрица: 13 сценариев × 2 политики, reference 9/13, wait 0/13; четыре
отрицательных случая сохранены. Source и извлечённый установленный wheel
отдельно выполнили по 26 native запусков и cold replay без runtime ошибок.
321 Python-модуль и два JSON fixture совпали с wheel побайтово; зависимости
использовались из текущего окружения, это не установка в чистое окружение.

Свежий momentum-nominal corpus содержит 1 RL episode и 1 native SFT
conversation, wire-SFT 0. Teacher — scripted_privileged, модель не обучалась.
Цель разгрузки успешна при 0/2 доставленных заявок.
[Подписанная хешем приёмка](acceptance/PHYSICAL_MISSION_GOALS_V1.json) связывает
текущие исходники, wheel, журналы и результаты в игнорируемом
`artifacts/modeling-goals-20261109/`. Пример генерации `examples.modeling`
запускается из checkout; packaged modeling suite работает из wheel без examples.
