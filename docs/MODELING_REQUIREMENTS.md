# Реестр моделей и требования задачи: M01-A

Текущее расширение: [M02-B2b finite thrust и matched forces](NAVIGATION_FINITE_FORCES.md)
добавляет opt-in inertial finite-burn estimator и declared J2/drag/SRP/Sun/Moon.
Прежние запреты finite/perturbed profiles ниже исторические; неизвестные
execution errors, body-frame thrust и остальные ограничения не сняты.

Текущее расширение: [M02-B2a known impulses и native sensor power](NAVIGATION_CONTROL_POWER.md).
Пассивные ограничения и цены измерений в исторических разделах ниже относятся
к прежнему этапу; текущий профиль поддерживает ideal inertial impulses и
continuous native power, но не finite burns/perturbed estimator или sensor CPU scheduling.

Текущее расширение: [M02-B1 sensor-aware tasks](SENSOR_AWARE_TASKS.md) уже
подключает passive measurements/EKF к TaskSession, Gym/PettingZoo и LLM.
M01 navigation features теперь enabled только при валидном sensor task recipe.
Исторические статусы/приёмки ниже относятся к своим исходникам; полная M02
и остальные комбинации моделей остаются открытыми.

Обновление M02-A: [отдельный NavigationSession](NAVIGATION_MEASUREMENTS.md)
уже поддерживает synthetic measurements/EKF. Это не общий TaskSession contract;
соответствующие M01 признаки здесь остаются not_implemented до M02-B.

9 ноября 2026. Первый исполнимый блок программы
[M01–M10](MODELING_COVERAGE_AND_ROADMAP.md). Это **не завершение всей M01** и не
доказательство превосходства над BSK-RL или другим движком. Здесь добавлен общий
контракт проверки конкретных физических моделей для каждого аппарата.

## Зачем это нужно

Глобальная capability среды не доказывает, что нужная модель есть у каждого
аппарата. Например, rigid ADCS у `a` не должен удовлетворять требованию к `b`,
а covariance prediction не должна считаться измерениями или estimator updates.
Теперь task recipe может потребовать эти свойства явно, и unsupported сочетание
будет отклонено при создании TaskSession, **до исполнения первой команды**.
Конструирование native runtime при этом выполняется; это не нулевой по стоимости
JSON linter и не гарантия успешного исполнения последующих команд.

## Recipe и SDK

```python
from space_env.tasks import ModelRequirement, TaskSession, inspect_task, model_registry

# scenario и task — существующие словари native scenario / task recipe.
task["model_requirements"] = [
    {"feature": "dynamic_orbit", "actors": ["a", "b"]},
    {"feature": "rigid_attitude", "actors": ["a"]},
    {"feature": "body_thrust", "actors": ["a"]},
]

preflight = inspect_task(scenario, task)  # конструирует и валидирует без advance
session = TaskSession(scenario, task)
profile = session.modeling_profile()     # trusted configuration metadata
```

`ModelRequirement` — strict/frozen schema. Actors должны быть уникальными,
непустыми и принадлежать ролям задачи. Неизвестное имя feature и повторное
требование той же actor/feature пары отклоняются. До 64 requirement groups и
64 actors в каждой; физические пределы конкретного backend не увеличиваются.

Пустой `model_requirements` не сериализуется, чтобы старые task JSON сохраняли
формат. Непустые требования входят в checkpoint и проверяются при cold restore;
их нельзя потерять при генерации dataset или замене evaluator.

## Проверяемые признаки

В текущем registry `/1` десять реализованных признаков (в исходном M01-A было восемь):

- `dynamic_orbit`: native managed dynamic orbit у указанного actor.
- `covariance_prediction`: navigation prediction включена именно для этого actor.
- `spacecraft_bus`: native coupled bus данного actor.
- `rigid_attitude`: реально созданная rigid model, а не rate-limited pointing.
- `body_thrust`: native engine из конфигурации actor.
- `rotating_body_thrust`: двигатель с явным rigid coupling.
- `momentum_unloading`: native RCS unloading actor.
- `thermal_network`: именно lumped thermal network, не любой thermal flag.
- `navigation_measurements`: synthetic received measurements в opt-in sensor task.
- `state_estimation`: passive central-gravity EKF в том же sensor task.

Два признака отклоняются как `not_implemented`: `continuous_relative_safety` и
`docking_contact`. Навигационные признаки без `navigation_sensors` имеют
`not_configured`, а с валидным sensor task — `enabled` для конкретного actor.
Отдельные legacy encounter/Pc kernels не дают continuous safety общему goal.


В profile каждая actor/feature клетка имеет `enabled`, `not_configured` или
`not_implemented`. `implemented` в registry значит наличие поддерживаемого
контракта, а не автоматическое включение во всех сценариях. Требования принимают
только `enabled`; ideal fallback не используется. Список проверяется по native
model objects/configuration после корректной backend construction, а не по
именам классов, текстовому поиску или capabilities другого аппарата.

## Доступ и физический смысл

Profile содержит **конфигурацию всех аппаратов** и предназначен для trusted
автора задачи/оператора. Он намеренно не включён в actor observations, LLM wire
context или Gym numerical features. Не передавать его агенту с ограниченными
правами без отдельной projection policy.

`inspect_task` не вызывает step, submit или forecast. При backend construction
могут рассчитываться исходные орбитальные дуги и другие initialization products;
поэтому это не обещание отсутствия численного распространения или минимальной
стоимости. Формулировка `no simulation` в scope результата означает отсутствие
исполненных команд/шагов миссии, а не запрет initialization computations. В отчёте нет будущих
траекторий, текущих координат, топлива/ошибок/здоровья. `modeling_profile()`
защищён session lock и проверкой external runtime mutation. Результаты profile
и registry — новые независимые структуры; их редактирование не меняет модель.

Совместимые requirements не доказывают одновременно:

- что действия допустимы при текущем заряде, температуре или занятости;
- что цель достижима, policy успешна или контроллер оптимален;
- что hardware параметры измерены и calibrated;
- что сочетание независимо qualified на долгой миссии;
- что физика полнее или быстрее другого симулятора.

Это отдельные уровни проверки. Legacy task/goal/native validation продолжают
действовать; registry их не отключает и не заменяет.

## CLI

После установки обновлённого wheel:

```bash
space-env-models --output model-registry.json
space-env-models --scenario scenario.json --task task.json --output compatibility.json
```

При разработке из checkout:

```bash
python -c 'from space_env.tasks.modeling import main; main()' --output model-registry.json
```

Scenario/task задаются вместе. Unsupported task вызывает ошибку, а не valid=true.
Output не может совпадать с input, включая symlink/hardlink aliases.
Вывод пишется atomically. CLI — локальная утилита, не сетевой endpoint.

## Приёмка и воспроизведение

```bash
python -m pytest tests/test_modeling_registry.py -q
python -m space_env.tasks.modeling_suite --output artifacts/new-model-registry-study
python -m space_env.tasks.modeling_suite --audit artifacts/new-model-registry-study
```

Новые tests включают native positive execution и cold restore для шести типов
requirements, partial actor covariance/thermal execution, отрицательные
требования чужой аппаратуры, неподдерживаемые признаки, recipe errors,
CLI alias protection и неизменность наблюдений/checkpoint от introspection.
[Receipt](acceptance/MODELING_REQUIREMENTS_V1.json) фиксирует текущие source hashes,
логи и installed-wheel результаты. Исторические physics receipts не переносятся
автоматически на изменившийся исходный код.

## Что остаётся в M01

Реестр пока не покрывает все 17 семейств из аудита. Следующий этап — explicit
RF/DTN/payload/processing/force-fidelity profiles, compatibility комбинаций,
versioned applicability envelopes и привязка каждой клетки к независимому
qualification bundle. Не ставить `qualified` по наличию одного smoke test.
После этого M02/M03 добавляют measurement/estimation и continuous safety
с отдельными native positive/negative cases и статистической/численной проверкой.

## Результаты текущего пакета

Прошли **265 различных локальных тестов**: 162 modeling/core regressions
(включая 27 новых) и 103 LLM/data regressions. Source и установленный wheel
отдельно выполнили и воспроизвели по 38 native запусков без runtime ошибок.
Reference success 12/19, wait 1/19; семь отрицательных reference случаев сохранены.

Из установленного wheel дополнительно проверены 19 task recipes с непустыми
требованиями всех включённых моделей, четыре явных unsupported отказа и CLI
registry. 335 Python/JSON файлов совпали побайтово с wheel; entrypoint
`space-env-models` присутствует в metadata. Это не clean-dependency install.
Артефакты: `artifacts/model-registry-20261109/`; receipt связывает их хеши с
322 Python-модулями текущего пакета. Новых внешних сравнений или обучения LLM
этот пакет не выполнял. CI/commit/push не добавлялись.
