# M02-B1: sensor-aware задачи в SDK, Gym/PettingZoo и LLM

Текущее расширение: [M02-B2b finite thrust и matched forces](NAVIGATION_FINITE_FORCES.md)
добавляет opt-in inertial finite-burn estimator и declared J2/drag/SRP/Sun/Moon.
Прежние запреты finite/perturbed profiles ниже исторические; неизвестные
execution errors, body-frame thrust и остальные ограничения не сняты.

Текущее расширение: [M02-B2a known impulses и native sensor power](NAVIGATION_CONTROL_POWER.md).
Пассивные ограничения и цены измерений в исторических разделах ниже относятся
к прежнему этапу; текущий профиль поддерживает ideal inertial impulses и
continuous native power, но не finite burns/perturbed estimator или sensor CPU scheduling.

9 ноября 2026. Измерения и EKF из [M02-A](NAVIGATION_MEASUREMENTS.md) подключены
к общему TaskSession без отдельного дублирующего native runtime. Для opt-in
задач собственные координаты — оценка, а не истинная орбита. Это **ограниченный
passive delivery profile**, не завершение всей M02-B: манёвры, perturbed estimator,
цена измерений и научно достоверная сенсорная аппаратура остаются отдельной работой.

## Конфигурация

`TaskSpec.navigation_sensors` — словарь actor → SensorSpec. Он должен покрывать
ровно всех аппаратов, которыми владеют роли задачи; до восьми аппаратов.
Применяются ограничения M02-A: passive `reference-dynamic`, central gravity,
не более 3600s native horizon и 512 samples на actor. Task decisions <=512.
Пустой словарь не сериализуется; обычные задачи сохраняют прежний observation
contract и численные пространства.

```python
from space_env.tasks import TaskSession

# Заполненные рецепты и физическая миссия; sensors содержат явные priors.
task["navigation_sensors"] = sensors
task["model_requirements"] = [
    {"feature": "navigation_measurements", "actors": ["a", "b"]},
    {"feature": "state_estimation", "actors": ["a", "b"]},
]
session = TaskSession(scenario, task)
observation = session.observe("a")
```

`SensorSpec.seed` задаётся явно и входит в checkpoint. Gym reset seed сам по себе
не переписывает sensor seeds; для независимого sensor experiment задавать новый
recipe/seed. Не считать повтор одинакового seed независимой геометрией.
Initial estimate/covariance и hidden `true_bias` — **trainer configuration**;
они не добавляются в capabilities, LLM messages или actor observation.

## Допустимые действия и цели

Разрешены coast/wait, downlink, processing и cancellation. Запрещены maneuvers,
injection и geometric pointing masks. Причина — estimator пока предполагает
пассивную центральную орбиту: разрешить burn при том же estimator значило бы
скрыто нарушить модель. Допустимые сочетания проверяются при конструировании.

`product_delivery` — новая native цель для **готовых бортовых продуктов**, а не
lifecycle customer requests:

```json
{
  "deadline_sec": 5.0,
  "conditions": [{
    "id": "ground-product",
    "kind": "product_delivery",
    "actor": "a",
    "product_ids": ["ready-data"]
  }]
}
```

Product IDs должны существовать в исходном inventory указанного controlled
actor, быть уникальными/непустыми. Успех — native status `delivered_on_time`
для каждого продукта и общие goal limits/deadline. Late или incomplete не дают
успех; ожидание, submit и частичная передача не заменяют этот критерий.
Это receipt семантики выбранного simplified downlink backend, не верификация
реальной наземной аппаратуры. Новый `kind_code=7` работает с прежними 34 goal
features; relative задачи сохраняют отдельные 37 features.

`product_delivery` разрешён и без sensor mode. В sensor mode другие goal kinds
пока отклоняются: exact attitude/orbit/relative errors могут раскрывать truth
через наблюдение цели. Будущая sensor-aware цель должна отдельно определить
trainer evaluator и доступный агенту прогресс, без автоматического копирования
истинных residuals.

Legacy `metrics.fulfilled/total` продолжают считать lifecycle requests. Поэтому
успешная product-delivery задача может иметь 0/0 таких requests; её success и
SFT gate определяются native `metrics.goal`, а не переименованием counters.

## Новая projection policy

View собирается по allowlist; прежний raw observation binding не передаётся.

- Actor position/velocity заменяются на текущую оценку EKF.
- Полная estimate covariance, фильтровая эпоха, возраст прогноза и последний
  полученный packet находятся в `navigation`, только для actor своей роли.
  Централизованный team view содержит sensors контролируемой команды.
- Остаются собственные product metadata и ownership, а также **идеальные**
  resource counters: заряд, память, температура, TX readiness/time. Это не
  реалистичная модель всех датчиков ресурсов.
- Удаляются точные geometry/link/weather/contact entities и relations,
  quaternion/mass orbit features и неизвестные новые поля. Слоты kept entities
  перенумеровываются: пропуски от скрытых links/stations не кодируют их число.
- Catalogue содержит статические action templates своей роли. Masks отражают
  **только право выбора**, не вычисленную по truth физическую допустимость.
  Native admission всё равно проверяет действие и может отклонить его.

Результат настоящей попытки передачи и reward остаются наблюдаемыми операционными
результатами. Это допустимая обратная связь, а не гарантия отсутствия любых
косвенных сведений об орбите. Требование здесь — не выдавать точную orbit truth,
геометрию и native goal residuals вместо заявленных измерений.

Known-beacon range/rate сохраняет явное допущение M02-A: beacon state считается
идеально известным и входит в packet. Это не скрытая оценка другого аппарата;
автор recipe должен выбирать такой профиль только при допустимости этого
знания. Новый delivery benchmark использует coordinate fixes у обоих аппаратов;
общий range/beacon profile не объявляется им дополнительно квалифицированным.

Task-level projection не является Python security sandbox. Trainer checkpoints
содержат hidden initial state/configuration; нельзя включать их в prompt агента.

## Gym/PettingZoo и LLM

Sensor-aware numerical observations дополнены:

- `navigation[max_entities,47]`;
- `navigation_mask[max_entities]`;
- `navigation_feature_mask[max_entities,47]`.

Row связан с actor entity index. Включены все 36 компонент covariance P, epoch,
age, sample count, packet acquisition/reception time, drop/accept/NIS и до трёх
значений измерения. Отсутствующее measurement отличается маской от нуля.
Position/velocity estimate лежат в существующих actor entity features.
Названия колонок — `NAVIGATION_FEATURES` и Gym reset info `navigation_feature_names`.
Полный structured view дополнительно содержит measurement noise covariance и
known beacon state; 47-column encoding не кодирует весь packet metadata.

PZ использует те же encoder/spaces, но navigation row только своего actor.
В LLM actor projection сохраняются navigation/belief и корректный binding;
brief/json wire context также сохраняет covariance/packet и уточняет, что
координаты estimated, а masks не гарантируют admission. `observe` не двигает
sensor clock и не создаёт новый шум.

## Транзакции, replay и данные

Измерения candidate шага и estimator updates рассчитываются на копии фильтров.
Они коммитятся только вместе с native executor и следующим observation.
Ошибка фильтра после native advance candidate не меняет committed episode.
Idempotent retry не дублирует пакеты; cold TaskSession/LLMEpisode replay
восстанавливает noise streams, posterior и projections из исходного рецепта.

Положительный native SFT требует фактического success и отсутствия rejected
transitions. Неуспех сохраняется в RL. Ни scenario truth, ни `true_bias`, ни
initial filter config не добавляются в SFT input. Episode/checkpoint JSON при
этом остаётся trainer audit artifact с privileged configuration.

Этот профиль **не оплачивает** sensor power/CPU/communication и не ограничивает
фиксы физической видимостью GNSS/маяка. Downlink/processing, напротив, исполняются
с существующими native ресурсными ограничениями. Полную hardware/resource fidelity
из такого сочетания выводить нельзя.

## Воспроизводимый пакет

```bash
python -m pytest tests/test_sensor_tasks.py -q
python -m space_env.navigation.task_suite --output artifacts/new-sensor-tasks
python -m space_env.navigation.task_suite --audit artifacts/new-sensor-tasks
```

Три fixed public cases: nominal, delayed/dropout и отсутствие ground contact.
Каждый исполняется reference и wait policy: всего шесть episodes. Готовый 1Mbit
продукт, два аппарата, одна orbital geometry, altered ground site в негативном
case, goal horizon 5s. Все split=train, teacher scripted; это не blind benchmark.
Reference policy заранее выбирает downlink: он не доказывает, что агент научился
использовать навигацию или что EKF повышает качество управления.

Выходы: native episode records, RL JSONL для всех шести и SFT JSONL только для
пригодных положительных траекторий. Нет общего corpus split registry в этой
малой suite; это отдельные demonstrational exports, не готовый benchmark corpus.
Audit повторяет native traces/metrics; source-bound receipt дополнительно
фиксирует export bytes. Никакая внешняя модель автоматически не вызывается.

[Приёмка](acceptance/SENSOR_AWARE_TASKS_V1.json). Старые source-bound receipts
относятся к прежним исходникам и не переписываются задним числом.

## Следующий блок M02-B2

1. Согласовать estimator dynamics с реальными maneuver/perturbation profiles,
   включая known controls, execution uncertainty и mass state по необходимости.
2. Добавить sensor scheduling, visibility и **native** power/CPU accounting.
3. Спроектировать sensor-safe navigation/orbit goals и observation/action policy
   для geometric control без truth-derived masks/residuals.
4. Расширить range/beacon права/uncertainty, clock/bias states и OOS handling.
5. Проверить learned policy на независимых орбитах; сравнивать accuracy/control
   quality при согласованных physics/observation/compute budgets.

## Результаты локальной приёмки

**331 различный локальный тест прошёл** (244 core/navigation + 87 LLM/data),
в том числе 24 новых integration tests. Шесть source episodes и шесть
installed-wheel episodes воспроизведены cold replay. В каждой среде получены
2 положительных native SFT conversation и 6 RL episodes, отрицательные сохранены.
Сообщения, события и метрики совпали между source/wheel; 340 Python/JSON файлов
wheel совпали с checkout, зависимости использовались из текущего окружения.

Полные record/checkpoint hashes отличаются: native identity фиксирует stale
source-environment distribution metadata `space-env=0.24.0`, а extracted wheel
имеет `0.134.0`. Это не замалчивается и не исправляется подменой provenance.
327 package source hashes совпадают, replay проверен **внутри каждой среды**;
перенос этих checkpoints между разными dependency identities не утверждается.

Receipt дополнительно реконструировал все RL/SFT rows из episode records
и проверил отсутствие hidden sensor configuration в exports. Полный repository
suite, clean install, обучение LLM и внешний comparator не запускались.
Артефакты: `artifacts/sensor-tasks-20261109/`; CI/commit/push не добавлялись.
