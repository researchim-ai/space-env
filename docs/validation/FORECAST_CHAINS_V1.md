# D2h-B1: прогнозные acquisition → processing → delivery chains

Добавлен операторский планировщик полного цикла без preload. Он использует
явно разрешённые прогнозы геометрии, составляет план **до исполнения** и затем
передаёт его обычному native TaskSession. Из геометрического proposal не следует
выполнимость: её проверяют PlanExecution и cold replay перед публикацией RL/SFT.
Это первый ограниченный профиль D2h-B; весь D2h-B не объявляется завершённым.

## Поддерживаемый профиль

`reference-spacecraft`, до 4 аппаратов, 16 станций, 64 запросов, 21600 секунд,
10000 decision cells с постоянным шагом 1–30 s. Гравитационные траектории с
опциональным J2, single-owner point requests, один consuming metadata transform
raw → final, фиксированная станция, ready/release/deadline/priority. Все запросы
сохраняются в знаменателе. Target и station объявляются до поиска расписания.

ADCS — rate-limited kinematic, без rigid dynamics, keepout или radio-pointing
constraint. Radio — ideal constant rate на терминал, без RF/shared bandwidth,
DTN, booking/events и failure injection. Нет area/image/quality/retry/TTL-моделей
в этом scheduler. Unsupported профиль отвергается, а не упрощается молча.
Вся цепочка консервативно занимает actor, включая ожидание downlink; station
занимается только во время передачи. Это ограничивает параллелизм и качество.

В native остаются coupled bus power и простой RC thermal. Scheduler не решает
thermal envelope: его predictions могут не реализоваться из-за model constraints,
и такой результат должен остаться failed native certificate. Нет flight calibration.

## Прогноз ориентации и оплата наведения

`MissionRuntime.forecast_pointing(actor, target_id, at_sec)` требует
`AccessPolicy(forecast=True)` и capability `forecast.pointing.geometry`.
Возвращает value-only future geometric GCRS quaternion, время и binding
scenario/trajectory/revision/geometry revision. Доступ не даётся обычному агенту.
Метод не продвигает физику, не меняет ориентацию, не вызывает admission и не
обещает возможность tracking/dwell/power. Прогноз условен на текущую траекторию.

Поиск использует только окна acquisition/contact и объявленные параметры.
После выбора времён операторский materializer получает quaternion к середине
будущей съёмки и записывает его в slew command. Контроллер держит GCRS goal;
это **не** автоматическое слежение за наземной целью. Native проверяет ошибки
наведения во время съёмки. Нельзя считать один midpoint quaternion доказательством
непрерывного tracking. Runtime сохраняет прежнюю семантику snapshot slew.

Консервативная длительность наведения:

`t_slew = Δt * ceil((180 / rate_deg_s + settling_s + ε) / Δt)`.

Она покрывает любой начальный quaternion при заявленной кинематической скорости.
Время съёмки округляется вверх для actor reservation, но команда imaging имеет
реальную acquisition duration. CPU reservation включает work, switching, read/write
`(raw_bits + final_bits) / memory_bandwidth_bps` и два lifecycle integration step
в качестве guard. TX duration округляется вверх по `final_bits / rate_bps`.
Это модельные консервативные бюджеты, не универсальные physical bounds.

## Поиск и ресурсы

Детерминированный greedy: deadline, priority, ID. На decision lattice ищется
последовательная цепочка slew → capture → transform → downlink. Capture должен
помещаться в acquisition window, downlink — в contact window и deadline.
Внутренние границы окон защищены guard 10 ms. Boundaries обрезанного forecast
at-now/horizon не сдвигаются. Actor calendar исключает перекрытия цепочек;
station calendar ограничивает число одновременных terminal reservations.

Энергия считается без будущего PV credit. Заряд на момент планирования берётся
из публичной текущей SOC. Base load резервируется до конца горизонта; stage
energy учитывает discharge efficiency. SOC floor — максимум порогов TX, slew,
payload, CPU и bus recovery. Peak stage load проверяется по max bus load.
TX quota учитывается отдельно. Память консервативно проверяется суммой
raw + final + workspace для всех actor requests. При таком ограничении storage
не переиспользуется при поиске, хотя native lifecycle может освобождать его.

До 2 млн search nodes; exceeded budget — явная ошибка, не partial best-solution.
Typed value schemas ограничивают окна, пары и запросы, запрещают nonfinite/private
поля, дубли, неверные ownership/bindings и inconsistent trajectory tokens.
Checksums защищают от повреждения, но не доказывают авторство чужого файла.

## Idle-boundary suffix replanning

`replan_chain(proposal, prefix_record)` — **явная операторская** граница.
Она заново проверяет прогноз исходного proposal и делает native replay prefix.
Поддерживается один replan исходного initial proposal; рекурсивная история
replans, новые demand, изменённый scenario и mid-operation interrupts пока запрещены.

Каждая затронутая цепочка должна быть реально `completed` в публичной native
telemetry. Незавершённая, failed или частично переданная цепочка отклоняется.
Завершённые запросы исключаются из нового поиска, но остаются в denominator.
Будущие forecasts получают текущие revision/time и SOC/used TX. Старые pointing
bindings использовать для нового suffix нельзя.

Все исполненные шаги и command templates замораживаются. Будущий catalogue
может менять времена и quaternion только ещё не исполненных действий. Новый
полный MissionPlan заново исполняет prefix, и canonical native backend report
должен совпасть с исходным. Checkpoint самого источника также проверяется через
cold TaskSession replay; произвольному «сообщению о completed» доверия нет.

Это не O(1) RAM restore и не online handling неизвестного отказа. При неизменной
обстановке suffix может совпасть с прежним либо ничего не добавить. Отдельный
`verify_chain_replan` всегда повторяет native prefix audit, даже без CLI `--native`.
Initial `verify_chain_proposal` без флага проверяет согласованность, с `--native`
заново получает разрешённые прогнозы. Ни тот ни другой не исполняет весь suffix:
`native_feasibility_validated=false` сохраняется до PlanExecution/bundle.

## CLI и воспроизводимый пример

```bash
python examples/planning/chain_mission.py /tmp/chain-input.json --write-spec
python examples/planning/chain_mission.py /tmp/chain-run --spec /tmp/chain-input.json --replan
space-env chain-plan scenario.json task.json proposal.json --goals goals.json
space-env verify-chain-plan proposal.json --native
# plan.json извлекается из поля proposal.plan
space-env mission-plan-run plan.json prefix.json --decisions 22
space-env chain-replan proposal.json prefix.json replanned.json
space-env verify-chain-plan replanned.json
```

Outputs не перезаписываются. Сохраняются исходные spec, forecast, pointings,
schedule, plan, prefix, replan и его native audit, execution и RL/SFT bundle.
В examples есть nominal/shifted/outage; геометрия не подбирается после outcomes.
Goal min_fulfilled=1 остаётся и в negative control, поэтому outage должен дать
failed certificate и RL-only. SFT — кандидаты по declared model-goal, не expert
teacher data. `expert_teacher_qualified=false`; teacher superiority не тестируется.

## Оставшийся D2h-B2

- Прерывание in-flight операций, отмена/перезапуск и частичные продукты, deadlocks.
- Versioned operator events/new demand и failure-aware forecast invalidation.
- Несколько последовательных replan epochs с bounded history и durable recovery.
- Менее консервативное interleaving payload/CPU/TX и accounting живой памяти.
- Equal-information forecast baseline, более сильный solver и frozen stress corpus
  по различным широтам, плотности сети, deadlines и releases. Нулевые результаты
  не исключать; действительное преимущество доказывать отдельно.
- Tracking/radio-pointing/rigid qualification и численная envelope-проверка;
  measured calibration C2 и hosted isolation Q7 остаются самостоятельными работами.

## Информационный бюджет данных и операторское владение

Quaternion в derived task catalogue получен из privileged future geometry.
Соответствующие SFT-кандидаты относятся к операторски скомпилированной постановке,
а не автоматически к агенту с обычными текущими наблюдениями. Transfer на такой
агент и равенство информационных бюджетов требуют отдельного benchmark.
Наличие корректного native trace не доказывает source-policy causality/expertise.

Operator bridge должен работать с эксклюзивно принадлежащим ему Runtime;
атомарная транзакция между несколькими concurrent bridge callers здесь не
реализована. CLI и replanner создают собственный Runtime. Изменять его во время
компиляции нельзя. Example exit 0 означает завершение workflow; реальный goal
status сохранён в result/certificate и может быть false у failed cases.
