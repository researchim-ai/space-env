# D2h-B2b-A: live actor policy, Gym/PettingZoo и flat incident epochs

Закрывает online/offline projection и bounded multi-epoch history. Новые customer
requests, runtime смена actor set и произвольные subsystem faults остаются
следующим блоком. Физика — прежний bounded kinematic/ideal-radio профиль B1/B2a;
новой flight calibration или teacher superiority этим не добавлено.

## Общая actor projection

`project_actor_observation` используется и B2a actor export, и live SDK. Она
проверяет native observation binding, оставляет только catalogue данной роли,
одинаково выбирает entries action_mask, сохраняет native_observation_binding
и рассчитывает новый binding. Schema остаётся `incident-actor-observation/1`.
Собственные cancel actions доступны; operator inject actions запрещены.

`ActorCoordinator(reference)` принимает initial chain proposal, single-incident
recovery или journal. Construction проверяет native ancestry и исполняет committed
prefix. Actor policy получает только `observe()` и собственные action IDs через
`step(choices, expected_bindings=..., idempotency_key=...)`.

Все исходные actor roles должны принять одно joint решение. Проверяются ownership,
slots, дубли, wait mixing, актуальность всех bindings. Неверные запросы не двигают
время. Повтор идентичного idempotency request возвращает прежний owned response;
другой body с тем же key отклоняется. Каталоги возвращаются копиями. Disabled,
но известное собственное действие допускается в native admission и может дать
rejected feedback — это нормальный отрицательный RL-опыт.

Native TaskSession делает один atomic joint step с фиксированным decision_sec.
Runtime idempotency использует `mission-plan/<index>`; actor request keys хранятся
отдельно. Это позволяет seal фактически выбранного policy prefix в исполнимый
fixed `MissionPlan` через `as_execution()`, сохранив ещё не исполненный suffix как
proposal. Его выполнимость до полного исполнения не обещается.

Reward общий, `reward_scope=joint_team`; он не является индивидуальной заслугой
каждого actor. Ни полный native replay, ни completed goal не доказывают causal
source-policy expertise или optimality.

## Checkpoint и восстановление

Checkpoint содержит reference один раз, actor requests/response hashes и native
TaskSession episode. Restore заново проверяет reference, проигрывает prefix и
все actor decisions, сравнивает responses, episode и полный canonical checkpoint.
Дублированный trace и bool вместо integer cursor не нормализуются молча.
Лимиты:128 MB checkpoint,10000 actor decisions, исходные limits B1.
Это cold audited replay, не O(1) RAM restore и не durable hosted job.

SDK RLock защищает наблюдение/step/checkpoint одного controller. Доступ к приватным
Python объектам не является security boundary: hostile in-process code и hosted
quotas/auth относятся к Q7, а не к этому интерфейсу.

## Gymnasium и PettingZoo

`ActorMissionEnv(reference)` — Gym joint Dict: отдельная Discrete/MultiDiscrete
action space и численное observation для каждой роли. Index→ID map фиксирован
в пределах данного epoch. Маска имеет длину собственного actor catalogue;
entity/relation arrays используют существующий fixed-capacity encoder.
Multi-slot нули служат wait padding; повторные индексы схлопываются перед decoder.
Маска общая для словаря IDs и может транслироваться на slots. Joint feasibility
повторно проверяет native TaskSession после объединения решений.

`ActorParallelEnv(reference)` — optional PettingZoo Parallel API, требует `[marl]`.
Operator role не включается в agents. Team reward одинаков у живых actors; episode
завершается совместно. Empty actions после завершения принимаются стандартно.

Reset seed управляет wrapper/action sampling; физический episode seed закреплён
проверяемым reference. Options — только игнорируемые metadata (Gym info возвращает
ignored_options), ими нельзя заменить scenario/seed/epoch. Новый epoch требует
нового Env, чтобы не менять action space во время episode. Gym joint Dict actions
не являются обещанием совместимости с любым сторонним single-agent trainer;
для multi-agent consumers предусмотрен ParallelEnv.

## Плоский incident journal

`create_journal(initial_proposal)` хранит origin один раз. `append_incident(journal,
execution_record, incident)` проверяет ancestry и native prefix. Record может
получиться от fixed PlanExecution либо `ActorCoordinator.as_execution()`.

Actor deviations после предыдущего committed cutoff сохраняются как delta edits:
fixed clock/action IDs. Историю до cutoff менять нельзя; operator injections
в actor edits запрещены. Native replay сверяет source episode SHA и physical
report SHA. Затем B2a phase-aware recovery обрабатывает новый committed incident.
Копии `source_record`/`source_proposal` из epoch result удаляются: node содержит
только hashes свидетелей, настройки и новый plan/forecast/state. Каждый node
связан с предыдущим и parent plan hash. Нет рекурсивных recovery payload.

До8 epochs,64 MB journal,1023 task actions и пределы B1. Полные ограниченные планы
хранятся на каждом epoch, поэтому объём не O(1); delta edits не дублируют уже
замороженный actor prefix. Native audit заново реконструирует каждый source fork,
проверяет episode/physical hashes и пересчитывает epoch. Цена replay растёт с
историей; low-latency reset/throughput SLA не заявлены.

`verify_journal(native=False)` — envelope/ancestry/clock/plan consistency.
`native=True` — повторная физика committed history. Ни один режим не исполняет
неизвестные будущие suffixes как поисковый oracle. Incident inputs и operator
commitments известны на момент branch, demand/trajectory/hardware не меняются.

## Actor data

`publish_actor_data(checkpoint, directory)` native-восстанавливает controller,
проверяет episode и declared goal. Actor suffix RL сохраняется всегда; SFT только
при выполненной полной model-goal и accepted transition. Форматы
`actor-online-rl/1` и `actor-online-sft/1`, reference SHA и shared projection.
Операторские этапы до cutoff не являются actor supervision. Для journal экспорт
относится к последнему committed suffix; предыдущие epoch trajectories можно
сохранять отдельными checkpoint, они не склеиваются скрыто.

1 GiB staging budget,16 MB на строку,10000 rows/file, exact file inventory,
row/file hashes и без overwrite. Native audit регенерирует каждую строку; без
флага — consistency. `expert_teacher_qualified=false` независимо от goal.
Operator-compiled future pointing catalogue остаётся отдельным information budget;
перенос на иной класс агентов требует собственного benchmark.

## Запуск

```bash
python -m examples.planning.actor_mission /tmp/actor-spec.json --write-spec
python -m examples.planning.actor_mission /tmp/actor-case --mode policy_fork
python -m examples.planning.actor_mission /tmp/eight-epochs --mode eight_epochs
space-env actor-controller-start reference.json controller.json
space-env actor-controller-observe controller.json observation.json
space-env actor-controller-step controller.json request.json next.json
space-env actor-controller-verify next.json
space-env actor-controller-execution next.json executed.json
space-env incident-journal-create proposal.json journal.json
space-env incident-journal-append journal.json executed.json next-journal.json --incident incident.json
space-env verify-incident-journal next-journal.json --native
space-env actor-data-export controller.json /tmp/actor-data
space-env verify-actor-data /tmp/actor-data --native
```

CLI writes новых outputs без перезаписи; actor step request имеет choices,
expected_bindings и idempotency_key. File CLI восстанавливает checkpoint перед
каждым шагом; SDK live object выполняет решения без такого полного restore.
Примеры policy_fork/eight_epochs/failed_policy — synthetic mechanism regression,
не независимый teacher holdout corpus. Failed goals и RL-only случаи удерживаются.

Далее: versioned new demand/catalogue, per-subsystem recovery, uncertainty-aware
forecast policies, equal-information stress corpus/solver comparison. Measured
calibration C2, long rigid qualification и hosted isolation Q7 остаются открыты.
