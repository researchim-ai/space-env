# Обязательные цели и явный критерий приёмки задачи

Обновление 15 ноября 2026 года добавляет опциональный `OrbitSession.acceptance`.
Он отделяет **выполнение объявленной задачи** от **полного выполнения всех native
целей**. Начальный контракт задаёт обязательные условия и минимальную взвешенную
долю, а физические цели, орбиты и native success не переписываются.

В общем каталоге 5 семейств и 51 стартовый профиль. Новые профили:
`orbit_and_deployment/contract_{priority,mandatory,infeasible}_{train,development,test}`.
Это варианты существующего orbital family, не девять новых физических моделей.

## Два разных результата

`metrics.success` и `metrics.goal` сохраняют прежнюю native all-goals семантику.
Новый `metrics.task_acceptance.accepted` отвечает на другой, заранее объявленный
вопрос. При natural native termination требуется одновременно:

\[
\text{task accepted}=\text{natural terminal}
\land \text{declared safety valid}
\land C\ge\tau
\land R\subseteq M,
\]

где C — weighted completion fraction, τ∈(0,1], R — обязательные condition IDs,
M — выполненные native условия. Цели, не входящие в R, могут быть необязательными,
но влиять на C. Неизвестные/повторяющиеся mandatory IDs, нулевой/нечисловой threshold
и acceptance без utility mode отклоняются.

В конфликтном примере с весами 9/1:

- τ=0.9 и обязательная цель с весом 9: её выполнение принимается, native full
  success остаётся false;
- τ=0.1 и обязательная цель с весом 1: нужно выполнить именно её; опциональная
  цель с весом 9 не заменяет обязательство;
- обе цели обязательны, а dispatch slot один: запрос остаётся валидной постановкой,
  но не принимается при текущих статических правилах. Он сохраняется как negative,
  а не удаляется из генератора.

Эта возможность поддержана для текущих orbital recipes с общей конечной эпохой.
Она не меняет lifecycle/termination других executors и не решает общий случай
разных сроков обязательных/необязательных заявок: native остановка по ранней
optional failure в другом семействе потребует отдельного адаптера.

## Зачем меняется награда

Одного флага приёмки мало: без дополнительного правила агент предпочёл бы
опциональные 0.9 обязательным 0.1. Поэтому opt-in контракт задаёт потенциал:

- safety latch сохраняет прежнее значение -2;
- без safety violations применяется weighted completion C;
- при native завершении с отсутствующей обязательной целью вычитается 1;
- native truncation также вычитает 1.

Session reward — изменение этого потенциала минус 0.1 за отклонённый batch.
Пропущенная mandatory цель имеет положительный вес, поэтому результат с такой
ошибкой меньше нуля даже при очень ценных выполненных optional целях. Выполнение
mandatory цели с ненулевой полезностью предпочтительнее. Safety не продаётся
за веса: -2 остаётся сильнее task penalty. Это reward accounting для объявленных
ограничений, не flight shield или модель всех космических рисков.

Threshold определяет **приёмку**, но частичный прогресс ниже threshold может
сохранять положительную C при выполненных mandatory целях. Такой эпизод не
становится успешным SFT example. Это graded objective, не бинарный success bonus.
Правила относятся к undiscounted returns; произвольный RL discount меняет задачу.

`reward_breakdown` в контрактном режиме аддитивен по полям `utility_delta`,
`mandatory_penalty`, `rejection_penalty`; `raw_native_reward` — диагностика,
не дополнительная награда. При native truncation task penalty также отражается
в этом поле. `metrics.reward_potential` показывает принятый task potential,
а `metrics.utility.score` сохраняет weighted safety-gated utility до task penalty.

## Завершение по бюджету агента

Публикуется бюджет вызовов и incomplete penalty=1. Если IndustrialEpisode
исчерпал call budget до native termination, последняя tool response получает
дополнительный `episode_penalty=-1`, включая случай invalid JSON. Это предотвращает
обход обязательств через остановку до конечной эпохи. Физическое время на invalid
call не продвигается; штраф не имитирует несуществующее исполнение.

Повторный вызов после episode done не разрешён. Native terminal и бюджет в одну
эпоху не дают двойной incomplete penalty. Hard safety failure также не получает
второй task penalty. Gym показывает `max_calls`, `remaining_calls` и penalty;
выход по call budget — truncation, не выполнение миссии.

Сумма rewards в episode/export включает invalid-call и episode penalties.
`utility.decision_return` по-прежнему учитывает только session decisions и может
отличаться от неё. Незавершённый эпизод никогда не проходит declared SFT eligibility.

## Текущая граница dispatcher — не неизменная физическая невозможность

Для single-epoch competition вычисляется cardinality upper bound из уже
запущенных аппаратов и текущей опубликованной dispatch capacity. Если обязательных
условий больше bound или лучший допустимый weighted fraction ниже threshold,
публикуется `infeasible_under_current_dispatch_rules`.

Это **условный** вывод. Будущие неизвестные обновления не читаются и не раскрываются.
Если позже в ту же dispatch эпоху оператор увеличит capacity с 1 до 2, bound
обновится, и прежний запрос может быть выполнен. Непротиворечивый bound получает
статус `not_ruled_out`, а не «физически доказана достижимость».

В тесте два мира имеют одинаковый публичный prefix: у второго на t=5 выпускается
capacity=2. Он выполняет обе native цели и declared acceptance, а исходный мир
до события не знает о таком будущем. Генератор не resample-ит infeasible запросы.

## Экспорт: старый default и отдельная версия схемы

```python
from space_env.industrial.missions import run_mission
from space_env.industrial.agent import export_training

record = run_mission(
    "orbit_and_deployment/contract_mandatory_train", policy="public",
)
assert not record["metrics"]["success"]
assert record["metrics"]["task_acceptance"]["accepted"]

export_training(
    record, "new-declared.sft.jsonl", format="sft", objective="declared",
)
```

Без `objective="declared"` используется прежняя native all-goals eligibility;
partial episode не становится SFT автоматически. У declared row другая схема:
`space-env.industrial-objective-training/1`. Она содержит:

- `success` — результат объявленной задачи;
- `native_success` — отдельный неизменённый native результат;
- `success_basis="declared_task_acceptance"`;
- опубликованный `task_objective` и `task_status`;
- causal messages и фактические rewards, без privileged configuration/checkpoints.

Контракт должен существовать в начальной source-bound конфигурации и первом
наблюдении. Нельзя выбрать threshold только после результата или изменить
`accepted` без провала replay. Integrity/replay не заменяют криптографическую
аттестацию оператора: setup остаётся trusted частью эксперимента.

## Воспроизведение

```bash
python -m space_env.industrial.phasing_corpus --windows --competition --utility \
  --acceptance mandatory --samples-per-split 2 --actors 2 \
  --output artifacts/new-mandatory-corpus

python examples/modeling/contract_comparisons.py \
  --corpus artifacts/new-mandatory-corpus --output artifacts/new-contract-comparisons

python examples/modeling/export_phasing.py \
  --corpus artifacts/new-mandatory-corpus --objective declared \
  --output artifacts/new-declared-training

python examples/modeling/export_contract_negatives.py \
  --corpus artifacts/new-mandatory-corpus \
  --comparisons artifacts/new-contract-comparisons \
  --output artifacts/new-contract-negatives
```

Presets `priority`, `mandatory`, `infeasible` используют одну физику. Mandatory
baseline ранжирует обязательные цели до weights. Priority-only negative control
намеренно игнорирует mandatory ranking; это проверка критерия, не сильный
промышленный конкурент. Train-only export сохраняет и неверные решения,
и resource failure, и budget truncation; development/test не копируются.

Текущий scope: 2–4 аппарата в competition, центральная гравитация, конечная
инерциальная тяга, ideal state и synthetic dispatch. Новых RF/engine-fault,
body-frame, J2/drag/SRP, collision или многодневных моделей этот блок не вводит.
Следующие этапы — отказ от невыполнимого запроса как отдельное проверяемое действие,
разные сроки/поступление задач, межэтапный физический runtime и обучение модели.

`objective` в экспортёре выбирает eligibility и success label, **не пересчитывает
физику или сохранённые rewards**. При acceptance-настроенной среде награда уже
учитывает контракт; default native export лишь сохраняет прежний способ маркировки.
Для таких данных потребителю следует явно выбрать нужную схему/метрику.

## Локальные проверки

Основной regression: **129 тестов прошли**, включая прежние reward/export контракты.
Ещё **4 теста** проверили guard-условия example сравнений — **133 разных теста**.
Тесты покрывают неизменность native success, explicit SFT mode, отказ от поддельного
accepted flag, mandatory-before-priority, budget exhaustion, текущие capacity bounds
и одинаковый публичный prefix при скрытом будущем обновлении.

В основных матрицах выполнены 36 эпизодов: priority/mandatory/infeasible, по два
seed на train/development/test, wait/public. Дополнительный example содержит
14 эпизодов: шесть priority-only negative controls, четыре четырёхаппаратных
сравнения, capacity update, native fuel violation и два budget cutoff.
Итого **50 записей с cold replay**, из них **48 с native termination и 2 truncation**.

- Public проходит declared acceptance во всех шести priority и шести mandatory
  случаях, сохраняя native full success=false при одном выполненном условии.
- В mandatory случаях игнорирование обязательной цели даёт return -0.1,
  корректный выбор — +0.1. В четырёхаппаратном варианте: -0.25 против +1/12.
- Infeasible запросы не проходят acceptance и не исключаются из матрицы.
- Capacity update на t=5 делает обе цели выполнимыми: accepted=true и native=true.
- Safety-negative имеет return -2; valid/invalid budget cutoffs — -1 и -1.1.

Train-only declared export даёт **4 SFT и 17 RL строк**. Это 17 разных train
records: четыре из них представлены и как SFT, и как RL. Held-out messages не
экспортированы. В successful partial SFT одновременно записаны success=true
по объявленному контракту и native_success=false; неуспех не скрывается.
Модель не обучалась, inverse/priority-only controls не являются промышленными
конкурентами, а малый synthetic набор не измеряет надёжность на реальных миссиях.

Данные: `artifacts/task-acceptance-20261115/`. Qualification использует
`final-tests.log` и именованные corpus/export directories; предварительные dev
прогоны не перепривязывались к окончательным исходникам.

Wheel 0.134.0 содержит **377 побайтово совпадающих Python/JSON файлов**.
Из отдельного cwd выполнены полные priority/mandatory/infeasible test-эпизоды
с replay и раздельными статусами. Это smoke с имеющимися зависимостями,
не чистая установка или полная installed qualification matrix.

[Приёмочная запись](acceptance/DECLARED_TASK_ACCEPTANCE_V1.json) связывает source
manifest, текущие examples и SHA-256 **90 локальных артефактов**. Исторические
receipts не перепривязаны. Seals подтверждают локальную целостность и согласованность,
не внешнюю сертификацию и не фактическое обучение модели.
