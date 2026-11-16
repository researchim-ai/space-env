# Приоритеты целей и полезность частичного выполнения

Следующий блок: [обязательные цели и явная приёмка задачи](DECLARED_TASK_ACCEPTANCE.md).
Он добавляет отдельную opt-in success basis и версию training schema; описанный
ниже native default не меняется.

Опциональный режим `OrbitSession.utility` добавляет явную нормированную ценность
выполненных целей и отдельный reward gate для объявленных ресурсных ограничений.
Новый конфликтный сценарий требует выбирать между индивидуально достижимыми
целями, конкурирующими за один dispatch slot. Общий каталог содержит 5 семейств
и 42 стартовых профиля; дополнительные профили переиспользуют ту же физику.

## Совместимость и уточнение прежнего описания

Старый native delivery reward уже учитывал число выполненных условий, когда
список failures пуст. Неуспех общей миссии не обязательно обнуляет эту награду:
в прежних critical примерах получалось положительное значение за 1/2 или 3/4
условий. При наличии failures действуют native failure/clawback правила.

Поэтому новый режим не следует понимать как первое появление любой частичной
награды. Он добавляет **явные веса, нормировку и документированный safety gate**.
Прежняя формулировка об отсутствии пропорциональной частичной награды была
неточной без этой оговорки. Исторические receipts не переписываются.

Без `utility` прежние награды, native цели, физика и признаки успеха сохранены.
Новый режим применяется только к текущим orbital recipes с native objective
`delivery`; чужие objective costs не отбрасываются молча.

## Контракт полезности

Для объявленных условий \(i\) задаются целые веса \(1\le w_i\le1000\).
Неуказанный вес равен 1. Неизвестные condition IDs и неподдержанные значения
отклоняются. Пусть \(m_i(t)\) — native признак выполнения:

\[
C(t)=\frac{\sum_i w_i m_i(t)}{\sum_i w_i},\qquad 0\le C\le1.
\]

В этом семействе native орбитальные условия имеют общую конечную эпоху.
До неё нет искусственного positive shaping за приближение к цели.

\[
U(t)=\begin{cases}
 C(t),&\text{объявленные safety constraints не нарушены},\\
 -2,&\text{нарушение зафиксировано хотя бы раз},
\end{cases}
\qquad
r_t=U(t+1)-U(t)-0.1\,\mathbb{1}_{\text{rejected batch}}.
\]

Latch сохраняется при последующих шагах и checkpoint replay. При текущем
начальном U=0 недисконтированная сумма session rewards не превышает 1 и при
нарушении safety не превышает -2 (за вычетом дополнительных отказов).
Произвольное увеличение весов не может купить нарушение safety через
положительную completion utility. Пропорции весов влияют на ценность, масштаб — нет.

Это определение **undiscounted task utility**, а не доказательство policy
invariance для произвольного RL discount или общего многовременного goal family.

### Какие ограничения учитываются

Проверяются объявленные native fuel reserve, energy budget и health requirements,
а также native keepout и неизвестные failure codes. Если измерение для требуемого
ресурсного bound отсутствует или нечисловое, полезность fail-closed.
Deadline и `orbit_window_missed` трактуются как невыполнение задания, а не
нарушение безопасности аппарата; полная mission success остаётся false.

Проверка safety не симулирует отсутствующую физику. Она не предотвращает
столкновения и не является flight shield. В текущих orbital recipes реально
проверен native перерасход топлива; energy/health/missing-value gate также
проверен контрактными unit-тестами. Полётной или аппаратной калибровки нет.

Отклонённый dispatch/native batch — recoverable ошибка стоимостью 0.1, не
автоматическая авария с penalty -2. Известные ограничения admission по-прежнему
проверяются до запуска; reward gate не заменяет эти проверки.

## Что сохраняется раздельно

- `metrics.success` — прежний native all-goals success. Оптимальное частичное
  выполнение само по себе не превращается в полный успех.
- `metrics.goal.completed` — число выполненных условий.
- `metrics.utility.completion_fraction` — взвешенная доля без safety gate.
- `metrics.utility.score` — итоговая потенциальная полезность с latch.
- `metrics.utility.decision_return` — сумма rewards валидных session decisions.
- `metrics.utility.raw_native_return` — сумма исходных TaskSession rewards.

Историческое `native_return` сохранено: в wrapper оно включает прежнюю dispatch
penalty. Для новых сравнений используйте явно именованные поля выше.

Некорректные JSON tool calls расходуют episode budget и дают дополнительный -0.1
на уровне IndustrialEpisode. Они не являются session decisions, поэтому не входят
в `decision_return`. Экспорт RL вычисляет фактическую сумму **всех** tool-call rewards.

Каждый utility transition содержит `reward_breakdown` с raw native reward,
изменением utility и refusal penalty. Это не сумма старой и новой награды:
в режиме utility старый goal reward заменяется выбранным протоколом.
Idempotency не начисляет utility повторно; bridge также защищает от повторного
model-call decision key. Остановка по call budget до достижения эпохи целей
не даёт награду за ещё не выполненное условие.

## Конфликт за слот

`generation.competition=True` требует windows и исключает disruption profiles.
Для всех аппаратов оставлено единственное окно [5,10) s. Decision grid — 5 s,
общий capacity — 1, каждое условие требует собственную выполненную команду.
Значит, в этой дискретной постановке можно выполнить не больше одного условия.

Каждая индивидуальная цель конструктивно вычислена RK4 по собственному burn.
Общего all-goals feasible reference нет. `reference_steps` пуст, а public
`dispatch_bound.maximum_completed_conditions` равен 1. При utility также
публикуется \(\max w_i/\sum w_i\) как upper bound взвешенной доли именно для
этих правил. Это комбинаторная граница dispatcher, не общий сертификат
физической недостижимости космической миссии.

Public policy использует тот же численный endpoint predictor для выбора
направления; готовые в одну эпоху команды ранжирует по опубликованному весу,
затем по residual. Это простой приоритетный baseline, не learned agent.

При двух целях с весами 9 и 1 приоритетный выбор даёт 0.9, обратный — 0.1,
wait — 0. Для четырёх аппаратов с весами 9/1/1/1 граница равна 0.75.
В обоих случаях native полная mission success остаётся false.

## Python и корпус

```python
from space_env.industrial.orbits import OrbitSession
from space_env.industrial.missions import make_episode

session = OrbitSession({
    "actors": 2,
    "generation": {
        "split": "train", "seed": 17, "windows": True, "competition": True,
    },
    "utility": {"priorities": {"slot-sat-00": 9, "slot-sat-01": 1}},
})

# Готовые стартовые профили:
partial = make_episode("orbit_and_deployment/priority_test")
recovery = make_episode("orbit_and_deployment/utility_recoverable_test")
critical = make_episode("orbit_and_deployment/utility_critical_test")
```

Utility применим и к прежним fixed/generated orbital recipes. Windowed и
competition варианты ограничены 2–4 аппаратами; immediate/fixed поддерживают
прежние 2–8. Setup копируется и проверяется. Агент не может передать веса или
goal statuses через action tool.

```bash
python -m space_env.industrial.phasing_corpus --windows --competition --utility \
  --samples-per-split 2 --actors 2 --output artifacts/new-priority-corpus

python -m space_env.industrial.phasing_corpus --windows --utility \
  --disruptions recoverable --samples-per-split 2 --actors 2 \
  --output artifacts/new-utility-recovery

python examples/modeling/utility_comparisons.py \
  --corpus artifacts/new-priority-corpus --output artifacts/new-priority-comparisons

python examples/modeling/export_phasing.py \
  --corpus artifacts/new-utility-recovery --output artifacts/new-utility-training
```

Runner при `--utility` задаёт вес 9 одному аппарату, остальным 1; выбор высокого
веса чередуется по seed. Веса публичны. Train/development/test остаются раздельными
по прежним диапазонам; из test нельзя сделать train. Корпуса recovery/critical
связаны общим synthetic происхождением и не являются независимыми задачами.

Пример сравнений включает inverse-priority negative control, четырёхаппаратный
случай и намеренный double-burn для native fuel violation. Обратное ранжирование
не считается сильным промышленным конкурентом или доказательством превосходства.

SFT требует прежнего **полного успешного** результата без отрицательной feedback.
Поэтому оптимальный partial priority episode по-прежнему не экспортируется как
успешная SFT-демонстрация. RL сохраняет частичные и аварийные исходы с фактической
полезностью. Явная отдельная success-семантика задачи максимизации utility,
если она потребуется для SFT, должна вводиться отдельным контрактом.

## Дальше

Следующие блоки: обязательные и необязательные цели, разные SLA и общие ресурсы,
онлайн-поступление/отмена задач, более общие оптимизационные baselines и измеренное
SFT/RL обучение. Сейчас физика остаётся центрально-гравитационной, а dispatch —
синтетическим ресурсом; новых моделей RF, двигателей, контакта или collision
avoidance это обновление не добавляет.

## Дополнительные обучающие примеры

```bash
python examples/modeling/export_utility_training.py \
  --corpus artifacts/new-priority-corpus \
  --comparisons artifacts/new-priority-comparisons \
  --output artifacts/new-utility-additional-training
```

Экспортёр проверяет исходные seals/source identity, матрицы попыток, соответствие
конфигураций и cold replay выбранных **train** записей. Дополнительно сохраняет
inverse-priority и native fuel-failure RL examples. Небезопасные case IDs не могут
использоваться для выхода за output directory; существующий output отклоняется.

Preference JSONL использует отдельную provider-independent схему:

- `prompt` — общий публичный prefix до первого различающегося действия;
- `chosen` / `rejected` — ровно по одному assistant tool-call message;
- `outcomes` — фактические episode returns вне prompt;
- `source_records` — хеши исходных проверенных записей.

Разные начальные условия, разные prompt/feedback до выбора, test/development,
небезопасные либо незавершённые эпизоды не используются как такие пары.
Последующие ответы инструментов и privileged configuration/checkpoints в prompt
не копируются. Ранжирование относится к **utility objective**, а не к полному
успеху миссии. Это не обход strict SFT eligibility и не готовая интеграция с
конкретным training provider; модель в этом обновлении не обучалась.

## Локальные измерения

Проведены **46 полных эпизодов с cold replay**: 36 в основных train/development/test
матрицах (competition, critical, recoverable), 6 inverse-priority сравнений,
3 четырёхаппаратных сравнения и 1 намеренный native fuel violation.

- В шести двухаппаратных competition случаях public даёт 0.9, inverse-priority —
  0.1, wait — 0; full native success false для всех этих политик.
- В четырёхаппаратном случае public достигает dispatch bound 0.75,
  inverse-priority даёт 1/12, wait — 0. Это маленькая дискретная задача с известной
  верхней границей, не результат сравнения с промышленными оптимизаторами.
- Все шесть recoverable public миссий выполнены полностью, utility равна 1.
- Critical public сохраняет безопасный partial результат с utility 0.1 либо 0.9
  в зависимости от приоритетов выполненной цели; полной mission success нет.
- Намеренный double-burn вызывает реальное native `fuel_reserve` failure;
  utility/decision return равны -2. Gate на нечисловых ресурсах, health,
  неизвестных failure codes и latched recovery проверен отдельно unit-тестами.

Обучающие данные: **2 SFT, 15 RL и 2 preference строки**, только из train.
В RL включены inverse-priority и safety-negative примеры; preference labels
сравнивают безопасную utility, не переименовывают partial в полный успех.
Веса/геометрия публичные synthetic; train и held-out режимы имеют общее модельное
происхождение. Обучения LLM и внешнего конкурентного benchmark не проводилось.

Основной regression: **101 тест прошёл**. Дополнительный экспортный набор:
**16 тестов прошли**, три из них пересекаются с основным набором — всего
**114 различных тестов**. Нормировочный тест дополнительно повторён после
рефакторинга тестового кода; в distinct count он учтён один раз.

Данные находятся в `artifacts/goal-utility-20261115/`, игнорируются Git.
`comparisons-final/` воспроизводится repo example; отдельно проверено совпадение
хеша одной записи при повторном запуске текущего example. Выполненные версии
export examples сохранены в артефактах; последующие guards идентификаторов
case отдельно проверены на выход за output directory. Исторические source-bound
receipts не перепривязывались к новым исходникам.

Wheel 0.134.0 содержит **376 побайтово совпадающих Python/JSON файлов**. Из отдельного
cwd выполнены полные priority-test и utility-recoverable-test эпизоды с replay:
полезность 0.9 при partial и 1 при полном выполнении. Это installed smoke с
существующими зависимостями, не clean-install или полная installed matrix.

[Приёмочная запись](acceptance/GOAL_UTILITY_V1.json) связывает source manifest,
текущие examples и SHA-256 **86 локальных артефактов**, включая выполненные версии
export drivers. Она также явно уточняет старую формулировку о partial credit,
не меняя историческую запись. Integrity seals не являются внешней сертификацией
или подтверждением запуска реальной LLM.
