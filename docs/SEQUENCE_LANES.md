# Sequence-lanes v1: наведение, параллелизм и полные миссии

8 ноября 2026. Новый opt-in режим:
`MissionWorkspace.network(configuration, planning_method="sequence_lanes")`.

## Что меняется

Планировщик использует предыдущую запланированную ориентацию аппарата вместо
повторного полного разворота. Новые наведения добавляются **после** предыдущей
съёмки данного аппарата: нельзя вставить новую цель в прошлое и тем самым
незаметно сделать ранее выбранный slew недопустимым.

Учитывается native tracking во время съёмки. После экспозиции реальная ориентация
может отличаться от midpoint quaternion. Для предыдущей съёмки сохраняется
угловая неопределённость не более `omega_max * reserved_capture_time`. Следующий
угол ограничивается сверху:

```text
theta_required <= min(180 deg,
    distance(previous_midpoint_goal, next_midpoint_goal)
    + omega_max * previous_reserved_capture_time)
slew_cells = ceil((theta_required / omega_max + settling + guard) / decision_sec)
```

Первое наведение идёт от известного initial quaternion, с нулевой неопределённостью.
После успешно исполненного slew ориентация снова привязана к его цели; tracking
экспозиции создаёт новый запас, а не бесконтрольно накапливает старые ошибки.
Это условный план для исходного kinematic profile, не сертификат при произвольной
перестановке команд, неисполненном предыдущем slew, rigid-body управлении или
неожиданных fault. Native admission и replay остаются окончательной проверкой.

В примере equatorial-relay-chain первый slew занимает 10 ячеек, последующие —
2 ячейки с tracking guard, а не 1 ячейку по неточному предположению о неподвижной
ориентации во время съёмки. На той же исходной постановке получена доставка 3/3.

## Ресурсы берутся из исполнителя

Compiler читает `MissionRuntime.command_claims`, declared power limits и
memory bandwidth исходного native профиля. В текущей реализации:

- slew: attitude;
- imaging: payload + attitude;
- transform: CPU;
- omni downlink: radio; при directed native claims сохраняется также attitude;
- ISL relay: все sender lanes; receiver резервируется консервативно эксклюзивно.

CPU может работать одновременно с последующим наведением или снимком, если
суммарные peak power и memory bandwidth допустимы. CPU остаётся один; два jobs
не исполняются на нём одновременно. Relay не превращается в независимый
«дешёвый radio» вопреки текущему runtime.

Проверяются окна контакта, ground terminal setup/ACK, число терминалов, TTL и
дедлайны. Энергетическая и lifetime-memory оценки остаются прежними консервативными
суммами. Raw/scratch/in-place reuse и освобождение памяти для дополнительной
упаковки не предполагаются. Native награда и правила выполнения заявок не меняются.

## Ограничение интерфейса агента

Дополнительно резервируется один **старт** команды на actor/decision tick.
Это ограничение интерфейса scalar `json_fixed_v1`, а не утверждение, что физически
нельзя запустить CPU и slew одновременно. Долгие операции продолжают работать
параллельно, но их команды стартуют в разных тактах. Role slots и fixed 5 s cadence
ради успешного benchmark не расширяются.

Это позволяет использовать существующие native tools и локальный Ollama adapter
без выдумывания новых multi-action wire targets. Конкретное поведение Qwen на
новом каталоге требует отдельного inference-эксперимента; в этой приёмке модель
не вызывается и не обучается.

## Поиск и проверка

Bounded portfolio проверяет priority, EDF, обратный порядок и ограниченное число
перестановок. `max_assignment_trials` ограничивает число порядков, `max_search_nodes`
задаёт общий бюджет. Внутри порядка owner/path/station и capture slot перебираются,
CPU и transport ищут ранние допустимые свободные интервалы. Принятый incumbent
может быть неполным при исчерпании бюджета.

Это эвристический order portfolio с append-only pointing. `optimal=false` всегда;
нет заявления о глобальном, непрерывном или native optimum. Интервалы ожидания
и возможность «ждать ради более позднего контакта» не перебираются исчерпывающе.

`verify_sequence_schedule` независимо от выбора поиска проверяет геометрию,
предшествующие наведения с tracking guard, stage order/width, lanes, общую
мощность/BW, стартовые ограничения, память/энергию, контакты и coverage.
Для source-bound audit абляции дополнительно перестраиваются compiler и planner:
seal сам по себе не доказывает соответствия геометрии физическим входам.

После dynamic fork initial-only sequence geometry удаляется; восстановление пока
использует conservative `serial_recovery`. Старый source checkpoint не меняется.

## Воспроизведение

```python
from space_env.workspace import MissionWorkspace
from space_env.llm.mission_suite import compile_suite

config = compile_suite()["cases"][0]["configuration"]
mission = MissionWorkspace.network(config, planning_method="sequence_lanes")
try:
    print(mission.state()["network"]["sequence_lanes"])
    print(mission.run()["network"]["fulfilled"])
finally:
    mission.close()
```

```bash
python -m examples.network.pointing_comparison \
  --output /tmp/new-sequence-comparison \
  --method serial --method exact_small \
  --method pointing_bounded --method sequence_lanes
python -m examples.network.pointing_comparison --audit /tmp/new-sequence-comparison
```

По умолчанию это шесть train workload-семейств на **одной** публичной геометрии,
четыре planner pipelines, одинаковый privileged reference executor. Не считать
их шестью независимыми геометриями или реальными LLM. Все ошибки остаются в
выбранной матрице, не получают выдуманной парной дельты.

## SFT/RL и локальная модель

Native LLMEpisode успешной миссии допускается в SFT только через прежний
complete-goal/error-free gate. Scripted teacher имеет provenance
`scripted_privileged`, quality `successful_execution_not_expert`. Никакого
автоматического статуса экспертности, fine-tuning или улучшения модели здесь нет.
Wire-SFT создаётся только из фактического provider-bound запроса/ответа.

В обоих локальных примерах теперь можно явно выбрать планировщик:

```bash
python examples/llm/run_local_qwen.py --configuration case.json \
  --planning-method sequence_lanes --protocol-profile json_fixed_v1 \
  --max-calls 64 --output /tmp/new-qwen-sequence

python -m examples.llm.generate_local --configuration train.json \
  --planning-method sequence_lanes --max-calls 64 \
  --reserved-splits reserved.json --output /tmp/new-qwen-sequence-train
```

Нужны отдельно запущенный локальный inference server и локальные веса. Команды не
выполнялись для оценки модели в этом пакете. Реально выбранный planning method
сохранён в native checkpoint и связан с manifest через episode hash; evaluation
runner дополнительно сверяет его с явным `case.planning_method`.

## Исправление предыдущей границы

При проверке выявлено, что native imaging сопровождает цель. Предыдущий
`pointing_bounded` учитывал finite midpoint goals, но не резервировал последующее
tracking-отклонение. Теперь к global cone добавлен rate-time guard, а доказательство
явно условно на успешных предыдущих наведениях. На equatorial случае его общий
резерв стал 11 ячеек вместо 10, а нынешний результат — 1/3, не прежние 2/3.

Исторические native 2/3 действительно были исполнены, но не доказывали полноту
угловой границы. Старый receipt сохранён с исходными hashes; его нельзя выдавать
за приёмку исправленной модели. Sequence-lanes с отдельным запасом предыдущей
экспозиции достигает 3/3 без обхода этого исправления и без изменения frozen задач.

## Финальная локальная приёмка

**176 passed / 253.32 s** в наборах sequence/pointing/joint/exact/dynamic/terminal,
LLM episode и corpus. Финальный модуль sequence отдельно: **19 passed / 15.52 s**,
включая проверку native CPU grants в такт с изменением quaternion; counts
пересекаются и не суммируются. Ruff, Python compile и diff-check passed.

На финальных исходниках с tracking guard повторены **24 native исполнения**:
шесть неизменных workload-семейств одной публичной геометрии, четыре режима.
Аудит выполнил native replay и отдельное повторное построение планов.

- relay-chain: serial/exact/pointing = 1/3, sequence = **3/3**;
- cpu-backlog: 1/3 → **2/3**;
- memory-pressure: 1/3 → 1/3;
- station-blackout: serial/exact 1/3, pointing 2/3, sequence **3/3**;
- link-blackout: 1/3 → **2/3**;
- deadline-pressure: 0/3 → 0/3.

Четыре улучшения против serial и против исправленного pointing_bounded, без
ухудшений на подматрице. Planned/native количество доставок совпало во всех
24 ячейках. Это не проверка всех 36 конфигураций/шести геометрий и не полная
physics qualification.

Все шесть sequence-эпизодов собраны в зарегистрированный и проверенный корпус:
**6 RL + 2 native SFT**, одна независимая геометрия, все train, teacher provenance
`scripted_privileged`. Wire-SFT = 0, модель не вызывалась и не обучалась. Всего
216 решений, из них 162 all-wait: такой маленький корпус не называется
разнообразным датасетом промышленного обучения.

[Машинный протокол, source hashes и точные пары](acceptance/SEQUENCE_LANES_V1.json).

Сборка corpus bridge из своего свежего отчёта, без инференса:

```python
import json
from pathlib import Path
from space_env.llm.corpus import build_corpus, register_corpus, verify_corpus

root = Path('/tmp/new-sequence-comparison')
report = json.loads((root / 'report.json').read_text())
inputs = [root / row['artifact'] for row in report['rows']
          if row['method'] == 'sequence_lanes' and row['status'] == 'completed']
build_corpus(inputs, '/tmp/new-sequence-corpus')
register_corpus('/tmp/new-sequence-corpus', '/tmp/shared-corpus.sqlite')
print(verify_corpus('/tmp/new-sequence-corpus'))
```

Ошибки без валидного episode остаются в benchmark matrix и не превращаются в
обучающие строки. Частично выполненные native episodes, напротив, включаются
для RL; они не отбрасываются ради высокого SFT success rate.
