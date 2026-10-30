# Записываемые инструменты агента и последовательная ретрансляция

Срез рабочего дерева0.134.0, 30 октября2026. Продолжение
[portable experiments](PORTABLE_EXPERIMENTS.md). Протокол не запускает внешнюю LLM:
он задаёт JSON schemas, исполняет вызовы и сохраняет проверяемую историю.

## Что запускается

Готовый пример фактической передачи5 Мбит между близкими аппаратами:

```bash
python examples/agent_tools/relay_demo.py /tmp/agent-demo
python -m space_env.cli replay-tools /tmp/agent-demo/agent-run.json
python -m space_env.cli export-dialogue /tmp/agent-demo/agent-run.json /tmp/dialogue.jsonl
```

Пример создаёт два аппарата и существующий продукт в памяти первого. Передача
занимает5с общих часов. Это задача передачи/опеки, не зачёт новой съёмки или
доставки на Землю. Данные в приёмнике появляются только в конце операции.

Можно записать заранее заданные JSON-вызовы:

```bash
python -m space_env.cli record-tools examples/portable_run/mission.yaml examples/agent_tools/calls.json /tmp/tools.json --lineage tools-demo --seed 7 --call-budget 20
python -m space_env.cli replay-tools /tmp/tools.json
```

Или запустить ограниченный эвристический baseline теми же инструментами:

```bash
python -m space_env.cli record-policy examples/portable_run/mission.yaml /tmp/policy.json --lineage policy-demo --policy priority --step 10 --max-steps 3
```

Последние два примера без станций/спроса проверяют pipeline, а не полезность плана.
Для рабочей миссии передать свой сценарий v2. Baseline предпочитает downlink при
контакте и буфере, иначе выбирает доступную заявку по fifo/deadline/priority/rolling.
Это эвристика, не constraint solver: плохие решения и отказы остаются в записи.
По достижении max_steps или ошибке бюджета выполнение останавливается без скрытого
завершения эпизода. Число отказов выводится отдельно; нулевое число отказов не
означает выполнения всех заявок.

## SDK и интерфейс

```python
from space_env.data_generators.agent_run import AgentRunRecorder, replay_agent_run, export_tool_dialogue
from space_env.mission.agent_protocol import tool_schemas
from space_env.mission.tools import MissionTools

recorder = AgentRunRecorder(MissionTools("family_nominal.yaml", seed=7, call_budget=20),
                            task="Collect declared products and deliver them on time.",
                            lineage="mission-family-1",
                            provenance={"teacher": "my-agent-v1", "expert": False})
view = recorder.invoke("observe", {}, call_id="read-1")
result = recorder.invoke("advance", {"dt_sec": 10, "expected_revision": 0})
recorder.save("agent-run.json")
assert replay_agent_run("agent-run.json")["ok"]
export_tool_dialogue("agent-run.json", "dialogue.jsonl")
schemas = tool_schemas()
```

`tool_schemas()` описывает шесть функций: observe, forecast, submit, cancel,
advance и relay. Схемы предназначены для JSON-интеграции; адаптеры конкретных
провайдеров/их strict modes не квалифицированы.
Аргументы проверяются Pydantic с запретом неизвестных полей, неконечных чисел,
отрицательных длительностей и bool вместо revision. Вызовы имеют уникальный call_id.

Schema error не вызывает executor и не расходует бюджет инструментов. Вызванный
метод расходует вызов даже при доменной ошибке; budget0 означает отсутствие лимита.
Это счётчик вызовов, не модель стоимости токенов/сетевой задержки LLM. RevisionError,
ошибка ресурса и ToolBudgetError записываются как `{ok:false,error:...}`.
Неожиданные ошибки программирования/ввода-вывода распространяются наружу, не получают
метку корректной демонстрации. Запись одновременно из нескольких потоков не поддержана.

## Информация, доступная агенту

- Контекст: инструкция пользователя, известные декларации заявок, список аппаратов,
  горизонт и schemas. Это профиль известного спроса, не скрытые будущие заявки.
- observe: SensorStream telemetry и текущие masks доступности. Masks — отдельно
  объявленная информация о текущей допустимости, не зашумлённое измерение.
- Каналы разных аппаратов имеют раздельные RNG/cache. Primary `tools.stream`
  сохранён для первого аппарата; остальные получают детерминированные каналы.
- advance возвращает revision/time/horizon/statuses. UI frames и точные значения
  SoC/орбиты из MissionSession.report в ответ нового протокола не попадают.
- forecast нового протокола — **номинальный геометрический** прогноз из текущей
  орбиты/модели. Он не исполняет будущие скрытые инжекты и не возвращает истинные
  SoC/память. Это не state-estimation filter и не прогноз с учётом будущих манёвров.
- Прямые прежние MissionTools.advance/forecast сохраняют прежние UI/truth semantics.
  Для указанного информационного контракта использовать invoke_tool/AgentRunRecorder;
  смешивать прямые вызовы и записываемый протокол нельзя.

## Переносимость и данные обучения

Session snapshot хранит world schema2, revision, session_id, task registry и
прожитые tracks/frames. Tools snapshot добавляет pending commands, бюджет/счётчик,
состояния сенсоров и их идентичность зарегистрированным каналам мира. Исходный YAML
не требуется. `MissionSession.from_world` и `MissionTools.from_session` позволяют
записать уже настроенный мир, не создавая его повторно через каталог.

`space-env.agent-run/1` содержит context/provenance/runtime manifest, начальный и
конечный tools snapshot, аргументы/ответы каждого вызова и состояния для проверки.
Replay проверяет каждый ответ и результат, бюджет, pending, RNG, ревизию,
конечное состояние/журнал/frames. Декларации context сверяются с начальной миссией.
SHA-256 проверяет целостность, не аутентичность источника; версии/источники/пакетные
данные фиксируются как в portable run. Межверсионная совместимость не предполагается.

`space-env.tool-dialogue/1` — одна JSONL-строка с messages/tools, lineage, split,
provenance и run checksum. Сообщения идут причинно: user context → assistant tool
call → tool response. Состояния `after`, initial/final snapshots и скрытые причины
отказов из verifier в model inputs не экспортируются. Неудачные действия не
выбрасываются из истории. Экспорт выполняется только после успешного replay того
же прочитанного record и атомарно заменяет файл; при отказе старый dataset сохраняется.

Это формат для обучения tool-call behavior/SFT после отбора teacher. Replay не
доказывает экспертность или оптимальность, не добавляет reward/предпочтения и не
квалифицирует адаптер тренера конкретного вендора. Lineage группирует варианты;
вызывающий отвечает за правильные семейства сценариев и отсутствие leakage.

## Профиль serial-sampled-relay/1

`relay(source,destination,dt_sec,expected_revision)` — блокирующий один хоп.
Участники должны быть idle, pending commands отсутствовать. Валидная операция
продвигает **весь мир**; при закрытой линии/пустом буфере она тоже расходует время,
возвращая no_transfer. Невалидные участники/длительность/горизонт/резервация не
меняют физический мир. Бюджет вызова всё равно расходуется.

Объём ограничен inventory, доступным у отправителя до начала, TTL обоих бортов,
выборочным минимальным ISL rate и свободной памятью получателя в конце. Участники
с запланированным событием/отказом/восстановлением внутри интервала отвергаются:
сначала вызывающий должен разделить интервал по этим границам. Активный burn,
радиосеанс, съёмка или занятый терминал запрещают операцию. После продвижения часов
проверяется фактический endpoint; передача фиксируется после него. Новые данные
не используются для downlink/следующего хопа в уже прошедшем интервале.

Ошибка после продвижения часов откатывает world, данные, резервации и сессионную
revision/frames. Payload identity/offset сохраняются. Этот профиль использует
упрощённую ISL RF-модель и sampled geometry, не точную непрерывную AOS/LOS границу.
Нет packet streaming, propagation delay, отдельного энергобаланса ISL-трансивера,
полной antenna-pointing модели, конкурентных hop schedules или квалифицированного
ACK/retry протокола. Часть custody metadata сохраняется по существующим правилам.
Legacy Fleet.relay/forward не переводились автоматически на эту семантику.

## Что дальше

Остаются полноценный resource-constrained planner, scheduled relay events вместо
блокирующих хопов, bounded/sharded хранение и независимый reference physics profile.
Артефакты находятся в памяти и сохраняют кумулятивные outcomes/frames; это ещё не
доказательство масштабирования до тысяч аппаратов или промышленной эксплуатации.
