# Приёмка агентного протокола и serial relay v1

30 октября2026. Пакет0.134.0, изменения рабочего дерева поверх portable run v1.
Руководство: [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md).

## Реализовано

- Шесть JSON tool schemas и проверяемая диспетчеризация: observe/forecast/submit/
  cancel/advance/relay. Validation/domain/budget failures сохраняются как ответы;
  неожиданные исключения не превращаются в успешные демонстрации.
- Session/tools snapshots сохраняют revision/session_id, pending commands, budget,
  frames/tracks и sensor RNG/cache/идентичность зарегистрированным каналам.
  Мир может поступать из публичного from_world/from_session без исходного YAML.
- Исправлен общий cache измерений нескольких аппаратов. Каналы теперь раздельны.
- Новый protocol advance не раскрывает UI truth. Forecast возвращает номинальную
  геометрию, не будущую истинную батарею/память/скрытые инжекты. Прежние прямые методы
  сохранены отдельно; информационный контракт относится к новому протоколу.
- Последовательный блокирующий relay продвигает весь мир до поступления данных.
  Учитываются ready inventory, TTL обоих бортов, sampled ISL rate, память,
  radio readiness и занятость участников. Ошибка откатывает передачу и время.
- World snapshot явно хранит engine_order: JSON canonical sorting больше не меняет
  порядок акторов и идентичность primary sensor после переноса.
- AgentRunRecorder сохраняет context/аргументы/ответы/ошибки и verifier state.
  Replay сверяет события, состояния и конечный tools snapshot. Context проверяется
  относительно исходной объявленной миссии. Экспорт tool-dialogue не включает
  скрытую истину verifier в сообщения для модели; split зависит от lineage.
- Recorded heuristic baseline и CLI record-policy/record-tools/replay-tools/
  export-dialogue. Эвристика не обходит инструменты и бюджет, не помечается экспертом.

## Проверки на финальных исходниках

- Новый `tests/test_agent_run.py`:38 passed,4.16с.
- CI fast:110 passed,12.93с.
- Интеграция agent/tools/session-dataset/fleet/ground/mission-contract/M0/finale:
  82 passed,13.68с. Наборы пересекаются и не складываются.
- Пример relay_demo:5 Мбит получены в момент5с;3 tool calls повторились;
  экспортирована одна строка с7 сообщениями. Проверены сохранение данных и запрет
  использования пришедших данных в предыдущем интервале downlink.
- CLI fresh-process replay после удаления YAML/calls, JSONL checksum, сохранение
  старого dataset при tamper/replay failure, бюджет, stale revision и отмена
  продолжительной передачи входят в новый набор.
- Новые проверки privacy: скрытая причина есть в verifier snapshot, но отсутствует
  в dialogue; advance не возвращает UI truth; nominal forecast не зависит от SoC
  и скрытого будущего отказа связи. Текущие validity masks объявлены публичными.
- Wheel установлен со всеми зависимостями в новый venv без system site packages.
  Проверка из `/tmp`, Python `-I`: четыре smoke cases, Dict rollout, Studio handlers,
  portable checkpoint/replay, serial relay1 Мбит, agent replay2 calls,
  dialogue5 messages и CLI validate прошли.
- Все149 Python-модулей wheel побайтово совпадают с checkout. SHA-256 и результаты:
  [AGENT_PROTOCOL_V1_VERIFICATION.json](AGENT_PROTOCOL_V1_VERIFICATION.json).
- Ruff прошёл для новых/изменённых файлов пакета, кроме engine.py, где не заявляется
  устранение прежних замечаний. `compileall` и `git diff --check` прошли.

Собрано446 тестов. Полный pytest,72ч/16 аппаратов, browser interaction,
сравнение с конкурентами и независимая физическая валидация для этого среза
не выполнялись. Более ранние результаты не объявляются новой полной квалификацией.

## Оставшиеся gates

Serial sampled relay — объявленный приближённый профиль, не полный радиотехнический
или packet/ACK/retry симулятор. Отдельный энергобаланс ISL и непрерывная геометрическая
валидация не добавлены. Legacy Fleet.relay/forward не мигрировали автоматически.

Recorded teacher — эвристика, не solver с доказанным соблюдением всех ограничений.
SFT dialogue содержит правильную причинную структуру, но квалификация teacher,
адаптера тренера и held-out обучения остаётся отдельной работой. Контекст объявляет
весь спрос заранее. Инструменты не являются production authenticated API.

Хранение пока in-memory с кумулятивными outcomes/frames; bounded/sharded storage
и большие workloads ещё не закрыты. Frame/time/reference physics, полный event
scheduler и официальная MARL qualification остаются в roadmap.
Commit/push/publication/deployment не выполнялись.
