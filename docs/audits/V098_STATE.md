# Проверка рабочего дерева 0.98.0

27 октября 2026. HEAD `9a95032`, поверх него существенные незакоммиченные изменения
пользователя. Версия в pyproject и пакете — 0.98.0. Проверяется именно рабочее дерево,
не только последний commit. Автор аудита продуктовый код не менял.

## Объём и подтверждения

Прочитаны diff и текущие world/commands/engine snapshot, Fleet reservation changes,
session/API, scenario validation, tasking/scoring, Gym и benchmark/dataset/CI.
AST parsing всех 119 Python-модулей пакета; зарегистрировано 36 сред.
Это проверка изменённых контрактов и важных границ, не независимая построчная
валидация всех физических моделей.

Выполнена команда:

```bash
python -m pytest -q tests/test_mission_contract.py tests/test_mission_roundtrip.py tests/test_fleet_tasking.py tests/test_scenario_v2.py tests/test_mission_benchmark.py tests/test_studio_api.py -k 'not http_server_answers_health and not api_serves_scenario_fleet_and_studio' --durations=5
```

Результат: **37 passed, 2 deselected, 2 warnings, 37.56 с**.
Суточная миссия — 26.94 с; smoke suite — 1.18 с в текущем окружении.
Собрано 252 теста. Полный suite, 72-часовые сценарии, HTTP socket и browser UI
в этой проверке не запускались; старые результаты 0.93 не выдаются за новую регрессию.
Clean-wheel install не проверялся. Force-include assets в конфигурации сборки есть.

## Подтверждённый прогресс

Новые тесты и код закрывают прежние простые случаи forecast health leak, horizon,
armed radio contention и смешения половин разных product_id. Появились постоянная
CommandBook, rollback engine step, world relay reservations, semantic hash,
MissionSession/revision, локальный session API, FleetWatch 1–4 аппарата/до16 заявок,
smoke benchmark и упаковка Studio в wheel configuration.

Не повторять прежний вывод «сессий нет» или «armed-сеанс не учитывается».
Наличие реализации не закрывает все граничные сочетания.

## Новые воспроизводимые дефекты

Команда: `python docs/audits/v098_probes.py`.
Результаты: [v098_probe_results.json](v098_probe_results.json).
Диагностический script сообщает наблюдения; это не passing acceptance suite.

1. **Batch не атомарен.** Две новые команды с одним ID и разным телом проходят
   предварительный check относительно старой книги; submit второй вызывает ошибку,
   но первая уже записана. Получено 0→1 records при now=0. Проверять внутренние
   конфликты batch и откатывать всю транзакцию.
2. **Адресная команда не исполняет заявку.** Команда imaging с request_id
   `not-a-request` записана как completed. Accept не разрешает request_id в реальную
   цель, settle учитывает station rejection, но не факт съёмки. Нужен authoritative
   executor с registry заявок и фактическим operation outcome.
3. **Оценка выбирает не ту acquisition.** Ранний продукт полностью закрывает100бит
   к t5 при deadline10; другой продукт120бит приходит t20. `_single_acquisition`
   выбирает наибольшее покрытие и `_completion_time` возвращает позднюю сдачу:
   on_time пуст. Выбирать earliest valid completion по всем acquisition.
4. **Снимок теряет наземный сегмент.** Мир со станцией Kiruna после save/load JSON
   имеет stations=[]. `load_world` собирает сценарий с attach_stations=False;
   snapshot не содержит станций. Тест одной орбиты без станций это не обнаруживает.

## Другие ограничения по чтению кода

- World snapshot берётся после `_one_antenna`, которое уже снимает armed radio.
  Поэтому показанный rollback не доказывает откат всех мутаций шага.
- MissionSession.advance вызывает accept до step, без session lock/транзакции;
  плохой dt/исключение может оставить изменённую книгу при прежней revision.
  Сетевой конкурентный тест не выполнялся, это риск по текущему контракту кода.
- Request/destination fields в orders снова сводятся к строкам; running/cancel
  полноценной длительной операции не реализованы общим executor.
- Арбитраж станций смотрит на текущий контакт, а не все переходы внутри dt.
  Relay всё ещё переносит данные до world.step; позднее сокращение dt исправлено,
  но полная временная причинность операции требует отдельного контракта.
- Снимок хранит внешний путь YAML; конфигурация/журнал/encounter не представлены
  самодостаточным переносимым mission artifact. Replay не сверяет весь final state.
- FleetWatch фиксирована на 4 слотах/16 заявках, flat vector; отрицательный slot
  не исключён до индексации. Seed не параметризует mission demand/world.
- Новые sessions доступны в API, но обращения `/api/sessions` в studio.html/studio.js
  не найдены. Это не доказательство отсутствия динамической интеграции вообще,
  но текущий frontend не следует считать проверенным живым mission console.
- Smoke suite — четыре известных регрессионных случая; нет независимого спроса,
  полноценной оценки обучения, hidden OOD и статистического сравнения алгоритмов.
- Stage-time RK4, TEME conversion, независимая Pc validation и прочие physics limits
  не закрыты изменениями этого дерева.

## Следующее действие

Актуальное задание: [MODELING_ROADMAP_V098.md](../MODELING_ROADMAP_V098.md).
M0 закрывает новые execution defects; M1 задаёт проверенную достоверность; M2–M7
последовательно развивают связанные подсистемы, сенсоры, сеть, неопределённость,
обучение/данные и масштаб. Новые возможности должны проходить общие gates.

Сравнение с конкурентами — только по совпадающему профилю и измеренным метрикам.
Проверены официальные [BSK-RL API](https://avslab.github.io/bsk_rl/api_reference/index.html)
и [Basilisk](https://avslab.github.io/basilisk/); собственных сравнительных прогонов
с ними в этом аудите не проводилось.
