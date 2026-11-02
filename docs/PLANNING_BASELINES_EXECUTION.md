# B07: исполняемый пакет планировщиков после B06

## Результат реализации B07 — 2 ноября 2026

Пакет реализован в bounded current-window profile: пять baselines, immutable public
input, resource-compatible subset search, native TaskSession runner, policy/RNG
checkpoint и cold replay, явный operator override, portable benchmark/semantic
verification, lineage/OOD и teacher gates, tiny native masked-tree oracle, CLI и
CI partitions. [Контракт](PLANNING_CONTRACT.md),
[приемка и точные результаты](acceptance/B07_PLANNING.json).

57 новых cases; reference 971, expanded regression 1372 passed. 29 длительных
неизменённых legacy cases не перезапускались для B07; отдельные CI jobs настроены,
удалённый workflow не запускался. Установленный wheel проверен вне checkout на
четырёх связанных EO/CPU/ISL/ground миссиях; positive 2/2, TTL 1/2 без ложного success.

Это не propagated multi-epoch optimizer: horizon — масштаб urgency, search —
текущие совместимые subsets, replanning — на каждой decision boundary. Forecast
capability, expert-policy evidence и глобальная оптимальность не заявлены. Ниже
сохранён исходный исполняемый scope; дальнейшее расширение forecasts отделено
от B08 data factory/B09 scale/B12 independent evidence.


Основа: `space_env.tasks`, контракт `AGENT_RUNTIME_CONTRACT.md`. Пакет должен
сравнивать стратегии на одной физике, а не добавлять параллельный упрощенный
симулятор ресурсов. B06 доказывает корректность исполнения и экспорта; B07 должен
добавить обоснованные решения и измерить их качество.

## 1. Публичный вход политики и граница знания

Создать `space_env.planning` с immutable planning inputs, содержащими только
task observation, известный task catalogue и разрешённые nominal estimates.
Политика не получает `TaskSession`, приватный runtime, scenario, seed, checkpoint
или будущие realized weather/fault/loss. Forecast service должен иметь явный
отдельный capability, horizon, assumptions и revision. Недоступный receiver buffer
или неизвестное окно не объявлять свободным.

Результат политики: выбранные action IDs, duration, публичные причины/assumptions.
Revision/idempotency добавляет runner. План нельзя напрямую записывать в backend.
Один общий runner исполняет результаты через `TaskSession.step`.

Приёмка: подмена не наступившего fault/seed при одинаковом публичном входе не
меняет решение детерминированной политики. Проверка должна сравнивать полное
решение и trace причин, а не только первую action ID.

## 2. Набор воспроизводимых baselines

Реализовать `wait`, `random_masked`, `priority`, `earliest_deadline` и
`route_aware_rolling`. RNG random baseline принадлежит политике и сохраняется в
её собственном checkpoint, независимо от физического RNG. Tie-break явно
лексикографический по stable IDs; порядок словаря не определяет качество решения.

Каждая политика выбирает совместимый набор по actor/action slots и static claims.
Mask не является гарантией совместной допустимости. Если комбинация отвергнута,
runner сохраняет отказ и потраченное время, а не повторяет бесплатно другие
комбинации до успеха.

Приёмка: одинаковые seed/input дают одинаковые trace; wait не выдаётся за
оптимальное решение, random не читает скрытое состояние, duplicate choice не
попадает в generated plan. Опциональные солверы импортируются лениво и имеют
детерминированные лимиты/публичный fallback.

## 3. Rolling-horizon EO/CPU/routing

Поддержать цепочку release → acquisition → processing precedence → custody →
ground ACK → fulfillment. Учитывать текущую SOC/память, retained/pinned products,
конечные CPU/radio/ADCS lanes, deadline, duration и terminal booking revisions.
При отсутствии необходимого параметра результат — conditional/unknown, не
гарантированная feasible миссия.

Начать с ограниченного поиска по конечным task templates, bounded horizon и
ограничением числа раскрытых узлов. Не заявлять глобальную оптимальность для
эвристики. Для маленьких детерминированных fixtures дать независимый exhaustive
oracle с явным budget; oracle существует в тесте/evaluator и не доступен политике.

Приёмка: routing не считает hop доставкой; partial ground TX не считается ACK;
просроченный TTL не воскресает при reroute; shared terminal не используется
сверх capacity. Планы не требуют неизвестного будущего ресурса как факта.

## 4. Replanning и recovery

Перепланировать на следующей decision boundary после наблюдаемого rejection,
fault/safe, TTL, изменения booking revision и исчезновения/появления local product.
Сначала отменять реально конфликтующий control window, затем делать новый выбор
через общую транзакцию. Не обнулять residual attitude, тепловую историю, долг по
энергии или уже потраченное время. Recovery action требует соответствующей роли.

Приёмка: reroute/fault/TTL cases из `examples/agent_task/` остаются воспроизводимы;
невыполнимая заявка честно остаётся невыполненной; отказ одной роли не раскрывает
другой её внутреннее состояние через подробный feedback.

## 5. Benchmark и качество teacher

Фиксировать публичный manifest до запуска: policy/version, task schema, lineage,
seeds, limits, objective, candidate forecast assumptions. Делить по lineage,
дополнительно держать отдельные workload families/параметры для OOD. Не подбирать
held-out seeds после просмотра результатов. Synthetic demonstration маркировать
scripted; лучшее выполнение на одном seed не делает её экспертной.

Метрики: fulfilled/total, on-time ground Mbits, lateness, completion latency,
native battery consumption, discarded/wasted bits, safe entries, rejection count,
decision wall time/RSS, число раскрытых search nodes. Для времени и ресурсов
указывать единицы и что измерено: physical или wall clock. Отдельно показывать
невыполнимые случаи; не исключать их из знаменателя после запуска.

Для teacher qualification заранее объявить минимальные пороги на каждом семействе
и сравнение с priority/EDF на тех же episodes. Оптимальность — только там, где
малый oracle её доказывает. SFT acceptance не означает expert quality автоматически.

## 6. Интеграция, артефакты и завершение

Добавить CLI для baseline run/evaluate; API может вызывать тот же runner.
Артефакты: manifest, policy trace, итоговые metrics, replay-verified task episode,
результат oracle только для разрешённых small fixtures. Политика не видит эти
evaluator-only поля во время исполнения. RL/SFT экспорт остаётся B06 exporter.

Длительные legacy mission tests исполнять отдельными CI jobs: campaign,
обычный tasking, 16 аппаратов/72 часа и остальной набор. Сверять объединение
с collected manifest без пропусков/пересечений. Прерванный SIGTERM процесс не
считать успешной проверкой даже при отсутствии `F` в частичном pytest output.

Проверить parity SDK/CLI и checkpoint resume политики вместе с task session;
сохранение только физического RNG без RNG/search state политики недостаточно.
Wheel проверять вне checkout. Финальные source fingerprints, regressions и
результаты записать в `docs/acceptance/B07_PLANNING.json` только после реального
завершения команд. Не коммитить и не публиковать автоматически.

## Следующие независимые gates

B08: parameterized scenario factory, data quality, diversity/duplicate checks,
held-out evaluation и версии датасетов. B09: streaming state/history, ограничение
памяти и fleet throughput. B12: независимые численные/аппаратные/operator fixtures.
Ни B06, ни этот план не подтверждают flight qualification или готовность к
непосредственному управлению настоящими аппаратами.
