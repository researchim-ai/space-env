# Следующие пакеты: проверяемый симулятор и продукт для обучения агентов

Актуально на 30 октября 2026. Основной план: [INDUSTRIAL_ROADMAP_V0134.md](INDUSTRIAL_ROADMAP_V0134.md).
Это последовательность реализации, а не заявление о промышленной готовности.

## Выполнено в рабочем дереве

Reference-планирование группировки: greedy baseline с ограничениями контактов,
очередей, терминалов и текущего SoC, исполнением и повторной проверкой плана.
Следующие gates: выбор продукта, rolling replanning, независимые optimum/regret
проверки и нагрузочные измерения — [Scheduling v1](validation/REFERENCE_SCHEDULING_V1.md).

Контракты команд/reset/restore/данных: [Execution v1](acceptance/EXECUTION_V1.md).
Наземные окна/отключения внутри шага: [Resource timeline v1](acceptance/RESOURCE_TIMELINE_V1.md).

## 1. Закончить причинное исполнение

Расширять диспетчер границ MissionWorld через публичные поставщики событий
подсистем. Включить processing-ready, TTL/ACK, fault/recovery. Для release/deadline
отделить момент изменения доступности заявки от момента принятия решения политикой.
Не вызывать policy повторно скрыто внутри шага. Определить порядок одновременных
событий, отсутствие бесконечного цикла на нулевом интервале и поведение отмены.

Приёмка: coarse/fine сравнение дискретных исходов и баланса данных; ошибка на
любом подинтервале откатывает шаг; replay из JSON повторяет события/квитанции.
Проверить накладные расходы на реальной миссии, затем полный pytest на финальном коде.

## 2. Самодостаточный артефакт эксперимента

Первый срез реализован: [Portable experiments](PORTABLE_EXPERIMENTS.md).
World schema2, run/1, direct-command replay и verified JSONL доступны через SDK/CLI.
Этот срез не закрывает весь этап: остаются tool/relay protocol, полный контекст SFT,
sharding/bounded storage и квалификация совместимости.

Сохранить schema version, полную конфигурацию миссии без зависимости от исходного
YAML, seeds, модель/данные и их hashes, observation/action/tool events, products,
receipts и итоговые KPI. Данные хранятся с checksums и атомарным завершением shard.
Не выдавать внутреннюю truth за доступную агенту телеметрию.

Приёмка: перенести run в другой каталог, запретить сеть и доступ к старому YAML;
продолжение и replay дают те же дискретные исходы и численные поля в объявленном допуске.
Из этого артефакта строятся RL/SFT records с контекстом и provenance, без leakage
между train/holdout по lineage миссии.

## 3. Квалифицированный физический профиль

Реализовано отдельное ядро [gcrs-reference/1](validation/GCRS_REFERENCE_V1.md):
время/EOP, GCRS/ITRS state/covariance, DOP853, force switches и измеренные проверки.
Добавлены [reference access и external formula checks](validation/REFERENCE_ACCESS_V1.md).
Сверка J2/drag/third-body имеет общие данные/кадры и не заменяет независимую
квалификацию всех сил. Переносимая эфемерида даёт геометрические окна станций.
Добавлен [reference-fixed-downlink/1](validation/REFERENCE_OPERATIONS_V1.md) для
исполнения готовых продуктов/Gymnasium на фиксированных reference-траекториях.
Его явные упрощения не являются полной миграцией общей миссии.
MissionWorld остаётся legacy-training/1. Следующий gate — независимые perturbed
reference fixtures и перенос всех связанных моделей миссии, не просто смена метки кадра.

Выполнить [FRAME_MIGRATION.md](validation/FRAME_MIGRATION.md): отделить legacy
от reference, закрепить время/EOP/backend, r/v/covariance и внешние reference fixtures.
Задать допуски до измерения, проверить несколько эпох и дуги 24/72 ч.
Сохранить старые задачи в versioned legacy suite, новый профиль получает свои цели.

Приёмка: измеренный error budget и applicability report, включая неподдерживаемые
режимы. Одного короткого двухтелого теста недостаточно для perturbed миссии.

## 4. Полноценная среда группировки и baseline

Первый общий tool protocol, записываемый heuristic baseline и блокирующий relay
реализованы: [AGENT_PROTOCOL.md](AGENT_PROTOCOL.md). Они не заменяют constraint solver,
официальную MARL qualification или конкурентный событийный relay scheduler.

Единый entities observation (аппараты, станции, заявки, операции), masks, явный
режим полной/частичной информации; официальный optional PettingZoo adapter.
Rolling-horizon planner учитывает энергию, память, pointing/setup, окна и сроки.
Начать с малых exact задач, затем неоднородной EO/ground/relay миссии.

Приёмка: официальные API tests, invalid-action атомарность, воспроизводимые seeds,
малый exact optimum и честный holdout отчёт. Невыполнимые заявки остаются в знаменателе.

## 5. Поставляемый пилотный продукт

Один CLI/SDK workflow: scenario → agent → run → dataset → replay → report.
Studio показывает фактическое и запланированное исполнение, причины отказов,
ресурсные конфликты, метрики и версии модели из того же run artifact.
Ввести bounded history, pagination, recovery и измерить реальные нагрузки
16/100, затем64/1000, затем256/10000 с работающими заявками и станциями.

Приёмка: воспроизводимый load/install report и проверенный браузерный workflow.
Для совместного hosted-сервиса отдельно реализовать ownership/auth/isolation,
quotas и backup. Пилот с оператором требует его данных, baseline и согласованных
критериев полезности. Превосходство над другими системами подтверждается только
сравнимым экспериментом; отсутствие такого эксперимента не скрывается.
