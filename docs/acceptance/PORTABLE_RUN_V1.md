# Приёмка переносимых экспериментов v1

Дата: 30 октября 2026. Рабочее дерево 0.134.0, поверх Execution v1 и Resource timeline v1.
Руководство и контракт: [PORTABLE_EXPERIMENTS.md](../PORTABLE_EXPERIMENTS.md).

## Реализованный пакет

1. Portable world schema2: эффективная конфигурация борта/сценария/constraints/goals,
   эпоха/кадр, состояние и полные журналы. Загрузка без YAML и обращения к каталогу.
   Schema1 остаётся читаемой. Запись JSON атомарна, с проверкой checksum.
2. RunRecorder/replay_run: исходное/конечное состояние, наблюдения, dt, поля команд,
   outcomes/продукты/квитанции; rollback не оставляет частично принятую команду.
   Проверяется каждый переход и финальное состояние/журналы. Версии dependencies,
   отпечатки исходников и пакетных данных фиксируют среду исполнения.
3. Проверенный экспорт behavior transitions JSONL с lineage-based split,
   provenance и checksum. Повреждённый или не сошедшийся run не заменяет старый dataset.
   Один прочитанный объект используется и для проверки, и для экспорта.
4. CLI `record-commands`, `replay-run`, `export-transitions`, минимальный пример
   двух аппаратов в `examples/portable_run`, руководство и installed-wheel smoke.

## Фактически выполненные проверки

- Новый набор portable: 14 passed,3.98с. Включены custom configuration, журналы и
  времена событий, промежуточный checkpoint, положительная сдача данных, два аппарата
  и общий терминал, tamper/runtime mismatch, неучтённое вмешательство, rollback,
  отказ atomic replacement, legacy schema1, fresh-process replay и CLI после удаления
  исходных YAML/plan, lineage split, checksum экспортированного JSONL.
- Расширенная регрессия portable/resource/integrity/M0/mission/ground:69 passed,9.32с.
- Старые mission roundtrip/session dataset/pipeline:20 passed,56.70с, включая
  суточный energy-limited baseline. Этот прогон предшествовал изменению только
  recorder rollback на внутренний checkpoint; финальный fast и новые тесты прошли после него.
- Обновлённый CI fast:72 passed,10.65с. Финальные дополнительные assertions
  fresh-process CLI/checksum проверены отдельным новым набором после fast.
- Сборка wheel, переустановка в clean venv, проверка из `/tmp` с Python `-I`:
  Dict rollout, четыре smoke cases, Studio handlers, schema2 snapshot, portable
  replay, один transition и CLI validate прошли.
- Все146 Python-модулей финального wheel побайтово совпадают с рабочими исходниками.
- Ruff для новых/изменённых файлов этого пакета и `git diff --check` прошли.
- В checkout собрано408 тестов. Полный pytest и сценарий16×72ч для этого дополнения
  не запускались; результаты предыдущих срезов не выдаются за новую полную приёмку.

Числа наборов пересекаются, складывать их нельзя. Машинные данные и hash wheel:
[PORTABLE_RUN_V1_VERIFICATION.json](PORTABLE_RUN_V1_VERIFICATION.json).

## Что ещё не закрыто

Run записывает direct MissionWorld commands; LLM tools, relay, session revision,
пользовательские callbacks и полноценный task/prompt context требуют общего protocol.
Observation — объявленная идеальная локальная телеметрия, не SensorStream с задержкой.
Успешный replay не означает экспертную стратегию, наличие reward или готовый LLM SFT.

In-memory steps/кумулятивные outcomes и полные журналы пока не масштабируемый
sharded store. Пакетные данные fingerprinted, но не вложены в каждый run; нужна
совместимая установка пакета. Нет подписи/аутентификации, межверсионной миграции run
или независимой физической квалификации. Пакет закрывает часть P3, не все P0/P1/P3.
Commit, push, публикация и развёртывание сервиса не выполнялись.
