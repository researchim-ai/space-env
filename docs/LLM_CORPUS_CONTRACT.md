# LLM Corpus v1 — проверяемая сборка данных миссий

Срез разработки: 8 ноября 2026. Локальная библиотека, без CI, публикации,
внешнего inference и автоматического обучения модели.

## Задача

Отдельный успешный LLM-эпизод ещё не является корректным корпусом. При объединении
нужно сохранить причинность, не обучаться на evaluation-геометриях, не считать
30 решений одной миссии 30 независимыми миссиями и не выдать повтор той же
траектории под новым именем за разнообразие.

Новый API: `space_env.llm.corpus.build_corpus`, `verify_corpus`, `register_corpus`.
Проверка provider-wire и генерация соответствующих строк перенесены из примера
в `space_env.llm.local_data`; прежние импорты из примера сохранены.

## Входы и результат

`build_corpus(inputs, output)` принимает пути к native LLM episode JSON либо
директориям `space-env.local-llm-data/1` из `examples.llm.generate_local`.
Смешивать эти два вида входов можно, но повтор одной semantic trajectory
отклоняется. Произвольные директории scripted demo не являются local-provider
bundle: для них передавайте отдельный `episode.json`.

Структура результата:

```text
corpus.json
  entries / summary / files / sha256
audit/0000/episode.json
  [manifest.json, inference.json, исходные JSONL provider-bundle]
data/rl.jsonl
data/sft.jsonl
data/wire-sft.jsonl
```

`audit/` — privileged trainer artifact: initial checkpoint и полный replay,
а не наблюдение, передаваемое политике. Корневые JSONL в `data/` содержат только
train-эпизоды. Evaluation сохраняется в audit для контроля split, но не попадает
в обучающие строки. Ошибочный/незавершённый train-эпизод сохраняется для RL;
положительный SFT требует успешной доставки без неуспешных вызовов. Wire-SFT
принимается только из provider-bound bundle с повторной проверкой prompt,
response schema, semantic target и native event. Native episode сам по себе
не позволяет восстановить фактический provider prompt — wire-строки не выдумываются.

## Контроль утечек и повторов

Группа геометрии вычисляется через **тот же `factory.semantic.split_group`**, что
и в factory registry. Пользовательские `configuration.group`, prose, пути и
названия файлов не определяют split identity. Это консервативный typed identity
fingerprint, а не математическое доказательство эквивалентности всех геометрий.

Внутри корпуса одна группа не может оказаться в разных splits. Идентичная
LLM-траектория не может повториться под другим group/текстом задачи. Семантика
траектории включает fingerprint native mission/task, последовательность
инструментов и аргументов без служебных revision/idempotency, применение перехода
и успех вызова. Это **LLM-specific trajectory identity**, не полная эквивалентность
другому формату factory trajectory. Общая геометрическая защита работает между
двумя производителями; полная cross-format дедупликация траекторий не заявляется.

`register_corpus` сначала независимо проверяет артефакт, затем в одной SQLite
`BEGIN IMMEDIATE` транзакции фиксирует все claims в существующем
`space-env.corpus-registry/1`. Конфликт откатывает всю транзакцию. Повтор регистрации
того же corpus content идемпотентен. Другая упаковка уже зарегистрированной
траектории отклоняется. Файл registry должен лежать вне immutable bundle.

Используйте **один и тот же файл registry** для factory и LLM. Если разные команды
укажут разные базы или пропустят регистрацию, глобальной защиты нет. Установленные
splits не объединяются автоматически: development/test и holdout/ood остаются
разными метками. Предварительно регистрируйте отложенные геометрии; обычный native
LLMEpisode с split=test/development допустим и без исполненных действий для такого
резервирования, но не является измеренной успешной evaluation-миссией.

## Атомарность, аудит и ресурсы

Сборка выполняется в соседней временной директории, очищаемой при ошибке.
Исходные provider-bundle файлы сначала копируются в ограниченный snapshot, затем
проверяются. Готовая директория появляется после проверки через rename в пределах
одной файловой системы. Уже существующий output отклоняется. Не запускайте внешние
процессы, изменяющие output или registry вручную во время операции.

Публикация директории и SQLite регистрация — **две отдельные операции**, без
обещания общей distributed transaction. После падения между ними готовый корпус
можно проверить и зарегистрировать повторно. Статус build: `not_registered`.

Лимиты: до 256 эпизодов, до 64 MB на файл и до 512 MB корпуса. Это ограниченный
локальный сборщик с удержанием проверяемых строк в памяти, не распределённый
streaming data lake. Symlink artifacts, лишние файлы, изменённые sealed metadata,
неверные JSONL targets и несовпадение native replay отклоняются. Source/dependency
binding сохраняется: исторический эпизод нужно проверять на зафиксированной
ревизии исходников, а не обходить проверку после обновления библиотеки.

Отчёт считает отдельно эпизоды, независимые геометрии, train/evaluation,
успешные train-миссии, строки каждого формата, происхождение политик и долю
ожидающих решений. Он не объявляет поведение экспертным и не доказывает эффект SFT.

## Использование

Из корня checkout, после установки reference зависимостей:

```bash
# Ранее полученные native episodes или реальные local-provider bundles:
python -m space_env.llm.corpus build /tmp/my-corpus \
  /tmp/train-run/episode.json /tmp/evaluation-run/episode.json
python -m space_env.llm.corpus verify /tmp/my-corpus
python -m space_env.llm.corpus register /tmp/my-corpus /tmp/shared-corpus.sqlite
```

Для сохранения wire-SFT передавайте `/tmp/train-run`, а не только его episode.json,
если это provider-bound `local-llm-data/1` bundle. Evaluation-геометрия должна
отличаться от train; иначе сборка намеренно завершится ошибкой.

Для factory используйте `space-env factory-register` с тем же registry.
Эти команды работают локально; никакие данные не отправляются в сеть.

Сквозной пример без inference:

```bash
python -m examples.llm.corpus_demo --output /tmp/new-corpus-demo
```

Он резервирует отдельную test-геометрию без исполнения, затем запускает
scripted privileged teacher для train-доставки, собирает native SFT/RL,
регистрирует и независимо проверяет результат. Это не LLM-score и не обучение.
