# Переносимые эксперименты и обучающие переходы

Реализовано в рабочем дереве 0.134.0. Эта возможность охватывает прямые команды
`MissionWorld`, не произвольный Python-код политики, LLM tools или relay-вызовы.

## Быстрый запуск

Из корня checkout, после установки пакета:

```bash
python -m space_env.cli record-commands examples/portable_run/mission.yaml examples/portable_run/plan.json /tmp/space-run.json --lineage portable-demo-1
python -m space_env.cli replay-run /tmp/space-run.json
python -m space_env.cli export-transitions /tmp/space-run.json /tmp/space-transitions.jsonl
```

Пример содержит два аппарата и ожидание30 секунд, без станций/заявок. Он проверяет
цепочку поставки данных, а не промышленную полезность или качество планировщика.
Положительная сдача продукта и конфликт общего терминала проверяются тестами.

`plan.json` содержит ровно ключ `steps`. Каждый шаг содержит `dt_sec` и `commands`;
команды — поля `MissionCommand`: `command_id`, `actor_id`, `kind`, опциональные
`request_id`, `destination_id`, `duration_sec`. Поддержаны действия текущей книги
команд; это не универсальный язык всех операций движка.

После записи run исходные YAML и plan больше не нужны. Потребуется совместимая
установка библиотеки и её пакетных данных. `replay-run` возвращает код1 при
несовпадении replay; повреждённый/неподдерживаемый формат вызывает явную ошибку.

## Python SDK

```python
from space_env.data_generators.portable_run import RunRecorder, replay_run, export_transitions
from space_env.simulation.commands import MissionCommand
from space_env.simulation.snapshot import save_world, load_world

# world — настроенный MissionWorld; можно начать с промежуточного состояния.
recorder = RunRecorder(world, lineage="customer-mission-family-1",
                       provenance={"teacher": "my-policy-v1", "expert": False})
recorder.step(10, [MissionCommand("pass-1", "sat-a", "comms", duration_sec=2)])
recorder.save("run.json")
assert replay_run("run.json")["ok"]
export_transitions("run.json", "transitions.jsonl")
save_world("checkpoint.json", world)
resumed = load_world("checkpoint.json")
```

Все действия записываемого мира проводить через recorder. Прямое внешнее изменение
состояния не записывается как действие; replay обязан выявить его последствия.
Recorder не владеет блокировкой мира: одновременно использовать его из нескольких
потоков или мутировать мир из другого клиента нельзя.

## Форматы и гарантии

**World schema2.** Содержит фактические SpacecraftConfig, ScenarioConfig,
Constraints, MissionGoal, seed, эпоху/кадр, snapshot мира и полные журналы двигателей.
Пасс-кэш, зарегистрированные SensorStream, RNG, targets, продукты/квитанции и
команды сохраняются через общий snapshot. Загрузка не использует каталог и не
пересчитывает окна станций. Настройки борта, изменённые после сборки сценария,
также сохраняются. Старый аргумент `scenario` в `save_world` принят для совместимости,
но больше не читается. Чтение schema1 сохранено и всё ещё требует старый YAML.
Это checkpoint мира, а не сериализация MissionSession/revision или пользовательских callbacks.

**Run `space-env.run/1`.** Содержит исходный portable world, lineage/provenance,
каждое наблюдение до команды, длительность/поля команд, следующее наблюдение,
состояния/продукты/квитанции/статусы операций и финальный portable world.
SHA-256 проверяет целостность содержимого; это не подпись и не доказательство
достоверности источника. Python/dependency versions, отпечаток Python-исходников
пакета и отпечатки пакетных CSV/JSON/YAML/TLE фиксируются в runtime manifest.
Изменение manifest даёт `runtime_mismatch`, без автоматической сертификации
совместимости разных версий. Зависимые пакетные данные не дублируются в run.

Проверка replay сравнивает каждый переход и полное конечное состояние/журналы.
Структуры и дискретные значения совпадают точно; конечные float сравниваются с
`rel_tol=1e-12`, `abs_tol=1e-8` в единицах соответствующего поля. Это допуск replay,
не физический error budget. Независимую проверку орбит он не заменяет.

**Transitions `space-env.transition/1`.** JSONL создаётся только после успешного
replay того же прочитанного объекта run, без повторного чтения изменяемого файла.
Запись содержит доступное наблюдение, команду и dt, следующее наблюдение, outcome,
provenance и run checksum. Observation contract `ideal-local-telemetry/1` —
пять полей на аппарат: время, SoC, память, контакт и станция. Это идеальная локальная
телеметрия, без шума/задержки; скрытые причины отказов в неё не добавляются.
Outcome — результат/цель, его нельзя подмешивать в входное наблюдение модели.

Split зависит только от переданного lineage, не варианта. Вызывающий отвечает за
объединение связанных сценариев под одним lineage; автоматического обнаружения
родственных миссий нет. Успешный replay не присваивает метку эксперта, reward или
оптимальность. Это данные для behavior cloning/анализа переходов; полный LLM SFT
требует контекста задачи, prompt/tool protocol и квалифицированного teacher.

World/run/JSONL записываются заменой временного файла в том же каталоге после
flush/fsync файла. При ошибке старый файл сохраняется. Каталог должен существовать;
облачные object stores, подписи и транзакция нескольких файлов не реализованы.
Функция экспорта возвращает checksum всего JSONL и число строк.

## Границы масштаба и следующие этапы

Журналы, steps и кумулятивные outcomes находятся в памяти; это пока не bounded
streaming/sharded storage для больших миссий. Нет обещания переносимости произвольных
подклассов движка, внешних сервисов, kernels/plugins или callbacks. Runtime manifest
не является lock-файлом всей ОС/нативных библиотек; сравнение replay остаётся обязательным.

Далее: единый протокол записи policy/tool/relay событий; контекст и teacher для SFT;
атомарные shards и bounded history; официальные MARL API tests; отдельный reference
physics profile. План: [PRODUCT_EXECUTION_NEXT.md](PRODUCT_EXECUTION_NEXT.md).
