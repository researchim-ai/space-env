# Установка, запуск и локальные тесты

Из корня проекта:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[reference,coverage,marl,dev]"
```

Запуск Studio и пульта оператора:

```bash
python -m space_env.api.server
```

Откройте [рабочую страницу миссии](http://127.0.0.1:8766/): задайте орбиты,
станции и заявки, нажмите «Составить план и открыть миссию», затем
«Прогон планировщика». План, фактические результаты и карты находятся здесь же.
`/operator` и `/workspace` открывают ту же страницу;
[Studio](http://127.0.0.1:8766/studio) сохраняет каталог демонстраций.
[Конфигурация, Python SDK и ограничения](MISSION_WORKSPACE.md).

Тот же сценарий без браузера, с результатом и данными в новом каталоге:

```bash
python -m examples.workspace.run --output /tmp/my-workspace-mission
```

Локальные тесты:

```bash
python -m pytest -q
```

Для отдельного изменения можно запустить только соответствующий файл:

```bash
python -m pytest tests/test_release_data.py -q
```

Пример миссии с новыми заявками во время выполнения:

```bash
python -m examples.planning.demand_mission /tmp/my-space-mission --mode two_updates
```

Каталог результата должен быть новым. Другие примеры находятся в `examples/`.
CI и удалённые проверки для работы с проектом не требуются. Benchmark и другие
диагностические инструменты необязательны и запускаются вручную по необходимости.
