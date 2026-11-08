# Pointing-bounded planning и полнота каталога

> Уточнение 8 ноября: добавлен запас на native tracking во время съёмки.
> Исторические числа 19→10 и 1→2 ниже относятся к прежней ревизии. Текущий
> global-cone режим в equatorial случае резервирует 11 ячеек и доставляет 1/3.
> Последовательный режим с отдельным tracking guard достигает 3/3:
> [актуальный контракт](SEQUENCE_LANES.md). Старый receipt — исторический.


8 ноября 2026. Opt-in режим `MissionWorkspace.network(...,
planning_method="pointing_bounded")`. Старый serial и его frozen benchmark не
подменяются новым планировщиком. Сравнение проводится на тех же конфигурациях.

## Измеренная причина неполного выполнения

В equatorial-relay-chain у трёх заявок единственный аппарат с sampled capture —
fleet-00. Прежний forecast закладывает для каждого нового наведения 180° при
скорости 2°/с и settling 1 с: 19 ячеек по 5 с, то есть 95 секунд. С учётом съёмки
и CPU обязательные префиксы требуют 64 ячейки при горизонте 36 ячеек. Поэтому
в консервативной дискретной модели все заявки совместно не помещаются.

Это **не доказательство физической недостижимости** миссии: исходный quaternion
и цели известны, реальный разворот меньше. У агента также нет цепочек для
невыбранных заявок: успех всех трёх недостижим внутри данного каталога, даже
если другая постановка могла бы выполнить физическую миссию.

## Безопасное сужение углового резерва

Операторский compiler перечисляет initial quaternion и все геометрические
quaternion-цели допустимых capture-grid midpoint для каждого аппарата.
Это конечный каталог: цель команды задаётся точно в том же GCRS и в тот же
момент, который будет использоваться при materialize. Future native execution
и результаты политик для получения целей не вызываются.

Знаки unit quaternions согласуются относительно initial pose. Если все попарные
скалярные произведения строго положительны, их минимум `m` задаёт границу
`theta_max = 2 arccos(m)` с численным запасом. Попарный минимум считается
блоками, без хранения полной NxN матрицы.

Почему учтены и прерванные развороты: shortest-arc SLERP в остром конусе является
нормированной положительной комбинацией endpoints. Если `u=sum(a_i*q_i)` и
`v=sum(b_j*q_j)` с неотрицательными коэффициентами, то
`u·v >= m * sum(a_i)*sum(b_j)`, а нормы не превосходят соответствующие суммы.
Следовательно, для нормированных u/v скалярное произведение не меньше m.
Это сохраняется при повторных прерываниях и выборе новой объявленной цели.

Если острый конус подтвердить нельзя — возврат к 180°. Это не эвристическая
подстановка «малого среднего угла». Невалидные/ненормированные quaternions и
переполнение бюджета целей отклоняются.

В equatorial примере граница около 92,73° даёт 10 ячеек, то есть 50 секунд.
Берётся максимум по допустимым owners; прежний расход энергии `owner_wh`
**не уменьшается**. Контакты, память, terminal acquisition/ACK, дедлайны,
длительность CPU и physical native admission сохраняются.

### Текущий tracking guard

Cone closure выше относится к чистым shortest-arc slew. Native imaging также
сопровождает цель. Поэтому текущий compiler добавляет к cone angle максимум
`omega_max * capture_cells * decision_sec` по допустимым предыдущим экспозициям,
с ограничением 180°. Граница условна на успешном выполнении предыдущих slew;
произвольное отклонение от плана или отказ не получает сертификат из старого
metadata. В equatorial случае 92,73° + 10° = 102,73°, то есть **11 ячеек**.
Исторический пример 10 ячеек ниже/выше не является текущим параметром.
Для более точного учёта именно предшествующей экспозиции см. sequence-lanes.

## Режим поиска и область применимости

Для <=6 заявок и <=48 ячеек используется существующий bounded exact-small
solver, если пользователь не задал request_order. Для более крупных задач или
явного порядка — serial на адаптированном forecast. Наличие exact-small не
означает, что enumeration/search завершились: сохранены budgets, upper bound,
`optimal` и `search_budget_exhausted`. Не доказанный optimum не объявляется.

Область применения: **fresh initial kinematic network workspace**, удержание
ориентации и кратчайшие rate-limited развороты к объявленным GCRS-целям.
Не распространяется на rigid-body/wheel dynamics, free-form attitude commands,
tracking, неизвестные faults или новые targets во время исполнения. Генератор
строит именно данный профиль; чужой или изменённый forecast не принимается без
повторной сверки с configuration и известной геометрией.

Dynamic fork использует прежний conservative serial_recovery. Initial-only
pointing proof и completeness не должны отображаться как действующие доказательства
после изменения сценария; они удаляются из current metadata восстановленной ветки.

## Диагностика до оценки агента

`planning.pointing_envelope.diagnose_network` сохраняет:

- какие заявки имеют цепочки в каталоге;
- какие actors имеют sampled capture;
- длительности slew и префикса;
- isolated single-request witness и признак исчерпания его search budget;
- превышение conservative memory-prefix bounds;
- необходимые ограничения суммарного EO/CPU prefix для единственного owner.

Провал singleton serial search — **unknown**, не сертификат infeasibility.
Успешный singleton при отсутствии совместной цепочки указывает на packing,
ограничения поиска или shared resources в данном forecast. Никакой результат
диагностики не устанавливает физическую невозможность миссии.

Эта информация находится в operator metadata и `state()["network"]`, а не
в observations/messages LLM. Privileged future geometry не выдаётся actor-policy.

## Использование и абляция

```python
from space_env.workspace import MissionWorkspace
from space_env.llm.mission_suite import compile_suite

config = compile_suite()["cases"][0]["configuration"]
mission = MissionWorkspace.network(config, planning_method="pointing_bounded")
try:
    print(mission.state()["network"]["completeness"])
    print(mission.run()["network"]["fulfilled"])
finally:
    mission.close()
```

```bash
# По умолчанию шесть train workload-семейств × три планировщика:
python -m examples.network.pointing_comparison --output /tmp/new-pointing-comparison
python -m examples.network.pointing_comparison --audit /tmp/new-pointing-comparison
```

В абляции serial / исходный exact_small / pointing_bounded используют одинаковые
physical configuration, но собственные выбранные назначения и каталоги. Исполняет
одинаковый privileged reference policy. Это сравнение pipeline планирования,
а не actor-only agent leaderboard. Аудит повторяет native episode **и отдельно
перестраивает планировщик/envelope**, проверяя metadata и диагностические claims.
Ошибка ячейки остаётся в матрице; пары с ошибками не получают выдуманную дельту.

Первый matched случай: serial 1/3, исходный exact_small 1/3, pointing_bounded 2/3
native доставок. Для нового плана bound поиска не закрыт, optimal=false.
После уменьшения префиксы всё ещё требуют 37 ячеек при 36 доступных: 3/3 не
обещаются. Частичный результат не проходит positive SFT gate.

Это инженерная проверка конкретного profile и регрессионных случаев, не hardware
калибровка, новое обучение LLM или доказательство превосходства над конкурентами.

## Историческая локальная приёмка до tracking guard

На финальных исходниках: **115 passed / 188.85 s** в наборах pointing envelope,
парной абляции, network dynamic/joint/exact и terminal sequence. В том числе
проверены quaternion interpolation/cancel bounds, fallback, неправильные цели,
подмена forecast/оптимальности, native 2/3 и удаление proof после dynamic fork.
Ruff и diff-check passed.

Парный native прогон: шесть frozen workload-семейств на **одной** геометрической
группе, три режима, **18/18 replayed** с независимым перестроением плана.
Три улучшения против обоих старых режимов (relay-chain, station-blackout,
link-blackout: 1→2 заявки), три неизменных результата, ноль ухудшений на этой
подматрице. Planned/native fulfilled совпали во всех 18 ячейках. Полных delivery
целей 3/3 не достигнуто; positive SFT из этих результатов не заявляется.

Это не приёмка всех 36 frozen конфигураций или других геометрий.
[Протокол с source hashes и точными парами](acceptance/POINTING_COMPLETENESS_V1.json).
