# Следующие крупные пакеты реализации


## B04-D2a — проверенное ядро вращения, 1 ноября 2026

Добавлено standalone rigid attitude ядро: физически допустимый тензор инерции,
Euler/quaternion integration, три идеальных ортогональных маховика, пределы torque
и momentum, DC/heat/mechanical-work ledger, внешний момент и `r cross F`.
21 новых тестов; JSON continuation и пример насыщения/разгрузки.
[Контракт](validation/RIGID_ATTITUDE_CONTRACT.md),
[приемка](acceptance/B04_RIGID_ATTITUDE_KERNEL.json),
[пример](../examples/rigid_attitude/demo.py).

**Это пока отдельный kernel:** MissionRuntime сохраняет rate-limited slew и
held-body burn. B04-D2 и B04 остаются in_progress. Следующий исполняемый блок —
B04-D2b: opt-in mission controller/state, реальная pointing/rate settling,
энергия/тепло и faults/cancel без сброса spin, off-center rotating thrust,
общие ownership/rollback/checkpoints и installed mission acceptance.
Подробные критерии интеграции — в контракте; затем calibration и B04-E/B05/B09.

Актуально на 1 ноября 2026.

**Единая очередь:** [INDUSTRIAL_AI_SIMULATOR_PLAN.md](INDUSTRIAL_AI_SIMULATOR_PLAN.md).
В ней находятся B01–B12, задачи, зависимости, критерии приемки и журнал закрытия.
Этот файл — короткая навигация, а не конкурирующий roadmap.

## B04-D1 — связанная тяга в системе корпуса, 1 ноября 2026

Добавлены chemical/electric engine schemas и `body_burn`: конечный gimbal,
settling/ignition, фактические quaternion → GCRS force, масса/топливо,
электричество/тепло, faults/cancel и совместная работа с CPU/omni-radio.
Прогноз учитывает только исполненную тягу; дальше — coast. Legacy GCRS команды
в coupled profile не открыты. Нет rigid-body моментов/маховиков и flight calibration.
[Контракт](validation/BODY_THRUST_CONTRACT.md),
[приемка](acceptance/B04_BODY_THRUST.json),
[пример](../examples/actuation/demo.py).

Следующий блок **B04-D2**: инерция, угловая скорость и Euler dynamics;
ограниченные torque actuators/маховики, насыщение и разгрузка momentum;
off-center thrust torque, связь с pointing/энергией/теплом; аналитические
conservation tests, fault/cancel/replay и installed mission acceptance.
B04-D и B04 остаются in_progress; затем B04-E thermal, B05 RF и B09 scale.


## Последовательность

1. B01: общий runtime и сценарий.
2. B02: управляемая reference-физика и независимая квалификация.
3. B03: request → acquisition → processing → delivery.
4. B04: связанный аппарат, ADCS/payload/энергия/здоровье.
5. B05: RF, наземные ресурсы и конкурентная ретрансляция.
6. B06: одна задача для RL/MARL/LLM/оператора.
7. B07: планирование, objectives и честный benchmark.
8. B08: воспроизводимая фабрика RL/SFT/preference данных.
9. B09: исполняемые нагрузки 16/100 → 64/1000 → отдельно 256/10000.
10. B10: Studio/оператор на общем runtime.
11. B11: release engineering и эксплуатация.
12. B12: данные промышленного пилота, калибровка и приемка.

## Реализованная база и доказательства

[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) фиксирует v1–v10 и проверки.
[Исходный аудит/roadmap](INDUSTRIAL_ROADMAP_V0134.md) сохраняет историю разрывов.
[Контракты и ограничения моделей](validation/) и [acceptance artifacts](acceptance/)
нужны для приемки; они не заменяются количеством функций или тестов.

v10: reference 249 passed, fast 110 passed, 679 collected, installed-wheel smoke.
Предыдущий полный v9: 665 passed, один sandbox socket failure, тот же HTTP-тест
отдельно passed с loopback-доступом. Это не непрерывный green-run на v10.

B01: runtime 38 passed, reference 287 passed, fast 110 passed, 717 collected;
installed wheel и console resume/replay подтверждены. Полный B01 regression не запускался.
B02: 18 новых тестов, reference 305 passed, fast 110 passed, 735 collected;
wheel 178/178 модулей и установленная dynamic CLI подтверждены. Полный regression
не перезапускался. B02 needs_external_evidence (исторический срез на завершении B02).
B02-Q1: 24 новых теста, reference 329 passed, fast 110 passed, 759 collected;
wheel 180/180 Python modules, 3 orbit fixtures, covariance CLI workflow проверены.
B02-Q2: 28 новых tests, reference 357 passed, fast 110 passed, 787 collected;
wheel 183/183 modules и installed SDK/CLI с binding/stale rejection проверены.
B03: 31 новых tests, reference 388 passed, fast 110 passed, 818 collected;
wheel 184/184 modules, installed EO lifecycle SDK/CLI и полная receipt проверены.
B04-A: 29 новых tests, reference 417 passed, fast 110 passed, legacy attitude
19 passed; 847 collected. Wheel 185/185 modules и 3 fixtures, installed SDK/CLI.
B04-B1: 32 новых cases, reference 449 passed, fast 110 passed, 879 collected;
wheel 186/186 Python modules и 3 frozen fixtures, isolated SDK/CLI acceptance.
B04-B2: 39 новых cases, reference 488 passed, fast 110 passed, 918 collected;
wheel 187/187 modules и 3 frozen fixtures, isolated optical/SAR SDK/CLI verified.
B04-B3: 37 новых cases, reference 525 passed, fast 110 passed, 955 collected;
wheel 188/188 modules + 3 fixtures; isolated SDK/CLI обоих режимов — в артефакте приемки.
B04-C: 34 новых cases, reference 559 passed, fast 110 passed, 989 collected;
wheel 189/189 modules + 3 fixtures; isolated resource-lanes SDK/CLI обоих режимов verified.
Full regression не запускался. Следующее «продолжаем» выполняет B04-D.
Commit/push/deployment — только по отдельной команде.

B04-D1: 39 новых cases; итоговые проверки — в acceptance artifact. Следующий шаг B04-D2.

B04-D1 validation: 39 новых cases, reference 598 passed (13 warnings), fast 110 passed (2 warnings), 1028 collected. Counts overlap; full regression не перезапускался. Wheel: 190/190 Python modules и 3 frozen fixtures. Chemical/electric installed CLI validate/run/resume/replay verified; обе миссии 2/2 delivered requests.

B04-D1 isolated extended SDK verification passed: chemical/electric, mid-gimbal и mid-firing restore, balances и console replay.

B04-D2a validation: 21 новых cases; reference 619 passed (13 warnings), fast 110 passed (2 warnings), 1049 collected. Counts overlap; full regression не запускался. Wheel 191/191 Python modules byte-matched, 3 frozen fixtures; isolated standalone demo passed.

Расширенный isolated installed SDK/CLI smoke также passed, включая прежние chemical/electric missions и mid-gimbal/mid-firing replay. Это проверка совместимости, не интеграция rigid kernel с MissionRuntime.
