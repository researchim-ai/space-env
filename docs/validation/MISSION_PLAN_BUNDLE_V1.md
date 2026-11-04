# D2g: исполнимые планы и сохраняемые native RL/SFT bundles

## Контракт плана

`planning.mission_plan.make_mission_plan` создаёт sealed JSON proposal, связанный
с exact scenario, нормализованным TaskSpec, episode seed и source/data/Python/
dependency identity. PlanStep содержит время, длительность решения и известные
action IDs. Полный action catalogue/параметры остаются в TaskSpec.

`t[i+1] = min(H, t[i] + advance[i])`.

Требуются непрерывная сетка от0 доH, не более10000 steps, отсутствие неизвестных
или повторных IDs и соблюдение declared joint/duration budgets. Гибкое управление
выражается несколькими catalogue actions; план не внедряет произвольный Python,
не пишет attitude/inventory/request status и не меняет admission rules.

`recorded_mission_plan(planning_checkpoint)` переводит записанные решения
существующего causal rolling baseline в candidate fixed schedule. Контрольные
transition/native hashes становятся ожиданиями, а не командами принудительно
получить нужный результат. Компиляция проверяет seals/structure, **не** является
cold replay source policy и не удостоверяет отсутствие oracle в чужом checkpoint.
Origin всегда `recorded-candidate`; source planning SHA — checksum, не подпись.
Отдельно поддерживаются планы `declared`, подготовленные до исполнения.

Это новый артефакт исполнения и проверки. Forecast optimizer, доказательство
optimality и generalization исходной policy этим не реализованы.

## Native исполнение

`PlanExecution` использует существующий `TaskSession.step` с revision/idempotency,
native masks, physical time и транзакционной обработкой joint actions. Baseline
policy в fixed executor не вызывается. Reject остаётся фактическим шагом с
coast/reward; divergence по ожидаемому transition сохраняется и останавливает
план. Уже выполненное действие не стирается задним числом.

Checkpoint связывает plan/build/episode. Resume восстанавливает TaskSession
через audited prefix replay, а не pickle или O(1) memory restore. Wrong-plan и
build mismatch отклоняются. PID metadata отличает same-process от reported
fresh-process recovery; само по себе не является OS/remote attestation.
Пример сверяет producer/consumer PID с parent-observed subprocess PID и ждёт
выхода producer до запуска consumer.

Plan привязан к initial scenario или собственному checkpoint; произвольная
пересадка на действующую operator session/изменённый мир не поддерживается.

## Сертификат выполнимости

Перед публикацией `VerifiedEpisode` **заново** исполняет native episode.
Quality gate проверяет bits/energy/thermal, ACK→reward, causal public views,
physical intervals/discount и accepted masks. Сертификат связывает plan,
execution, episode, final native report и объявленные цели:

- завершение всей сетки и горизонта;
- отсутствие execution error и совпадение expected transition/native hashes;
- min/max fulfilled и rejection limit;
- при необходимости coupled acquisition→CPU→delivery/epoch gates;
- optional запрет failed operations.

Неуспешные/незавершённые операции сохраняются по слоям. Достижение mission
minimum не означает успех каждого slew, выполнение всех requests или industrial
SLA. `all_reported_operations_successful` отражает это отдельно. По умолчанию
failed intermediate control windows не скрываются и не запрещаются, если
объявленный goal их допускает. Denominator исходной миссии сохраняется.

Operation diagnostics относятся к managed maps attitude/lifecycle/engine/
unloading. Radio delivery проверяется через native ACK/reward и mission quota,
а не обещанием отсутствия каждого отдельного wire/receiver failure. Для такого
SLA потребуется отдельный command/ACK-level gate.

Это executable native lower-bound witness на данной модели и fixed inputs,
не независимый физический oracle. `expert_teacher_qualified=false` всегда.

## Сохраняемый dataset

`publish_plan_bundle` атомарно публикует новую директорию:

- `plan.json`, `execution.json`, `certificate.json`;
- central-role `rl.jsonl`, включая failure/reject trajectory;
- central-role `sft.jsonl` только при пройденных declared mission gates;
- `bundle.json`: exact file/row hashes, counts, lineage/split и cross-bindings.

Каждая строка содержит plan/certificate/episode hashes. Промпт SFT использует
только существующие public observations и selected action; blueprint, будущие
расписания, truth и seed не вкладываются в user messages. Hash metadata находится
вне messages. Native quality остаётся `verified_execution_not_expert_qualification`.
Даже successful zero-floor plan не превращается в expert teacher.

Failed declared goal сохраняет RL и failed certificate, не выдаёт SFT. Partial
prefix не считается full-horizon teacher. Новая директория строится через staging
и rename; IO/byte-budget failure убирает staging и сохраняет существующий target.
Budget включает все файлы и manifest; это local atomic visibility, не distributed
transaction или hard OS quota. JSON<=256MB, row<=32MiB, steps<=10000,
default bundle budget1GiB с пределом4GiB. Native checkpoint не удаляется после
export timing — его можно повторно проверить и получить другие existing exports.

Split остаётся existing deterministic lineage-family split. Это не frozen
train/dev/test benchmark и не доказательство отсутствия semantic duplicates.
Local-role RL можно получить существующим VerifiedEpisode API из сохранённого
episode; новая bundle schema пока central-role only.

## Аудит

`verify_plan_bundle(..., native=False)` проверяет точный inventory, symlinks,
размеры, duplicate/nonfinite JSON, seals/cross-bindings, порядок/полноту строк,
actions→plan, feedback→certificate и запрет privileged public fields.
Пересчитываются certificate goals из retained reports. Это **integrity audit**:
переписавший все hashes может подделать физическое утверждение.

`native=True` требует compatible build, повторяет native replay, quality gates
и сравнивает каждую RL/SFT строку с заново полученными observations/reward/action.
Например, re-sealed подмена reward может пройти integrity mode, но отклоняется
native mode. Возвращаемый `native_execution_replayed` явно разделяет режимы.
Строки и certificate сравниваются по каноническому JSON, а не Python numeric
equality (`1 == 1.0`, `0 == False`). Каждый seal, включая собственный seal
certificate, проверяется отдельно. Execution checkpoints имеют compact `/2`
episode; полный source planning record допускается при компиляции candidate.
Оба режима не удостоверяют автора и не доказывают causality исходной policy.

## CLI

```bash
space-env mission-plan-from-record proposal.json plan.json --goals goals.json
space-env mission-plan-run plan.json prefix.json --decisions 45
space-env mission-plan-resume plan.json prefix.json execution.json
space-env mission-plan-export plan.json execution.json dataset
space-env verify-mission-plan-bundle dataset
space-env verify-mission-plan-bundle dataset --native
python examples/planning/mission_bundle.py /tmp/retained-plan --native-audit
python examples/planning/mission_bundle.py /tmp/retained-negative --negative --native-audit
python examples/planning/mission_bundle.py /tmp/retained-two-orbit --full-orbit
```

CLI exit0 — корректное действие/пройденный goal,1 — retained failed mission
goal/execution,2 — invalid input/corrupt bundle/IO contract error. Prefix export
с failed goal остаётся полноценным negative RL bundle. Existing outputs не
перезаписываются. Полный пример сохраняет исходный proposal и результаты.

## Следующий раздел

Forecast-based planner с independently frozen holdouts и baseline comparison;
отдельные teacher gates на completion/rejections/robustness; multi-role retained
exports; long rigid refinement; measured calibration C2 и Q7 durable/auth/OS
isolation. Текущий блок не закрывает эти задачи названием «сертификат».

## Приёмка — 2026-11-05

[B11_MISSION_PLAN](../acceptance/B11_MISSION_PLAN.json) связывает17 артефактов.
Wheel SHA `2f7811dbad94ea670a7420e0cc71342f76febe2c5d8ae2a3d5eb6e877461d46d`:
263 Python modules/296 source+data files и README byte-matched. Installed runs
выполнены Python-I вне checkout, venv system-site-packages; dependency-clean
локальная установка и remote CI не заявляются.

Три frozen workflows:

- positive180s:2/24 fulfilled,90 RL+90 SFT;
- заранее infeasible outage goal180s:0/24,90 RL, failed certificate, SFT отсутствует;
- full12600s:8/32 fulfilled в обоих epochs,1260 RL+1260 SFT.

Все3 fixed executions совпали с final native state исходного baseline.
Все3 producer/consumer пары прошли parent-observed fresh-process continuation:
prefix90s,90s и6300s. Для каждого bundle выполнен обязательный независимый
native cold replay при публикации. Short positive/negative имеют также отдельные
native audit и повторные native audit после byte-exact archive reconstruction.
Full bundle дополнительно прошёл integrity audit; отдельный full native audit
поверх обязательного native export replay не запускался.

Суммарно сохранены1440 RL и1350 SFT строк. У full retained47 noncompleted
managed operations; min-service goal пройден, но all-operations success и expert
teacher qualification не заявляются. Complete32/32 или industrial SLA этим
результатом не доказаны. Негативный goal остаётся явной неудачей, даже когда
сам expected-negative workflow успешно проверен.

1761 reference,129 targeted,110 fast passed;33 новых plan tests. Наборы
пересекаются. Выбранные lint checks прошли. Weekly/manual installed-wheel CI
добавлен, не запускался remote. Pre-hardening short runs и намеренно прерванные
full/reference runs сохранены отдельно; не входят в final denominator.

[Точный архив](../acceptance/B11_MISSION_PLAN_BUNDLES.tar.gz) содержит68 файлов,
22,068,429 bytes compressed; [индекс](../acceptance/B11_MISSION_PLAN_ARCHIVE.json)
сохраняет SHA/размер каждого entry. Все entries проверены байт-в-байт, root и
linked hashes проверены. В `.gitignore` исключены только эти named acceptance
archives; случайные build/run tar files остаются ignored.

```bash
mkdir -p /tmp/retained-plan-acceptance
tar -xzf docs/acceptance/B11_MISSION_PLAN_BUNDLES.tar.gz -C /tmp/retained-plan-acceptance
space-env verify-mission-plan-bundle /tmp/retained-plan-acceptance/full/dataset
# Для native audit нужен matching source/data/Python/dependency build:
space-env verify-mission-plan-bundle /tmp/retained-plan-acceptance/positive/dataset --native
```
