# D2h-B2a: committed incident и восстановление оставшихся стадий

Оператор может прервать активные команды и объявить отключение станций на
decision boundary. Восстановление строится по **реальной публичной telemetry**:
остатку final-продукта, наличию raw и прогрессу CPU. Съёмка и обработка не
пересоздаются из предсказанного trace. Это ограниченное расширение D2h-B1,
не полное закрытие dynamic mission planning.

## Модель события и полномочия

Поддерживается один incident исходного `chain-proposal/1` при неизменных demand,
траекториях, скоростях и аппаратных параметрах B1. `IncidentSpec` содержит до
8 уникальных станций и absolute end_sec в пределах горизонта. Время начала
вычисляется из реальной границы исполнения. Неизвестные станции, backdated end,
новые заявки, recursive incidents и изменение rate отвергаются.

1. Исходный checkpoint проходит cold replay.
2. Активные команды всех actors отменяются одним joint step; шаг занимает Δt.
3. `station_outage_added` выполняется отдельным operator step, также Δt.
4. После этих **заранее объявленных операторских действий** считываются текущие
   inventory/lifecycle и новые privileged contact/acquisition forecasts.
5. Future suffix ищется по значениям, без rollout/admission oracle.

Шагов отмены/отключения может не быть при отсутствии соответствующих действий.
Один непустой incident обязателен; должна остаться хотя бы одна planning cell.
Операции до фактической отмены продолжают действовать: никакой backdating или
мгновенной бесплатной остановки между decision boundaries не вводится.

TaskSpec разрешает `station_outage_added` только роли с полномочием на данную
станцию. Компилятор создаёт отдельную роль `@incident`, недоступную actor roles.
Это внешнее операторское/сценарное воздействие, не автономное действие агента.
Payload дополнительно проверяется native transactional event implementation.

## Состояние восстановления

Все исходные запросы остаются в denominator, включая потерянные и невыполнимые.
Классификация после committed prefix:

- completed: уже подтверждённая доставка, не планируется заново;
- final: реально существующий доставляемый final-продукт с remaining_bits > 0;
- raw: raw существует, job не terminal; повторяются только transform + downlink;
- fresh: raw/final ещё нет, acquisition attempt не потрачен;
- terminal: failed/cancelled/expired либо acquisition attempt без живого raw.

Отмена downlink освобождает control window, но не удаляет данные и не сбрасывает
счётчик переданных бит. Для final резервируется
`Δt * ceil(remaining_bits / rate_bps / Δt)`. Используется исходная постоянная rate;
RF/DTN, station rate changes и shared bandwidth сюда не входят.

Отмена CPU переводит существующий job в paused; native продолжает его с сохранённым
remaining_sec. Scheduler консервативно резервирует полную исходную CPU duration
с guard B1, но **не сбрасывает native work/provenance**. Отмена незавершённой
съёмки при max_attempts=1 делает запрос terminal: нет магического восстановления
несуществующего raw или дополнительной попытки.

Для fresh остаётся paid slew → imaging → transform → downlink. Новый quaternion
получается к будущему midpoint съёмки с явной forecast permission. Для raw/final
повторной съёмки/наведения не добавляется; radio pointing в этом профиле unrestricted.

Actor занят всей оставшейся цепочкой, station — только TX cells. Геометрические
окна и deadlines защищаются guard. Энергетический search budget консервативно
резервирует полную первоначальную chain energy даже для укороченных стадий.
Это может отбрасывать допустимые решения; фактический native ledger оплачивает
только реально исполненные операции. SOC и использованная TX quota считываются
после incident, future PV credit при поиске не используется. Thermal feasibility,
tracking, optimality и аппаратная калибровка не доказываются.

## История и проверяемость

Исполненные templates/steps не меняются. Для recovery создаются новые action IDs,
поэтому старый slew target нельзя подменить новым будущим quaternion в прошлом.
Полученный full MissionPlan дважды проверяет physical report: до operator
controls и после committed incident. Добавление будущего catalogue не должно
менять выполненный native prefix. Поиск suffix остаётся value-only; проигрывание
уже выполненного/зафиксированного prefix не является поисковым lookahead.

`verify_incident` заново воспроизводит эти префиксы, собирает telemetry/forecasts
и пересчитывает proposal. Он не исполняет весь suffix и честно возвращает
`native_feasibility_validated=false`. Для full model-goal feasibility необходимы
PlanExecution и native replay при `publish_plan_bundle`/`publish_incident_data`.
Checksums не являются доказательством авторства внешнего файла.

Цель `require_no_failed_operations` не ослабляется автоматически. Примеры заранее
устанавливают false, разрешая намеренные operator cancellations, но требуют
min_fulfilled=2, zero rejection и coupled conservation/lineage. При failed service
цель остаётся failed; SFT-кандидаты не выдаются. Диагностика отмен сохраняется.

## Данные для ML: оператор отдельно от actor policy

`publish_incident_data` сохраняет полную проверенную историю в `operator_trace/`.
Её central SFT содержит operator controls и **не предназначена для обучения
автономного спутникового агента**. Manifest помечает это явно; training_files
перечисляет только actor suffix файлы после committed_prefix_steps.

Для каждой исходной actor role экспортируются:

- RL: наблюдения/действия/next observation только данной роли, team reward;
- SFT action prediction: только если mission goal выполнен и transition accepted.

Из catalogue удаляются чужие action IDs, mask пересчитывается с той же выборкой.
Наблюдение получает отдельную schema `incident-actor-observation/1`, новый binding
и ссылку native_observation_binding. Действия декодируются по IDs; coordinator
собирает совместный TaskSession step. Это не переименование стандартного native
наблюдения: детерминированная derived projection проверяется отдельно.
Operator injection не появляется ни в catalogue агента, ни в его target.

RL schema `incident-role-rl/1`, SFT schema `incident-role-sft/1` используют
`reward_scope=joint_team`, `expert_teacher_qualified=false`. Derived future
pointing catalogue скомпилирован оператором; перенос на иной information budget
не квалифицирован. Цель выполнена ≠ source policy expert/optimal.

Publisher staging/rename не перезаписывает output; общий budget 1 GiB. Аудит
проверяет exact inventory, hashes/paths/symlinks, goal и binding к control plan.
`--native` дополнительно пересоздаёт каждую actor строку из cold replay и заново
проверяет incident prefix. Без флага — consistency audit, не повторная физика.

## Запуск

```bash
python -m examples.planning.incident_mission /tmp/incident-input.json --mode partial_tx --write-spec
python -m examples.planning.incident_mission /tmp/incident-demo --mode partial_tx
space-env chain-incident source-proposal.json prefix.json recovery.json --incident incident.json
space-env verify-chain-incident recovery.json
# recovery.plan сохраняется в plan.json и исполняется обычным mission-plan-run
space-env incident-data-export recovery.json execution.json /tmp/incident-role-data
space-env verify-incident-data /tmp/incident-role-data --native
```

Примеры: partial_tx, paused_cpu, lost_capture, blocked, slew. Все имеют заранее
заданные inputs; outage/recovery сроки не подбираются после результатов.
Example exit 0 означает завершение workflow, не успешность mission goal.
Operator bridge владеет Runtime эксклюзивно; concurrent bridge callers не
являются атомарной multi-call транзакцией. Resume — native prefix replay,
не O(1) RAM snapshot и не hosted durable job.

## Следующий самостоятельный блок

Новые customer requests и versioned catalogue; несколько incident/replan epochs
с bounded history; typed faults/recovery отдельных подсистем; более эффективный
interleaving CPU/TX/storage; equal-information optimizer comparisons и независимый
stress corpus. Rigid ADCS/radio tracking, измеренная калибровка C2 и hosted Q7
остаются отдельными критериями промышленной готовности.

Role dataset projection — отдельный интерфейс, не drop-in observation численного
TaskEnv. Online coordinator должен применять ту же фильтрацию каталога/mask и
передавать ответы по action IDs в joint step. Готовый новый Gym wrapper для
этой projection не входит в B2a; исходные task environments остаются доступны.
