# B11-Q5-D1b: восстановление миссии через границу процессов

Дата: 2026-11-03. Исходники0.134.0; evidence привязан также к wheel/source/data SHA.

## Включение

```bash
python examples/calibration/qualification.py /tmp/ensemble-process --new-process
space-env verify-ensemble /tmp/ensemble-process
```

В EnsembleQualificationSpec задаются `resume_after_decisions: 1` и
`resume_in_new_process: true`. Остальные поля — по
[контракту ансамблей](NATIVE_ENSEMBLE_QUALIFICATION_V1.md).
В прямом ScaleSpec эти же поля задаются в workload. В ensemble новый процесс
включается только для restored/cold half; direct/warm остаётся сравнительным run.

Default false сохраняет прежнее in-process восстановление. Новые false fields
не включаются в canonical сериализацию Workload/EnsembleSpec, сохраняя legacy
input hashes. Сама кодовая сборка новая: старые native receipts не квалифицируют её.

## Протокол и проверка связи

1. Родитель фиксирует Workload и отдельный invocation nonce, не меняющий seeds.
2. Producer interpreter создаёт PlanningRun, выполняет ровно заданное число
   решений и атомарно публикует compact JSON planning checkpoint в packet.
3. Родитель ждёт успешного выхода producer и сверяет packet с наблюдавшимся PID.
   Одного наличия файла при ненулевом exit code недостаточно.
4. Только затем запускается continuation interpreter. Он проверяет packet и
   запрещает совпадение producer/consumer PID. Python objects, mutable simulator
   graph, pickle и process caches через границу не передаются.
5. Восстановление использует `PlanningRun.from_checkpoint`: существующий API
   **аудированно воспроизводит prefix**, сверяя decisions/transitions/native state
   и policy state, а затем продолжает миссию. Это не O(1) загрузка memory snapshot.
6. Родитель сверяет consumer PID, nonce и packet hash с фактически запущенным child.
   В result остаются обе phase outcomes и native-integrity gate `process_restart`.

Packet связывает canonical Workload, scenario, task с ожидаемым policy origin,
episode seed, policy config, точное число решений и checkpoint digest. Operator
override в этом протоколе запрещён. Сверяются whole-package source/data digest,
package/Python и numpy/scipy/pydantic/pyerfa versions. Подменённые или resealed
workload/build/seed/task/policy fields не обходят эти проверки.

Проверяются typed finite timing/RSS observations и distinct PIDs. Повторное
использование PID консервативно даёт failure. Hash/nonce обеспечивают связь
локальных артефактов, не авторскую подпись или аутентификацию удалённого host.
Родительский runner и файловая система остаются доверенной границей.

## Бюджеты и отказ

`timeout_sec` — один общий deadline для prefix и continuation. Вторая фаза
не получает заново полный timeout. При timeout child kill+wait/communicate
завершается до возврата error row. Distributed cancellation здесь не реализована.

Source input producer ограничен2MB. Packet ограничен меньшим из
`max_checkpoint_bytes` и64MB; symlinks, oversized/truncated packets отвергаются.
Нет подмены сценария меньшим для прохождения бюджета. Точка восстановления,
попавшая после фактического завершения эпизода, не переносится автоматически.

`execution_sec` в новом режиме включает prefix execution/checkpoint preparation,
consumer restore с повторным prefix replay и continuation execution. OS startup,
packet I/O и final replay/export дополнительно видны через phase/end-to-end wall
timing. Execution gate не заменяет общий subprocess deadline.
Top-level RSS использует максимум доступных prefix/consumer high-water snapshots
на моментах замера. Это наблюдение, не OS-enforced memory limit; отсутствие
поддержанного RSS measurement не объявляется passed RSS gate.

В `native/workers/<index>/` сохраняются `restart.json`, `prefix-progress.json`
и continuation `progress.json`, если стадия успела начаться. При раннем отказе
сохраняется последняя доступная диагностика. У error row `restart_protocol.status`
остаётся failed; completed prefix не означает completed mission.

Logical executions и OS launches разделены: restored repeat использует два
последовательных интерпретатора. `planned_process_launches` считает OS launches для admitted workloads и baseline.
Structural rejections остаются в attempt/slot denominator, но в этот launch count
не включаются. Одновременно child процессов не больше declared workers, максимум4.
Inventory расширен до400 файлов, с прежними64MB/file,256MB суммарно и32MB receipt.
Превышение не выдаёт successful final receipt; partial artifacts сохраняются.
Automatic retry/resume nonempty job directory не добавлен.

## Audit и результат

Offline verifier требует packet в inventory, связывает его с observed PIDs,
nonce, workload, policy/seed и recorded build identity. Native manifest, progress
и raw execution rows по-прежнему сверяются. Это identity/aggregation audit,
не fresh physics replay и не qualification изменённой текущей сборки.

Structural rejection, prefix/continuation failure, timeout, resource/mission gate
failure и несовпавшие pair signatures остаются в знаменателе D1. Timeout оставляет
mission outcome ненаблюдавшимся. Проваленный resource budget может сосуществовать
с выполненными customer requests; эти статусы не смешиваются.

## Проверки и границы

Новые тесты: guards/подмена packet, build/seed/policy/task/PID bindings,
запрет local fallback, byte limits, terminated prefix, corruption до consumer,
настоящий timeout с reaping, retained error rows и real process-boundary restore
в сравнении с direct run. Legacy ensemble/progress tests проверяются отдельно.

Installed EO/CPU/relay/ACK example содержит baseline и две assumed battery
вариации: шесть logical executions, девять OS launches. Сравниваются полный
causal trace и native backend report каждого direct/restored pair. Это конкретная
synthetic service/recovery mission, не flight qualification произвольного сценария.

Приёмка: `docs/acceptance/B11_PROCESS_RESTART.json`. Следующий крупный этап:
независимые fixed-site/multi-plane closed-loop workloads, pointing/resource/error
budgets и mission holdout. C2 измерительная калибровка, Q6 teacher/benchmark
и Q7 durable jobs с принудительной OS isolation остаются открытыми.

## Результат текущей приёмки

[B11_PROCESS_RESTART.json](../acceptance/B11_PROCESS_RESTART.json): 41 новых tests,
91 targeted,1581 reference,110 fast passed,2011 collected; наборы пересекаются.
Reference завершился в одном запуске. 30 logical attempts в6 installed cases
дали9 successful separate-process restores, включая prefix после3 решений и
random-masked policy. Отрицательные structural/budget/timeout outcomes сохранены.
Все6 offline/CLI audits проверены; audit не перезапускал физику.

251 modules /284 source-data files byte-matched; wheel SHA:
`9e6ffa7b22eb130ecdb107d1acc97d0c4837a141c1987a7f6c57f83fe6994412`.
Сохранены exact UTF-8 diagnostic files, включая packets и observed phase outcomes,
в6 gzip JSON bundles. Исторические6 receipt families сохранены без изменения.
Прежние numerical/workflow проверки на этой сборке также прошли.
Venv system-site-packages; нет clean remote CI, expanded full suite или общей
hardware/independent mission qualification. C2/D2/Q6/Q7 остаются открытыми.
