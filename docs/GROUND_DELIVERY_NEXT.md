# B11-Q4: исходный план ground delivery и его реализация

## Что показал B11-Q3

Три независимых процесса, включая prefix recovery, завершили прежний frozen
профиль 12 аппаратов / 40 заявок / 120 секунд. Native replay, quality/accounting,
witness, execution/step latency, RSS и checkpoint gates прошли. Execution —
856.4–889.2 с при лимите 1200; полный worker — 1700.4–1758.3 с при timeout 2400.
**Общая приёмка не пройдена:** 28/40 заявок, 14 из 20 Мбит подтверждены на земле.
Все 40 acquisition, CPU и custody операций завершены; фактические ground senders —
все четыре relay. Остаток — 6 Мбит без подтверждённого ground completion к horizon.
Не утверждается, что все эти биты находились в одном и том же transport state.

Все три raw native/trace signature пары совпали. Это наблюдаемая повторяемость,
а не успешная формальная comparison group: min completion нарушен, поэтому все
три повтора остаются failed и группа остаётся unqualified. Точные receipts:
[приёмка](acceptance/B11_NATIVE_COPY.json),
[исходный результат](acceptance/B11_NATIVE_COPY_LARGE.json).

## Почему текущий baseline получает именно 28

В frozen `scaling.campaign` профиль задаёт imaging=2 с, transform=2 с, relay=4 с,
**downlink=4 с**, один Ground terminal и decision interval=2 с. Preloaded продуктов
нет. В `planning.policy.BaselinePolicy.feasible` активный downlink занимает станцию
до конца окна; активный relay дополнительно блокирует destination actor. Native
reference adapter не сообщает early-complete для downlink, а runtime сохраняет
control до `until_sec`. Каталог этого профиля не содержит cancellation actions.

При этих консервативных правилах контроллера первая ground control window может
начаться не раньше 2+2+4=8 с. Окна не пересекаются и admission требует, чтобы
4-секундное control window целиком помещалось в horizon. Поэтому baseline-bound:

```text
floor((120 - 8) / 4) = 28 full ground control windows
```

Результат достигает этого bound. Это **controller-specific** ограничение под
перечисленными assumptions. Оно не является доказательством физической
невыполнимости 40 заявок для любого агента, другим набором actions или другим
управлением терминалом. Native RF setup/serialization/ACK и фактическая доступность
терминала должны проверяться отдельно; полное control window не равно реальному
времени передачи полезных битов.

## Пакет реализации, без изменения старого результата

1. Разделить в отчётности технические runtime/replay/accounting gates и mission
   service objective. Сохранить overall `ok`/denominators для старого schema и
   явно показывать оба результата. Наблюдаемое совпадение raw signatures нельзя
   выдавать за прошедшую formal comparison group.
2. Добавить preflight-диагностику controller-specific window bounds с assumptions;
   не маркировать ею произвольный сценарий физически infeasible. Проверять отсутствие
   preload, cancel/early completion, достаточную station capacity, pipeline chains,
   decision-grid и horizon containment прежде, чем применять простой bound.
3. Ввести версионированную настройку ground control duration в generator. Старый
   профиль остаётся 4 с, его frozen inputs/failed receipt сохраняются. Новый кандидат
   с 2 с получает отдельный manifest/name; 40 заявок, horizon 120 и completion 1.0
   не ослаблять. Это изменение operational task setup, не force/RF physics.
4. До крупного запуска проверить native 2-секундные окна на разных текущих
   range/rate/setup состояниях: передача и ACK должны реально успевать. Полные
   charged energy, terminal/radio contention, setup, dead time, delayed ACK,
   receiver half-duplex, product custody и очереди нельзя заменять формулой rate.
5. Для общих задач добавить causal cancel-and-reschedule завершённых downlink
   controls: только по наблюдаемому ACK/delivered state, не по одному remaining=0.
   Не отменять незавершённый relay/CPU или чужие команды. Использовать существующий
   atomic cancellation→native re-admission контракт и независимые role registries.
   При отклонении новой команды cancellation также должна откатиться.
6. Уточнить планирование shared-ground windows/очереди в baseline. Изменения
   feasibility должны учитывать выбранные cancellations, role action slots,
   реальные claims и ресурсные ограничения; single-action mask не является joint
   admission oracle. Не передавать future truth/rollout в policy.
7. Small native probes → late arrivals/multi-relay/fault matrix → frozen large
   profile straight×2 + prefix restore×1. Все прежние технические пороги, quality,
   conservation и verified RL/SFT export сохранить. Фиксировать новый manifest до
   измерений; старый case не подменять успешным более лёгким случаем.
8. Сохранить 4-секундный case как исторический operational stress result. Отдельно
   измерять baseline score и environment validity: корректный симулятор может
   показать, что конкретный controller не выполнил поставленную задачу.

B11-Q4 реализован; см. `GROUND_SCHEDULING_CONTRACT.md` и
`acceptance/B11_GROUND.json` для фактической приёмки. История по-прежнему полностью копируется; immutable
science storage, более широкие/длительные/многоплоскостные миссии и независимая
отраслевая калибровка остаются отдельными направлениями.
