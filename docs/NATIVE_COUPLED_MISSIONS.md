# Native сервис миссии и нелинейное относительное движение

Реализация от 11 ноября 2026 года расширяет предыдущие три пакета двумя
профилями над существующим физическим executor. Она не заменяет математическую
CW-модель или быстрый slotted-service профиль и не объявляет все комбинации
моделей совместно квалифицированными.

## 1. Потоковые заявки в одном физическом runtime

`NativeServiceSession` использует `MissionRuntime` с `reference-spacecraft`,
spacecraft bus, RF и DTN. В этом профиле **нет второго ledger энергии или битов**.
Орбиты, ориентация, память, питание, тепло, двигатели, radio acquisition,
наземные терминалы и custody продолжают исполняться существующим native кодом.

```python
from space_env.industrial import NativeServiceSession
from space_env.industrial.native_service import native_service_recipe

session = NativeServiceSession(native_service_recipe("nominal"))
observation = session.observe()
transition = session.step(
    {"commands": []},
    expected_revision=observation["revision"],
    idempotency_key="decision-0",
)
```

### Источники трафика и реальные ограничения

Stream объявляет source/destination, общий размер, размер chunk, cadence,
начальную эпоху, приоритет и deadline. Каждый chunk поступает через native
`onboard_product_added` строго на своей границе. Это уже приобретённые байты
приложения, **не новый физический сенсор или изображение**. Stream появляется
в наблюдении только после своего release. До первой заявки её декларация и
product IDs остаются в привилегированном setup. После release параметры
этого stream известны контроллеру, включая объявленную cadence.

- В native inventory действительно отсутствует ещё не поступивший chunk.
- При отказе admission по памяти событие не создаёт продукт. Отказ сохраняется
  в denominator; он не превращается в успешную обработку или исчезнувший пакет.
- Память проверяется с учётом native reservations и staging.
- Custody меняет владельца по native протоколу с fragment/ACK/retry/TTL.
- Ground-доставка учитывается по native ACK. Переданные биты и выполненная
  вовремя заявка — разные метрики.
- Cancel останавливает контролирующие команды, но не удаляет закреплённые
  за in-flight custody-transfer байты. Discard применяется после drain,
  когда native перестаёт pin продукт. Ранее доставленные биты сохраняются.
- Priority/cancel для уже полностью доставленного вовремя stream игнорируются:
  они не переписывают заработанную ценность или прошлые receipts.
- Поздняя доставка сохраняется как поздняя, а не автоматически как успех.

Внутренний metadata-журнал tracks stream/chunk admission и приоритеты. Он не
дублирует количество оставшихся байтов, charge, mass или физические receipts.
Балансы source inventory, receiver staging и wire data берутся из native.

```text
inventory balance = initial + injected + custody_in - custody_out
                    - ground_delivered - remaining - discarded
receiver balance  = unique_received - custody_in - discarded_staging - staging
wire balance      = serialized - received_unique - duplicates - lost
                    - wasted_partial - abandoned_inflight - inflight - partial
```

### RF, наведение и общие ресурсы

Доступность канала не берётся из вручную заданного `bits_per_slot`. Public view
содержит текущие ideal-local состояния акторов, inventory и текущие sampled
RF budgets. Native учитывает geometry, occultation, диапазон частот, Doppler,
pointing, weather, terminal acquisition, aggregate capacity и доступные ресурсы.
Текущий budget — не обещание будущей связи и не oracle успешного admission.

В демонстрационной сети камеры передают на 2.2 ГГц, Ground принимает
8.1–8.3 ГГц, а шлюзы передают на 8.2 ГГц. Поэтому прямой camera→Ground путь
не работает по модели band compatibility; требуется ISL и gateway. У двух
шлюзов общая наземная станция. В `pointing`-варианте используются body antennas
10°, начальная ориентация намеренно не направлена на Ground. Slew меняет
реальную ориентацию, solar/thermal состояние и последующий RF budget.

Radio/thermal/actuation остаются объявленными инженерными профилями.
Нет waveform/BER/interference solver, аппаратной калибровки антенн,
полного TCP/BPv7 wire stack или доказанной модели массового пользовательского
beam hopping. Это packetized application traffic поверх native custody.

### Продолжение после манёвра без обнуления ресурсов

Вместо `scenario` можно передать **native MissionRuntime checkpoint**:

```python
cfg = {
    "prefix_checkpoint": native_runtime.checkpoint(compact=True),
    "streams": new_stream_declarations,
    "decision_sec": 2.0,
}
continued = NativeServiceSession(cfg)
```

Префикс пересоздаётся штатным native replay. Service сохраняет абсолютное
время, source identity, состояния аппаратов, топливо, тепло, энергию, продукты,
pending commands и transport. Новые releases нельзя backdate до начала service.
Горизонт и native budgets не продлеваются. TaskSession goal/reward автоматически
не импортируются: этот адаптер задаёт собственную цель по новым streams;
старые продукты и процессы остаются в native, доступны контроллеру и не удаляются.

Фиксированный `prefix`-пример выполняет body-frame включение химического
двигателя, включая ignition, вспомогательное питание и тепло, затем продолжает
service с физического времени 4 s. Engine settings синтетические и явно так
помечены. Это один исполнитель, без переноса ресурсов из абстрактного
`ServiceSession` и без перезапуска часов. Полное развёртывание группировки
по орбитальным слотам этим примером не квалифицируется.

### Действия и область входа

`commands` — bounded native RuntimeCommand payloads. Поле `command_id` можно
опустить: adapter назначает уникальный ID. Forecast bindings запрещены;
никакие события не доступны в agent action. Допустимые виды команд публикуются
как `supported_commands`, а реальный admission повторно проверяет текущий native
executor. Отмена задаётся через `cancel_actors`.

Весь control-batch атомарен. Отклонённая native команда стоит один decision
interval и penalty 0.1; внешние события всё равно применяются. События разделяют
интервал точно на своих временных границах, без искусственной обязательной
двухсекундной задержки, использованной предыдущим operations driver.

Текущие contract bounds: до 16 actors, 32 streams, 256 chunks и 512 решений;
шаг 0.01–10 s при достаточном decision budget. Это пределы допустимого входа,
не измеренная масштабируемость полной физики. Контроллер центральный, имеет
объединение текущей идеальной телеметрии; распределённый обмен оплачиваемыми
сообщениями здесь не добавлен.

Награда: прирост суммы приоритетов полностью выполненных вовремя streams,
минус 0.1 за отклонённый native batch. Все original streams остаются в строгом
all-streams success denominator; cancel, memory admission failure и late delivery
показаны отдельно. Эта метрика не должна подменять договорную SLA конкретного
оператора без адаптации определения требований.

## 2. Нелинейные GCRS орбиты и оценивание вместо CW truth

`NativeProximitySession` использует две самостоятельные native GCRS траектории
и существующий `NavigationSession`. Для каждого аппарата есть свой synthetic
Cartesian-position sensor и шестимерный EKF. Relative RTN state вычисляется
из двух текущих **оценок**, с корректным вращением координатной системы.

```python
from space_env.industrial import NativeProximitySession
from space_env.industrial.native_proximity import native_proximity_recipe

session = NativeProximitySession(native_proximity_recipe("latency"))
transition = session.step(
    {"kind": "inspect"}, expected_revision=0, idempotency_key="decision-0",
)
```

В этом профиле truth не распространяется через CW STM:

```text
r'_i = v_i
v'_i = -mu * r_i / |r_i|^3
```

Поддержаны central coasts и импульсы chaser на границах решений. Команда
задаётся в RTN, переводится в GCRS по оценённому chief и исполняется native.
Fuel depletion учитывается native rocket model. Sensor power включён в native
base load. События acquisition/reception разделяют advance; поздние измерения
сохраняют свою эпоху и известную им историю управлений.

### Что именно видит агент

- Оценки GCRS и covariance каждого аппарата, age, полученные packets/NIS.
- Relative RTN estimate, оставшийся delta-v и задача inspection/hold/retreat.
- Relative **GCRS** covariance как сумма двух независимых фильтров.
  Cross-covariance не оценивается; RTN covariance не выдаётся за эту сумму.
- Нет истинной траектории, истинных residuals цели, future packets или
  frame transforms на основе скрытого истинного chief.

Датчики — зашумлённые Cartesian position measurements, не camera pose,
лидарный point cloud или полный GNSS receiver. Инспекция позиционная:
проверяются положение/скорость в допустимой зоне на границах решений,
но не изображения поверхности и не её optical coverage.

### Проверка интервала без CW

Для каждой unforced central дуги перицентр вычисляется из начального состояния:

```text
h = r × v
p = |h|² / mu
e = |(v × h)/mu - r/|r||
rp = p / (1+e)
```

С численным запасом вниз для `rp` получаем:

```text
|a_rel| <= mu/rp_1² + mu/rp_2² = A
|v_rel(t)| <= |v_rel(0)| + A*T = V
min_distance >= min(distance_left, distance_right) - V*h/2 - epsilon
```

Иерархическое деление проверяет подынтервалы до `safe`, `unsafe` или
`indeterminate`. Последний не засчитывается как безопасный успех. Проверка
читает уже исполненную native дугу, не даёт агенту будущие состояния и не
откатывает физическое время после нарушения. Это evaluator, **не flight safety
shield**, гарантирующий предотвращение столкновений.

Область этого профиля преднамеренно ограничена: два аппарата, central gravity,
impulse-separated coasts; initial/post-impulse pericenter >=6500 km, native
horizon <=3600 s, fine integration settings и шаг управления до 10 s.
Generic central interval checker ограничен 30 s. J2/drag/SRP/third-body,
continuous thrust и spacecraft-bus actuation здесь отвергаются: этот bound
не расширяется на них молча. Native сервис связи использует более богатый
spacecraft executor; их совместная sensor-aware квалификация ещё открыта.

Default numerical allowance 1 cm относится к численному сравнительному
профилю. Это не формальная интервальная арифметика, не доказанная сверху
глобальная ошибка ODE solver и не физическая погрешность реального спутника.
Публичная baseline по-прежнему использует простую PD/CW planning heuristic
на оценках; actual simulation и safety evaluator нелинейны. Heuristic sigma
margin не является доказанной Gaussian вероятностью безопасности.

## 3. Ограниченное восстановление и изоляция отказа

В `OperationsSession` добавлены:

- `max_reset_attempts` (1 по умолчанию, 1–3) на actor/subsystem за миссию;
- наблюдаемый журнал фактических reset attempts/results;
- `isolate`, выполняющий native safe-mode transition;
- policy, которая прекращает сбросы после исчерпания лимита, изолирует аппарат,
  продолжает другие работы и периодически проверяет возможность восстановления;
- единый внешний revision в actor views; исходный native counter теперь отдельно
  помечен `native_revision`.

Ideal BIT/reset остались идеализированными ground maintenance services.
Внешнее снятие отказа в сценарии — объявленное причинное событие оператора,
не физически смоделированный ремонт и не действие, доступное агенту.
Обновление не означает hardware FDIR calibration или модель резервных схем.

## 4. Общий контракт, данные и локальная проверка

Новые `kind="native_service"` и `kind="native_proximity"` подключены к
`IndustrialEpisode`, source-bound replay, SFT/RL export и `IndustrialEnv`.
Шаг, idempotency и policy call budget используют прежний контракт. Gym seed
для native proximity меняет два sensor streams, не геометрию орбит.
Наблюдения возвращают detached JSON; checkpoints и native truth — trainer-only.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m space_env.industrial.native_study --output artifacts/new-native-coupled
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m space_env.industrial.native_study --audit artifacts/new-native-coupled
python -m pytest -q tests/test_native_service.py tests/test_native_proximity.py tests/test_native_recovery_contracts.py tests/test_native_study.py
```

Manifest фиксируется до исполнения: 7 native service, 5 native proximity и
2 FDIR конфигурации × wait/public = 28 попыток. Два spawn workers не делят
runtime. Audit заново исполняет всю матрицу и сравнивает полный agent record,
включая решения политики, сообщения, rewards и конечный witness. Ошибки
инфраструктуры и неуспехи миссий остаются в denominator. `valid=true` само по
себе не означает, что все миссии решены или нет retained error-строк.

Варианты публичны и происходят от общих progenitors; это regression coverage,
не независимый industrial/OOD benchmark и не сравнение с внешними движками.
Установка: `pip install -e '.[reference,coverage,dev]'`. CI не требуется.

## 5. Проверено на текущих исходниках

Артефакты: `artifacts/native-coupled-20261111/`.

- **445 целевых и регрессионных тестов прошли**, включая новые профили,
  прежние industrial контракты, checkpoints/copy ownership, navigation,
  dynamic workspace/network, planning, actuation, resource lanes, radio и bus.
  Это не весь тестовый набор репозитория.
- Матрица **28/28** завершилась без ошибок инфраструктуры и без invalid
  tool calls. Fresh-process audit пересчитал решения политик и полные
  agent records всех 28 попыток. `errors_retained=0`.
- Native service, public: nominal, outage, receiver failure, low memory и
  body-burn prefix выполнили оба stream: 700000 bits, семь chunks, priority 11.
  Low-memory вариант под wait действительно создаёт admission rejections;
  public в этой конфигурации успевает освобождать память.
- Cancellation сохранила отрицательный исход по исходному all-streams
  критерию: routine отменён, urgent доставлен (300000 bits, priority 9).
- **Pointing-вариант не решён вовремя**: 700000 bits доставлены поздно,
  `on_time_chunks=0`. Он не удалён из study и не назван успехом по числу битов.
  Это недостаток данной эвристики, не доказанная невозможность соблюсти сроки.
- Source, receiver и wire balance errors во всех native-service строках
  study равны нулю в данном численном профиле. Это не universal exact-integer
  arithmetic: native wire/resource integration использует float.
- Native proximity: public выполнила positional inspection nominal/latency,
  blind case и отдельный retreat goal. В crossing сохранилась безопасность
  интервалов, но к 30 s не завершены ни инспекция, ни критерий полного retreat;
  success остаётся false. Wait пересекает keepout в crossing.
- Blind nominal-relative case решается из хорошего prior без манёвра:
  этот набор не доказывает необходимость навигационных измерений или
  трудность миссий для сильного агента. Здесь проверяется исполнение контрактов.
- Permanent-radio FDIR: один reset, изоляция и продолжение дали 3/4 доставки,
  fulfilled priority 19, native return 22. При объявленном external clear на
  28 s политика диагностировала восстановление, вышла из safe и выполнила
  4/4 доставки. Один native rejected batch сохранился: return 29.9, не 30.
  Общие maintenance attempts включают диагностику/isolation/recovery;
  это не число resets. Протокол/политика изменены, поэтому сравнение с прежним
  прогоном не является независимым head-to-head алгоритмическим исследованием.

### Статистическая согласованность nonlinear navigation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/modeling/native_navigation_study.py --seeds 32 --output artifacts/new-native-navigation.json
```

Сохранены 64 двухаппаратные траектории: 32 seed × nominal/latency с dropout.
Truth initial states для двух аппаратов независимо выбираются из объявленного
Gaussian prior; sensor seeds отделены от prior seed. Управление coast,
process PSD нулевой, модели сил совпадают. Между профилями seeds парные.
Все 64 попытки дошли до 16 s и сохранили безопасные интервалы.

- Mean relative-GCRS NEES: nominal 6.0972, latency 6.1979;
  95% interval среднего [4.8600, 7.2583].
- Mean actor NEES: 5.8577 и 5.6933; interval [5.1813, 6.8779].
- Mean actor NIS: 3.5312 и 3.2888; interval [2.4300, 3.6291].

Шесть средних попали в заранее выбранные marginal 95% интервалы.
Это matched-model EKF consistency, не physical sensor calibration,
не доказательство cross-covariance при неизвестных общих ошибках,
не гарантия одновременного 95% покрытия всех проверок и не OOD.
Неподходящие/ранние/ошибочные попытки сохраняются в denominator.

### Демонстрация обучающих форматов

Четыре свежих train-прогона (без изменения split у test-records): успешные
body-prefix service и delayed navigation, а также failed-pointing и partial
permanent-fault. Сохранены две SFT и четыре episodic RL строки. Неуспехи
сохраняются в RL и не выдаются за положительные SFT demonstrations.
Каждый экспорт проходит фактический replay. В training rows нет checkpoints,
будущих событий, hidden initial states или simulator truth witnesses.

Эти train-примеры происходят от **тех же публичных regression progenitors**,
что study. Они проверяют формат/причинность, не независимый train/test split,
не реальный model inference и не улучшение обученных весов.

## Что ещё не закрыто

1. Общий sensor-aware spacecraft-bus профиль, соединяющий nonlinear
   navigation, ADCS/actuation, RF, thermal и EO в одной квалификации.
2. Доказанные interval bounds для возмущённых/finite-thrust дуг, uncertainty-aware
   encounter management, CDM/Pc workflow, sensor-based pose и контактная динамика.
3. Реальные hardware FDIR, резервные схемы, неизвестные actuator errors,
   калибровочные данные и независимые сравнения в области применимости.
4. Сильные planners, новые независимые орбитальные семейства, распределённое
   управление и обучение с закрытым holdout. Большое число вариантов одного
   progenitor не является таким holdout.
5. Длительные миссии, полное развёртывание/пополнение группировки и квалификация
   масштаба. Stream/event deadlines данного wrapper ограничены 3600 s от origin;
   существование native более длинного горизонта само по себе этот предел
   не расширяет. Multi-day physical service этим study не проверен.

Новый код существенно соединяет физику и задачу агента, но не даёт основания
объявлять симулятор промышленно/полётно сертифицированным или превосходящим
все альтернативные системы.

### Продолжение контроллера и владение конфигурацией

Полный `NativeServiceSession.checkpoint()` сохраняет также private stream/event
schedule контроллера. Для продолжения **той же** сервисной постановки используйте
`NativeServiceSession.from_checkpoint(...)`.

`prefix_checkpoint` принимает только native runtime checkpoint: он сохраняет
физическое состояние и уже поступившие native события/продукты, но не будущие
неопубликованные заявки прежнего внешнего контроллера. Это запуск нового service
этапа с явно переданными новыми streams/events, а не неявный перенос чужой цели.

Command IDs, event IDs и idempotency keys нового этапа выделяются с учётом
native revision и уже занятых имён префикса. Они не сбрасываются к прежним
именам первого шага и не вычисляются из скрытых будущих событий. Проверены
новые команды и новое station-recovery событие поверх префикса уже работавшего
service-контроллера. Fresh controller сохраняет его inventory/ресурсы и не
натыкается на старые command/cache keys.

Конструкторы владеют копиями входных моделей, включая изменяемые event payloads
и sensor mappings: последующее изменение caller-owned Pydantic model не меняет
миссию или её replay input.

### Сборка и воспроизводимость

Wheel использует ту же версию 0.134.0; Python/JSON payload сверяется побайтно.
Installed smoke запускается из отдельного cwd с явным путём распакованного
wheel: body-prefix service и delayed native proximity выполняют по два шага,
затем восстанавливаются из JSON checkpoint. Это smoke с существующими
зависимостями, не полная clean-install или installed qualification matrix.

Distribution metadata исходного рабочего окружения сообщает 0.24.0;
metadata собранного wheel — 0.134.0. Checkpoints связывают фактическую среду,
поэтому данные двух окружений проверяются в своих средах без перепривязки.
Старые receipts остаются историческими: новый source digest не подставляется
в старые результаты. Данные текущего финального прогона находятся в `final/`,
`final-audit.json`, `final-regression.log`, `final-navigation-consistency.json`
и `export-final/` внутри artifact root.

Финальный локальный прогон: **445 тестов прошли**, все **28 эпизодов** повторены
в свежих процессах с совпадением политики и записи эпизода; инфраструктурных
ошибок нет. Дополнительно выполнены **64 двухаппаратные траектории** проверки
согласованности фильтров. В wheel побайтно совпадают **366 Python/JSON файлов**;
installed smoke обоих новых профилей прошёл в указанном выше ограниченном объёме.

[Приёмочная запись](acceptance/NATIVE_COUPLED_MISSIONS_V1.json) связывает digest
исходников и SHA-256 47 локальных артефактов, включая отрицательные результаты,
экспорты и wheel. Артефакты находятся в игнорируемой Git директории
`artifacts/native-coupled-20261111/`; запись не заменяет их публикацию или внешнюю
независимую проверку.
