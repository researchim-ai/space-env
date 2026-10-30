# Приёмка reference downlink operations v1

31 октября2026, рабочее дерево0.134.0. Контракт:
[REFERENCE_OPERATIONS_V1.md](../validation/REFERENCE_OPERATIONS_V1.md).

## Реализованный пакет

- ReferenceDownlinkMission исполняет fixed-trajectory GCRS/ITRS downlink, а не только
  выдаёт список visibility windows. Учитывает готовый inventory, readiness/deadline,
  destination, memory, terminal capacity, outages, battery/base/tx loads и явный
  ideal-sun-tracking generation profile. Квитанции являются результатом передачи.
- Конец продукта и battery floor разбивают внешний шаг. Канал отдаётся следующему
  eligible актору сразу после освобождения терминала. Данные не создаются повторно.
- Ledger энергии записывает генерацию, served consumption, curtailment, unserved
  base demand и balance residual. Solar grid фиксирован относительно начала миссии,
  не внешнего dt. Ошибка шага откатывает состояние и append-only log lengths.
- ReferenceDownlinkEnv даёт joint MultiDiscrete actions, structured observation
  ресурсов/контактов/продуктов со сроками, reset/seed/done и reward только за полностью
  своевременный продукт. Использует обычный rollout_episode writer.
- Self-contained reference-operations-run/1 хранит spec/ephemerides/actions/results;
  replay проверяет каждый исход и final report. CLI reference-run/reference-replay.

## Измеренные проверки

- Новые tests/test_reference_operations.py:15 passed,1.83с.
- CI reference:79 passed,8.34с; CI fast:110 passed,12.28с. Наборы пересекаются.
- Gymnasium check_env прошёл. Есть предупреждение о равных Box bounds у постоянных
  полей product metadata; это не ошибка API и bounds не расширялись ради предупреждения.
- Проверены step partitions10/(1×10)/(3+7), передача терминала при completion,
  battery cutoff на1с, реальные memory/ready/outage/deadline/destination ограничения,
  два терминала, атомарный rollback, энергобаланс и tiny-step progress guard.
- CLI replay работает после удаления исходных spec/plan. Изменённый result даже
  после пересчёта checksum обнаруживается replay.
- Сквозной пример:2 аппарата,1 терминал, outage[2,4). Продукты2/3Мбит завершены
  на2/7с;5Мбит on time. Replay совпал, RL writer записал успешный переход с reward5.
- Wheel переустановлен в отдельный venv, проверен из `/tmp` с Python `-I`:
  положительная reference delivery2Мбит/reward2/replay и прежние smoke/physics/
  access/agent/legacy data paths прошли. Все159 Python-модулей совпадают с checkout.
- Ruff для изменённых файлов, compileall и diff check прошли.

В checkout собрано509 тестов. Единого полного pytest, большого endurance run,
обучения baseline и независимой калибровки resource hardware на этом срезе не было.
Машинные результаты: [REFERENCE_OPERATIONS_V1_VERIFICATION.json](REFERENCE_OPERATIONS_V1_VERIFICATION.json).

## Что остаётся

MissionWorld не переключён автоматически: этот профиль исполняет только доставку
существующего inventory по фиксированным reference orbits. Constant rate и идеальное
наведение панелей — объявленные упрощения. Нет imaging/processing/relay/ACK, burn,
ADCS, thermal coupling, battery ageing/current/voltage и полного RF budget.

Gymnasium facade — joint agent, не официальный PettingZoo adapter. Hardware constants
предоставлены task spec; не вся конфигурация дублируется в tensor. Нет заявления о
state-estimation realism, оптимальном arbitration или обобщающемся обученном агенте.
Отчёты и архивы пока накапливаются в памяти; пределы входа не являются load qualification.

Следующий gate — планировщик/динамические reference operations, единый backend
interface и дальнейшая независимая физическая проверка. Не было commit/push,
публикации, развёртывания или объявления industrial/flight qualification.
