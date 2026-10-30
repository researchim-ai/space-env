# Приёмка физического ядра GCRS reference v1

Дата30 октября2026; рабочее дерево0.134.0. Руководство:
[GCRS_REFERENCE_V1.md](../validation/GCRS_REFERENCE_V1.md).

## Что изменилось

- Явные `legacy-training/1` и `gcrs-reference/1` вместо предположения, что
  историческая строка GCRF означает квалифицированный GCRS.
- Epoch с двухчастным TAI, UTC/TAI/TT/UT1 и SI arithmetic через leap second.
- Источник/покрытие/статус EOP явны. Предсказания opt-in; вне покрытия/пробела
  более суток нет автоматической подстановки нулевого DUT1/pole.
- ERFA c2t06a, преобразование r/v с Rdot и covariance полным6×6 Jacobian.
- DOP853 orbit profile с отдельными tolerances позиции/скорости, force flags
  и manifest конфигурации/исходников/данных. J2/drag вычисляются согласованно с ITRS;
  Sun/Moon/SRP используют отдельные объявленные reference conventions.
- CLI physics-reference/reference-propagate, публичный SDK, sealed JSON trajectory
  и24ч пример. CI reference включает новые тесты; installed-wheel job ставит extra reference.
- Profile guard в engine и portable snapshots; mixed-profile world restore
  отклоняется до изменения любого актора. Legacy scenarios не пересчитаны под новый кадр.

## Фактические проверки

- Новый набор:28 passed,3.81с.
- CI reference:44 passed,6.44с; CI fast:110 passed,12.18с.
- Reference/portable/agent/mission contracts:93 passed,8.35с.
- Все наборы пересекаются. В checkout474 теста; единого полного запуска для этого
  дополнения нет. Предыдущая миссия16×72ч не выдаётся за reference-проверку.
- Frozen ERFA upstream matrix: максимальный элементный остаток около3.64e-15
  при заранее заданном допуске1e-12. Это проверка backend/conventions, не независимая
  физическая истина ERFA. Astropy сравнивается с теми же явно заданными EOP.
- Circular two-body24/72ч проверены по закрытой форме; inclined eccentric24ч —
  по уравнению Кеплера. Допуски:1м позиции и1e-3м/с скорости для circular cases;
  они не переносятся на perturbations. [Measured values](GCRS_REFERENCE_V1_MEASUREMENTS.json)
  и [результаты проверки](GCRS_REFERENCE_V1_VERIFICATION.json) сохранены отдельно.
- J2 force сверена с градиентом отдельно записанного потенциала;24ч convergence
  сравнивает два max_step с теми же силами. Это не независимый внешний force oracle.
- Проверены SI leap arithmetic, EOP leap interpolation, r/v/covariance roundtrip,
  stationary ground velocity, strict profile/input/error/output guards и знаки drag.
- Отдельно исполнена24ч CLI траектория с J2/SRP/third-body,145 выходных состояний.
  Абсолютная accuracy относительно внешней совпадающей force configuration не измерена.
- Wheel переустановлен в изолированный venv; проверка из `/tmp` с Python `-I`:
  все прежние smoke/portable/agent/relay paths и7 reference checks прошли.
  Все154 Python-модуля wheel совпадают с checkout; fixture JSON присутствует в wheel.
- Ruff прошёл для reference/report/CLI/snapshot/world/изменённых тестов и scripts.
  Прежние замечания engine.py не объявляются исправленными. Compileall и diff check прошли.

При совместном physics-прогоне новая Astropy-сверка сначала падала: два старых
теста оставляли глобально подменённый get_polar_motion и настройки IERS. Они теперь
используют pytest monkeypatch с восстановлением. Допуск новой проверки не ослаблялся.

## Что эта приёмка не означает

MissionWorld всё ещё legacy. Новое ядро не квалифицирует power/thermal/payload/RF,
не добавляет6-DOF ADCS, manoeuvre/массообмен, higher-degree gravity, dX/dY corrections,
полную uncertainty propagation или независимую perturbed24/72ч validation.
Следующий gate — external force/trajectory fixtures и согласованная миграция
миссионного executor. Не было commit/push, публикации или развёртывания сервиса.
