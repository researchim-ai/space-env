# Приёмка reference access / force crosscheck v1

30 октября2026, рабочее дерево0.134.0. Контракт и запуск:
[REFERENCE_ACCESS_V1.md](../validation/REFERENCE_ACCESS_V1.md).

## Реализовано

1. ReferenceEphemeris читает sealed GCRS orbit без исходного input/EOP файла:
   вложенные EOP, контроль UTC/TAI/backend/leaps, bounded cadence и отсутствие
   extrapolation. Cubic Hermite интерполирует r/v по TAI SI-секундам.
2. Глядя на неподвижную ITRS станцию, возвращает elevation/azimuth, range/range-rate,
   необязательный классический Doppler. Поиск AOS/LOS и peak, clipping границ,
   настройки и применимость сохраняются в отчёте.
3. CLI reference-access обрабатывает до128 явных станций, проверяет поля/имена и
   пишет атомарный sealed network report. Пример суток с Kiruna/Svalbard дал26 окон.
4. Внешние Boinor J2/drag/third-body formulas сравниваются в SI для3 состояний;
   J2 дуги24/72ч сверены с LSODA вместо production DOP853. Source/version/harness
   hashes, inputs, actual/expected vectors и общие зависимости записываются явно.
5. Physics CLI получил --crosscheck; CI reference и installed-wheel smoke включают
   новый путь. Существующий MissionWorld не переключён на новые кадры автоматически.

## Выполненные проверки

- Новые тесты:20 passed,1.33с. Проверены interpolation против circular closed form,
  AOS/LOS против отдельно вычисленной аналитической геометрии, range-rate против
  численной производной дальности, Doppler sign, clipped intervals, перенос без
  package EOP, corrupt/frame/time/cadence/backend/policy guards и sealed CLI output.
- Намеренная ошибка J2 обнаруживается crosscheck; ошибочный input не заменяет
  прежний output. Эталонная geometry comparison разделяет использованный frame provider.
- CI reference:64 passed,7.04с; CI fast:110 passed,12.39с. Наборы пересекаются.
- Длинная J2 сверка:24ч max position difference0.00639254м,72ч0.01924907м.
  Предварительно заданный tolerance10м /0.01м/с не изменялся по результату.
  Force-difference tolerance1e-12м/с². Это numerical comparison, не operational truth.
- Полный report с входами/версиями/общими данными:
  [REFERENCE_ACCESS_V1_MEASUREMENTS.json](REFERENCE_ACCESS_V1_MEASUREMENTS.json).
- Wheel переустановлен в изолированное окружение; проверен из `/tmp`, Python `-I`.
  Положительный reference access, external force crosscheck, старые smoke cases,
  Dict/portable/agent replay, serial relay и dataset export прошли.
- Все156 Python-модулей wheel совпадают с checkout. Ruff для изменённых файлов,
  compileall и diff check прошли.

Собрано494 теста. Полный pytest, большая миссия16×72ч, независимая сторонняя
frame/EOP/атмосферная истина и browser interaction для этого среза не проверялись.
Данные: [REFERENCE_ACCESS_V1_VERIFICATION.json](REFERENCE_ACCESS_V1_VERIFICATION.json).

## Открытые gates

В сравнении сил общие ERFA/EOP, density, Sun/Moon positions и constants. Boinor
проверяет внешнюю реализацию формул, не заменяет независимую физическую калибровку.
SRP/eclipse, полная комбинированная perturbed дуга с независимыми данными, covariance
uncertainty и hardware models не квалифицированы этим пакетом.

Access — геометрический кандидат контакта, не RF availability, наземное бронирование
или доставленный продукт. Short/grazing passes между scan nodes могут быть пропущены;
root tolerance не является общей physical accuracy. Интерполятор работает по
зафиксированной траектории, изменение миссии/манёвр требует её пересчёта.

MissionWorld остаётся legacy-training/1. Reference event/mission executor,
ресурсный планировщик и перенос power/thermal/payload/RF остаются следующими этапами.
Commit/push/publication/deployment не выполнялись.
