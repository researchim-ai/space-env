# Reference access и внешняя сверка сил v1

Рабочее дерево0.134.0. Продолжение [GCRS reference](GCRS_REFERENCE_V1.md).
Добавлены прикладная геометрия станции и сравнение вычислительных реализаций сил.
MissionWorld остаётся legacy-training/1; автоматической миграции агентных сред нет.

## Запуск

```bash
python -m space_env.cli physics-reference --long --crosscheck
python -m space_env.cli reference-propagate examples/reference_access/state.json /tmp/orbit.json
python -m space_env.cli reference-access /tmp/orbit.json examples/reference_access/stations.json /tmp/access.json
```

Пример задаёт полярную орбиту с J2, сутки полёта и выходную сетку30с. Для двух станций,
Кируны и Свальбарда, получены26 геометрических окон в проверенном запуске.
Это возможность связи по углу места, а не число переданных продуктов или успешно
выполненных заявок. Частота, RF rate, погода, бронирования и терминалы здесь не
ограничивают окна. Исходный state.json после получения orbit.json не нужен.

## Переносимая эфемерида

```python
import json
from space_env.core.ground import KIRUNA
from space_env.reference.access import ReferenceEphemeris

ephemeris = ReferenceEphemeris(json.load(open('/tmp/orbit.json')))
state = ephemeris.state(1234.5)  # GCRS r/v, SI
look = ephemeris.look(KIRUNA, 1234.5, frequency_hz=2.2e9)
passes = ephemeris.access(KIRUNA, scan_sec=30, root_tolerance_sec=.001)
```

Из результата SDK `propagate` строить `ReferenceEphemeris.from_result(result)`.
Файл CLI уже содержит checksum. При чтении проверяются schema, GCRS/profile,
checksum, размер/порядок/конечность сетки, согласованность UTC/TAI, ERFA version,
leap table и policy EOP. Для работы используются **вложенные EOP rows**;
таблица из исходного каталога/файла заново не читается.

Интерполяция — cubic Hermite: положения и скорости GCRS, независимая переменная —
TAI SI-секунды. По умолчанию интервалы исходной сетки не больше60с; явное увеличение
лимита допускается максимум до120с. Более редкий файл требуется пересчитать, а не
молчаливо считать высокоточным. Это ограничение применимости интерполятора, не
универсальный допуск в метрах. Экстраполяция запрещена. Maximum duration72ч и
maximum100001 state nodes унаследованы от orbit profile.

`look` преобразует r/v в ITRS полным state Jacobian. Станция неподвижна в ITRS;
возвращаются azimuth/elevation, slant range и range rate. Необязательный Doppler
имеет знак `-frequency * range_rate / c`: сближение даёт положительный сдвиг.
Это мгновенная классическая геометрия: light time, relativistic Doppler,
рефракция, plate motion и ошибки измерений не моделируются.

`access` сканирует visibility margin и уточняет найденные переходы Brent solver.
Интервалы, обрезанные началом/концом эфемериды, имеют clipped_start/clipped_end.
Peak сначала локализуется сканированием, затем уточняется локально. Максимум
между крайне редкими узлами не гарантирован. Короткие/grazing passes без изменения
знака на скан-сетке могут быть пропущены. Root tolerance не означает такую же
физическую точность AOS/LOS; ошибки исходной орбиты/EOP/интерполяции остаются.

Сетевой CLI принимает JSON `{stations:[...]}` с явными GroundStation fields,
отвергает неизвестные поля и повтор имён, ограничивает число станций128.
Output содержит checksum, hash исходной эфемериды, настройки поиска и ограничения.
При ошибке старый output не заменяется. Это не ground scheduler и не RF/data executor.

## Внешняя сверка формул

`force_crosscheck` вызывает внешние реализации из Boinor:
J2_perturbation, atmospheric_drag и third_body. Формулы не копируются в пакет.
[Первичная документация J2](https://docs.boinor.space/en/0.20/quickstart.html).
Для3 пространственных состояний сравниваются9 векторов ускорения в м/с².
Согласованы единицы: km→m, km³/s²→m³/s², kg/km³ и km²/kg для drag.

Для J2 дуги600с либо24/72ч сравнивают production DOP853 с отдельной LSODA
интеграцией, использующей внешний J2. Tolerances заданы заранее:1e-12м/с²
для force differences,10м/0.01м/с для максимальных расхождений длинной дуги.
В проверенном случае J224ч расхождение позиции около6мм,72ч — около2см.
Это **разница вычислений**, не ошибка траектории реального аппарата.

Общие inputs: ERFA/EOP frames, density, Sun/Moon positions и constants.
Следовательно, проверка не независима относительно этих данных и соглашений.
SRP/eclipse не сравниваются этим oracle; нет полной независимой проверки
суммы всех perturbations, атмосферы или реальной миссии. Ни Orekit, ни Basilisk
в этом пакете не объявляются подключённым внешним oracle.

Report содержит epoch, states/density, actual/expected vectors, EOP manifest,
версии и hashes внешнего source/harness/production models, tolerances и unqualified
scope. Test с намеренно испорченной J2 силой подтверждает отказ проверки.

## Следующий gate

Калиброванные независимые данные/конфигурации сил и согласованный reference backend
для выполнения миссии. Полученные contact windows можно использовать как вход
планирования/анализа, но они не отвечают на действия агента сами по себе.
После манёвра/изменения модели эфемериды и планы нужно пересчитать.
Сквозные power/thermal/payload/RF/task outcomes остаются задачей следующего этапа.
