# Отраслевые библиотеки

Базовый расчёт не требует JDK и ядер SPICE. Дополнительные пакеты ставятся extras из `pyproject.toml` и подключаются только в момент вызова.

## Всегда в зависимостях

| Пакет | Зачем в space-env |
|-------|-------------------|
| [boinor](https://github.com/boinor/boinor) | Двухтелая пропагация Земли, кеплеровы элементы и векторы. Обёртка — `space_env.core.orbit.Orbit` |
| gymnasium | Интерфейс четырёх сред и `gym.make("space_env/...")` |
| pydantic | Конфиги КА, миссий и снимки симуляции |
| numpy, pandas, pyyaml | Векторы, сводка бенчмарка, YAML сценариев и наград |

Положение Солнца и GMST посчитаны своими рядами (`core/sun.py`, `core/frames.py`), чтобы тень и окна связи не зависели от эфемерид.

## Опционально

### sgp4 — `pip install 'space-env[sgp4]'`

`space_env.integrations.orbit_from_tle(line1, line2)` вызывает `sgp4.api.Satrec` и возвращает обычный `Orbit` в метрах. Так читаются TLE NORAD/CelesTrak. Дальше шаг симуляции идёт двухтелой моделью boinor, а не повторным SGP4.

### spiceypy — `pip install 'space-env[spice]'`

`load_spice_kernels` и `sun_position_spice_m(et)`. `et` — эфемеридное время SPICE, секунды TDB от J2000. Нужны ядра, как минимум планетарное SPK и часы LSK (`furnsh`). Без них spiceypy бросает свою ошибку. Учебный контур по-прежнему берёт аналитическое Солнце.

### ccsdspy и spacepackets — `pip install 'space-env[ccsds]'`

Свой код собирает первичный заголовок Space Packet: `space_env.core.packets.encode_space_packet`. `integrations.ccsds_backend.variable_length_definition` строит `ccsdspy.VariableLength`, если пакет установлен, и читает файл через `parse_packet_file`.

### orekit-jpype — `pip install 'space-env[orekit]'`

`integrations.propagate_keplerian_m` поднимает JVM, строит `KeplerianOrbit` и `KeplerianPropagator`. Нужен JDK 11+ (или extra `orekit-jdk`) и данные Orekit: без `orekit-data` шкалы времени Orekit не инициализируются. Для обычных эпизодов RL этот путь не вызывается.

## Чего нет

pykep (ESA, GPL-3) в зависимости не входит: другая лицензия, ставится отдельно и в код не импортируется.
