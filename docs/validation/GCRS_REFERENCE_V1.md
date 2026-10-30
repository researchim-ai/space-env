# Reference physics v1: GCRS, время и адаптивная пропагация

Реализованный срез рабочего дерева0.134.0, 30 октября2026. Это отдельное ядро,
не автоматическая замена физики всех сред и не завершение промышленной квалификации.

## Профили разделены явно

MissionWorld/SimulationEngine продолжают использовать `legacy-training/1`:
прежние модели, календарное float-JD время и историческую метку GCRF.
Профиль теперь записывается в engine capture и portable world definitions.
Несовместимый профиль отвергается до восстановления любого актора мира.
Старые snapshots без поля профиля читаются как historical legacy.

`space_env.reference` использует `gcrs-reference/1`, `ReferenceState(frame="GCRS")`
и собственное двухчастное время. Векторы legacy GCRF не переименовываются в GCRS.
Обычный Orbit/MissionWorld не принимает этот профиль через неявную подстановку.
Обратная миграция mission executor, сенсоров, целей и каталога остаётся отдельной работой.

## Установка и запуск

```bash
pip install -e '.[reference]'
python -m space_env.cli physics-reference --long
python -m space_env.cli reference-propagate examples/reference_physics/state.json /tmp/reference-orbit.json
```

Первый запуск измеряет ошибку матрицы относительно frozen ERFA fixture и ошибки
круговой двухтелой орбиты на600с/24ч/72ч. Проверки не являются общим сертификатом
возмущённой орбиты. Второй строит24ч траекторию с J2/SRP/third-body из явного
GCRS состояния и сохраняет JSON с checksum. Для прогнозных EOP нужен явный флаг
`--allow-predictions`; он не превращает прогнозные данные в измеренные.

```python
import numpy as np
from space_env.reference import EOPSeries, Epoch, FrameContext, ReferenceState, ForceModel, propagate

context = FrameContext(EOPSeries.packaged())
epoch = Epoch.utc('2026-07-18T12:00:00Z')
state = ReferenceState(epoch, np.array([7e6, 0, 0]), np.array([0, 5000, 5700]))
result = propagate(state, 86400, context, model=ForceModel(j2=True), sample_sec=60)
fixed = context.transform(np.r_[state.r_m, state.v_m_s], epoch,
                          source='GCRS', destination='ITRS', covariance=np.eye(6))
```

Состояния в СИ: r в м, v в м/с. Блоки covariance имеют соответственно м²,
м²/с и м²/с². Rotation matrix преобразует GCRS → ITRS. В обратном направлении
используется полный обратный state Jacobian, не только transpose для блока скорости.

## Время и данные ориентации

Epoch хранит две части TAI JD. UTC вводится явно строкой ISO с Z; SI-секунды
добавляются в TAI. ERFA выполняет UTC↔TAI↔TT/UT1 и поддерживает високосные секунды.
Допустимость этого профиля ограничена1972–2026; более поздние эпохи отвергаются,
а не получают молчаливое предположение о будущих leap seconds. Старый Time не менялся.

EOPSeries требует источник и минимум два упорядоченных узла; пробел больше суток
(с учётом leap second) отвергается. UT1-TAI/xp/yp линейно интерполируются по TAI,
чтобы UTC leap не размазывался на сутки вращения Земли. Нет zero-EOP fallback и
экстраполяции. Использование prediction узлов требует явного разрешения.
Packaged snapshot IERS начинается16 июля2026; узлы после2026 не включаются в
применимость профиля. Реальные границы и статусы есть в manifest rows.

В manifest сохраняются источник/hash пакетного EOP, использованные узлы, policy
прогнозов и hash таблицы leap seconds. EOP covariance/uncertainty не распространяется.
Celestial pole offsets dX/dY пока **не применяются**, это объявленное ограничение.

## Преобразование положения/скорости/ковариации

ERFA `c2t06a` задаёт IAU2006/2000A rotation, Earth rotation и polar motion.
Преобразование скорости включает Rdot*r; Rdot вычисляется пятиузловой центральной
разностью по SI времени с h=0.5с. Для covariance применяется полный6×6 Jacobian.
Нужны EOP также в окрестности±1с для производной; неподдерживаемая граница отвергается.

Проверки: frozen numerical fixture из
[upstream ERFA test](https://github.com/liberfa/erfa/blob/master/src/t_erfa_c.c#L2354),
rotation orthogonality, наземная неподвижная точка с ненулевой инерциальной скоростью,
6D roundtrip, covariance symmetry/PSD и Astropy transform при одинаковых явных EOP.
Astropy и этот backend используют ERFA: это независимый API-путь проверки
velocity/конвенций, **не независимая физическая истина** IAU/EOP модели.

## Силы и интегрирование

ForceModel всегда включает центральное поле. Опционально:

- J2 в земной системе координат с возвратом ускорения в GCRS;
- drag относительно вращающейся атмосферы: скорость из полного ITRS state transform,
  геодезические lat/lon/height, table или явно запрошенный NRLMSISE;
- cannonball SRP с геометрической тенью и направлением от Солнца к аппарату;
- Newtonian third-body Sun/Moon. Sun — ERFA epv00 с geocentric TT→TDB,
  Moon — ERFA moon98/TT; без переключения на Meeus при отсутствии зависимости.

Drag-профиль ограничен150–800км; extrapolation крайних узлов таблицы запрещена.
Это не калибровка атмосферы и Cd реальной миссии. Необходимый NRLMSISE пакет
должен присутствовать; его datetime-интерфейс не поддерживает leap-second input.
Эфемериды/силы — объявленное Newtonian approximation, без полного relativistic model.

DOP853 из SciPy имеет отдельные atol для r/v, rtol и max_step. По умолчанию:
rtol1e-11, atol позиции1e-4м, скорости1e-7м/с, max_step60с. Численный tolerance
не равен физической точности. Максимальная дуга72ч; максимум100000 выходных интервалов.
Нет burn/массообмена, ADCS, higher-degree gravity, tides, winds, альбедо-пертурбаций
или квалифицированного spacecraft geometry model. Earth intersection исключена.

Результат содержит profile, время/кадр/единицы, force settings, версии SciPy/ERFA/
NumPy, hashes исходников reference и используемых legacy force helpers, EOP/leaps
и integration settings. CLI пишет его атомарно и добавляет checksum содержимого.
Это не новый формат portable MissionWorld run и не гарантия cross-version replay.

## Что подтверждено и что остаётся

Закрытые проверки включают circular24/72ч и eccentric inclined24ч против решения
уравнения Кеплера, J2 против численного градиента независимо записанного потенциала,
знак drag, исполнение остальных perturbations и24ч J2 convergence при разных max_step.
Последнее сверяет интегратор с теми же силами, не заменяет внешний force oracle.

Точная геометрия и force-model accuracy возмущённых24/72ч дуг должны быть проверены
по независимым данным/сопоставимой конфигурации Orekit/Basilisk/другого эталона.
Затем нужен migration всего mission executor: access, power, thermal, catalogue,
события, snapshots и scoring должны использовать выбранный профиль согласованно.
До этого reference orbit нельзя объявлять high-fidelity средой обучения агента.
