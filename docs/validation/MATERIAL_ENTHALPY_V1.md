# B11-Q5-C1b: температурная теплоёмкость и энергия тепловой сети

Дата: 2026-11-03. Версия исходников: 0.134.0. Модель включается явно;
постоянная теплоёмкость остаётся режимом существующих сценариев.

## Назначение и источник

Тепловые узлы теперь могут использовать C(T)=m·cp(T). Первый материал —
[NIST 6061-T6 Aluminum](https://trc.nist.gov/cryogenics/materials/6061%20Aluminum/6061_T6Aluminum_rev.htm),
Specific Heat column, коэффициенты a–i. Публикуемая корреляция:

$$c_p(T)=10^{\sum_{i=0}^{8}a_i(\log_{10}T)^i},\qquad
C(T)=m c_p(T).$$

Единицы: K, kg, J/(kg·K), J/K. Допустимы 4–300 K включительно.
Исходный HTML SHA-256:
`a2c62b7f4f774db34903bfcaf1479df326ab42de7a7e734a664aa7a0e76c972f`.
Дата загрузки 2026-11-03; оригинальная страница в package не распространяется.
Опубликованная ошибка аппроксимации cp относительно данных — 5%: это **не sigma**,
не доверительный интервал и не точность прогноза температуры спутника.

Это опубликованная корреляция материала. Нет нового ряда измерений, аппаратной
калибровки, автоматически известной массы/геометрии или подтверждения условий
настоящего спутника. Q5-C2 остаётся открытым. Данный пакет закрывает отдельное
расширение физического ядра, а не заменяет аудит измерений из C2-A.

## Контракт узла

Ровно одно из `capacity_j_per_k` и `material_capacity` должно быть задано:

```json
{
  "initial_k": 200.0,
  "trip_k": 295.0,
  "resume_k": 290.0,
  "material_capacity": {
    "material_id": "nist-al6061-t6-cp/1",
    "mass_kg": 0.01,
    "assumption": "Homogeneous synthetic Al6061-T6 lump; mass and geometry assumed"
  }
}
```

`mass_kg` положительна и ограничена 10⁶ kg; rationale обязателен. Неизвестные
material IDs, обе модели сразу, отсутствие обеих моделей и initial T вне
диапазона отвергаются. Переменная модель относится к `thermal_network.nodes`,
не к старой аналитической transmitter RC surrogate. Статус всей сети не
превращается в calibrated/published автоматически от выбора материала.

Пределы trip/resume являются отдельными safety settings. Если они выходят за
диапазон корреляции, модель может завершить операцию ошибкой раньше trip.
Нужно выбирать safety thresholds и рабочий запас с учётом диапазона материала.
Физический диапазон нельзя расширить настройкой trip или solver tolerance.

## Уравнения и учёт энергии

Для каждого узла:

$$m c_p(T)\dot T=Q_{internal}+Q_{external}-G_b(T-T_b)
-\epsilon\sigma A T^4+\sum_jG_{ij}(T_j-T).$$

Запасённая энергия считается по энтальпии:

$$\Delta U=m\int_{T_0}^{T_1}c_p(T)\,dT.$$

Выражение C(T₁)·(T₁−T₀) для переменной теплоёмкости неверно. Новый интеграл
используется в kernel balances, суммарном `network.stored`, observation balance
и общем энергетическом балансе bus. Межузловые, bath и radiation ledgers
по-прежнему интегрируются отдельно. В observation доступны текущая
`heat_capacity_j_per_k` и `node_stored_j` для каждого узла.

Функция cp вычисляется Horner evaluation в log10 T; signed enthalpy — bounded
QUADPACK quadrature с epsabs=10⁻⁸ J/kg, epsrel=10⁻¹⁰ и limit=100 subdivisions.
Nonconvergence/error budget violation дают ValueError. Это оценка численной
квадратуры, не оценка ошибки исходной материальной корреляции. Масса масштабирует
численную ошибку, поэтому общий energy tolerance нужно выбирать для своей миссии.

Radau/DOP853 вычисляют cp на каждой стадии. Экстраполяции и clipping нет, в том
числе на промежуточных стадиях. Слишком крупный solver step около границы может
отвергнуть операцию даже при допустимой предполагаемой конечной температуре:
нужен запас по области и меньший step. В MissionRuntime такая ошибка откатывает
всю транзакцию, включая энергию, продукты, время и thermal state. Возврата
частичной успешной операции или последней допустимой температуры нет.

Ограничения derivative evaluations остаются прежними; quadrature имеет отдельный
конечный бюджет на вызов. Это не OS CPU/memory quota. Phase changes, cp(T) смесей,
контактные сопротивления и temperature-dependent conductance здесь не добавлены.

## Проверки и воспроизведение

```bash
space-env reference-qualify-materials /tmp/materials.json
python examples/materials/demo.py /tmp/materials-native
python -m pytest -q tests/test_material_capacity.py tests/test_thermal_network.py
```

`scripts/generate_material_vectors.py` — отдельный stdlib-only oracle:
Decimal precision45, polynomial sum вместо native Horner, adaptive Simpson
вместо native QUADPACK. Коэффициенты общие с источником; это независимая
численная реализация, а не независимый физический эксперимент. Нормальная
проверка читает frozen fixture с pinned SHA, не генерирует его заново.

34 численных сравнения: 12 cp points, 6 signed-integral reference magnitudes,
8 native isolated heating runs на двух solvers (endpoint и energy balance).
Верхняя точка 300 K проверяется через cp/integral; ODE heating к самой границе
не заявляется проверенным из-за возможного stage overshoot. Native runs с
независимо заданной энергией проверяют интервалы 4→10→77→150→250 K по отдельности.
Пороги: cp relative 5·10⁻¹⁰, enthalpy 10⁻⁴ J/kg, endpoint 2·10⁻⁵ K,
node energy residual 10⁻⁴ J. Это numerical gates, не физические error bars.

Дополнительные тесты: энергия замкнутой двухузловой сети, partition consistency,
излучение/bath, нагреватель с оплатой энергии, PV extraction с батареей при
двух SOC, domain rejection, native rollback и cold checkpoint/replay.
Native example отдельно сохраняет prefix/final/scenario и краткую приёмку;
это integration smoke, не полная спутниковая миссия с customer service.

Сборка и текущая приёмка: `docs/acceptance/B11_MATERIALS.json`. Исторические
CALIBRATION/POINTING/ENVELOPES/GROUND receipts сохраняют исходные build hashes.

## Следующий этап

[OPEN_DATA_CALIBRATION_NEXT.md](../OPEN_DATA_CALIBRATION_NEXT.md): получить
измерительные данные с проверенными правами и протоколом, выбрать соответствующий
estimator, затем проверить связанную миссию на независимом workload. Для
материалов отдельно нужны specimen/composition variability, контактная геометрия,
обоснованная неопределённость и более широкий рабочий диапазон, если он требуется.

## Результат приёмки этой сборки

[Приёмка](../acceptance/B11_MATERIALS.json): 34 numerical comparisons и installed
native replay/domain rollback прошли. 31 новый material test, 158 targeted
material/thermal/calibration и 110 fast прошли (наборы пересекаются), 1931 collected.
Первый reference-run: 1500 passed, 1 PermissionError из-за запрета localhost bind.
Единственный упавший HTTP-тест повторён с разрешением local server и прошёл;
оба журнала сохранены. Единого зелёного reference-run не заявляется.

Wheel SHA `eddf4f1947c244349fc957887ca33ce8825a3e63a28867a461f76ea555560687`;
249 modules/282 inventory files сверены с checkout. Все четыре установленных
numerical/workflow reports имеют один source/data digest. Предыдущие receipts
не изменялись. Venv использует system-site-packages; remote clean CI, весь full
suite и большая миссия на этой сборке не запускались.
