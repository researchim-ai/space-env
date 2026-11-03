# Следующий пакет: открытые измерения и проверка целых миссий

Дата: 2026-11-03. Основа: B11-Q5-C1, исходники 0.134.0.
Приёмка: [B11_CALIBRATION.json](acceptance/B11_CALIBRATION.json).

## Граница текущего результата

Есть шесть опорных профилей компонентов из спецификаций, explicit adapters,
три условных scalar estimators и воспроизводимые sensitivity ensembles.
14 workflow checks и три коротких native-варианта проверены установленным wheel.
Это основа для подготовки экспериментов, а не подтверждённая аппаратная калибровка.
Собственных измерений пока нет; следующий пакет работает с открытыми источниками.

## Дополнительный физический пакет C1b

Реализована температурная теплоёмкость NIST Al6061-T6 в thermal nodes с
энтальпийным учётом энергии, строгим диапазоном и native rollback/replay.
[Контракт](validation/MATERIAL_ENTHALPY_V1.md). Это опубликованная корреляция,
а не принятый измерительный dataset: перечисленные ниже C2-A/C2-B не закрыты.

## C2-A. Принять один измерительный набор, прежде чем расширять каталог

1. Выбрать одну подсистему и конкретный эксперимент: преобразование солнечной
   энергии, батарея, тепловой узел или динамика ориентации. Не объединять разные
   эксперименты только ради количества строк.
2. Проверить доступность самих файлов, условия использования, первичный источник,
   приборы, протокол, единицы, времена, specimen IDs и рабочие условия. Страница
   с описанием без доступных файлов не является принятым dataset.
3. Сохранить upstream SHA и manifest подготовки; сделать отдельный детерминированный
   converter с версией, dimensional checks, причинами исключений и числом строк
   до/после каждого шага. Публиковать данные только при разрешённом распространении.
4. Задать область применимости и физическую наблюдаемость параметров. Для батареи
   различать Ah и Wh; для теплового опыта учитывать известность net absorbed heat;
   для вращения — независимость знания момента и инерции.

**Готовность:** воспроизводимый import из зафиксированных исходных bytes, понятные
rights/provenance и отчёт о полноте. При недостаточных данных результат — явный
data gap; запрещено заполнять отсутствующие измерения synthetic строками с measured label.

## C2-B. Добавить estimator под реальный протокол

1. Оценить offsets, ошибки входов/выходов, корреляции, временную зависимость,
   nuisance parameters и идентифицируемость. Текущий known-noise y=θx применять
   только если его предпосылки соответствуют эксперименту.
2. Зафиксировать train/validation/holdout по независимым аппаратам, опытам или
   условиям до подбора модели. Соседние точки одного ряда не дают независимый holdout.
3. Выбрать estimator, loss и uncertainty method; не выводить sigma из nominal
   datasheet tolerance без обоснования. Сохранить preprocessing и все residuals.
4. Проверить recovered parameters на synthetic truth, затем prediction error и
   uncertainty coverage на независимых измерениях. Пороги определить заранее;
   недостаточную выборку или domain coverage отразить как ограничение.
5. Экспортировать model card: источник, область условий, fitted parameters,
   uncertainty assumptions, исключения и воспроизводимую команду проверки.

**Готовность:** отчёт об ошибках в физических единицах, negative cases и независимый
holdout без повторной подгонки. Scope лабораторного опыта не расширять до flight qualification.

## D1. Native execution bridge реализован

`qualify-ensemble` запускает baseline и все admitted variants через fresh-process
workers с replay/quality/service/resource gates и warm/cold comparison. Отказы
compiler/runtime остаются в знаменателе, offline audit проверяет frozen artifacts.
[Контракт](validation/NATIVE_ENSEMBLE_QUALIFICATION_V1.md). Constructive EO example
не закрывает независимую fixed-site/multi-plane квалификацию из D2 ниже.

### D1b: отдельный process между prefix и continuation реализован

Опциональный process-isolated protocol проверяет disk handoff, PIDs/exit codes,
source/build/seed/policy identity и shared deadline. Default false сохраняет
legacy режим. [Контракт](validation/PROCESS_RESTART_V1.md). Это отдельная recovery
приёмка; весь D2 и эксплуатационный Q7 этим не закрываются.

### D2a: independent fixed-site delivery profile реализован

Входные planes/sites не выводятся из друг друга. Full actor×site matrix,
conditional exact-dual capacity certificate и native receipt/resource witnesses
работают на орбитальном интервале; negative cases и process restore сохранены.
[Контракт](validation/INDEPENDENT_GEOMETRY_V1.md). Это готовые onboard products,
constant-rate geometric service, не qualification связанной съёмки/наведения.

## D2b. Замкнутая EO/pointing-миссия на независимой геометрии

1. Собрать fixed-site multi-plane workload: независимо заданные станции и цели,
   реальные slew/tracking intervals, окна связи, ограничения мощности/тепла,
   насыщение маховиков и доступность наземных ресурсов.
2. Сохранить baseline, assumptions, seeds, reference inputs и бюджеты сравнения.
   Новые sensitivity draws не подменяют исходные условия после неудачи.
3. Запустить каждый заявленный member: сохранить admission rejection, controller
   failure, timeout, service outcomes и resource violations в общем знаменателе.
4. Проверить cold restore/replay, остановку при выходе из допустимой области,
   баланс ресурсов и численную сходимость выбранных физических величин.
5. Разделить parameter-fit holdout и mission/controller holdout. Измерить влияние
   ошибок компонентов на mission outcomes; не переносить qualification старой сборки.

**Готовность:** проверяемый manifest всех попыток, native checkpoints, результаты
по каждому quality gate и привязка к wheel/source/data hashes. Большую миссию
переаттестовать отдельно: короткий smoke её не заменяет.

## Q6. Выпустить ограниченный, проверяемый AI benchmark

1. Включать в teacher corpus только эпизоды с пройденными native/service gates;
   отдельно сохранять объяснимые failure/recovery задачи.
2. Проверить public observations: latent draws, hidden truth и diagnostic
   manifests не должны попадать в actor prompts или признаки политики.
3. Зафиксировать разделение scenario/parameter/controller seeds и held-out
   operating conditions. Сравнить heuristic, planner и обучаемый baseline.
4. Публиковать success, safety/resource violations, latency и стоимость вычислений
   вместе с размером выборки и протоколом оценки, включая неудачные попытки.

**Готовность:** воспроизводимые RL/SFT exports и benchmark card с конкретной
областью применения. Общий industrial-readiness claim остаётся закрыт до
независимой проверки физических данных и эксплуатационных требований.

## Порядок исполнения

Сначала C2-A и C2-B для одной подсистемы, затем D на ограниченном наборе миссий,
затем Q6. Если открытые измерения не проходят входной аудит, параллельно можно
реализовывать D на явно synthetic/mixed-evidence моделях, сохраняя C2 открытым.
Новые defaults, версии контрактов и ранее выпущенные receipts не переписывать
ради прохождения проверки. Каждый пакет завершать отдельной проверяемой приёмкой.
