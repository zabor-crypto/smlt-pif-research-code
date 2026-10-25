# SMLT / ПИФ — исходный код исследовательского прототипа

## English brief

An **academic portfolio-risk research prototype** for a fund with an existing
SMLT equity position. The engineering question is how a proposed hedge remains
feasible across the whole book: information available at decision time, later
execution, cash, integer contracts, margin, costs and settlement.

This is a **source-only academic export**, with a deliberately small synthetic
test entry point. It does not reproduce the full market study: market data,
economic-run configurations and generated results are excluded. Comparative
economic effectiveness, market out-of-sample performance and live readiness are
not established by this release.

- **Start with the narrative:** [English project brief](https://zabor-crypto.github.io/smlt-pif-research-site/en.html)
  or the [presentation source](https://github.com/zabor-crypto/smlt-pif-research-site).
- **Inspect the implementation:** [data checks](src/smlt_hedge/validation/checks.py),
  [option mechanics](src/smlt_hedge/strategies/options.py), and the selected tests below.
- **Attribution:** this is a six-person academic project. Boris Zabavnikov's
  documented contribution is project leadership, quantitative modelling,
  hedging/optimization algorithms and the execution mechanism. Engineering and
  testing were shared; see the [team's contribution breakdown](https://github.com/zabor-crypto/smlt-pif-research-site/blob/main/team.html).
  Repository ownership is not a claim of sole authorship of every component.
- **Provenance:** [SOURCE_VERSION.json](SOURCE_VERSION.json) identifies the
  exported source snapshot and per-file hashes. The library and original tests
  remain unchanged; this public wrapper adds documentation and bounded CI.
- **Rights:** no open-source license has been granted. Rights remain with the
  respective holders; public visibility does not grant reuse rights. Dependency
  licenses remain separate. No market-data redistribution rights are implied.

### Reproduce the bounded synthetic checks

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependency installation may
access package registries; after synchronization the selected tests use only
in-code arrays/tables and a temporary directory, without market files or APIs.

```sh
uv sync --frozen --group dev
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --offline --frozen python -m pytest tests/test_data.py tests/test_options.py -q -p no:cacheprovider
```

CI runs these two modules and verifies exported-file hashes. Passing them covers
their data-validation, storage and option-formula examples; it does not validate
the full adaptive policy, historical research conclusions or fund suitability.

## Описание исследования

Цель исследования — управление риском всей книги учебного ПИФа с существующей позицией SMLT. Финансовая версия 1.0 объединяет постановку, математическую структуру и программный прототип; сравнительная экономическая эффективность не установлена.

## Точная версия и область
Экспорт исходников R10 `e11952dac3dd0703e63f29d269ad51dee97bdf0c`, дерево `fde3d5beca830215b2a4c4c5afa3d8c8c4b452ef`. Файлы библиотеки и четырёх тестовых модулей скопированы без изменения байтов; полный список — `SOURCE_VERSION.json`. В библиотеке сохраняются исторические движки и экспериментальные пути. Присутствие функции или теста не означает принятия её результата. Режимная модель находится в `src/smlt_hedge/regime_v2/`, адаптивный контур — в `adaptive/`; ранние хеджевые реализации не заменяют правила текущей полной книги.

Исходный полный снимок содержал 1 751 файл и 44 697 005 байт, включая данные. Это не объём Python и не размер данного публичного экспорта. Из экспорта исключены все данные, outputs, конфигурации экономических запусков, рабочие скрипты, частные контракты и история прежнего репозитория. Это версия для чтения кода и ограниченного синтетического воспроизведения; полный рыночный расчёт из этого набора не воспроизводится.

## Начальные условия и метод
Модельная позиция: 40 000 SMLT ×250₽ плюс отдельные 10млн₽; исходная СЧА20млн₽. 250₽ — модельная отметка, не подтверждённая цена покупки. Origin25.09.2026 18:50МСК имеет ограниченную причинную квалификацию; октябрьские сведения нельзя использовать для сентябрьского решения.

Четыре семейства совместного риска, синхронные блоки10сессий и горизонт126сессий связаны с полным контролем кандидата, поздним исполнением и непрерывным учётом. Бюджет ES95≤15%СЧА действует при допуске; ≥50% вне денежного рынка — на исполнимой активной цели; лимит ОФЗ≤40%исходной СЧА — на применимых активных стадиях. Обязательны деньги, целые контракты, IM/VM, ликвидность, расходы и расчёты. Намерение в t не означает исполнение по тому же закрытию. BTC/RUB — учебное допущение прямого доступа с прокси BTC/USDT×USD/RUB.

## Среда и синтетические примеры
Требуется Python3.12, диапазон зафиксирован в `pyproject.toml`; версии зависимостей — в `uv.lock`. Основные зависимости: NumPy, pandas, SciPy, scikit-learn, matplotlib, openpyxl, PyYAML, requests, tabulate; разработка — pytest. Их код не включён в экспорт, лицензии зависимостей сохраняются у поставщиков.

Для чистого окружения и ограниченной синтетической проверки:

```sh
uv sync --frozen --group dev
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --offline --frozen python -m pytest tests/test_data.py tests/test_options.py -q -p no:cacheprovider
```

Два выбранных модуля используют построенные в коде массивы/таблицы и временный каталог. Они иллюстрируют выплаты/проверки данных, хранение и формулы опционов. Ещё два включённых модуля содержат синтетические примеры риска, оптимизации и исторических хеджевых реализаций; они не доказывают экономику текущей политики. Не запускайте автоматически все библиотечные пути или сетевые сборщики. Установщик зависимостей обращается к публичному реестру пакетов; сами указанные примеры не требуют рыночных файлов.

## Что проверено и чего не утверждается
В Task179 исходники проверены статическим разбором синтаксиса и сравнением хешей, без импорта библиотеки, запуска pytest, Quant, AWS или экономических экспериментов. Это историческая область проверки исходного экспорта. Добавленный публичный CI ограничен двумя синтетическими модулями выше и сравнением файлов с манифестом; его результат относится к конкретному проверенному коммиту. Это не завершённый G2. Исторические 2 175 собранных случаев и частичные 9PASS/1FAIL относятся к прежней ограниченной проверке R08, не к этому экспорту. Принятие исходников/условного маршрута R10 не является полным нативным запуском. Рыночного OOS, доказанной прибыльности, действующей брокерской интеграции или правовой пригодности реального ПИФа выпуск не заявляет.

## Права
Открытая лицензия пока не выбрана. Все права сохраняются за соответствующими правообладателями; этот текст не выдаёт лицензию на повторное использование. Публичный доступ к экспорту не заменяет согласование прав; выбор открытой лицензии требует отдельного решения владельца с учётом соавторства. Вопросы прав на рыночные данные не решаются раскрытием кода; данные отсутствуют.
