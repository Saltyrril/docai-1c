# docai-1c

Локальный ИИ-сервис для первичных бухгалтерских документов и 1С:Бухгалтерии.

- **Извлечение реквизитов** из УПД, счетов на оплату, актов, ТОРГ-12 и счетов-фактур (vision-модель Qwen3.5-2B, работает локально через Ollama).
- **Проверка**: контрольные цифры ИНН, КПП, БИК, расчётный счёт, суммы и ставки НДС.
- **Создание документов в 1С** через внешнюю обработку.
- **RAG-поиск по договорам** с цитатами из исходного текста.

Проект одновременно научная работа: эксперименты воспроизводимы (seed, YAML-конфиг), метрики и отчёты лежат в `reports/`.

## Быстрый старт (Windows, PowerShell)

```powershell
uv sync
uv run pytest
uv run docai version
```

## Команды разработки

```powershell
uv sync                            # установить окружение по uv.lock
uv run pytest                      # тесты
uv run ruff check                  # линтер
uv run ruff format                 # форматирование
uv run pre-commit install          # один раз после клонирования
uv run pre-commit run --all-files  # все проверки перед коммитом
uv run docai --help                # CLI проекта
```

Шаг считается готовым, когда проходят `uv run pytest` и `uv run ruff check`.

## Структура

| Папка | Что внутри |
|---|---|
| `src/docai/schemas` | Модели документов (Pydantic) |
| `src/docai/validation` | Проверки ИНН, КПП, БИК, НДС |
| `src/docai/synth` | Генератор синтетических данных |
| `src/docai/render` | HTML-шаблоны → PDF → «сканы» |
| `src/docai/edo` | УПД в XML формата ФНС |
| `src/docai/extract` | Извлечение полей |
| `src/docai/rag` | Поиск по договорам |
| `src/docai/service` | HTTP-сервис (FastAPI) |
| `src/docai/eval` | Метрики и отчёты |
| `onec/` | Исходники внешней обработки 1С |
| `notebooks/` | Ноутбуки для Google Colab (дообучение) |
| `data/` | Данные: synth, real, rag (в git не попадают) |
| `reports/` | Результаты экспериментов |
| `docs/` | Прогресс, решения, эксперименты |
| `tests/` | Тесты |

Подробности — в [docs/PROGRESS.md](docs/PROGRESS.md) и [docs/DECISIONS.md](docs/DECISIONS.md).
