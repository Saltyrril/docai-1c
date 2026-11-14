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
uv run playwright install chromium   # один раз: браузер для печатных форм
uv run pytest
uv run docai version
```

## Печатные формы (PDF)

```powershell
uv run docai synth generate --n 100 --seed 42 --out data/synth/json
uv run docai render --in data/synth/json --out data/synth/pdf
```

Для каждого `NNNNN_<тип>.json` создаются `NNNNN_<тип>.pdf` и `NNNNN_<тип>.layout.json`
(страница и bbox каждого поля, в пунктах PDF). Вариант вёрстки и заглушки логотипа, печати и подписи
выбираются по `--seed`; `--variant 1|2|3` фиксирует вариант, `--limit N` и `--stem <имя>` ограничивают набор.
Chromium хранится в `%LOCALAPPDATA%\ms-playwright`, в репозиторий не входит.

## Извлечение полей

```powershell
uv run docai extract path\to\file.pdf                 # auto: XML → парсер, PDF с текстом → text, иначе vision
uv run docai extract scan.jpg --mode vision --out result.json
uv run docai extract p1.jpg p2.jpg                    # несколько картинок — страницы одного документа
uv run docai extract file.pdf --mode ocr              # для сравнения: Tesseract → текст → LLM
```

Нужна Ollama с моделью `qwen3.5:2b-q8_0`; для `--mode ocr` — Tesseract с русским языком:
`winget install --id UB-Mannheim.TesseractOCR -e` и `rus.traineddata` в папку `tessdata`.
На выходе JSON: `document` (по схеме `docai.schemas.Document`), `warnings`, `errors` и `meta`
(маршрут, версия промпта, время этапов, токены, пиковая VRAM). Настройки — `configs/extract.yaml`.

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
| `src/docai/labeling` | Разметка реальных документов (Gradio) и обезличивание |
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
