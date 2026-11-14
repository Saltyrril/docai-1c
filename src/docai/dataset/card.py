"""Статистика собранного датасета и карточка `docs/DATASET.md` (таблицы считаются по манифесту)."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from docai.dataset.build import BUILD_INFO_NAME, MANIFEST_NAME
from docai.dataset.split import SPLITS

LINE_BUCKETS = ((1, 3), (4, 8), (9, 15), (16, 25))
TYPE_TITLES = {
    "upd": "УПД",
    "bill": "Счёт на оплату",
    "act": "Акт",
    "torg12": "ТОРГ-12",
    "vat_invoice": "Счёт-фактура",
}


def read_manifest(dataset: Path) -> list[dict[str, Any]]:
    text = (dataset / MANIFEST_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _bucket(n: int) -> str:
    for low, high in LINE_BUCKETS:
        if low <= n <= high:
            return f"{low}–{high}"
    return f"{LINE_BUCKETS[-1][1]}+"


def compute_stats(dataset: Path) -> dict[str, Any]:
    rows = read_manifest(dataset)
    docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        docs.setdefault(row["doc_id"], row)
    json_docs = {
        doc_id: json.loads((dataset / row["json"]).read_text(encoding="utf-8"))
        for doc_id, row in docs.items()
    }

    by_type_split: dict[str, Counter[str]] = defaultdict(Counter)
    lines_by_type: dict[str, list[int]] = defaultdict(list)
    bucket_split: dict[str, Counter[str]] = defaultdict(Counter)
    line_rates: Counter[str] = Counter()
    doc_rates: Counter[str] = Counter()
    rate_split: dict[str, Counter[str]] = defaultdict(Counter)
    years: Counter[int] = Counter()
    sellers: dict[str, set[str]] = defaultdict(set)
    buyers_in_other: Counter[str] = Counter()
    seller_split = {row["seller_inn"]: row["split"] for row in docs.values()}
    for doc_id, row in docs.items():
        data = json_docs[doc_id]
        split = row["split"]
        by_type_split[row["doc_type"]][split] += 1
        n_lines = len(data["lines"])
        lines_by_type[row["doc_type"]].append(n_lines)
        bucket_split[_bucket(n_lines)][split] += 1
        rates = [line.get("vat_rate") or "—" for line in data["lines"]]
        line_rates.update(rates)
        for rate in set(rates):
            doc_rates[rate] += 1
            rate_split[rate][split] += 1
        years[int(data["date"][:4])] += 1
        sellers[split].add(row["seller_inn"])
        buyer_split = seller_split.get(data["buyer"]["inn"])
        if buyer_split is not None and buyer_split != split:
            buyers_in_other[split] += 1

    split_docs = Counter(row["split"] for row in docs.values())
    split_errors = Counter(row["split"] for row in docs.values() if row["has_errors"])
    pages = Counter(row["pages"] for row in docs.values())
    sizes = {name: _dir_size(dataset / name) for name in ("json", "pdf", "images")}
    return {
        "info": json.loads((dataset / BUILD_INFO_NAME).read_text(encoding="utf-8")),
        "documents": len(docs),
        "rows": len(rows),
        "images": sum(len(r["images"]) for r in rows),
        "split_docs": dict(split_docs),
        "split_sellers": {s: len(sellers[s]) for s in SPLITS},
        "split_errors": dict(split_errors),
        "by_type_split": {t: dict(c) for t, c in by_type_split.items()},
        "lines_by_type": {
            t: {"mean": sum(v) / len(v), "min": min(v), "max": max(v)}
            for t, v in lines_by_type.items()
        },
        "bucket_split": {b: dict(c) for b, c in bucket_split.items()},
        "line_rates": dict(line_rates),
        "doc_rates": dict(doc_rates),
        "rate_split": {r: dict(c) for r, c in rate_split.items()},
        "years": dict(years),
        "pages": dict(pages),
        "level_images": dict(
            Counter(
                {
                    lvl: sum(len(r["images"]) for r in rows if r["level"] == lvl)
                    for lvl in {r["level"] for r in rows}
                }
            )
        ),
        "error_kinds": dict(Counter(k for r in docs.values() for k in r["error_kinds"])),
        "buyer_in_other_split": dict(buyers_in_other),
        "sizes_bytes": sizes,
    }


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "—"


def _table(header: list[str], body: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines)


def _gib(value: int) -> str:
    return f"{value / 1024**3:.2f} ГиБ" if value >= 1024**3 else f"{value / 1024**2:.0f} МиБ"


def _split_table(title: str, groups: dict[str, dict[str, int]], order: list[str]) -> str:
    header = [title, *SPLITS, "всего", "доля"]
    total = sum(sum(v.values()) for v in groups.values())
    body = []
    for key in order:
        counts = groups.get(key, {})
        all_ = sum(counts.values())
        body.append([key, *(str(counts.get(s, 0)) for s in SPLITS), str(all_), _pct(all_, total)])
    return _table(header, body)


def render_card(stats: dict[str, Any]) -> str:
    cfg = stats["info"]["config"]
    n = stats["documents"]
    sd = stats["split_docs"]
    rates_order = sorted(stats["line_rates"], key=lambda r: -stats["line_rates"][r])
    types_order = [t for t in TYPE_TITLES if t in stats["by_type_split"]]
    type_groups = {f"{TYPE_TITLES[t]} (`{t}`)": stats["by_type_split"][t] for t in types_order}
    buckets = [_bucket(low) for low, _ in LINE_BUCKETS]
    total_size = sum(stats["sizes_bytes"].values())
    split_rows = [
        [
            s,
            str(sd.get(s, 0)),
            _pct(sd.get(s, 0), n),
            str(stats["split_sellers"][s]),
            str(stats["split_errors"].get(s, 0)),
        ]
        for s in SPLITS
    ]
    lines_rows = [
        [
            f"{TYPE_TITLES[t]} (`{t}`)",
            f"{stats['lines_by_type'][t]['mean']:.1f}",
            str(stats["lines_by_type"][t]["min"]),
            str(stats["lines_by_type"][t]["max"]),
        ]
        for t in types_order
    ]
    rate_rows = [
        [
            f"`{r}`",
            str(stats["line_rates"][r]),
            _pct(stats["line_rates"][r], sum(stats["line_rates"].values())),
            str(stats["doc_rates"][r]),
            _pct(stats["doc_rates"][r], n),
        ]
        for r in rates_order
    ]
    other = stats["buyer_in_other_split"]
    leak = ", ".join(f"{s}: {other.get(s, 0)}" for s in SPLITS)
    return f"""# Карточка датасета {cfg["name"]} {cfg["version"]}

> Файл создан командой `uv run docai dataset card`; таблицы посчитаны по `manifest.jsonl` собранного датасета. Правьте тексты в `src/docai/dataset/card.py`, а не таблицы.

## Назначение

Синтетические первичные документы для обучения и проверки извлечения реквизитов с картинок: УПД, счёт на оплату, акт, ТОРГ-12, счёт-фактура. У каждого документа есть эталонный JSON по схеме `docai.schemas`, PDF с настоящим текстовым слоем, bbox каждого поля и картинки трёх уровней искажения («чисто», «скан», «фото»). Данные не содержат реальных организаций и персональных данных.

## Состав

- Документов: **{n}**. Строк манифеста: {stats["rows"]} (документ × уровень × dpi). Картинок: {stats["images"]}.
- Seed: {cfg["seed"]}. Доля документов с намеренной ошибкой (арифметика, ИНН): {cfg["error_rate"]:.0%}; ошибки названы в манифесте (`has_errors`, `error_kinds`), в эталонном JSON их пометок нет.
- Даты документов — 2025–2026 годы ({", ".join(f"{y}: {c}" for y, c in sorted(stats["years"].items()))}), ставка НДС проверяется по дате документа.
- Уровни: {", ".join(f"`{k}` — {v} картинок" for k, v in sorted(stats["level_images"].items()))}. Разрешение растеризации: {", ".join(str(d) for d in cfg["dpi"])} dpi. Страниц на документ: {", ".join(f"{k} стр. — {v}" for k, v in sorted(stats["pages"].items()))}.
- Размер на диске: JSON {_gib(stats["sizes_bytes"]["json"])}, PDF {_gib(stats["sizes_bytes"]["pdf"])}, картинки и их разметка {_gib(stats["sizes_bytes"]["images"])}; всего {_gib(total_size)}.

### Структура папки

```text
manifest.jsonl      одна строка на (документ, уровень, dpi)
build_info.json     конфиг сборки
json/               эталонные документы (+ _manifest.jsonl с метаданными генерации)
pdf/                <id>.pdf и <id>.layout.json (bbox полей в пунктах PDF)
images/             <id>_p<стр>_<уровень>_<dpi>dpi.(png|jpg) и .layout.json (bbox, quad в пикселях)
```

Поля строки манифеста: `id`, `doc_id`, `doc_type`, `split`, `seed`, `level`, `dpi`, `pages`, `json`, `pdf`, `pdf_layout`, `images`, `image_layouts`, `seller_inn`, `has_errors`, `error_kinds`. Пути относительны папки датасета. Seed картинки выводится из `seed`, имени документа, страницы, dpi и уровня (D24).

## Разбиение train / val / test

Разбиение идёт **по продавцам** (по ИНН): все документы одного продавца попадают в одну часть. Доли заданы по документам ({cfg["split"]["train"]:.0%} / {cfg["split"]["val"]:.0%} / {cfg["split"]["test"]:.0%}), фактические отличаются, потому что делится не документ, а продавец целиком.

{_table(["Часть", "Документов", "Доля", "Продавцов", "С ошибкой"], split_rows)}

Утечки по продавцу нет (проверяется тестом). Остаётся утечка по покупателю: организации в справочнике из ~350 штук выступают и продавцом, и покупателем, поэтому покупатель документа из части может быть продавцом документов из другой. Документов, у которых покупатель — продавец из другой части: {leak}. Для моделей, которым важны реквизиты именно продавца, этого достаточно; для строгой оценки по покупателю потребуется отдельное разбиение.

## Распределения

Документы по типам:

{_split_table("Тип", type_groups, list(type_groups))}

Число строк в таблице (документы):

{_split_table("Строк", stats["bucket_split"], buckets)}

{_table(["Тип", "в среднем", "минимум", "максимум"], lines_rows)}

Ставки НДС. «Строк» — число строк таблиц документов с этой ставкой, «документов» — в скольких документах есть хотя бы одна такая строка:

{_table(["Ставка", "Строк", "Доля строк", "Документов", "Доля документов"], rate_rows)}

Намеренные ошибки: {", ".join(f"`{k}` — {v}" for k, v in sorted(stats["error_kinds"].items())) or "нет"}.

## Как воспроизвести

```powershell
uv sync
uv run playwright install chromium   # один раз
uv run docai dataset build --config configs/dataset_v1.yaml
uv run docai dataset card --dataset data/datasets/v1 --out docs/DATASET.md
```

Тот же seed и та же версия кода дают побайтно тот же `manifest.jsonl` и те же JSON. Манифест намеренно не содержит хэшей PDF и картинок: байты PDF могут зависеть от версии Chromium и шрифтов системы. Версии библиотек закреплены в `uv.lock`.

## Лицензия и ограничения

- Данные полностью синтетические: названия организаций придуманы, ИНН и счета сгенерированы по алгоритму, а не взяты из реестров; совпадения случайны. Предлагаемая лицензия данных — CC BY 4.0. Лицензию кода владелец репозитория пока не выбрал.
- Датасет не коммитится (`data/` в `.gitignore`). PDF и картинки отрисованы системными шрифтами Windows (Times New Roman, Arial, Calibri), поэтому перед публикацией файлов PDF нужно проверить условия лицензий этих шрифтов; картинки и JSON с разметкой публикации не мешают.
- Печати, подписи и логотипы — заглушки; параметры искажений подобраны «на глаз» и на реальных сканах не калибровались (D24); формы повторяют структуру, но не пиксель в пиксель (D20). Результаты на синтетике нельзя переносить на реальные документы без проверки на `data/real`.
"""


def write_card(dataset: Path, out: Path) -> dict[str, Any]:
    stats = compute_stats(dataset)
    out.write_text(render_card(stats), encoding="utf-8", newline="\n")
    return stats
