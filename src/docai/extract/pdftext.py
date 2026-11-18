"""Текстовый путь: текст и таблицы из текстового слоя PDF (pdfplumber) с сохранением порядка.

Страница режется по вертикали на полосы «текст — таблица — текст …». Текст вне
таблиц берётся как есть, таблица — одним из двух способов (`TableStyle`):
- grid — строками вида `| ячейка | ячейка |`: так модель видит, какие числа стоят
  в одной строке и в одном столбце (P9);
- labeled — табличная часть документа построчно «графа: значение» (P13, D52): шапка
  из одной-двух строк склеивается в подпись графы, строка с номерами граф («А | Б | 1 | 1а»)
  и пустые ячейки («», «--») отбрасываются. Модели не нужно считать столбцы: в широкой
  таблице УПД (16 граф) она сдвигала значения на графу. Таблицы без шапки (банковские
  реквизиты, стороны) остаются в виде grid;
- labeled_v2 — то же, но строки «Итого», «Всего по накладной» и т. п. (без номера позиции)
  выводятся отдельно: «Итоговая строка «Итого»: …», а позиции — «Позиция N: …». В labeled
  они выглядели как позиции, и модель добавляла их в lines (эксперимент p13_val_e3_tables).
"""

import re
from pathlib import Path
from typing import Any, Literal

import pdfplumber

TableStyle = Literal["grid", "labeled", "labeled_v2"]


def _cell(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split()).replace("|", "/")


def table_to_text(rows: list[list[str | None]]) -> str:
    """Таблица → строки `| a | b |`; полностью пустые строки пропускаются."""
    lines = []
    for row in rows:
        cells = [_cell(c) for c in row]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Ячейка с номером графы: «А», «Б», «1», «1а», «10а».
_COLUMN_NUMBER = re.compile(r"^(?:[А-ЯA-Z]|\d{1,2}[а-яa-z]?)$")
_ROW_NUMBER = re.compile(r"^\d{1,4}\.?$")
_EMPTY = frozenset({"", "-", "--", "—", "–"})
_HYPHEN_BREAK = re.compile(r"(?<=\w) [-‐] (?=\w)")
_PARENS = re.compile(r"\s*\([^()]*\)")


def _short(label: str) -> str:
    """Подпись графы короче: без переносов «наимено - вание» и пояснений в скобках."""
    short = _PARENS.sub("", _HYPHEN_BREAK.sub("", label))
    return " ".join(short.replace(" ,", ",").split()) or label


def _is_column_numbers(cells: list[str]) -> bool:
    filled = [c for c in cells if c]
    return len(filled) >= 3 and all(_COLUMN_NUMBER.match(c) for c in filled)


def _is_data_row(cells: list[str]) -> bool:
    """Строка позиции: начинается с номера по порядку и это не строка номеров граф."""
    first = next((c for c in cells if c), "")
    return bool(_ROW_NUMBER.match(first)) and not _is_column_numbers(cells)


def _labels(header: list[list[str]], width: int) -> list[str]:
    """Подписи граф из строк шапки: «верхняя / нижняя». Объединённая ячейка верхней строки
    (пусто справа от подписи при непустой нижней) продолжает подпись слева."""
    parts: list[list[str]] = [[] for _ in range(width)]
    for row in header:
        carry = ""
        for i in range(width):
            cell = row[i] if i < len(row) else ""
            below = any(r[i] for r in header[header.index(row) + 1 :] if i < len(r))
            if cell:
                carry = cell
                parts[i].append(cell)
            elif carry and below:
                parts[i].append(carry)
            else:
                carry = ""
    labels = []
    for i, p in enumerate(parts):
        label = _short(" / ".join(dict.fromkeys(p)))
        labels.append(label or f"графа {i + 1}")
    return labels


def table_to_labeled(rows: list[list[str | None]], *, mark_totals: bool = False) -> str | None:
    """Табличная часть → строки «графа: значение; …». None — у таблицы нет шапки и строк.

    mark_totals: позиции — «Позиция N: …», строки без номера (итоги) — «Итоговая строка
    «<текст>»: …» без ячеек «X» (labeled_v2).
    """
    cleaned = [[_cell(c) for c in row] for row in rows]
    cleaned = [r for r in cleaned if any(r)]
    start = next((i for i, r in enumerate(cleaned) if _is_data_row(r)), None)
    if not start or len(cleaned[0]) < 4:
        return None
    header = [r for r in cleaned[:start] if not _is_column_numbers(r)]
    if not header:
        return None
    width = max(len(r) for r in cleaned)
    labels = _labels(header, width)
    lines = []
    for row in cleaned[start:]:
        if _is_column_numbers(row):
            continue
        if not mark_totals:
            pairs = [f"{labels[i]}: {v}" for i, v in enumerate(row) if v not in _EMPTY]
            if pairs:
                lines.append("- " + "; ".join(pairs))
            continue
        first = next((i for i, c in enumerate(row) if c), None)
        if first is None:
            continue
        rest = [
            f"{labels[i]}: {v}"
            for i, v in enumerate(row)
            if i != first and v not in _EMPTY and v.upper() != "X"
        ]
        if _is_data_row(row):
            lines.append(f"Позиция {row[first].rstrip('.')}: " + "; ".join(rest))
        else:
            lines.append(f"Итоговая строка «{row[first]}»: " + "; ".join(rest))
    return "\n".join(lines)


def render_table(rows: list[list[str | None]], style: TableStyle = "grid") -> str:
    if style in ("labeled", "labeled_v2"):
        labeled = table_to_labeled(rows, mark_totals=style == "labeled_v2")
        if labeled is not None:
            return labeled
    return table_to_text(rows)


def _inside(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def page_text(page: pdfplumber.page.Page, tables_style: TableStyle = "grid") -> str:
    """Текст одной страницы: полосы текста и таблицы в порядке сверху вниз."""
    tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
    if not tables:
        return (page.extract_text() or "").strip()

    boxes = [t.bbox for t in tables]
    outside = page.filter(
        lambda obj: not any(_inside(obj, b) for b in boxes) if "x0" in obj else True
    )
    parts: list[str] = []
    top = 0.0
    for table in tables:
        _, t_top, _, t_bottom = table.bbox
        if t_top > top:
            band = outside.crop((0, top, page.width, t_top), strict=False)
            text = (band.extract_text() or "").strip()
            if text:
                parts.append(text)
        rendered = render_table(table.extract(), tables_style)
        if rendered:
            parts.append(rendered)
        top = max(top, t_bottom)
    if top < page.height:
        band = outside.crop((0, top, page.width, page.height), strict=False)
        text = (band.extract_text() or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def pdf_pages_text(path: Path, tables_style: TableStyle = "grid") -> list[str]:
    """Текст каждой страницы PDF."""
    with pdfplumber.open(path) as pdf:
        return [page_text(page, tables_style) for page in pdf.pages]
