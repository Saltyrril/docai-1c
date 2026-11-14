"""Текстовый путь: текст и таблицы из текстового слоя PDF (pdfplumber) с сохранением порядка.

Страница режется по вертикали на полосы «текст — таблица — текст …». Текст вне
таблиц берётся как есть, таблица — строками вида `| ячейка | ячейка |`: так модель
видит, какие числа стоят в одной строке и в одном столбце.
"""

from pathlib import Path
from typing import Any

import pdfplumber


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


def _inside(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def page_text(page: pdfplumber.page.Page) -> str:
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
        rendered = table_to_text(table.extract())
        if rendered:
            parts.append(rendered)
        top = max(top, t_bottom)
    if top < page.height:
        band = outside.crop((0, top, page.width, page.height), strict=False)
        text = (band.extract_text() or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def pdf_pages_text(path: Path) -> list[str]:
    """Текст каждой страницы PDF."""
    with pdfplumber.open(path) as pdf:
        return [page_text(page) for page in pdf.pages]
