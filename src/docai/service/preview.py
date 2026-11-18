"""Картинки страниц для формы проверки в 1С: `preview_<N>.png` рядом с исходным файлом.

Рисуются в потоке воркера сразу после извлечения: pdfium не рассчитан на параллельные
вызовы из разных потоков, а воркер — единственный, кто его использует.
"""

import logging
from pathlib import Path

import pypdfium2 as pdfium

from docai.extract.images import fit_long_side, image_files_pages
from docai.extract.router import InputKind

log = logging.getLogger("docai.service")


def preview_path(job_dir: Path, page: int) -> Path:
    return job_dir / f"preview_{page}.png"


def render_previews(
    source: Path, kind: InputKind, job_dir: Path, max_side: int, max_pages: int
) -> int:
    """Сохранить страницы как PNG; вернуть число сохранённых. XML картинок не даёт."""
    if kind is InputKind.XML:
        return 0
    try:
        if kind is InputKind.PDF:
            return _render_pdf(source, job_dir, max_side, max_pages)
        page = fit_long_side(image_files_pages([source])[0], max_side)
        page.save(preview_path(job_dir, 1), format="PNG")
        return 1
    except Exception:
        log.exception("не удалось нарисовать страницы для превью")
        return 0


def _render_pdf(source: Path, job_dir: Path, max_side: int, max_pages: int) -> int:
    document = pdfium.PdfDocument(source)
    try:
        count = min(len(document), max_pages)
        for index in range(count):
            page = document[index]
            width_pt, height_pt = page.get_size()
            scale = max_side / max(width_pt, height_pt)
            image = page.render(scale=scale).to_pil().convert("RGB")
            image.save(preview_path(job_dir, index + 1), format="PNG")
        return count
    finally:
        document.close()
