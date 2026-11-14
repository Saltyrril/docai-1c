"""Растеризация PDF в изображения через pypdfium2."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

DPI_CHOICES = (150, 200, 300)
PDF_DPI = 72.0  # пунктов PDF в дюйме


@dataclass(frozen=True)
class RasterPage:
    """Страница как картинка BGR (uint8) и коэффициенты перевода пунктов в пиксели."""

    image: np.ndarray
    page: int  # с 1
    dpi: int
    size_pt: tuple[float, float]
    scale: tuple[float, float]  # пикселей на пункт по x и по y


def rasterize_pdf(pdf: bytes | Path, dpi: int) -> list[RasterPage]:
    """Все страницы PDF в заданном разрешении. Фон белый, без прозрачности."""
    source = pdf.read_bytes() if isinstance(pdf, Path) else pdf
    document = pdfium.PdfDocument(source)
    pages: list[RasterPage] = []
    try:
        for index in range(len(document)):
            page = document[index]
            width_pt, height_pt = page.get_size()
            bitmap = page.render(scale=dpi / PDF_DPI)
            # Формат BGR по умолчанию, как у OpenCV. copy — буфер принадлежит bitmap.
            image = np.ascontiguousarray(bitmap.to_numpy()[:, :, :3])
            height_px, width_px = image.shape[:2]
            pages.append(
                RasterPage(
                    image=image,
                    page=index + 1,
                    dpi=dpi,
                    size_pt=(width_pt, height_pt),
                    # Реальный размер в пикселях округляется, поэтому коэффициенты считаем по нему.
                    scale=(width_px / width_pt, height_px / height_pt),
                )
            )
    finally:
        document.close()
    return pages
