"""Страницы документа как картинки: из PDF (pypdfium2) или файлов изображений (Pillow)."""

import base64
import io
from collections.abc import Sequence
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from docai.augment.raster import PDF_DPI


def pdf_pages_images(path: Path, dpi: int) -> list[Image.Image]:
    """Все страницы PDF в RGB с заданным разрешением."""
    document = pdfium.PdfDocument(path)
    try:
        return [page.render(scale=dpi / PDF_DPI).to_pil().convert("RGB") for page in document]
    finally:
        document.close()


def pdf_pages_fit(path: Path, max_side: int) -> list[Image.Image]:
    """Страницы PDF, отрисованные сразу под нужную длинную сторону (без лишнего пересжатия)."""
    document = pdfium.PdfDocument(path)
    try:
        images = []
        for page in document:
            width_pt, height_pt = page.get_size()
            scale = max_side / max(width_pt, height_pt)
            images.append(page.render(scale=scale).to_pil().convert("RGB"))
        return images
    finally:
        document.close()


def image_files_pages(paths: Sequence[Path]) -> list[Image.Image]:
    """Картинки-страницы; у многостраничного TIFF каждая страница отдельно.

    Поворот по EXIF применяется: фото с телефона часто сохранены «боком» с меткой.
    """
    pages: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            for index in range(getattr(image, "n_frames", 1)):
                image.seek(index)
                pages.append(ImageOps.exif_transpose(image).convert("RGB"))
    return pages


def fit_long_side(image: Image.Image, max_side: int) -> Image.Image:
    """Уменьшить так, чтобы длинная сторона была не больше `max_side`. Не увеличивает."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    ratio = max_side / longest
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def to_base64_png(image: Image.Image) -> str:
    return base64.b64encode(to_png_bytes(image)).decode("ascii")
