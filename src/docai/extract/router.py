"""Маршрутизатор: какой путь извлечения выбрать для входного файла.

Правила (режим auto):
- XML формата ФНС → парсер `docai.edo` без LLM;
- PDF, в каждой странице которого есть текстовый слой → текстовый путь;
- PDF без текста (хотя бы одна страница — скан) или картинка → vision-путь.

Тип файла определяется по первым байтам, а не по расширению: «скан.pdf», который
на деле JPEG, уйдёт по пути картинки.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pypdfium2 as pdfium


class InputKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    XML = "xml"


class Mode(StrEnum):
    """Режим, который просит пользователь."""

    AUTO = "auto"
    TEXT = "text"
    VISION = "vision"
    OCR = "ocr"


class Route(StrEnum):
    """Путь, по которому реально пойдёт документ."""

    XML = "xml"
    TEXT = "text"
    VISION = "vision"
    OCR = "ocr"


class RoutingError(ValueError):
    """Файл не подходит для выбранного режима или не распознан."""


_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",  # JPEG
    b"II*\x00",  # TIFF (little-endian)
    b"MM\x00*",  # TIFF (big-endian)
    b"BM",  # BMP
)


def sniff_kind(head: bytes) -> InputKind:
    """Тип файла по первым байтам (достаточно 64)."""
    if head.startswith(b"%PDF-"):
        return InputKind.PDF
    if head.startswith(_IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return InputKind.IMAGE
    text = head.lstrip(b"\xef\xbb\xbf").lstrip()
    if text.startswith(b"<"):
        return InputKind.XML
    raise RoutingError("неизвестный формат: ожидаются PDF, PNG/JPEG/TIFF/BMP/WEBP или XML")


def read_kind(path: Path) -> InputKind:
    with path.open("rb") as file:
        head = file.read(64)
    try:
        return sniff_kind(head)
    except RoutingError as exc:
        raise RoutingError(f"{path.name}: {exc}") from exc


def pdf_text_chars(path: Path) -> list[int]:
    """Число непробельных символов текстового слоя на каждой странице PDF."""
    document = pdfium.PdfDocument(path)
    try:
        counts = []
        for page in document:
            text = page.get_textpage().get_text_range()
            counts.append(sum(1 for ch in text if not ch.isspace()))
        return counts
    finally:
        document.close()


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    kind: InputKind
    reason: str
    text_chars: tuple[int, ...] = ()  # по страницам, только для PDF


def choose_route(
    kind: InputKind,
    mode: Mode,
    text_chars: Sequence[int] = (),
    min_text_chars: int = 30,
) -> RouteDecision:
    """Чистое решение без чтения файлов: по типу входа, режиму и тексту страниц PDF."""
    chars = tuple(text_chars)
    has_text = bool(chars) and all(c >= min_text_chars for c in chars)

    if kind is InputKind.XML:
        if mode not in (Mode.AUTO,):
            raise RoutingError(
                f"XML разбирается парсером ЭДО без LLM; режим {mode} к нему неприменим"
            )
        return RouteDecision(Route.XML, kind, "XML: парсер формата ФНС без LLM")

    if mode is Mode.AUTO:
        if kind is InputKind.PDF and has_text:
            reason = f"PDF с текстовым слоем (символов по страницам: {list(chars)})"
            return RouteDecision(Route.TEXT, kind, reason, chars)
        if kind is InputKind.PDF:
            reason = (
                f"PDF без текстового слоя (символов по страницам: {list(chars)}, "
                f"порог {min_text_chars})"
            )
            return RouteDecision(Route.VISION, kind, reason, chars)
        return RouteDecision(Route.VISION, kind, "изображение", chars)

    if mode is Mode.TEXT:
        if kind is not InputKind.PDF:
            raise RoutingError(
                "текстовый путь работает только с PDF; для картинки — vision или ocr"
            )
        if not has_text:
            raise RoutingError(
                f"в PDF нет текстового слоя (символов по страницам: {list(chars)}); "
                "используйте --mode vision или --mode ocr"
            )
        return RouteDecision(Route.TEXT, kind, "режим text", chars)

    route = Route.VISION if mode is Mode.VISION else Route.OCR
    return RouteDecision(route, kind, f"режим {mode}", chars)


def route_inputs(paths: Sequence[Path], mode: Mode, min_text_chars: int = 30) -> RouteDecision:
    """Решение для одного документа: один файл или несколько картинок-страниц."""
    if not paths:
        raise RoutingError("не указан входной файл")
    for path in paths:
        if not path.is_file():
            raise RoutingError(f"файл не найден: {path}")
    kinds = [read_kind(p) for p in paths]
    if len(paths) > 1 and any(k is not InputKind.IMAGE for k in kinds):
        raise RoutingError(
            "несколько файлов допустимы только как картинки-страницы одного документа"
        )
    kind = kinds[0]
    chars = pdf_text_chars(paths[0]) if kind is InputKind.PDF else ()
    return choose_route(kind, mode, chars, min_text_chars)
