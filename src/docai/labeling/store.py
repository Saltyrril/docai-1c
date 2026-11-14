"""Хранилище разметки: исходные файлы, эталонные JSON и служебные сведения.

Раскладка (всё внутри `data/real/`, которая не коммитится):

    data/real/files/<id>.pdf|jpg|png|...   исходник документа
    data/real/files/<id>/                  или папка со страницами-картинками (1.jpg, 2.jpg ...)
    data/real/labels/<id>.json             эталон в нашей схеме (только поля схемы)
    data/real/labels/_meta/<id>.json       служебное: вид съёмки, время, замечания проверок

Эталон не содержит служебных полей, чтобы его можно было читать `parse_document` и
сравнивать с ответом модели так же, как синтетическую разметку.
"""

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from docai.schemas.documents import AnyDocument, parse_document

DEFAULT_ROOT = Path("data/real")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
PDF_SUFFIX = ".pdf"
SOURCE_SUFFIXES = IMAGE_SUFFIXES | {PDF_SUFFIX}
# Длинная сторона картинки в интерфейсе. Для разметки глазами хватает; модель получит свою.
DISPLAY_MAX_SIDE = 2200
PDF_DPI = 130

CAPTURE_KINDS = ("PDF (цифровой)", "скан", "фото", "скриншот", "другое")

SourceKind = Literal["pdf", "image", "folder"]


class StoreError(RuntimeError):
    """Не удалось прочитать или записать данные разметки."""


@dataclass(frozen=True, slots=True)
class Source:
    """Один документ для разметки: файл или папка со страницами."""

    id: str
    path: Path
    kind: SourceKind


def _page_files(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: (len(p.stem), p.stem),  # 2.jpg раньше 10.jpg
    )


def _load_image(path: Path) -> Image.Image:
    """Открыть картинку: повернуть по EXIF (фото с телефона), привести к RGB."""
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        image.load()
    return _fit(image.convert("RGB"))


def _fit(image: Image.Image) -> Image.Image:
    longest = max(image.size)
    if longest > DISPLAY_MAX_SIDE:
        scale = DISPLAY_MAX_SIDE / longest
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    return image


class LabelStore:
    """Работа с папкой `data/real`: список документов, страницы, сохранение эталонов."""

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.labels_dir = self.root / "labels"
        self.meta_dir = self.labels_dir / "_meta"

    # ---- исходники ---------------------------------------------------------------------

    def list_sources(self) -> list[Source]:
        """Документы из `files/` по имени. Два исходника с одним именем — ошибка."""
        if not self.files_dir.is_dir():
            return []
        found: dict[str, Source] = {}
        for path in sorted(self.files_dir.iterdir(), key=lambda p: p.name.lower()):
            if path.is_dir() and _page_files(path):
                source = Source(path.name, path, "folder")
            elif path.suffix.lower() == PDF_SUFFIX:
                source = Source(path.stem, path, "pdf")
            elif path.suffix.lower() in IMAGE_SUFFIXES:
                source = Source(path.stem, path, "image")
            else:
                continue
            if source.id in found:
                raise StoreError(
                    f"Два исходника с одним именем «{source.id}»: "
                    f"{found[source.id].path.name} и {path.name}. Переименуйте один из них."
                )
            found[source.id] = source
        return list(found.values())

    def get_source(self, doc_id: str) -> Source:
        for source in self.list_sources():
            if source.id == doc_id:
                return source
        raise StoreError(f"Нет исходного файла для «{doc_id}» в {self.files_dir}")

    def page_count(self, source: Source) -> int:
        if source.kind == "pdf":
            pdf = pdfium.PdfDocument(source.path)
            try:
                return len(pdf)
            finally:
                pdf.close()
        if source.kind == "folder":
            return len(_page_files(source.path))
        return 1

    def render_page(self, source: Source, page: int = 0) -> Image.Image:
        """Страница для показа в интерфейсе (нумерация с нуля)."""
        if source.kind == "image":
            return _load_image(source.path)
        if source.kind == "folder":
            return _load_image(_page_files(source.path)[page])
        pdf = pdfium.PdfDocument(source.path)
        try:
            return _fit(pdf[page].render(scale=PDF_DPI / 72).to_pil().convert("RGB"))
        finally:
            pdf.close()

    def add_files(self, paths: list[Path]) -> list[str]:
        """Скопировать файлы в `files/`. Существующие файлы не перезаписываются."""
        import shutil

        self.files_dir.mkdir(parents=True, exist_ok=True)
        added: list[str] = []
        for path in paths:
            path = Path(path)
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            target = self.files_dir / path.name
            counter = 2
            while target.exists():
                target = self.files_dir / f"{path.stem}_{counter}{path.suffix}"
                counter += 1
            shutil.copy2(path, target)
            added.append(target.stem)
        return added

    # ---- эталоны -----------------------------------------------------------------------

    def _label_path(self, doc_id: str) -> Path:
        if not doc_id or Path(doc_id).name != doc_id or doc_id.startswith("_"):
            raise StoreError(f"Недопустимый идентификатор документа: «{doc_id}»")
        return self.labels_dir / f"{doc_id}.json"

    def is_labeled(self, doc_id: str) -> bool:
        return self._label_path(doc_id).is_file()

    def load_label(self, doc_id: str) -> AnyDocument | None:
        path = self._label_path(doc_id)
        if not path.is_file():
            return None
        return parse_document(path.read_text(encoding="utf-8"))

    def load_meta(self, doc_id: str) -> dict[str, Any]:
        path = self.meta_dir / f"{self._label_path(doc_id).stem}.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, doc_id: str, document: AnyDocument, meta: dict[str, Any] | None = None) -> Path:
        """Записать эталон (и служебные сведения) атомарно: либо всё, либо ничего."""
        path = self._label_path(doc_id)
        payload = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)
        _write_atomic(path, payload + "\n")
        if meta is not None:
            record = {"saved_at": dt.datetime.now().isoformat(timespec="seconds"), **meta}
            meta_text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
            _write_atomic(self.meta_dir / path.name, meta_text)
        return path

    def label_ids(self) -> list[str]:
        if not self.labels_dir.is_dir():
            return []
        return sorted(p.stem for p in self.labels_dir.glob("*.json") if not p.name.startswith("_"))

    def progress(self) -> tuple[int, int]:
        """(размечено, всего исходников)."""
        sources = self.list_sources()
        done = sum(1 for s in sources if self.is_labeled(s.id))
        return done, len(sources)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
