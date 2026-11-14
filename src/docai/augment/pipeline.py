"""Сквозной проход: PDF + разметка полей → картинки и разметка в пикселях."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from docai.augment.levels import AugmentResult, Level, augment_page
from docai.augment.raster import RasterPage, rasterize_pdf

LAYOUT_SUFFIX = ".layout.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Written:
    image: Path
    layout: Path
    level: Level
    dpi: int
    page: int


def make_rng(seed: int, *parts: object) -> np.random.Generator:
    """Генератор, зависящий только от seed и «адреса» картинки (устойчив между запусками)."""
    key = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def load_layout(pdf_path: Path) -> dict[str, Any] | None:
    path = pdf_path.with_name(pdf_path.stem + LAYOUT_SUFFIX)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def page_fields(layout: dict[str, Any] | None, page: RasterPage) -> list[dict[str, Any]]:
    """Поля одной страницы с bbox, переведёнными из пунктов PDF в пиксели картинки."""
    if layout is None:
        return []
    sx, sy = page.scale
    result = []
    for entry in layout["fields"]:
        if entry["page"] != page.page:
            continue
        x0, y0, x1, y1 = entry["bbox"]
        result.append(
            {
                "field": entry["field"],
                "text": entry["text"],
                "box": (x0 * sx, y0 * sy, x1 * sx, y1 * sy),
            }
        )
    return result


def build_layout(
    *,
    source: Path,
    layout: dict[str, Any] | None,
    page: RasterPage,
    level: Level,
    seed: int,
    image_name: str,
    fields: list[dict[str, Any]],
    result: AugmentResult,
) -> dict[str, Any]:
    """Разметка искажённой картинки: bbox в пикселях, начало слева сверху, ось Y вниз."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": source.name,
        "doc_type": layout.get("doc_type") if layout else None,
        "variant": layout.get("variant") if layout else None,
        "level": level.value,
        "dpi": page.dpi,
        "page": page.page,
        "page_count": layout.get("page_count") if layout else None,
        "seed": seed,
        "text_layer": False,
        "image": {"file": image_name, "width": result.width, "height": result.height},
        "params": result.params,
        "homography": [[round(v, 8) for v in row] for row in result.homography.tolist()],
        "fields": [
            {
                "field": f["field"],
                "bbox": [round(v, 2) for v in moved.bbox],
                "quad": [[round(x, 2), round(y, 2)] for x, y in moved.quad.tolist()],
                "text": f["text"],
            }
            for f, moved in zip(fields, result.boxes, strict=True)
        ],
    }


def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    levels: list[Level],
    dpis: list[int],
    seed: int,
) -> list[Written]:
    """Растеризовать PDF и записать картинку и `.layout.json` для каждой комбинации."""
    layout = load_layout(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Written] = []
    for dpi in dpis:
        for page in rasterize_pdf(pdf_path, dpi):
            fields = page_fields(layout, page)
            for level in levels:
                rng = make_rng(seed, pdf_path.stem, page.page, dpi, level.value)
                result = augment_page(page.image, [f["box"] for f in fields], level, rng, dpi)
                base = f"{pdf_path.stem}_p{page.page}_{level.value}_{dpi}dpi"
                image_path = out_dir / f"{base}{result.extension}"
                layout_path = out_dir / f"{base}{LAYOUT_SUFFIX}"
                image_path.write_bytes(result.data)
                payload = build_layout(
                    source=pdf_path,
                    layout=layout,
                    page=page,
                    level=level,
                    seed=seed,
                    image_name=image_path.name,
                    fields=fields,
                    result=result,
                )
                layout_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                written.append(Written(image_path, layout_path, level, dpi, page.page))
    return written
