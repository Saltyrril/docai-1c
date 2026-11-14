"""Примеры P5: сетки 3×3 по уровням и рамки полей поверх картинок.

Запуск из корня репозитория (нужны PDF из `docai render` в data/synth/pdf):
    uv run python reports/samples/make_samples.py
"""

import json
from pathlib import Path

import cv2

from docai.augment.levels import Level, augment_page
from docai.augment.photometric import decode
from docai.augment.pipeline import load_layout, make_rng, page_fields
from docai.augment.preview import draw_fields, make_grid
from docai.augment.raster import rasterize_pdf

PDF_DIR = Path("data/synth/pdf")
OUT = Path(__file__).parent
SEED = 42
DPIS = (150, 200, 300)
QUOTA = {"bill": 2, "upd": 2, "act": 2, "torg12": 2, "vat_invoice": 1}  # всего 9
OVERLAY_INDEXES = (0, 4, 7)  # три разных типа документа и три разных разрешения
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 82]


def pick_documents() -> list[Path]:
    left = dict(QUOTA)
    chosen: list[Path] = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        layout = load_layout(pdf)
        kind = layout["doc_type"] if layout else None
        if left.get(kind, 0) > 0:
            left[kind] -= 1
            chosen.append(pdf)
    return chosen


def main() -> None:
    documents = pick_documents()
    assert len(documents) == 9, documents
    manifest: dict[str, object] = {"seed": SEED, "documents": []}
    for level in Level:
        plain, overlays = [], []
        for index, pdf in enumerate(documents):
            dpi = DPIS[index % 3]
            layout = load_layout(pdf)
            page = rasterize_pdf(pdf, dpi)[0]
            fields = page_fields(layout, page)
            rng = make_rng(SEED, pdf.stem, page.page, dpi, level.value)
            result = augment_page(page.image, [f["box"] for f in fields], level, rng, dpi)
            image = decode(result.data)
            plain.append(image)
            if index in OVERLAY_INDEXES:
                marked = draw_fields(image, _as_layout(result), thickness=max(1, dpi // 150))
                half = 900 / marked.shape[1]
                marked = cv2.resize(marked, None, fx=half, fy=half, interpolation=cv2.INTER_AREA)
                name = f"bbox_{level.value}_{pdf.stem}_{dpi}dpi.jpg"
                cv2.imwrite(str(OUT / name), marked, JPEG)
                overlays.append(name)
            if level is Level.CLEAN:
                manifest["documents"].append(  # type: ignore[union-attr]
                    {"cell": index + 1, "pdf": pdf.name, "dpi": dpi, "type": layout["doc_type"]}
                )
        grid = make_grid(plain, cols=3, cell=(420, 590), gap=10)
        cv2.imwrite(str(OUT / f"grid_{level.value}.jpg"), grid, JPEG)
        print(level.value, "grid + overlays:", overlays)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


def _as_layout(result) -> dict:
    return {"fields": [{"bbox": list(b.bbox), "quad": b.quad.tolist()} for b in result.boxes]}


if __name__ == "__main__":
    main()
