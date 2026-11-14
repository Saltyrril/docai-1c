"""Растеризация PDF в PNG: uv run --with pillow python png.py <pdf> <префикс> [масштаб]."""

import sys

import pypdfium2 as pdfium

pdf = pdfium.PdfDocument(sys.argv[1])
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.6
for i in range(len(pdf)):
    pdf[i].render(scale=scale).to_pil().save(f"{sys.argv[2]}_p{i + 1}.png")
