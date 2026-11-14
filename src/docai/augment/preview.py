"""Просмотр результата: рамки полей поверх картинки и сетка миниатюр."""

from typing import Any

import cv2
import numpy as np

FRAME = (0, 0, 255)  # красный (BGR)
QUAD = (0, 160, 0)  # зелёный


def draw_fields(image: np.ndarray, layout: dict[str, Any], thickness: int = 1) -> np.ndarray:
    """Копия картинки с рамками: красный — bbox, зелёный — точный четырёхугольник поля."""
    canvas = image.copy()
    for entry in layout["fields"]:
        x0, y0, x1, y1 = (round(v) for v in entry["bbox"])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), FRAME, thickness)
        quad = np.round(np.array(entry["quad"])).astype(np.int32)
        cv2.polylines(canvas, [quad], isClosed=True, color=QUAD, thickness=thickness)
    return canvas


def make_grid(
    images: list[np.ndarray], cols: int = 3, cell: tuple[int, int] = (520, 700), gap: int = 12
) -> np.ndarray:
    """Сетка из cols столбцов; картинка вписывается в ячейку cell=(ширина, высота)."""
    rows = -(-len(images) // cols)
    cw, ch = cell
    sheet = np.full((rows * ch + (rows + 1) * gap, cols * cw + (cols + 1) * gap, 3), 235, np.uint8)
    for index, image in enumerate(images):
        h, w = image.shape[:2]
        k = min(cw / w, ch / h)
        small = cv2.resize(
            image, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA
        )
        r, c = divmod(index, cols)
        y = gap + r * (ch + gap) + (ch - small.shape[0]) // 2
        x = gap + c * (cw + gap) + (cw - small.shape[1]) // 2
        sheet[y : y + small.shape[0], x : x + small.shape[1]] = small
    return sheet
