"""Геометрия: гомографии 3×3 и перенос прямоугольников полей вместе с картинкой.

Все гомографии заданы в «непрерывных» координатах: левый верхний угол левого верхнего
пикселя — (0, 0), центр пикселя — (0.5, 0.5). Так же заданы bbox из разметки.
OpenCV считает центр пикселя в целых координатах, поэтому для warp матрица сопрягается
со сдвигом на полпикселя (`to_cv`).
"""

import math

import numpy as np

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


def identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def to_cv(h: np.ndarray) -> np.ndarray:
    """Матрица для cv2.warpPerspective из матрицы в непрерывных координатах."""
    shift_in = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64)
    shift_out = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], dtype=np.float64)
    return shift_out @ h @ shift_in


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def translate_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def rotate_matrix(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Поворот на angle_deg градусов вокруг (cx, cy); положительный — по часовой стрелке."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]],
        dtype=np.float64,
    )


def quad_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Гомография, переводящая четыре точки src в dst (оба массива 4×2)."""
    return _homography(src, dst)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Точное решение системы 8×8 в float64 (cv2.getPerspectiveTransform считает в float32)."""
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    solution = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    return np.append(solution, 1.0).reshape(3, 3)


def apply_points(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Применить гомографию к точкам N×2."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def box_corners(box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def transform_box(h: np.ndarray, box: Box) -> tuple[Box, np.ndarray]:
    """Образ прямоугольника: (охватывающий осевой прямоугольник, четырёхугольник 4×2)."""
    quad = apply_points(h, box_corners(box))
    return (
        (
            float(quad[:, 0].min()),
            float(quad[:, 1].min()),
            float(quad[:, 0].max()),
            float(quad[:, 1].max()),
        ),
        quad,
    )


def clip_box(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    return (
        min(max(x0, 0.0), width),
        min(max(y0, 0.0), height),
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
    )


def scale_about(s: float, cx: float, cy: float) -> np.ndarray:
    """Масштаб s относительно точки (cx, cy)."""
    return translate_matrix(cx, cy) @ scale_matrix(s, s) @ translate_matrix(-cx, -cy)
