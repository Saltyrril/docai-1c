"""Фотометрические искажения: фон, свет, тень, размытие, шум, JPEG.

Всё здесь не двигает пиксели, поэтому bbox не пересчитываются. Работаем в float32 BGR (0..255).
"""

import cv2
import numpy as np

Rng = np.random.Generator


def smooth_field(rng: Rng, width: int, height: int, cells: int, amplitude: float) -> np.ndarray:
    """Плавное случайное поле около 0 (height×width): шум на сетке cells×cells, растянутый."""
    small = rng.standard_normal((cells, max(1, round(cells * width / height)))).astype(np.float32)
    field = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    field -= field.mean()
    peak = float(np.abs(field).max()) or 1.0
    return field / peak * amplitude


def make_background(
    rng: Rng, width: int, height: int, level: float, tint: float, texture: float, fine: float
) -> np.ndarray:
    """Фон вокруг листа: средний уровень, оттенок по каналам, крупная и мелкая текстура."""
    channel_gain = 1.0 + rng.uniform(-tint, tint, size=3).astype(np.float32)
    base = np.full((height, width, 3), level, dtype=np.float32) * channel_gain
    base += smooth_field(rng, width, height, 6, texture)[:, :, None]
    base += rng.standard_normal((height, width, 1), dtype=np.float32) * fine
    return base


def composite(
    page: np.ndarray, background: np.ndarray, h_cv: np.ndarray, interpolation: int
) -> np.ndarray:
    """Положить страницу на фон, предварительно преобразовав матрицей h_cv (для OpenCV)."""
    height, width = background.shape[:2]
    warped = cv2.warpPerspective(
        page.astype(np.float32),
        h_cv,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    coverage = cv2.warpPerspective(
        np.full(page.shape[:2], 1.0, dtype=np.float32),
        h_cv,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[:, :, None]
    return warped * coverage + background * (1.0 - coverage)


def lighting_gain(rng: Rng, width: int, height: int, strength: float) -> np.ndarray:
    """Неравномерный свет: наклонный градиент и одно «пятно» (height×width×1, около 1)."""
    lw, lh = 64, max(2, round(64 * height / width))
    ys, xs = np.mgrid[0:lh, 0:lw].astype(np.float32)
    xs = xs / (lw - 1) * 2 - 1
    ys = ys / (lh - 1) * 2 - 1
    angle = rng.uniform(0, 2 * np.pi)
    ramp = xs * np.cos(angle) + ys * np.sin(angle)
    cx, cy = rng.uniform(-0.8, 0.8, size=2)
    sigma = rng.uniform(0.5, 1.1)
    blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    gain = 1.0 + strength * ramp + rng.uniform(-0.8, 0.8) * strength * (blob - 0.5)
    gain = cv2.resize(gain.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    return gain[:, :, None]


def shadow_gain(rng: Rng, width: int, height: int, darkness: float, softness: float) -> np.ndarray:
    """Тень: полуплоскость с мягким краем, иногда с овальным «пальцем» (множитель 0..1)."""
    sw, sh = 128, max(2, round(128 * height / width))
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xs = xs / (sw - 1)
    ys = ys / (sh - 1) * (height / width)
    angle = rng.uniform(0, 2 * np.pi)
    px, py = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75) * (height / width)
    distance = (xs - px) * np.cos(angle) + (ys - py) * np.sin(angle)
    edge = np.clip(distance / softness + 0.5, 0.0, 1.0)
    shade = edge * edge * (3 - 2 * edge)  # smoothstep: 0 вне тени, 1 внутри
    if rng.random() < 0.5:
        ex, ey = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9) * (height / width)
        rx, ry = rng.uniform(0.08, 0.2), rng.uniform(0.04, 0.12)
        oval = np.exp(-(((xs - ex) / rx) ** 2 + ((ys - ey) / ry) ** 2))
        shade = np.maximum(shade, oval)
    gain = 1.0 - darkness * shade
    gain = cv2.resize(gain.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(gain, 0.0, 1.0)[:, :, None]


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else image


def add_noise(rng: Rng, image: np.ndarray, sigma: float) -> np.ndarray:
    """Шум матрицы: общий для каналов (яркостный) и слабый цветовой."""
    height, width = image.shape[:2]
    luma = rng.standard_normal((height, width, 1), dtype=np.float32) * sigma
    chroma = rng.standard_normal((height, width, 3), dtype=np.float32) * (sigma * 0.3)
    return image + luma + chroma


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image + 0.5, 0, 255).astype(np.uint8)


def encode(image: np.ndarray, extension: str, jpeg_quality: int | None) -> bytes:
    """Закодировать в PNG или JPEG. Результат — только пиксели, без текстового слоя."""
    params = [] if jpeg_quality is None else [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    ok, buffer = cv2.imencode(extension, image, params)
    if not ok:
        raise RuntimeError(f"не удалось закодировать изображение в {extension}")
    return buffer.tobytes()


def decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
