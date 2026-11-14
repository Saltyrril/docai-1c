"""Три уровня искажения: clean, scan, photo. Каждый возвращает картинку и пересчитанные bbox."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

from docai.augment import geometry as geo
from docai.augment import photometric as ph

Box = geo.Box


class Level(StrEnum):
    CLEAN = "clean"
    SCAN = "scan"
    PHOTO = "photo"


@dataclass(frozen=True)
class AugmentedBox:
    """Поле на искажённой картинке: охватывающий прямоугольник и точный четырёхугольник."""

    bbox: Box
    quad: np.ndarray  # 4×2, обход от левого верхнего угла исходного прямоугольника


@dataclass
class AugmentResult:
    data: bytes  # закодированный файл (PNG или JPEG)
    extension: str  # ".png" или ".jpg"
    width: int
    height: int
    boxes: list[AugmentedBox]
    homography: np.ndarray  # исходная картинка → итоговая, непрерывные координаты
    params: dict[str, Any] = field(default_factory=dict)


def _fit_into_canvas(h: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    """Уменьшить и сдвинуть, чтобы образ всей страницы целиком лежал внутри кадра."""
    corners = geo.box_corners((0, 0, width, height))
    quad = geo.apply_points(h, corners)
    bw = quad[:, 0].max() - quad[:, 0].min()
    bh = quad[:, 1].max() - quad[:, 1].min()
    s = min(1.0, (width - 2 * margin) / bw, (height - 2 * margin) / bh)
    h = geo.scale_about(s, width / 2, height / 2) @ h
    quad = geo.apply_points(h, corners)
    dx = max(margin - quad[:, 0].min(), 0.0) + min(width - margin - quad[:, 0].max(), 0.0)
    dy = max(margin - quad[:, 1].min(), 0.0) + min(height - margin - quad[:, 1].max(), 0.0)
    return geo.translate_matrix(dx, dy) @ h


def _scan_homography(rng: np.random.Generator, width: int, height: int) -> tuple[np.ndarray, float]:
    angle = float(rng.uniform(-3.0, 3.0))
    h = geo.rotate_matrix(angle, width / 2, height / 2)
    shift = geo.translate_matrix(
        rng.uniform(-0.01, 0.01) * width, rng.uniform(-0.01, 0.01) * height
    )
    return _fit_into_canvas(shift @ h, width, height, margin=0.004 * width), angle


def _photo_homography(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Лист в кадре «сверху под углом»: вписанный четырёхугольник со сдвинутыми углами."""
    ix, iy = 0.07 * width, 0.07 * height
    base = np.array([[ix, iy], [width - ix, iy], [width - ix, height - iy], [ix, height - iy]])
    jitter = np.array([0.045 * width, 0.045 * height])
    dst = base + rng.uniform(-1, 1, size=(4, 2)) * jitter
    # Дополнительный поворот листа в кадре.
    rotation = geo.rotate_matrix(float(rng.uniform(-5, 5)), width / 2, height / 2)
    dst = geo.apply_points(rotation, dst)
    src = geo.box_corners((0, 0, width, height))
    return _fit_into_canvas(geo.quad_matrix(src, dst), width, height, margin=0.01 * width)


def _scan(
    image: np.ndarray, rng: np.random.Generator, dpi: int
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    height, width = image.shape[:2]
    h, angle = _scan_homography(rng, width, height)
    tone = float(rng.uniform(0.90, 0.97))
    bg_level = float(rng.uniform(150, 205))
    background = ph.make_background(rng, width, height, bg_level, 0.02, 4.0, 2.0)
    out = ph.composite(image.astype(np.float32) * tone, background, geo.to_cv(h), cv2.INTER_CUBIC)
    sigma = float(rng.uniform(0.4, 1.0)) * dpi / 200
    out = ph.blur(out, sigma)
    noise = float(rng.uniform(3.0, 8.0))
    out = ph.add_noise(rng, out, noise)
    quality = int(rng.integers(70, 91))
    params = {
        "rotation_deg": round(angle, 3),
        "paper_tone": round(tone, 3),
        "background_level": round(bg_level, 1),
        "blur_sigma_px": round(sigma, 3),
        "noise_sigma": round(noise, 3),
        "jpeg_quality": quality,
    }
    return out, params, h


def _photo(
    image: np.ndarray, rng: np.random.Generator, dpi: int
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    height, width = image.shape[:2]
    h = _photo_homography(rng, width, height)
    tone = float(rng.uniform(0.88, 0.98))
    bg_level = float(rng.uniform(50, 140))
    background = ph.make_background(rng, width, height, bg_level, 0.12, 14.0, 4.0)
    out = ph.composite(image.astype(np.float32) * tone, background, geo.to_cv(h), cv2.INTER_CUBIC)
    light = float(rng.uniform(0.15, 0.30))
    gain = ph.lighting_gain(rng, width, height, light)
    darkness = float(rng.uniform(0.20, 0.45))
    softness = float(rng.uniform(0.02, 0.08))
    shadow = ph.shadow_gain(rng, width, height, darkness, softness)
    colour = 1.0 + rng.uniform(-0.04, 0.04, size=3).astype(np.float32)
    exposure = float(rng.uniform(0.92, 1.05))
    out = out * np.clip(gain * shadow, 0.35, 1.3) * colour * exposure
    sigma = float(rng.uniform(0.6, 1.4)) * dpi / 200
    out = ph.blur(out, sigma)
    noise = float(rng.uniform(2.0, 5.0))
    out = ph.add_noise(rng, out, noise)
    quality = int(rng.integers(50, 81))
    params = {
        "paper_tone": round(tone, 3),
        "background_level": round(bg_level, 1),
        "light_gradient": round(light, 3),
        "shadow_darkness": round(darkness, 3),
        "shadow_softness": round(softness, 3),
        "exposure": round(exposure, 3),
        "blur_sigma_px": round(sigma, 3),
        "noise_sigma": round(noise, 3),
        "jpeg_quality": quality,
    }
    return out, params, h


def augment_page(
    image: np.ndarray, boxes: list[Box], level: Level, rng: np.random.Generator, dpi: int
) -> AugmentResult:
    """Исказить страницу и перенести bbox (в пикселях исходной картинки) на результат."""
    height, width = image.shape[:2]
    if level is Level.CLEAN:
        params: dict[str, Any] = {}
        h = geo.identity()
        data, extension = ph.encode(image, ".png", None), ".png"
    else:
        build = _scan if level is Level.SCAN else _photo
        floating, params, h = build(image, rng, dpi)
        data = ph.encode(ph.to_uint8(floating), ".jpg", params["jpeg_quality"])
        extension = ".jpg"
    moved: list[AugmentedBox] = []
    for box in boxes:
        bbox, quad = geo.transform_box(h, box)
        moved.append(AugmentedBox(geo.clip_box(bbox, width, height), quad))
    return AugmentResult(
        data=data,
        extension=extension,
        width=width,
        height=height,
        boxes=moved,
        homography=h,
        params=params,
    )
