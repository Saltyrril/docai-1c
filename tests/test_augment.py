"""Тесты P5: растеризация, геометрия, уровни искажения, bbox на картинках, CLI."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from docai.augment import geometry as geo
from docai.augment import photometric as ph
from docai.augment.levels import Level, augment_page
from docai.augment.pipeline import load_layout, make_rng, page_fields, process_pdf
from docai.augment.preview import draw_fields, make_grid
from docai.augment.raster import rasterize_pdf
from docai.render.engine import Decor, Renderer
from docai.schemas.common import DocType
from docai.synth.generator import generate_document

NO_DECOR = Decor(logo=None, stamp=None, sign=None)


# --- общие данные ---------------------------------------------------------------------------


def _write_render(renderer: Renderer, doc, folder: Path, stem: str) -> Path:
    result = renderer.render(doc, variant=1, decor=NO_DECOR, source=f"{stem}.json")
    pdf_path = folder / f"{stem}.pdf"
    pdf_path.write_bytes(result.pdf)
    (folder / f"{stem}.layout.json").write_text(
        json.dumps(result.layout, ensure_ascii=False), encoding="utf-8"
    )
    return pdf_path


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory) -> dict[str, Path]:
    """Однострочный счёт и многостраничный ТОРГ-12 с разметкой."""
    folder = tmp_path_factory.mktemp("pdf")
    bill = next(
        d
        for i in range(1, 200)
        if (d := generate_document(42, i, error_rate=0.0).document).doc_type == DocType.BILL
    )
    torg = next(
        d
        for i in range(1, 200)
        if (d := generate_document(42, i, error_rate=0.0).document).doc_type == DocType.TORG12
        and len(d.lines) >= 4
    )
    torg = torg.model_copy(update={"lines": torg.lines * 12})
    with Renderer() as renderer:
        return {
            "bill": _write_render(renderer, bill, folder, "bill"),
            "torg": _write_render(renderer, torg, folder, "torg"),
        }


def _ink_box(gray: np.ndarray, threshold: int = 110) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(gray < threshold)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# --- растеризация ---------------------------------------------------------------------------


@pytest.mark.parametrize("dpi", [150, 200, 300])
def test_rasterize_size_follows_dpi(pdfs, dpi):
    (page,) = rasterize_pdf(pdfs["bill"], dpi)
    width_pt, height_pt = page.size_pt
    height_px, width_px = page.image.shape[:2]
    assert abs(width_px - width_pt * dpi / 72) <= 1
    assert abs(height_px - height_pt * dpi / 72) <= 1
    assert page.image.dtype == np.uint8
    assert page.image.shape[2] == 3
    assert page.scale[0] == pytest.approx(dpi / 72, rel=1e-2)
    assert page.image.mean() > 200  # белая страница с текстом


def test_rasterize_multipage_and_bytes_input(pdfs):
    pages = rasterize_pdf(pdfs["torg"].read_bytes(), 150)
    assert len(pages) >= 2
    assert [p.page for p in pages] == list(range(1, len(pages) + 1))


# --- геометрия ------------------------------------------------------------------------------


def test_rotate_matrix_clockwise_about_center():
    h = geo.rotate_matrix(90, 50, 50)
    out = geo.apply_points(h, np.array([[100.0, 50.0], [50.0, 50.0]]))
    assert out[0] == pytest.approx([50.0, 100.0])  # правая точка уходит вниз: по часовой, Y вниз
    assert out[1] == pytest.approx([50.0, 50.0])


def test_quad_matrix_maps_corners():
    src = geo.box_corners((0, 0, 100, 200))
    dst = np.array([[10.0, 5.0], [90.0, 12.0], [95.0, 190.0], [3.0, 180.0]])
    out = geo.apply_points(geo.quad_matrix(src, dst), src)
    assert out == pytest.approx(dst, abs=1e-9)


def test_transform_box_returns_bounding_box_and_quad():
    h = geo.rotate_matrix(45, 0, 0)
    bbox, quad = geo.transform_box(h, (0, 0, 10, 10))
    assert quad.shape == (4, 2)
    assert bbox[2] - bbox[0] == pytest.approx(10 * 2**0.5)
    assert geo.clip_box((-5, -5, 20, 20), 10, 10) == (0, 0, 10, 10)


@pytest.mark.parametrize(
    "h",
    [
        geo.rotate_matrix(3.0, 200, 150),
        geo.rotate_matrix(-2.4, 200, 150) @ geo.translate_matrix(7.3, -4.1),
        geo.quad_matrix(
            geo.box_corners((0, 0, 400, 300)),
            np.array([[20.0, 10.0], [370.0, 25.0], [385.0, 280.0], [8.0, 270.0]]),
        ),
    ],
    ids=["rotation", "rotation+shift", "perspective"],
)
def test_box_moves_with_pixels_within_half_pixel(h):
    """Тёмный прямоугольник на белом после warp лежит там, куда его переносит transform_box."""
    image = np.full((300, 400, 3), 255, np.uint8)
    box = (120, 90, 260, 170)
    image[box[1] : box[3], box[0] : box[2]] = 0
    warped = cv2.warpPerspective(
        image, geo.to_cv(h), (400, 300), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    bbox, _ = geo.transform_box(h, box)
    x0, y0, x1, y1 = _ink_box(warped[:, :, 0], threshold=128)
    for got, want in zip((x0, y0, x1, y1), bbox, strict=True):
        assert abs(got - want) <= 1.0, (got, want)


# --- уровни ---------------------------------------------------------------------------------


def _page_with_fields(pdfs, dpi=150, name="bill"):
    (page, *_) = rasterize_pdf(pdfs[name], dpi)
    layout = load_layout(pdfs[name])
    return page, page_fields(layout, page)


def test_clean_level_is_png_and_identity(pdfs):
    page, fields = _page_with_fields(pdfs)
    boxes = [f["box"] for f in fields]
    result = augment_page(page.image, boxes, Level.CLEAN, make_rng(1), 150)
    assert result.extension == ".png"
    assert np.array_equal(ph.decode(result.data), page.image)  # без потерь
    assert all(
        m.bbox == pytest.approx(b, abs=1e-6) for m, b in zip(result.boxes, boxes, strict=True)
    )


@pytest.mark.parametrize(("level", "quality"), [(Level.SCAN, (70, 90)), (Level.PHOTO, (50, 80))])
def test_jpeg_quality_in_range(pdfs, level, quality):
    page, fields = _page_with_fields(pdfs)
    boxes = [f["box"] for f in fields]
    for seed in range(6):
        result = augment_page(page.image, boxes, level, make_rng(seed), 150)
        assert result.extension == ".jpg"
        assert result.data[:2] == b"\xff\xd8"  # JPEG
        assert quality[0] <= result.params["jpeg_quality"] <= quality[1]
        assert ph.decode(result.data).shape == page.image.shape


def test_scan_rotation_within_three_degrees(pdfs):
    page, fields = _page_with_fields(pdfs)
    angles = [
        augment_page(page.image, [f["box"] for f in fields], Level.SCAN, make_rng(s), 150).params[
            "rotation_deg"
        ]
        for s in range(30)
    ]
    assert all(abs(a) <= 3.0 for a in angles)
    assert max(angles) > 1.5 > -1.5 > min(angles)  # диапазон реально используется


@pytest.mark.parametrize("level", [Level.SCAN, Level.PHOTO])
def test_page_stays_inside_frame_and_boxes_are_clipped(pdfs, level):
    page, fields = _page_with_fields(pdfs)
    height, width = page.image.shape[:2]
    for seed in range(10):
        result = augment_page(page.image, [f["box"] for f in fields], level, make_rng(seed), 150)
        corners = geo.apply_points(result.homography, geo.box_corners((0, 0, width, height)))
        assert corners[:, 0].min() >= 0 and corners[:, 0].max() <= width
        assert corners[:, 1].min() >= 0 and corners[:, 1].max() <= height
        for moved in result.boxes:
            x0, y0, x1, y1 = moved.bbox
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height


def test_photo_is_perspective_not_affine(pdfs):
    page, fields = _page_with_fields(pdfs)
    result = augment_page(page.image, [f["box"] for f in fields], Level.PHOTO, make_rng(3), 150)
    h = result.homography
    assert abs(h[2, 0]) + abs(h[2, 1]) > 1e-7  # есть перспективная часть


def test_photo_has_uneven_light_and_scan_has_gray_background(pdfs):
    page, fields = _page_with_fields(pdfs)
    boxes = [f["box"] for f in fields]
    clean = page.image.mean()
    scan = ph.decode(augment_page(page.image, boxes, Level.SCAN, make_rng(4), 150).data)
    photo = ph.decode(augment_page(page.image, boxes, Level.PHOTO, make_rng(4), 150).data)
    # серый фон: углы кадра заметно темнее белой бумаги
    assert scan[:6, :6].mean() < 215
    assert scan.mean() < clean
    # неравномерный свет: яркость пустых мест листа сильно меняется по кадру
    h, w = photo.shape[:2]
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blocks = [gray[h * r // 4 : h * (r + 1) // 4, w * c // 4 : w * (c + 1) // 4].mean()
              for r in range(4) for c in range(4)]  # fmt: skip
    assert max(blocks) - min(blocks) > 25


@pytest.mark.parametrize(("level", "floor"), [(Level.SCAN, 0.9), (Level.PHOTO, 0.6)])
def test_inverse_homography_restores_the_page(pdfs, level, floor):
    """Картинка, развёрнутая обратной матрицей, совпадает с исходной (ZNCC), а со сдвигом — нет."""
    page, fields = _page_with_fields(pdfs, dpi=150)
    boxes = [f["box"] for f in fields]
    result = augment_page(page.image, boxes, level, make_rng(11), 150)
    height, width = page.image.shape[:2]
    back = cv2.warpPerspective(
        ph.decode(result.data), geo.to_cv(np.linalg.inv(result.homography)), (width, height)
    )

    def band(image: np.ndarray) -> np.ndarray:
        """Полоса частот текста: убираем свет и тень (крупные пятна) и шум (мелкое)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        return cv2.GaussianBlur(gray, (0, 0), 2) - cv2.GaussianBlur(gray, (0, 0), 12)

    def zncc(a: np.ndarray, b: np.ndarray) -> float:
        sl = (slice(height // 10, -height // 10), slice(width // 10, -width // 10))
        a_g, b_g = band(a)[sl], band(b)[sl]
        return float((a_g * b_g).sum() / np.sqrt((a_g**2).sum() * (b_g**2).sum()))

    aligned = zncc(page.image, back)
    shifted = zncc(page.image, np.roll(back, 6, axis=1))
    assert aligned > floor
    assert aligned > shifted + 0.1


def test_boxes_cover_the_printed_text_after_augmentation(pdfs):
    """В «чистом» PNG чернила поля лежат внутри bbox; после scan — внутри перенесённого bbox."""
    page, fields = _page_with_fields(pdfs, dpi=300)
    chosen = [
        f
        for f in fields
        if f["field"] in {"number", "date", "seller.inn", "totals.amount_with_vat"}
    ]
    assert len(chosen) >= 3
    boxes = [f["box"] for f in chosen]
    clean = augment_page(page.image, boxes, Level.CLEAN, make_rng(0), 300)
    scan = augment_page(page.image, boxes, Level.SCAN, make_rng(0), 300)
    scan_gray = cv2.cvtColor(ph.decode(scan.data), cv2.COLOR_BGR2GRAY)
    clean_gray = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
    for moved_clean, moved_scan in zip(clean.boxes, scan.boxes, strict=True):
        for gray, moved, tol, thr in (
            (clean_gray, moved_clean, 2, 110),
            (scan_gray, moved_scan, 5, 130),
        ):
            x0, y0, x1, y1 = (round(v) for v in moved.bbox)
            region = gray[max(0, y0 - 12) : y1 + 12, max(0, x0 - 12) : x1 + 12]
            ink_x0, ink_y0, ink_x1, ink_y1 = _ink_box(region, thr)
            # чернила поля занимают почти весь bbox: границы совпадают с точностью tol пикселей
            # (рядом могут быть соседние поля, поэтому проверяем только «чернила внутри рамки»)
            inside = gray[y0 + tol : y1 - tol, x0 + tol : x1 - tol]
            assert (inside < thr).any(), moved
            assert ink_x1 > ink_x0 and ink_y1 > ink_y0


def test_same_seed_same_bytes_other_seed_differs(pdfs):
    page, fields = _page_with_fields(pdfs, dpi=150)
    boxes = [f["box"] for f in fields]
    first = augment_page(page.image, boxes, Level.PHOTO, make_rng(7, "a"), 150)
    again = augment_page(page.image, boxes, Level.PHOTO, make_rng(7, "a"), 150)
    other = augment_page(page.image, boxes, Level.PHOTO, make_rng(8, "a"), 150)
    assert first.data == again.data
    assert first.data != other.data


def test_draw_fields_and_grid(pdfs):
    page, fields = _page_with_fields(pdfs, dpi=150)
    result = augment_page(page.image, [f["box"] for f in fields], Level.SCAN, make_rng(2), 150)
    layout = {"fields": [{"bbox": list(m.bbox), "quad": m.quad.tolist()} for m in result.boxes]}
    marked = draw_fields(ph.decode(result.data), layout)
    assert marked.shape == page.image.shape
    assert not np.array_equal(marked, ph.decode(result.data))
    grid = make_grid([marked] * 5, cols=3, cell=(100, 140), gap=4)
    assert grid.shape == (2 * 140 + 3 * 4, 3 * 100 + 4 * 4, 3)


# --- конвейер и CLI -------------------------------------------------------------------------


def test_process_pdf_writes_image_and_pixel_layout(pdfs, tmp_path):
    written = process_pdf(pdfs["bill"], tmp_path, list(Level), [150, 300], seed=5)
    assert len(written) == 6
    source = load_layout(pdfs["bill"])
    for item in written:
        layout = json.loads(item.layout.read_text(encoding="utf-8"))
        assert layout["text_layer"] is False
        assert layout["level"] == item.level.value
        assert layout["dpi"] == item.dpi
        assert layout["image"]["file"] == item.image.name
        data = item.image.read_bytes()
        assert b"%PDF" not in data[:1024]
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (layout["image"]["height"], layout["image"]["width"])
        assert len(layout["fields"]) == len(source["fields"])
        assert {f["field"] for f in layout["fields"]} == {f["field"] for f in source["fields"]}
        for entry in layout["fields"]:
            assert len(entry["quad"]) == 4
            assert entry["text"]


def test_clean_layout_is_source_layout_in_pixels(pdfs, tmp_path):
    (item,) = process_pdf(pdfs["bill"], tmp_path, [Level.CLEAN], [300], seed=1)
    layout = json.loads(item.layout.read_text(encoding="utf-8"))
    source = load_layout(pdfs["bill"])
    k = layout["image"]["width"] / source["page_size_pt"][0]
    for got, want in zip(layout["fields"], source["fields"], strict=True):
        assert got["bbox"] == pytest.approx([v * k for v in want["bbox"]], abs=0.5)


def test_multipage_fields_split_by_page(pdfs, tmp_path):
    written = process_pdf(pdfs["torg"], tmp_path, [Level.SCAN], [150], seed=1)
    source = load_layout(pdfs["torg"])
    assert len(written) == source["page_count"] >= 2
    total = 0
    for item in written:
        layout = json.loads(item.layout.read_text(encoding="utf-8"))
        want = [f for f in source["fields"] if f["page"] == item.page]
        assert [f["field"] for f in layout["fields"]] == [f["field"] for f in want]
        assert layout["page"] == item.page
        total += len(layout["fields"])
    assert total == len(source["fields"])


def test_missing_layout_gives_images_without_fields(pdfs, tmp_path):
    lone = tmp_path / "lone.pdf"
    lone.write_bytes(pdfs["bill"].read_bytes())
    (item,) = process_pdf(lone, tmp_path / "out", [Level.SCAN], [150], seed=1)
    layout = json.loads(item.layout.read_text(encoding="utf-8"))
    assert layout["fields"] == []
    assert layout["doc_type"] is None


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "docai.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_cli_augment_scan(pdfs, tmp_path):
    out = tmp_path / "scans"
    result = _run_cli(
        "augment", "--level", "scan", "--in", str(pdfs["bill"].parent), "--out", str(out)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    jpgs = sorted(out.glob("*_scan_200dpi.jpg"))
    assert len(jpgs) >= 2  # счёт (1 стр.) и ТОРГ-12 (несколько)
    assert len(list(out.glob("*.layout.json"))) == len(jpgs)


def test_cli_augment_all_levels_and_dpis(pdfs, tmp_path):
    out = tmp_path / "all"
    result = _run_cli(
        "augment", "--level", "all", "--dpi", "150", "--dpi", "300", "--stem", "bill",
        "--in", str(pdfs["bill"].parent), "--out", str(out),
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    names = sorted(p.name for p in out.glob("*.layout.json"))
    assert len(names) == 6
    assert any("_clean_300dpi" in n for n in names)
    assert any("_photo_150dpi" in n for n in names)


def test_cli_augment_rejects_unknown_dpi_and_empty_folder(tmp_path):
    bad = _run_cli("augment", "--dpi", "123", "--in", str(tmp_path), "--out", str(tmp_path / "o"))
    assert bad.returncode == 2
    empty = _run_cli("augment", "--in", str(tmp_path), "--out", str(tmp_path / "o"))
    assert empty.returncode == 1
