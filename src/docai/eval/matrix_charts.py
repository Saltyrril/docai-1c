"""Графики матрицы P12: качество против скорости и качество по уровням входа (SVG без зависимостей).

Цвета — палитра Окабе–Ито (различима при дальтонизме); путь кодируется формой маркера,
модель — цветом. Вертикальные «усы» — 95% интервалы.
"""

import math
from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
LEVEL_COLORS = {"clean": "#0072B2", "scan": "#E69F00", "photo": "#D55E00"}
LEVEL_TITLES = {"clean": "clean", "scan": "scan", "photo": "photo"}
FONT = 'font-family="Segoe UI, Arial, sans-serif"'
INK = "#222222"
GRID = "#dddddd"


def model_colors(models: Sequence[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _marker(path: str, x: float, y: float, color: str) -> str:
    if path == "text":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="white"/>'
    if path == "vision":
        return f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" fill="{color}" stroke="white"/>'
    pts = f"{x:.1f},{y - 7:.1f} {x + 7:.1f},{y + 6:.1f} {x - 7:.1f},{y + 6:.1f}"
    return f'<polygon points="{pts}" fill="{color}" stroke="white"/>'


def _text(
    x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = ""
) -> str:
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{INK}" {FONT}{w}>{escape(text)}</text>'
    )


def _frame(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="white"/>' + "".join(body) + "</svg>\n"
    )


def _legend(x: float, y: float, models: dict[str, str], paths: Sequence[str]) -> list[str]:
    out = [_text(x, y, "Модель", 12, weight="bold")]
    for i, (name, color) in enumerate(models.items()):
        yy = y + 20 + i * 20
        out.append(f'<rect x="{x}" y="{yy - 10}" width="12" height="12" fill="{color}"/>')
        out.append(_text(x + 20, yy, name))
    y2 = y + 40 + len(models) * 20
    out.append(_text(x, y2, "Путь", 12, weight="bold"))
    for i, path in enumerate(paths):
        yy = y2 + 20 + i * 20
        out.append(_marker(path, x + 6, yy - 4, "#777777"))
        out.append(_text(x + 20, yy, {"ocr": "ocr + LLM"}.get(path, path)))
    return out


def scatter_svg(
    points: Sequence[dict[str, Any]], metrics: Sequence[tuple[str, str]], title: str
) -> str:
    """Качество (по вертикали, %) против секунд на страницу (по горизонтали, логарифм).

    points: {model, path, x (с/стр), y: {metric: {value, low, high}}}; metrics: (имя, подпись).
    """
    models = model_colors(list(dict.fromkeys(p["model"] for p in points)))
    paths = list(dict.fromkeys(p["path"] for p in points))
    xs = [p["x"] for p in points if p["x"]]
    lo = 10 ** math.floor(math.log10(min(xs))) if xs else 1
    hi = 10 ** math.ceil(math.log10(max(xs))) if xs else 10
    if lo == hi:
        hi = lo * 10
    panel_w, panel_h, left, top, gap = 420, 300, 60, 50, 70
    width = left + len(metrics) * (panel_w + gap) + 170
    height = top + panel_h + 90
    body = [_text(left, 24, title, 15, weight="bold")]

    def px(x: float, ox: float) -> float:
        return ox + (math.log10(x) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * panel_w

    for index, (metric, label) in enumerate(metrics):
        ox = left + index * (panel_w + gap)
        body.append(_text(ox + panel_w / 2, top - 14, label, 13, "middle", "bold"))
        for tick in range(0, 101, 20):
            y = top + panel_h - tick / 100 * panel_h
            body.append(f'<line x1="{ox}" y1="{y}" x2="{ox + panel_w}" y2="{y}" stroke="{GRID}"/>')
            body.append(_text(ox - 8, y + 4, f"{tick}", 11, "end"))
        decade = lo
        while decade <= hi * 1.0001:
            x = px(decade, ox)
            body.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_h}" stroke="{GRID}"/>'
            )
            body.append(_text(x, top + panel_h + 18, f"{decade:g}", 11, "middle"))
            decade *= 10
        body.append(
            _text(
                ox + panel_w / 2,
                top + panel_h + 40,
                "секунд на страницу (логарифмическая шкала)",
                11,
                "middle",
            )
        )
        body.append(
            f'<rect x="{ox}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="{INK}"/>'
        )
        for p in points:
            est = p["y"].get(metric)
            if not est or est.get("value") is None or not p["x"]:
                continue
            x = px(p["x"], ox)
            y = top + panel_h - est["value"] * panel_h
            color = models[p["model"]]
            if est.get("low") is not None:
                y0 = top + panel_h - est["low"] * panel_h
                y1 = top + panel_h - est["high"] * panel_h
                body.append(
                    f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="1.5"/>'
                )
            body.append(_marker(p["path"], x, y, color))
    body.append(_text(16, top + panel_h / 2, "%", 12))
    body += _legend(left + len(metrics) * (panel_w + gap) - 30, top, models, paths)
    return _frame(width, height, body)


def level_bars_svg(
    rows: Sequence[dict[str, Any]], metrics: Sequence[tuple[str, str]], title: str
) -> str:
    """Столбцы по уровням clean/scan/photo. rows: {label, levels: {уровень: {метрика: оценка}}}."""
    levels = ["clean", "scan", "photo"]
    bar, inner, outer = 13, 2, 22
    group_w = len(levels) * bar + (len(levels) - 1) * inner
    left, top, panel_h, gap = 60, 50, 220, 120
    plot_w = len(rows) * (group_w + outer)
    width = left + plot_w + 40
    height = top + len(metrics) * (panel_h + gap) + 20
    body = [_text(left, 24, title, 15, weight="bold")]
    for index, (metric, label) in enumerate(metrics):
        oy = top + index * (panel_h + gap)
        body.append(_text(left, oy - 14, label, 13, weight="bold"))
        for tick in range(0, 101, 20):
            y = oy + panel_h - tick / 100 * panel_h
            body.append(
                f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="{GRID}"/>'
            )
            body.append(_text(left - 8, y + 4, f"{tick}", 11, "end"))
        for i, row in enumerate(rows):
            gx = left + outer / 2 + i * (group_w + outer)
            for j, level in enumerate(levels):
                est = (row["levels"].get(level) or {}).get(metric)
                if not est or est.get("value") is None:
                    continue
                x = gx + j * (bar + inner)
                h = est["value"] * panel_h
                body.append(
                    f'<rect x="{x:.1f}" y="{oy + panel_h - h:.1f}" width="{bar}" height="{h:.1f}" '
                    f'fill="{LEVEL_COLORS[level]}"/>'
                )
                if est.get("low") is not None:
                    cx = x + bar / 2
                    y0 = oy + panel_h - est["low"] * panel_h
                    y1 = oy + panel_h - est["high"] * panel_h
                    body.append(
                        f'<line x1="{cx:.1f}" y1="{y0:.1f}" x2="{cx:.1f}" y2="{y1:.1f}" stroke="{INK}" stroke-width="1"/>'
                    )
            lx, ly = gx + group_w / 2, oy + panel_h + 14
            body.append(
                f'<text x="{lx:.1f}" y="{ly}" font-size="11" fill="{INK}" {FONT} '
                f'text-anchor="end" transform="rotate(-40 {lx:.1f} {ly})">{escape(row["label"])}</text>'
            )
    lx = left + plot_w - 200
    for j, level in enumerate(levels):
        body.append(
            f'<rect x="{lx + j * 70}" y="14" width="12" height="12" fill="{LEVEL_COLORS[level]}"/>'
        )
        body.append(_text(lx + j * 70 + 18, 25, LEVEL_TITLES[level]))
    return _frame(width, height, body)
