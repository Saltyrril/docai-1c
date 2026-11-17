"""Калибровка уверенности: насколько уверенность из P10 предсказывает ошибки (D45).

Точка — одно поле, которому постобработка поставила уверенность, и верно ли оно.
- ECE (Expected Calibration Error; Naeini et al., 2015; Guo et al., «On Calibration of
  Modern Neural Networks», ICML 2017): 10 корзин равной ширины по уверенности,
  ECE = Σ_b (n_b / N) · |доля верных_b − средняя уверенность_b|;
- MCE — наибольшее из этих расхождений по непустым корзинам;
- Brier = среднее (уверенность − верно)²;
- AUROC «уверенность отличает ошибку»: вероятность, что у случайного неверного поля
  уверенность ниже, чем у случайного верного (равные — пополам).
ECE, MCE и Brier раскладываются на суммы по документам и бутстрепятся вместе с прочими
метриками; AUROC — отдельным циклом по тем же перевыборкам документов.
"""

from collections.abc import Sequence

import numpy as np

from docai.eval.stats import Sums


def bin_index(confidence: float, n_bins: int) -> int:
    return min(int(confidence * n_bins), n_bins - 1)


def bin_keys(n_bins: int) -> list[str]:
    return [f"cal:{b}:{part}" for b in range(n_bins) for part in ("n", "conf", "correct")]


def ece_fn(n_bins: int):
    def fn(s: Sums) -> np.ndarray:
        gap = sum(
            np.abs(np.asarray(s[f"cal:{b}:correct"]) - s[f"cal:{b}:conf"]) for b in range(n_bins)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return gap / np.asarray(s["conf:n"], float)

    return fn


def mce_fn(n_bins: int):
    def fn(s: Sums) -> np.ndarray:
        gaps = []
        for b in range(n_bins):
            n = np.asarray(s[f"cal:{b}:n"], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                gap = np.abs(np.asarray(s[f"cal:{b}:correct"]) - s[f"cal:{b}:conf"]) / n
            gaps.append(np.where(n > 0, gap, -np.inf))
        best = np.max(np.stack(gaps), axis=0)
        return np.where(np.isfinite(best), best, np.nan)

    return fn


def auroc(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """AUROC уверенности как признака «поле верно» (Манн — Уитни, равные — пополам)."""
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n_pos, n_neg = int(ok.sum()), int((~ok).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty(len(conf))
    sorted_conf = conf[order]
    i = 0
    while i < len(conf):
        j = i
        while j + 1 < len(conf) and sorted_conf[j + 1] == sorted_conf[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # средний ранг для равных
        i = j + 1
    return float((ranks[ok].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_bootstrap(
    per_doc: Sequence[tuple[Sequence[float], Sequence[bool]]],
    weights: np.ndarray,
    ci: float,
) -> tuple[float, float, float]:
    """AUROC и процентильный интервал по перевыборкам документов (weights — B×n)."""
    conf_all = [np.asarray(c, float) for c, _ in per_doc]
    ok_all = [np.asarray(k, bool) for _, k in per_doc]
    point = auroc(
        np.concatenate(conf_all) if conf_all else [], np.concatenate(ok_all) if ok_all else []
    )
    values = []
    for row in weights:
        reps = row.astype(int)
        conf = np.concatenate(
            [np.repeat(c, r) for c, r in zip(conf_all, reps, strict=True) if r] or [np.empty(0)]
        )
        ok = np.concatenate(
            [np.repeat(k, r) for k, r in zip(ok_all, reps, strict=True) if r] or [np.empty(0, bool)]
        )
        values.append(auroc(conf, ok))
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return point, float("nan"), float("nan")
    alpha = (1 - ci) / 2
    low, high = np.quantile(arr, [alpha, 1 - alpha])
    return point, float(low), float(high)


def reliability_bins(sums: Sums, n_bins: int) -> list[dict]:
    out = []
    for b in range(n_bins):
        n = float(sums[f"cal:{b}:n"])
        out.append(
            {
                "low": b / n_bins,
                "high": (b + 1) / n_bins,
                "fields": int(n),
                "mean_confidence": round(float(sums[f"cal:{b}:conf"]) / n, 4) if n else None,
                "share_correct": round(float(sums[f"cal:{b}:correct"]) / n, 4) if n else None,
            }
        )
    return out


def reliability_svg(bins: Sequence[dict], title: str) -> str:
    """График надёжности: доля верных полей по корзинам уверенности против диагонали.

    Статичная картинка для report.md: одна серия (столбцы), пунктир — идеальная калибровка,
    под столбцами — число полей в корзине. Цвета и тёмная тема — через CSS внутри SVG.
    """
    w, h = 560, 400
    left, right, top, bottom = 56, 20, 44, 72
    pw, ph = w - left - right, h - top - bottom
    n = len(bins)
    slot = pw / n
    gap = 2

    def x(v: float) -> float:
        return left + v * pw

    def y(v: float) -> float:
        return top + (1 - v) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{title}">',
        "<style>"
        ".bg{fill:#fcfcfb}.bar{fill:#2a78d6}.grid{stroke:#e4e3df;stroke-width:1}"
        ".axis{stroke:#8a8984;stroke-width:1}.diag{stroke:#52514e;stroke-width:1.5;"
        "stroke-dasharray:5 4;fill:none}.t{fill:#0b0b0b;font:13px sans-serif}"
        ".s{fill:#52514e;font:11px sans-serif}"
        "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.bar{fill:#3987e5}"
        ".grid{stroke:#383835}.axis{stroke:#8a8984}.diag{stroke:#c3c2b7}"
        ".t{fill:#f0efec}.s{fill:#c3c2b7}}"
        "</style>",
        f'<rect class="bg" width="{w}" height="{h}"/>',
        f'<text class="t" x="{left}" y="24">{title}</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line class="grid" x1="{x(0)}" x2="{x(1)}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
        )
        parts.append(
            f'<text class="s" x="{left - 8}" y="{y(tick) + 4:.1f}" '
            f'text-anchor="end">{f"{tick:.2f}".replace(".", ",")}</text>'
        )
    for b, cell in enumerate(bins):
        x0 = left + b * slot + gap / 2
        share = cell.get("share_correct")
        if share:
            top_y = y(share)
            height = y(0) - top_y
            r = min(4.0, height / 2, (slot - gap) / 2)
            bw = slot - gap
            # Скругление 4 px только сверху, низ стоит на нулевой линии.
            parts.append(
                f'<path class="bar" d="M{x0:.1f},{y(0):.1f} V{top_y + r:.1f} '
                f"Q{x0:.1f},{top_y:.1f} {x0 + r:.1f},{top_y:.1f} H{x0 + bw - r:.1f} "
                f'Q{x0 + bw:.1f},{top_y:.1f} {x0 + bw:.1f},{top_y + r:.1f} V{y(0):.1f} Z">'
                f"<title>уверенность {cell['low']:.1f}–{cell['high']:.1f}: "
                f"полей {cell['fields']}, верных {share:.0%}, "
                f"средняя уверенность {cell['mean_confidence']:.2f}</title></path>"
            )
        parts.append(
            f'<text class="s" x="{x0 + (slot - gap) / 2:.1f}" y="{y(0) + 30:.1f}" '
            f'text-anchor="middle">{cell["fields"]}</text>'
        )
    parts.append(f'<path class="diag" d="M{x(0)},{y(0)} L{x(1)},{y(1)}"/>')
    parts.append(f'<line class="axis" x1="{x(0)}" x2="{x(1)}" y1="{y(0)}" y2="{y(0)}"/>')
    for tick in (0, 0.5, 1):
        parts.append(
            f'<text class="s" x="{x(tick):.1f}" y="{y(0) + 16:.1f}" '
            f'text-anchor="middle">{f"{tick:.1f}".replace(".", ",")}</text>'
        )
    parts.append(
        f'<text class="s" x="{left + pw / 2}" y="{h - 22}" text-anchor="middle">'
        "уверенность поля (корзины по 0,1); цифры под осью — число полей</text>"
    )
    parts.append(
        f'<text class="s" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">'
        "столбец — доля верных полей; пунктир — идеальная калибровка</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
