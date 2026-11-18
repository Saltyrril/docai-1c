"""Сводка матрицы P12: таблица по ячейкам, по уровням, парное сравнение с лучшей, графики.

Берётся общее для всех ячеек начало списка документов, которое уже посчитано, поэтому
ячейки сравнимы друг с другом (одни и те же документы, парный бутстреп). Реальные документы
считаются отдельной таблицей и попадают в отчёт только метриками: ни значений, ни имён.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml

from docai.eval.matrix import (
    Cell,
    MatrixConfig,
    active_cells,
    build_cells,
    cached_predictions,
    cell_items,
    load_vram_check,
    placement_summary,
    plan_docs,
    real_items,
    skip_reason,
)
from docai.eval.matrix_charts import level_bars_svg, scatter_svg
from docai.eval.report import fmt, fmt_metric, fmt_p
from docai.eval.run import build_records, compare_runs, evaluate_records

TABLE_METRICS = (
    "doc_ok", "cells_f1", "header_accuracy", "rows_f1", "text_anls", "doc_type_accuracy",
    "extracted", "silent_error", "ece", "sec_per_page",
)  # fmt: skip
LEVELS = ("clean", "scan", "photo")
PATH_TITLES = {"text": "text", "vision": "vision", "ocr": "ocr + LLM"}


def report_dir(cfg: MatrixConfig) -> Path:
    date = cfg.report.date or dt.date.today().isoformat()
    return cfg.report.out_dir / f"{date}_{cfg.name}"


def common_prefix(cells: list[Cell], items: dict[str, list]) -> int:
    """Сколько первых документов готовы во всех ячейках (синтетика, по порядку)."""
    n = None
    for cell in cells:
        done = cached_predictions(cell, [r for r in items[cell.id] if not r[1].real])
        first_missing = next(
            (order for order, it in items[cell.id] if not it.real and it.item_id not in done),
            None,
        )
        ready = (
            max((order for order, it in items[cell.id] if not it.real), default=-1) + 1
            if first_missing is None
            else first_missing
        )
        n = ready if n is None else min(n, ready)
    return n or 0


def _pick(est: dict, names=TABLE_METRICS) -> dict[str, Any]:
    return {k: est[k] for k in names if k in est}


def aggregate(cfg: MatrixConfig, *, docs: int | None = None) -> dict[str, Any]:
    cells = active_cells(cfg)
    plan = plan_docs(cfg)
    real = real_items(cfg)
    items = {c.id: cell_items(c, plan, real) for c in cells}
    ready = common_prefix(cells, items)
    n_docs = ready if docs is None else min(docs, ready)
    if n_docs == 0:
        raise ValueError("нет документов, готовых во всех ячейках: сначала docai eval-matrix run")

    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    real_rows: list[dict[str, Any]] = []
    for cell in cells:
        ecfg = cell.eval_config()
        syn = [(o, it) for o, it in items[cell.id] if not it.real and o < n_docs]
        preds = cached_predictions(cell, syn)
        selected = [it for _, it in syn]
        records = build_records(selected, preds, ecfg)
        result = evaluate_records(records, ecfg)
        result["config"] = {"name": cell.id}
        results[cell.id] = result
        levels = {
            lvl: {"documents": v["documents"], **_pick(v["metrics"])}
            for lvl, v in result["groups"]["level"]["values"].items()
        }
        rows.append(
            {
                "cell": cell.id,
                "model": cell.model.id,
                "tag": cell.model.tag,
                "path": cell.path_name,
                "items": len(selected),
                "overall": _pick(result["overall"]),
                "levels": levels,
                "performance": result["performance"],
                "placement": placement_summary(cell),
            }
        )
        # Реальные: что уже посчитано; только метрики, без документов.
        rsel = [(o, it) for o, it in items[cell.id] if it.real]
        rpreds = cached_predictions(cell, rsel)
        if rpreds:
            rrec = build_records([it for _, it in rsel if it.item_id in rpreds], rpreds, ecfg)
            rres = evaluate_records(rrec, ecfg)
            real_rows.append(
                {
                    "cell": cell.id,
                    "model": cell.model.id,
                    "path": cell.path_name,
                    "items": len(rrec),
                    "overall": _pick(rres["overall"]),
                    "levels": {
                        k: {"documents": v["documents"], **_pick(v["metrics"])}
                        for k, v in rres["groups"]["level"]["values"].items()
                    },
                }
            )

    best = max(rows, key=lambda r: _key(r))
    best_result = results[best["cell"]]
    for row in rows:
        if row["cell"] == best["cell"]:
            row["vs_best"] = None
            continue
        comp = compare_runs(results[row["cell"]], best_result, cfg_eval(cfg, cells))
        row["vs_best"] = {
            "common_documents": comp["common_documents"],
            "doc_ok": comp["metrics"]["doc_ok"],
            "cells_f1": comp["metrics"]["cells_f1"],
            "mcnemar_doc_ok": comp["mcnemar_doc_ok"],
        }
    candidates = [
        r for r in rows
        if r["vs_best"] is None or all(
            r["vs_best"][m]["high"] is None or r["vs_best"][m]["high"] >= 0
            for m in ("doc_ok", "cells_f1")
        )
    ]  # fmt: skip
    fastest = min(candidates, key=lambda r: r["overall"]["sec_per_page"]["value"] or 1e9)
    check = load_vram_check(cfg)
    skipped = [
        {"model": m.id, "tag": m.tag, "paths": m.paths, "reason": reason}
        for m in cfg.models
        if (reason := skip_reason(cfg, m, check)) is not None
    ]
    return {
        "name": cfg.name,
        "documents": n_docs,
        "documents_planned": len(plan),
        "ready_in_all_cells": ready,
        "cells": rows,
        "best_quality": best["cell"],
        "fastest_not_worse_than_best": fastest["cell"],
        "skipped": skipped,
        "real": real_rows,
        "real_items_available": len(real),
    }


def cfg_eval(cfg: MatrixConfig, cells: list[Cell]):
    return cells[0].eval_config()


def _key(row: dict[str, Any]) -> tuple[float, float]:
    o = row["overall"]
    return (o["doc_ok"]["value"] or 0.0, o["cells_f1"]["value"] or 0.0)


# ---- отчёт -------------------------------------------------------------------------------


def _sec(row: dict[str, Any]) -> str:
    est = row["overall"]["sec_per_page"]
    return fmt(est, percent=False, digits=1)


def _vram(row: dict[str, Any]) -> str:
    peak = row["performance"].get("peak_vram_mib_max")
    place = row.get("placement")
    share = f"{place['vram_share_median']:.0%}" if place else "—"
    return f"{peak / 1024:.2f}".replace(".", ",") if peak else "—", share


def _main_table(data: dict[str, Any]) -> list[str]:
    out = [
        "| Модель | Путь | Док. | Без правок [95% ДИ] | Ячейки F1 | Поля шапки acc | Тип верен "
        "| Молча с ошибкой | с/стр [95% ДИ] | Пик VRAM, ГиБ | Модель в VRAM |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in data["cells"]:
        o = r["overall"]
        peak, share = _vram(r)
        out.append(
            f"| {r['model']} | {PATH_TITLES.get(r['path'], r['path'])} | {r['items']} "
            f"| {fmt_metric('doc_ok', o['doc_ok'])} | {fmt_metric('cells_f1', o['cells_f1'])} "
            f"| {fmt_metric('header_accuracy', o['header_accuracy'])} "
            f"| {fmt_metric('doc_type_accuracy', o['doc_type_accuracy'])} "
            f"| {fmt_metric('silent_error', o['silent_error'])} | {_sec(r)} | {peak} | {share} |"
        )
    return out


def _level_table(data: dict[str, Any], metric: str, title: str) -> list[str]:
    out = [
        f"**{title}**",
        "",
        "| Модель | Путь | clean | scan | photo |",
        "|---|---|---|---|---|",
    ]
    for r in data["cells"]:
        cols = []
        for lvl in LEVELS:
            v = r["levels"].get(lvl)
            cols.append(f"{fmt_metric(metric, v[metric])} (n={v['documents']})" if v else "—")
        out.append(
            f"| {r['model']} | {PATH_TITLES.get(r['path'], r['path'])} | " + " | ".join(cols) + " |"
        )
    return out


def _vs_best_table(data: dict[str, Any]) -> list[str]:
    out = [
        "| Модель | Путь | Общих док. | Δ без правок, п.п. [ДИ] | Макнемар (p) "
        "| Δ ячейки F1, п.п. [ДИ] |",
        "|---|---|---|---|---|---|",
    ]

    def d(est):
        if est["low"] is None:
            return "—"
        f = lambda x: f"{x * 100:+.1f}".replace(".", ",")  # noqa: E731
        return f"{f(est['diff'])} [{f(est['low'])}; {f(est['high'])}]"

    for r in data["cells"]:
        v = r["vs_best"]
        if v is None:
            out.append(
                f"| {r['model']} | {PATH_TITLES.get(r['path'], r['path'])} | лучшая | — | — | — |"
            )
            continue
        out.append(
            f"| {r['model']} | {PATH_TITLES.get(r['path'], r['path'])} | {v['common_documents']} "
            f"| {d(v['doc_ok'])} | {fmt_p(v['mcnemar_doc_ok']['p_value'])} | {d(v['cells_f1'])} |"
        )
    return out


def render_report(cfg: MatrixConfig, data: dict[str, Any]) -> str:
    best = next(r for r in data["cells"] if r["cell"] == data["best_quality"])
    fast = next(r for r in data["cells"] if r["cell"] == data["fastest_not_worse_than_best"])
    lines = [
        f"# {cfg.name}: матрица «модель × путь × качество входа»",
        "",
        cfg.description.strip(),
        "",
        f"Документов в таблице: **{data['documents']}** из {data['documents_planned']} "
        "запланированных (общее для всех ячеек готовое начало списка; уровни раздаются по кругу, "
        f"по ≈{data['documents'] // 3} "
        "документов на уровень). Все числа — после постобработки P10, 95% интервалы — кластерный "
        "бутстреп по документам (D45). Скорость — секунд на страницу, замер от сети; пик VRAM — "
        "по всему GPU, «модель в VRAM» — доля весов и кэша, лежащая на GPU (`/api/ps`).",
        "",
        "## Сводная таблица",
        "",
        *_main_table(data),
        "",
        "## По уровням входа",
        "",
        *_level_table(data, "doc_ok", "Без правок в обязательных полях"),
        "",
        *_level_table(data, "cells_f1", "Ячейки: F1"),
        "",
        "## Сравнение с лучшей ячейкой (парное, общие документы)",
        "",
        f"Лучшая по доле «без правок» (при равенстве — по F1 ячеек): **{best['model']} / "
        f"{PATH_TITLES.get(best['path'], best['path'])}**. Самая быстрая из тех, что статистически "
        "не хуже лучшей (верхняя граница 95% интервала разности ≥ 0 и по «без правок», "
        "и по F1 ячеек): "
        f"**{fast['model']} / {PATH_TITLES.get(fast['path'], fast['path'])}**.",
        "",
        *_vs_best_table(data),
        "",
        "![Качество против скорости](quality_vs_speed.svg)",
        "",
        "![Качество по уровням](quality_by_level.svg)",
        "",
    ]
    if data["skipped"]:
        lines += ["## Пропущено", ""]
        lines += [
            f"- **{s['model']}** ({s['tag']}), пути {', '.join(s['paths'])}: {s['reason']}"
            for s in data["skipped"]
        ]
        lines.append("")
    lines += ["## Реальные документы", ""]
    if data["real"]:
        lines += [
            "Размеченные реальные документы (только метрики, значения не публикуются).",
            "",
            "| Модель | Путь | Док. | Без правок [95% ДИ] | Ячейки F1 |",
            "|---|---|---|---|---|",
        ]
        for r in data["real"]:
            o = r["overall"]
            lines.append(
                f"| {r['model']} | {PATH_TITLES.get(r['path'], r['path'])} | {r['items']} "
                f"| {fmt_metric('doc_ok', o['doc_ok'])} | {fmt_metric('cells_f1', o['cells_f1'])} |"
            )
    else:
        lines.append(
            "Размеченных реальных документов в `data/real/labels` нет — в таблице только "
            "синтетика. Когда разметка появится (P8), `eval-matrix run` досчитает их, "
            "остальное не пересчитывается."
        )
    lines.append("")
    return "\n".join(lines)


def write_matrix_report(cfg: MatrixConfig, data: dict[str, Any]) -> Path:
    out = report_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    cells = data["cells"]
    points = [
        {
            "model": r["model"],
            "path": r["path"],
            "x": r["overall"]["sec_per_page"]["value"],
            "y": r["overall"],
        }
        for r in cells
    ]
    quality = (("cells_f1", "Ячейки: F1, %"), ("doc_ok", "Без правок, %"))
    files = {
        "quality_vs_speed.svg": scatter_svg(points, quality, "Качество против скорости"),
        "quality_by_level.svg": level_bars_svg(
            [{"label": f"{r['model']} / {r['path']}", "levels": r["levels"]} for r in cells],
            quality,
            "Качество по уровням входа",
        ),
        "report.md": render_report(cfg, data),
        "summary.json": json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        "config.yaml": yaml.safe_dump(
            json.loads(cfg.model_dump_json()), allow_unicode=True, sort_keys=False
        ),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return out


__all__ = ["aggregate", "build_cells", "render_report", "write_matrix_report"]
