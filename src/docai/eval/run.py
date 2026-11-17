"""Оценка целиком: выборка → ответы модели (кэш) → постобработка → сравнение → метрики.

`evaluate(config)` возвращает словарь, который пишется в metrics.json; отчёт строит
`docai.eval.report`. Постобработка применяется здесь, к кэшированным «сырым» ответам,
поэтому смена её констант (D42) не требует повторного прогона модели.
"""

import json
import logging
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from docai import __version__
from docai.eval.calibration import auroc_bootstrap, reliability_bins
from docai.eval.compare import compare_documents
from docai.eval.config import EvalConfig
from docai.eval.metrics import (
    GROUP_METRICS,
    GROUPINGS,
    DocRecord,
    count_matrix,
    field_metrics,
    main_metrics,
    record_counts,
)
from docai.eval.predict import Prediction, run_predictions
from docai.eval.sample import EvalItem, select_items
from docai.eval.stats import bootstrap, mcnemar, paired_bootstrap, resample_weights, totals
from docai.extract import finish
from docai.schemas.documents import parse_document
from docai.schemas.fields import FIELD_SPEC

log = logging.getLogger("docai.eval")

# Метрики, по которым сравниваются ответ модели до и после постобработки.
RAW_VS_POST = (
    "doc_ok",
    "header_accuracy",
    "required_header_accuracy",
    "text_anls",
    "rows_f1",
    "cells_f1",
)


def build_records(
    items: Sequence[EvalItem], predictions: dict[str, Prediction], cfg: EvalConfig
) -> list[DocRecord]:
    extract_cfg = cfg.extract.build()
    m = cfg.metrics
    opts = {"anls_threshold": m.anls_threshold, "row_match_threshold": m.row_match_threshold}
    records = []
    for item in items:
        truth = parse_document(item.truth.read_text(encoding="utf-8"))
        pred = predictions[item.item_id]
        record_kw: dict[str, Any] = {"truth": truth, "preview": item.preview}
        if pred.raw is None:
            post_cmp = raw_cmp = compare_documents(truth, None, **opts)
            status, route = None, None
        else:
            raw_result = pred.raw.result
            post = finish(pred.raw, extract_cfg.postprocess)
            raw_doc = parse_document(raw_result.document) if raw_result.document else None
            post_doc = parse_document(post.document) if post.document else None
            fields = post.postprocess.fields if post.postprocess else []
            post_cmp = compare_documents(truth, post_doc, confidences=fields, **opts)
            raw_cmp = compare_documents(truth, raw_doc, **opts)
            status = post.postprocess.status.value if post.postprocess else None
            meta = post.meta
            route = meta.route
            # Время из кэша: извлечение и второй путь. Постобработка (доли секунды) не входит:
            # она пересчитывается при каждой сборке отчёта, и отчёт перестал бы повторяться.
            seconds = meta.timings_s.get("total", 0.0) + (meta.cross_check_total_s or 0.0)
            record_kw |= {
                "pred": post_doc,
                "seconds": round(seconds, 3),
                "pages": meta.pages or item.pages,
                "peak_vram_mib": meta.vram.peak_mib if meta.vram else None,
                "prompt_tokens": meta.tokens.prompt,
                "completion_tokens": meta.tokens.completion,
                "errors": list(post.errors),
            }
        records.append(
            DocRecord(
                item_id=item.item_id,
                doc_id=item.doc_id,
                doc_type=item.doc_type,
                level=item.level,
                route=route,
                comparison=post_cmp,
                raw=raw_cmp,
                status=status,
                exception=pred.exception,
                real=item.real,
                has_errors=item.has_errors,
                **record_kw,
            )
        )
    return records


def _estimates(matrix, keys, metrics, cfg: EvalConfig) -> dict[str, dict]:
    b = cfg.bootstrap
    est = bootstrap(matrix, keys, metrics, n_resamples=b.n_resamples, ci=b.ci, seed=cfg.seed)
    return {name: e.as_dict() for name, e in est.items()}


def _doc_summary(r: DocRecord, counts: dict[str, float]) -> dict[str, Any]:
    c = r.comparison
    return {
        "item_id": r.item_id,
        "doc_id": r.doc_id,
        "doc_type": r.doc_type,
        "pred_doc_type": c.pred_doc_type,
        "level": r.level,
        "route": r.route,
        "status": r.status,
        "ok": c.ok,
        "ok_raw": r.raw.ok,
        "f1": round(c.f1, 4),
        "required_errors": c.required_errors,
        "wrong_header": [f.field for f in c.header if not f.correct],
        "rows_ref": c.n_ref_rows,
        "rows_pred": c.n_pred_rows,
        "rows_matched": c.rows_matched,
        "rows_exact": c.rows_exact,
        "seconds": r.seconds,
        "pages": r.pages,
        "peak_vram_mib": r.peak_vram_mib,
        "has_errors": r.has_errors,
        "exception": r.exception,
        "counts": {k: v for k, v in counts.items() if v},
    }


def _performance(records: Sequence[DocRecord]) -> dict[str, Any]:
    timed = [r for r in records if r.seconds is not None and r.pages]
    per_page = [r.seconds / r.pages for r in timed]
    per_doc = [r.seconds for r in timed]
    peaks = [r.peak_vram_mib for r in records if r.peak_vram_mib is not None]

    def q(values: list[float], p: float) -> float | None:
        return round(float(np.quantile(values, p)), 2) if values else None

    return {
        "documents_timed": len(timed),
        "pages": sum(r.pages for r in timed),
        "seconds_total": round(sum(per_doc), 1),
        "seconds_per_doc_mean": round(statistics.fmean(per_doc), 2) if per_doc else None,
        "seconds_per_doc_median": q(per_doc, 0.5),
        "seconds_per_page_median": q(per_page, 0.5),
        "seconds_per_page_p90": q(per_page, 0.9),
        "seconds_per_page_max": round(max(per_page), 2) if per_page else None,
        "peak_vram_mib_max": max(peaks) if peaks else None,
        "peak_vram_mib_median": round(float(np.median(peaks))) if peaks else None,
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
    }


def evaluate_records(records: Sequence[DocRecord], cfg: EvalConfig) -> dict[str, Any]:
    n_bins = cfg.metrics.calibration_bins
    metrics = main_metrics(n_bins)
    fns = {m.name: m.fn for m in metrics}
    counts = [record_counts(r, n_bins) for r in records]
    counts_raw = [record_counts(r, n_bins, raw=True) for r in records]
    matrix, keys = count_matrix(counts)
    matrix_raw, keys_raw = count_matrix(counts_raw)
    b = cfg.bootstrap

    overall = _estimates(matrix, keys, fns, cfg)
    raw_overall = _estimates(matrix_raw, keys_raw, {k: fns[k] for k in RAW_VS_POST}, cfg)

    # До и после постобработки — одни и те же документы: парный бутстреп и Макнемар.
    shared = sorted(set(keys) | set(keys_raw))
    align = {k: i for i, k in enumerate(keys)}
    align_raw = {k: i for i, k in enumerate(keys_raw)}
    a = np.column_stack(
        [matrix_raw[:, align_raw[k]] if k in align_raw else np.zeros(len(records)) for k in shared]
    )
    p = np.column_stack(
        [matrix[:, align[k]] if k in align else np.zeros(len(records)) for k in shared]
    )
    paired = paired_bootstrap(
        a, p, shared, {k: fns[k] for k in RAW_VS_POST},
        n_resamples=b.n_resamples, ci=b.ci, seed=cfg.seed,
    )  # fmt: skip
    raw_vs_post = {
        "metrics": {k: v.as_dict() for k, v in paired.items()},
        "mcnemar_doc_ok": mcnemar(
            [r.raw.ok for r in records], [r.comparison.ok for r in records]
        ).as_dict(),
    }

    groups: dict[str, Any] = {}
    for gname, (title, key_fn) in GROUPINGS.items():
        values = sorted({key_fn(r) or "—" for r in records})
        table = {}
        for value in values:
            idx = [i for i, r in enumerate(records) if (key_fn(r) or "—") == value]
            sub = matrix[idx]
            table[value] = {
                "documents": len(idx),
                "metrics": _estimates(sub, keys, {k: fns[k] for k in GROUP_METRICS}, cfg),
            }
        groups[gname] = {"title_ru": title, "values": table}

    names = [s.name for s in FIELD_SPEC if any(f"f:{s.name}:ref" in c for c in counts)]
    point = totals(matrix, keys)
    per_field = {}
    field_est = _estimates(matrix, keys, field_metrics(names), cfg)
    for name in names:
        spec = next(s for s in FIELD_SPEC if s.name == name)
        per_field[name] = {
            "title_ru": spec.title_ru,
            "line": spec.is_line_field,
            "reference_values": int(point[f"f:{name}:ref"]),
            **{k.split("|")[1]: v for k, v in field_est.items() if k.split("|")[0] == name},
        }

    weights = resample_weights(len(records), b.n_resamples, cfg.seed)
    per_doc_conf = [
        (
            [pt.confidence for pt in r.comparison.confidence],
            [pt.correct for pt in r.comparison.confidence],
        )
        for r in records
    ]
    auc, auc_low, auc_high = auroc_bootstrap(per_doc_conf, weights, b.ci)
    calibration = {
        "points": int(point["conf:n"]),
        "bins": reliability_bins(point, n_bins),
        "auroc": {"value": round(auc, 6), "low": round(auc_low, 6), "high": round(auc_high, 6)},
        "review_threshold": cfg.extract.build().postprocess.review_threshold,
    }

    return {
        "overall": overall,
        "metric_titles": {m.name: m.title_ru for m in metrics},
        "higher_is_better": {m.name: m.higher_is_better for m in metrics},
        "raw": raw_overall,
        "raw_vs_post": raw_vs_post,
        "groups": groups,
        "fields": per_field,
        "calibration": calibration,
        "performance": _performance(records),
        "documents": [_doc_summary(r, c) for r, c in zip(records, counts, strict=True)],
    }


def compare_runs(current: dict[str, Any], other: dict[str, Any], cfg: EvalConfig) -> dict[str, Any]:
    """Парное сравнение с другим metrics.json на общих документах (B — текущий прогон)."""
    mine = {d["item_id"]: d for d in current["documents"]}
    theirs = {d["item_id"]: d for d in other["documents"]}
    common = sorted(set(mine) & set(theirs))
    if not common:
        return {"common_documents": 0}
    n_bins = cfg.metrics.calibration_bins
    fns = {m.name: m.fn for m in main_metrics(n_bins)}
    keys = sorted({k for i in common for k in (*mine[i]["counts"], *theirs[i]["counts"])})
    a = np.array([[theirs[i]["counts"].get(k, 0.0) for k in keys] for i in common], dtype=float)
    b_ = np.array([[mine[i]["counts"].get(k, 0.0) for k in keys] for i in common], dtype=float)
    b = cfg.bootstrap
    paired = paired_bootstrap(a, b_, keys, fns, n_resamples=b.n_resamples, ci=b.ci, seed=cfg.seed)
    return {
        "other": other.get("config", {}).get("name"),
        "common_documents": len(common),
        "metrics": {k: v.as_dict() for k, v in paired.items()},
        "mcnemar_doc_ok": mcnemar(
            [bool(theirs[i]["ok"]) for i in common], [bool(mine[i]["ok"]) for i in common]
        ).as_dict(),
    }


ProgressFn = Callable[[int, int, EvalItem, Prediction, bool], None]


def evaluate(
    cfg: EvalConfig,
    *,
    progress: ProgressFn | None = None,
    notice: Callable[[str], None] | None = None,
    extract_fn=None,
) -> tuple[dict[str, Any], list[EvalItem], list[DocRecord]]:
    items = select_items(cfg)
    if not items:
        raise ValueError("выборка пуста: проверьте data в конфиге")
    kwargs = {"extract_fn": extract_fn} if extract_fn is not None else {}
    predictions = run_predictions(
        items,
        cfg.extract.build(),
        cfg.extract.mode,
        cfg.cache_dir,
        cross_check=cfg.extract.cross_check,
        progress=progress,
        power_guard=cfg.power_guard,
        notice=notice,
        **kwargs,
    )
    records = build_records(items, predictions, cfg)
    result = evaluate_records(records, cfg)
    extract_cfg = cfg.extract.build()
    result = {
        "docai_version": __version__,
        "config": json.loads(cfg.model_dump_json()),
        "extract_config": extract_cfg.model_dump(mode="json"),
        "sample": {
            "items": len(items),
            "documents": len({it.doc_id for it in items}),
            "real": any(it.real for it in items),
        },
        **result,
    }
    if cfg.compare_with is not None:
        other = json.loads(Path(cfg.compare_with).read_text(encoding="utf-8"))
        result["comparison"] = compare_runs(result, other, cfg)
    return result, items, records
