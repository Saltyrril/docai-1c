"""Счётчики по документам и метрики из них (определения — D44, D45).

Каждый документ превращается в словарь счётчиков (сколько эталонных полей, сколько верных,
сколько строк и т. д.). Любая метрика — функция от сумм счётчиков, поэтому одна и та же
матрица «документы × счётчики» даёт и точечную оценку, и бутстреп, и парное сравнение.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from docai.eval.calibration import bin_index, bin_keys, ece_fn, mce_fn
from docai.eval.compare import TEXT_NORMS, DocComparison
from docai.eval.stats import MetricFn, f1, ratio
from docai.schemas.common import BaseDocument
from docai.schemas.fields import FIELDS_BY_NAME


@dataclass(slots=True)
class DocRecord:
    """Всё, что известно об одном прогоне: элемент выборки, сравнение, время, статус."""

    item_id: str
    doc_id: str
    doc_type: str
    level: str
    route: str | None
    comparison: DocComparison  # после постобработки (то, что получит 1С)
    raw: DocComparison  # ответ модели до постобработки
    status: str | None  # upload | review | reject (None — извлечение упало)
    seconds: float | None = None  # извлечение + постобработка
    pages: int | None = None
    peak_vram_mib: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    exception: str | None = None
    errors: list[str] = field(default_factory=list)
    real: bool = False
    has_errors: bool = False
    # Документы для худших примеров отчёта (в metrics.json не пишутся).
    truth: BaseDocument | None = None
    pred: BaseDocument | None = None
    preview: Path | None = None


def comparison_counts(c: DocComparison, n_bins: int) -> dict[str, float]:
    out: dict[str, float] = {
        "docs": 1,
        "extracted": c.extracted,
        "ok": c.ok,
        "type_ok": c.type_ok,
        "hdr:ref": len(c.header),
        "hdr:correct": c.header_correct,
        "hdr:req_ref": sum(f.required for f in c.header),
        "hdr:req_correct": sum(f.required and f.correct for f in c.header),
        "hdr:empty_ref": len(c.empty_ref),
        "hdr:halluc": sum(pred is not None for _, pred in c.empty_ref),
        "rows:ref": c.n_ref_rows,
        "rows:pred": c.n_pred_rows,
        "rows:matched": c.rows_matched,
        "rows:exact": c.rows_exact,
        "txt:ref": 0,
        "txt:anls": 0.0,
        "txt:correct": 0,
    }
    cells = c.cells
    out |= {"cells:ref": cells.ref, "cells:pred": cells.pred, "cells:correct": cells.correct}
    for f in c.header:
        out[f"f:{f.field}:ref"] = 1
        out[f"f:{f.field}:correct"] = f.correct
        if f.anls is not None:
            out[f"f:{f.field}:anls"] = f.anls
            out["txt:ref"] += 1
            out["txt:anls"] += f.anls
            out["txt:correct"] += f.correct
    for name, counts in c.line_fields.items():
        out[f"f:{name}:ref"] = counts.ref
        out[f"f:{name}:pred"] = counts.pred
        out[f"f:{name}:correct"] = counts.correct
        if FIELDS_BY_NAME[name].norm in TEXT_NORMS:
            out[f"f:{name}:anls"] = counts.anls
            out["txt:ref"] += counts.ref
            out["txt:anls"] += counts.anls
            out["txt:correct"] += counts.correct
    for key in bin_keys(n_bins):
        out[key] = 0.0
    out["conf:n"] = len(c.confidence)
    out["conf:correct"] = sum(p.correct for p in c.confidence)
    out["conf:sum"] = sum(p.confidence for p in c.confidence)
    out["conf:brier"] = sum((p.confidence - p.correct) ** 2 for p in c.confidence)
    for p in c.confidence:
        b = bin_index(p.confidence, n_bins)
        out[f"cal:{b}:n"] += 1
        out[f"cal:{b}:conf"] += p.confidence
        out[f"cal:{b}:correct"] += p.correct
    return out


def record_counts(r: DocRecord, n_bins: int, *, raw: bool = False) -> dict[str, float]:
    c = r.raw if raw else r.comparison
    out = comparison_counts(c, n_bins)
    status = r.status
    out |= {
        "st:upload": status == "upload",
        "st:review": status == "review",
        "st:reject": status == "reject",
        "st:failed": status is None,
        "st:upload_ok": status == "upload" and c.ok,
        "st:upload_bad": status == "upload" and not c.ok,
        "st:flag_bad": status in ("review", "reject") and not c.ok,
        "st:bad": not c.ok,
    }
    has_perf = r.seconds is not None and bool(r.pages)
    out |= {
        "perf:docs": has_perf,
        "perf:seconds": r.seconds if has_perf else 0.0,
        "perf:pages": r.pages if has_perf else 0,
    }
    return out


def count_matrix(rows: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, list[str]]:
    keys = sorted({k for row in rows for k in row})
    matrix = np.array([[float(row.get(k, 0.0)) for k in keys] for row in rows], dtype=float)
    return matrix.reshape(len(rows), len(keys)), keys


@dataclass(frozen=True, slots=True)
class MetricDef:
    name: str
    title_ru: str
    fn: MetricFn
    # Чем больше, тем лучше (для подписи разности в сравнении).
    higher_is_better: bool = True


def _brier(s):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(s["conf:brier"], float) / s["conf:n"]


def main_metrics(n_bins: int) -> list[MetricDef]:
    """Основные метрики: все с 95% интервалами в отчёте."""
    m = MetricDef
    return [
        m("doc_ok", "Документов без ошибок в обязательных полях (можно загружать без правок)",
          ratio("ok", "docs")),
        m("doc_type_accuracy", "Тип документа верен", ratio("type_ok", "docs")),
        m("extracted", "Документ собран (валидный JSON)", ratio("extracted", "docs")),
        m("header_accuracy", "Поля шапки: accuracy", ratio("hdr:correct", "hdr:ref")),
        m("required_header_accuracy", "Обязательные поля шапки: accuracy",
          ratio("hdr:req_correct", "hdr:req_ref")),
        m("hallucination", "Выдумки: значение при пустом эталоне",
          ratio("hdr:halluc", "hdr:empty_ref"), higher_is_better=False),
        m("text_anls", "Текстовые поля: ANLS", ratio("txt:anls", "txt:ref")),
        m("text_accuracy", "Текстовые поля: accuracy", ratio("txt:correct", "txt:ref")),
        m("rows_precision", "Строки: precision (найдена пара)", ratio("rows:matched", "rows:pred")),
        m("rows_recall", "Строки: recall (найдена пара)", ratio("rows:matched", "rows:ref")),
        m("rows_f1", "Строки: F1 (найдена пара)", f1("rows:matched", "rows:pred", "rows:ref")),
        m("rows_exact_f1", "Строки: F1 (строка верна целиком)",
          f1("rows:exact", "rows:pred", "rows:ref")),
        m("cells_precision", "Ячейки: precision", ratio("cells:correct", "cells:pred")),
        m("cells_recall", "Ячейки: recall", ratio("cells:correct", "cells:ref")),
        m("cells_f1", "Ячейки: F1", f1("cells:correct", "cells:pred", "cells:ref")),
        m("upload_rate", "Статус «можно загружать»", ratio("st:upload", "docs")),
        m("upload_precision", "Среди «можно загружать» — действительно без ошибок",
          ratio("st:upload_ok", "st:upload")),
        m("silent_error", "Молча загружен с ошибкой (доля всех документов)",
          ratio("st:upload_bad", "docs"), higher_is_better=False),
        m("bad_flagged", "Документ с ошибкой отправлен на проверку или отклонён",
          ratio("st:flag_bad", "st:bad")),
        m("conf_accuracy", "Доля верных среди полей с уверенностью",
          ratio("conf:correct", "conf:n")),
        m("conf_mean", "Средняя уверенность поля", ratio("conf:sum", "conf:n")),
        m("ece", "ECE (10 корзин)", ece_fn(n_bins), higher_is_better=False),
        m("mce", "MCE (10 корзин)", mce_fn(n_bins), higher_is_better=False),
        m("brier", "Brier", _brier, higher_is_better=False),
        m("sec_per_page", "Секунд на страницу", ratio("perf:seconds", "perf:pages"),
          higher_is_better=False),
    ]  # fmt: skip


# Метрики для таблиц по группам (тип документа, уровень, путь): короче основного списка.
GROUP_METRICS = (
    "doc_ok",
    "doc_type_accuracy",
    "header_accuracy",
    "text_anls",
    "rows_f1",
    "cells_f1",
    "silent_error",
    "ece",
    "sec_per_page",
)


def field_metrics(field_names: Sequence[str]) -> dict[str, MetricFn]:
    """Accuracy каждого поля (и ANLS для текстовых, precision для полей строк)."""
    out: dict[str, MetricFn] = {}
    for name in field_names:
        spec = FIELDS_BY_NAME[name]
        out[f"{name}|accuracy"] = ratio(f"f:{name}:correct", f"f:{name}:ref")
        if spec.norm in TEXT_NORMS:
            out[f"{name}|anls"] = ratio(f"f:{name}:anls", f"f:{name}:ref")
        if spec.is_line_field:
            out[f"{name}|precision"] = ratio(f"f:{name}:correct", f"f:{name}:pred")
    return out


GroupKey = Callable[[DocRecord], str | None]

GROUPINGS: dict[str, tuple[str, GroupKey]] = {
    "doc_type": ("Тип документа", lambda r: r.doc_type),
    "level": ("Уровень качества", lambda r: r.level),
    "route": ("Путь извлечения", lambda r: r.route or "—"),
}
