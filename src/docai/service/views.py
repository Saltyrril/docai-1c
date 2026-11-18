"""Из результата конвейера — плоское представление для 1С (см. models.py и docs/API.md)."""

from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from docai.extract.result import ExtractionResult
from docai.schemas.common import BaseDocument
from docai.schemas.documents import parse_document
from docai.service.models import (
    CorrectionView,
    FieldView,
    IssueView,
    JobView,
    LineView,
)
from docai.service.store import JobRow

_LINE_COLUMNS = (
    "name",
    "unit_name",
    "unit_code",
    "quantity",
    "price",
    "amount_without_vat",
    "vat_rate",
    "vat_amount",
    "amount_with_vat",
)


def number_str(value: Decimal | float) -> str:
    """Число строкой с точкой, без экспоненты: Decimal('1E+2') → '100'."""
    return format(value if isinstance(value, Decimal) else Decimal(str(value)), "f")


def _scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return number_str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def flat_values(document: BaseDocument) -> dict[str, str | None]:
    """Все значения документа по путям: `seller.inn`, `lines[0].price`."""
    out: dict[str, str | None] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                walk(f"{prefix}.{key}" if prefix else key, child)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(f"{prefix}[{index}]", child)
        else:
            out[prefix] = _scalar(node)

    walk("", document.model_dump())
    return out


def extracted_values(result: ExtractionResult | None) -> dict[str, str | None]:
    """Значения, которые сервис отдал 1С: с ними сравниваются правки человека."""
    if result is None or result.document is None:
        return {}
    return flat_values(parse_document(result.document))


def same_value(before: str | None, after: str | None) -> bool:
    """Значения совпадают: пустые равны, числа сравниваются как числа («10» = «10.00»)."""
    left = (before or "").strip()
    right = (after or "").strip()
    if left == right:
        return True
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def _conf(value: float | None) -> str | None:
    return None if value is None else f"{value:.2f}"


def _fill_result(view: JobView, result: ExtractionResult) -> None:
    meta = result.meta
    view.valid = result.valid
    view.route = meta.route
    view.model = meta.model
    view.pages = meta.pages
    total = meta.timings_s.get("total")
    view.seconds = None if total is None else number_str(total)
    view.warnings = list(result.warnings)
    view.errors = list(result.errors)

    document = parse_document(result.document) if result.document is not None else None
    values = flat_values(document) if document is not None else {}
    if document is not None:
        view.doc_type = _scalar(document.doc_type)
        view.upd_status = _scalar(values.get("upd_status"))

    report = result.postprocess
    review_names: set[str] = set()
    confidence: dict[str, float] = {}
    if report is not None:
        view.document_status = report.status.value
        view.document_status_ru = report.status_ru
        view.status_reasons = list(report.reasons)
        view.mean_confidence = _conf(report.mean_confidence)
        review_names = {f.field for f in report.review_fields}
        corrected = {f.field for f in report.fields if f.corrected}
        for item in report.fields:
            confidence[item.field] = item.confidence
            view.fields.append(
                FieldView(
                    name=item.field,
                    value=values.get(item.field, item.value),
                    confidence=_conf(item.confidence),
                    needs_review=item.field in review_names,
                    corrected=item.field in corrected,
                )
            )
        view.issues = [
            IssueView(
                code=issue.code.value,
                severity=issue.severity.value,
                field=issue.field,
                message=issue.message,
            )
            for issue in report.issues_after
        ]
        view.corrections = [CorrectionView(**c.model_dump()) for c in report.corrections]
    else:
        view.fields = [
            FieldView(name=n, value=v, confidence=None, needs_review=False, corrected=False)
            for n, v in values.items()
        ]

    if document is not None:
        for index in range(len(document.lines)):
            prefix = f"lines[{index}]."
            confs = [c for name, c in confidence.items() if name.startswith(prefix)]
            line = LineView(
                n=index + 1,
                min_confidence=_conf(min(confs)) if confs else None,
                needs_review=any(name.startswith(prefix) for name in review_names),
                **{col: values.get(f"{prefix}{col}") for col in _LINE_COLUMNS},
            )
            view.lines.append(line)


def build_job_view(
    row: JobRow,
    result: ExtractionResult | None,
    *,
    queue_ahead: int | None,
    feedback_count: int,
) -> JobView:
    view = JobView(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        queue_ahead=queue_ahead,
        filename=row.filename,
        source_kind=row.source_kind,
        mode=row.mode,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        error=row.error,
        preview_pages=row.preview_pages,
        feedback_count=feedback_count,
    )
    if result is not None:
        _fill_result(view, result)
    return view
