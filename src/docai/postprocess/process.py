"""Постобработка документа после LLM: нормализация → проверки → автоисправления →
уверенность → статус. Порядок и правила — D41, D42, D43."""

from collections.abc import Sequence

from docai.postprocess.autofix import autofix_document
from docai.postprocess.confidence import score_fields
from docai.postprocess.config import PostprocessConfig
from docai.postprocess.evidence import SourceKind, SourceText
from docai.postprocess.models import (
    DocStatus,
    FieldConfidence,
    PostprocessReport,
    PostprocessResult,
)
from docai.postprocess.normalize import normalize_document
from docai.schemas.common import BaseDocument
from docai.validation import Severity, validate_document


def decide_status(
    doc: BaseDocument | None,
    fields: Sequence[FieldConfidence],
    errors_left: int,
    config: PostprocessConfig,
) -> tuple[DocStatus, list[str], list[FieldConfidence]]:
    """Статус, причины и поля, которые нужно проверить глазами.

    Отклонить: документа нет, в нём нет строк или не прочитана половина полей
    (уверенность ниже `reject_threshold` у доли `reject_low_share` и больше).
    Нужна проверка: есть поле ниже `review_threshold` или остались ошибки проверок.
    Иначе можно загружать.
    """
    review = sorted(
        (f for f in fields if f.confidence < config.review_threshold), key=lambda f: f.confidence
    )
    if doc is None:
        return DocStatus.REJECT, ["документ не собран"], review
    if not doc.lines:
        return DocStatus.REJECT, ["в документе нет строк таблицы"], review
    unread = [f for f in fields if f.confidence < config.reject_threshold]
    if fields and len(unread) / len(fields) >= config.reject_low_share:
        share = len(unread) / len(fields)
        reason = f"не прочитана половина полей: {len(unread)} из {len(fields)} ({share:.0%})"
        return DocStatus.REJECT, [reason], review
    reasons = []
    if errors_left:
        reasons.append(f"остались ошибки проверок: {errors_left}")
    if review:
        reasons.append(f"полей с низкой уверенностью: {len(review)}")
    if reasons:
        return DocStatus.REVIEW, reasons, review
    return DocStatus.UPLOAD, ["ошибок нет, уверенность во всех полях достаточна"], review


def postprocess(
    doc: BaseDocument | None,
    *,
    route: str,
    texts: Sequence[tuple[SourceKind, str]] = (),
    alternates: Sequence[BaseDocument] = (),
    config: PostprocessConfig | None = None,
) -> PostprocessResult:
    """Нормализовать, проверить и при возможности исправить документ, оценить уверенность.

    route: xml | text | vision | ocr. Для XML исправлений нет: данные взяты у отправителя
    как есть, расхождение в них — ошибка документа, а не чтения.
    texts: тексты-источники (текстовый слой PDF, OCR) для проверки «значение дословно есть».
    alternates: документы, извлечённые другими путями из того же файла.
    """
    cfg = config or PostprocessConfig()
    if doc is None:
        status, reasons, _ = decide_status(None, [], 0, cfg)
        report = PostprocessReport(status=status, status_ru=status.title_ru, reasons=reasons)
        return PostprocessResult(document=None, report=report)

    sources = [SourceText.build(kind, text) for kind, text in texts]
    current, corrections = normalize_document(doc)
    issues_before = validate_document(current)
    if cfg.autofix and route != "xml" and any(i.severity is Severity.ERROR for i in issues_before):
        current, fixes = autofix_document(current, sources)
        corrections += fixes
    issues_after = validate_document(current)

    corrected = frozenset(c.field for c in corrections if c.kind == "autofix")
    fields = score_fields(
        current,
        issues_after,
        route=route,
        sources=sources,
        alternates=alternates,
        corrected=corrected,
        config=cfg,
    )
    errors_left = sum(1 for i in issues_after if i.severity is Severity.ERROR)
    status, reasons, review = decide_status(current, fields, errors_left, cfg)
    scored = [f.confidence for f in fields]
    report = PostprocessReport(
        status=status,
        status_ru=status.title_ru,
        reasons=reasons,
        mean_confidence=round(sum(scored) / len(scored), 4) if scored else None,
        corrections=corrections,
        issues_before=issues_before,
        issues_after=issues_after,
        review_fields=review,
        fields=fields,
    )
    return PostprocessResult(document=current, report=report)
