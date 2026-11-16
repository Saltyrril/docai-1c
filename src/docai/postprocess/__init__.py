"""Постобработка после LLM: нормализация, автоисправления, уверенность, статус документа."""

from docai.postprocess.autofix import autofix_document
from docai.postprocess.confidence import score_fields
from docai.postprocess.config import PostprocessConfig
from docai.postprocess.evidence import SourceKind, SourceText
from docai.postprocess.models import (
    Correction,
    DocStatus,
    FieldConfidence,
    PostprocessReport,
    PostprocessResult,
)
from docai.postprocess.normalize import normalize_document
from docai.postprocess.process import decide_status, postprocess

__all__ = [
    "Correction",
    "DocStatus",
    "FieldConfidence",
    "PostprocessConfig",
    "PostprocessReport",
    "PostprocessResult",
    "SourceKind",
    "SourceText",
    "autofix_document",
    "decide_status",
    "normalize_document",
    "postprocess",
    "score_fields",
]
