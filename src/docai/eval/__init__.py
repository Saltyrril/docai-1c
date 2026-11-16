"""Метрики качества и отчёты экспериментов."""

from docai.eval.catch import (
    CleanStats,
    CorruptionStats,
    corrupt,
    evaluate_clean,
    evaluate_corruptions,
    same_document,
)
from docai.eval.corruptions import CORRUPTIONS, CORRUPTIONS_BY_NAME, Corruption

__all__ = [
    "CORRUPTIONS",
    "CORRUPTIONS_BY_NAME",
    "CleanStats",
    "Corruption",
    "CorruptionStats",
    "corrupt",
    "evaluate_clean",
    "evaluate_corruptions",
    "same_document",
]
