"""Результат постобработки: исправления, уверенность по полям, статус документа."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from docai.schemas.common import BaseDocument
from docai.validation import ValidationIssue


class DocStatus(StrEnum):
    UPLOAD = "upload"  # можно загружать
    REVIEW = "review"  # нужна проверка
    REJECT = "reject"  # отклонить

    @property
    def title_ru(self) -> str:
        return _TITLES[self]


_TITLES = {
    DocStatus.UPLOAD: "можно загружать",
    DocStatus.REVIEW: "нужна проверка",
    DocStatus.REJECT: "отклонить",
}


class Correction(BaseModel):
    """Одно изменение значения после LLM. Пишется в журнал всегда."""

    kind: Literal["normalize", "autofix"]
    field: str
    before: str | None
    after: str | None
    rule: str
    reason: str


class FieldConfidence(BaseModel):
    """Уверенность в одном поле и из чего она сложилась."""

    field: str
    value: str | None
    confidence: float
    # Множители правдоподобия по сигналам (произведение по одноимённым сигналам).
    signals: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    corrected: bool = False


class PostprocessReport(BaseModel):
    status: DocStatus
    status_ru: str
    reasons: list[str] = Field(default_factory=list)
    mean_confidence: float | None = None
    corrections: list[Correction] = Field(default_factory=list)
    # Замечания проверок до автоисправлений (после нормализации) и после них.
    issues_before: list[ValidationIssue] = Field(default_factory=list)
    issues_after: list[ValidationIssue] = Field(default_factory=list)
    review_fields: list[FieldConfidence] = Field(default_factory=list)
    fields: list[FieldConfidence] = Field(default_factory=list)


class PostprocessResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    document: BaseDocument | None
    report: PostprocessReport
