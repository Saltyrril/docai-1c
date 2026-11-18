"""Модели ответов и запросов API.

Контракт для 1С: плоский JSON без вложенных объектов там, где это возможно; числа
(суммы, количество, уверенность) — строки с точкой, даты — ISO 8601 (ГГГГ-ММ-ДД),
отсутствующее значение — null. Целые технические счётчики (страницы, номер строки) —
обычные числа JSON. Подробное описание — docs/API.md.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

JobStatus = Literal["queued", "running", "done", "failed"]

# Имя поля: seller.inn, totals.amount_with_vat, lines[0].price. Такие же имена в ответе.
FIELD_NAME = re.compile(r"^[a-z_]+(\.[a-z_]+)*$|^lines\[\d{1,4}\]\.[a-z_]+$")


class JobCreated(BaseModel):
    id: str = Field(description="Идентификатор задания")
    status: JobStatus
    queue_ahead: int = Field(description="Сколько заданий выполняется или ждёт раньше этого")


class FieldView(BaseModel):
    name: str = Field(description="Путь поля: number, seller.inn, lines[0].price")
    value: str | None = Field(description="Значение; null, если не найдено")
    confidence: str | None = Field(description="Уверенность 0.00–1.00 строкой")
    needs_review: bool = Field(description="Уверенность ниже порога: показать человеку")
    corrected: bool = Field(description="Значение исправлено автоматически после модели")


class LineView(BaseModel):
    n: int = Field(description="Номер строки, с 1")
    name: str | None = None
    unit_name: str | None = None
    unit_code: str | None = None
    quantity: str | None = None
    price: str | None = None
    amount_without_vat: str | None = None
    vat_rate: str | None = None
    vat_amount: str | None = None
    amount_with_vat: str | None = None
    min_confidence: str | None = Field(default=None, description="Наименьшая уверенность в строке")
    needs_review: bool = False


class IssueView(BaseModel):
    code: str = Field(description="Код проверки, например INN_CHECKSUM")
    severity: Literal["error", "warning"]
    field: str | None
    message: str


class CorrectionView(BaseModel):
    kind: Literal["normalize", "autofix"]
    field: str
    before: str | None
    after: str | None
    rule: str
    reason: str


class JobView(BaseModel):
    """Состояние задания и, когда оно выполнено, результат. Набор ключей всегда один."""

    id: str
    status: JobStatus
    queue_ahead: int | None = Field(default=None, description="Заданий впереди; null вне очереди")
    filename: str
    source_kind: str | None = Field(description="pdf | image | xml")
    mode: str
    created_at: str = Field(description="ISO 8601, UTC")
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = Field(default=None, description="Причина, если status = failed")

    # Результат (status = done).
    document_status: Literal["upload", "review", "reject"] | None = Field(
        default=None, description="upload — можно загружать, review — проверить, reject — отклонить"
    )
    document_status_ru: str | None = None
    status_reasons: list[str] = Field(default_factory=list)
    valid: bool | None = Field(default=None, description="Документ собран по схеме")
    doc_type: str | None = Field(
        default=None, description="upd | bill | act | torg12 | vat_invoice"
    )
    upd_status: str | None = Field(default=None, description="Статус УПД, «1» или «2»")
    route: str | None = Field(default=None, description="xml | text | vision | ocr")
    model: str | None = None
    pages: int | None = None
    preview_pages: int = Field(default=0, description="Сколько страниц доступно в /preview")
    seconds: str | None = Field(default=None, description="Время обработки, секунды строкой")
    mean_confidence: str | None = None
    fields: list[FieldView] = Field(default_factory=list)
    lines: list[LineView] = Field(default_factory=list)
    issues: list[IssueView] = Field(default_factory=list)
    corrections: list[CorrectionView] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    feedback_count: int = 0


class FeedbackIn(BaseModel):
    fields: dict[str, str | None] = Field(
        min_length=1,
        max_length=500,
        description="Поля, которые показаны человеку и им подтверждены или исправлены: "
        "имя → значение (числа строкой с точкой, даты ISO)",
    )
    author: str | None = Field(default=None, max_length=100, description="Кто проверял")
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("fields")
    @classmethod
    def _names(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        for name in value:
            if not FIELD_NAME.match(name):
                raise ValueError(f"недопустимое имя поля: {name!r}")
        return value


class FeedbackOut(BaseModel):
    job_id: str
    feedback_id: int
    total: int = Field(description="Сколько значений прислано")
    changed: int = Field(description="Из них отличаются от извлечённых")


class OllamaHealth(BaseModel):
    reachable: bool
    model: str
    model_available: bool = False
    model_loaded: bool = False
    error: str | None = None


class GpuHealth(BaseModel):
    available: bool
    vram_total_mib: int | None = None
    vram_free_mib: int | None = None
    error: str | None = None


class HealthView(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    worker_alive: bool
    queued: int
    running: int
    ollama: OllamaHealth
    gpu: GpuHealth
