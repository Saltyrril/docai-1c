"""Результат извлечения: документ по нашей схеме и всё, что нужно для воспроизведения и замеров."""

from typing import Any

from pydantic import BaseModel, Field

from docai.extract.vram import VramStats


class LlmCall(BaseModel):
    """Один запрос к модели: какая часть документа, токены и длительности Ollama (с)."""

    part: str
    prompt_tokens: int
    completion_tokens: int
    done_reason: str | None
    total_s: float
    load_s: float
    prompt_eval_s: float
    eval_s: float
    image_size: tuple[int, int] | None = None
    text_chars: int | None = None


class Tokens(BaseModel):
    prompt: int = 0
    completion: int = 0


class Meta(BaseModel):
    docai_version: str
    started_at: str
    source: list[str]
    mode: str
    route: str | None = None
    route_reason: str | None = None
    input_kind: str | None = None
    pages: int | None = None
    model: str | None = None
    prompt_version: str | None = None
    ocr_engine: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)
    tokens: Tokens = Field(default_factory=Tokens)
    llm_calls: list[LlmCall] = Field(default_factory=list)
    vram: VramStats | None = None


class ExtractionResult(BaseModel):
    """Итог извлечения.

    `document` — JSON по схеме `docai.schemas.Document` (уже проверен Pydantic) или None,
    если собрать документ не удалось; `valid` — документ есть. Ошибки отдельных частей
    (например, одной страницы из трёх) остаются в `errors`, даже если документ собран.
    """

    document: dict[str, Any] | None
    valid: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    meta: Meta
