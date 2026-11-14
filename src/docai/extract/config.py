"""Настройки конвейера извлечения. Значения по умолчанию совпадают с `configs/extract.yaml`."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class OllamaOptions(BaseModel):
    """Параметры генерации, которые уходят в `options` запроса Ollama.

    Все параметры сэмплинга задаются явно: в Modelfile qwen3.5 стоят temperature 1
    и presence_penalty 1.5, а результат не должен от них зависеть (D38).
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.0
    seed: int = 42
    top_k: int = 20
    top_p: float = 0.95
    presence_penalty: float = 0.0
    num_ctx: int = 8192
    num_predict: int = 4096


class ExtractConfig(BaseModel):
    """Всё, от чего зависит результат извлечения; копия попадает в результат."""

    model_config = ConfigDict(extra="forbid")

    model: str = "qwen3.5:2b-q8_0"
    ollama_url: str = "http://127.0.0.1:11434"
    # Режим рассуждений Qwen3.5 в Ollama включён по умолчанию; выключается `think: false` (D38).
    think: bool = False
    keep_alive: str = "5m"
    timeout_s: float = 600.0
    options: OllamaOptions = Field(default_factory=OllamaOptions)

    # Версии промптов: имя файла без расширения в src/docai/extract/prompts/.
    text_prompt: str = "text_v1"
    vision_prompt: str = "vision_v1"

    # Маршрутизация: страница PDF считается текстовой, если в ней столько непробельных символов.
    min_text_chars_per_page: int = 30
    # Vision-путь: длинная сторона картинки перед моделью.
    max_image_side: int = 1280
    # Текстовый путь: если текст всего документа длиннее, страницы идут в модель по одной.
    max_text_chars: int = 12000

    # OCR-путь (Tesseract).
    tesseract_cmd: str | None = None
    ocr_lang: str = "rus"
    ocr_psm: int = 3
    ocr_dpi: int = 300

    # Мониторинг VRAM: период опроса nvidia-smi и порог, после которого пишется предупреждение.
    vram_poll_s: float = 0.5
    vram_limit_gib: float = 3.8


def load_config(path: Path | None) -> ExtractConfig:
    """Конфиг из YAML; без пути — значения по умолчанию."""
    if path is None:
        return ExtractConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExtractConfig.model_validate(data)
