"""Конфиг оценки (P11): что извлекаем, на какой выборке и как считаем метрики.

Запуск: `uv run docai eval --config configs/eval/<имя>.yaml`. Один конфиг — один
эксперимент: модель, путь извлечения, версии промптов, выборка, seed.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from docai.extract import ExtractConfig, Mode
from docai.extract import load_config as load_extract_config

Level = Literal["clean", "scan", "photo"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    """Выборка документов."""

    # dataset — синтетический датасет с манифестом (P6); real — разметка data/real (P8).
    source: Literal["dataset", "real"] = "dataset"
    dataset: Path = Path("data/datasets/v1")
    split: str = "test"
    # Сколько разных документов взять (None — все). Выборка стратифицирована по типу.
    n_docs: int | None = 100
    levels: list[Level] = Field(default_factory=lambda: ["clean", "scan", "photo"])
    # rotate — каждый документ на одном уровне (уровни по кругу, поровну);
    # all — каждый документ на каждом уровне (парное сравнение уровней, ×len(levels) времени).
    level_assignment: Literal["rotate", "all"] = "rotate"
    doc_types: list[str] | None = None
    # Документы, которые нельзя брать в оценку (например, по ним подбирались константы).
    exclude_doc_ids: list[str] = Field(default_factory=list)
    dpi: int = 200
    # Для real: папка с разметкой и исходниками (data/real).
    real_root: Path = Path("data/real")


class ExtractSection(_Strict):
    """Что и как извлекаем. Поля, кроме config и mode, перекрывают значения из config."""

    config: Path | None = Path("configs/extract.yaml")
    mode: Mode = Mode.AUTO
    cross_check: Mode | None = None
    model: str | None = None
    text_prompt: str | None = None
    vision_prompt: str | None = None
    date_format: Literal["iso", "printed"] | None = None
    # Чистый уровень: подавать PDF (путь auto → текстовый слой) или картинку.
    clean_input: Literal["pdf", "image"] = "pdf"

    def build(self) -> ExtractConfig:
        base = load_extract_config(self.config)
        overrides = {
            key: value
            for key, value in (
                ("model", self.model),
                ("text_prompt", self.text_prompt),
                ("vision_prompt", self.vision_prompt),
                ("date_format", self.date_format),
            )
            if value is not None
        }
        return base.model_copy(update=overrides)


class MetricsConfig(_Strict):
    # Порог ANLS (Biten et al., 2019): при нормированном расстоянии ≥ τ оценка 0.
    anls_threshold: float = 0.5
    # Пара строк считается сопоставленной, если сходство не ниже порога (D44).
    row_match_threshold: float = 0.5
    calibration_bins: int = 10


class BootstrapConfig(_Strict):
    n_resamples: int = 2000
    ci: float = 0.95


class ReportConfig(_Strict):
    out_dir: Path = Path("reports")
    # Дата в имени папки отчёта; None — сегодня.
    date: str | None = None
    worst: int = 5
    # Длинная сторона картинки худшего примера в отчёте, px.
    image_side: int = 1000


class EvalConfig(_Strict):
    name: str
    description: str = ""
    seed: int = 2026
    data: DataConfig = Field(default_factory=DataConfig)
    extract: ExtractSection = Field(default_factory=ExtractSection)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    # Кэш ответов модели: <predictions_dir>/<name>/<item_id>.json (в git не попадает).
    predictions_dir: Path = Path("data/experiments/eval")
    # metrics.json другой конфигурации для парного сравнения на общих документах.
    compare_with: Path | None = None
    # Считать только от сети: ждать зарядку, пересчитывать документы с батареей или сном (D47).
    power_guard: bool = True

    @model_validator(mode="after")
    def _check(self) -> "EvalConfig":
        if not self.name or any(c in self.name for c in '\\/:*?"<>| '):
            raise ValueError('name: имя без пробелов и символов \\/:*?"<>|')
        if self.extract.mode is Mode.TEXT and set(self.data.levels) - {"clean"}:
            raise ValueError("mode text читает текстовый слой PDF: допустим только уровень clean")
        if self.extract.mode is Mode.TEXT and self.extract.clean_input == "image":
            raise ValueError("mode text несовместим с clean_input: image")
        return self

    @property
    def cache_dir(self) -> Path:
        return self.predictions_dir / self.name


def load_eval_config(path: Path) -> EvalConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvalConfig.model_validate(data)
