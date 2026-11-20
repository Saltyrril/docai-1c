"""Конфиг сборки SFT-датасета (YAML → Pydantic)."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Вид примера: путь text (текстовый слой PDF) или картинка страницы одного из уровней P5.
Kind = Literal["text", "clean", "scan", "photo"]
KINDS: tuple[Kind, ...] = ("text", "clean", "scan", "photo")


class Mix(BaseModel):
    """Доли видов примеров. Каждый документ попадает в датасет одним видом."""

    model_config = ConfigDict(extra="forbid")
    text: float = Field(ge=0)
    clean: float = Field(ge=0)
    scan: float = Field(ge=0)
    photo: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum_is_one(self) -> Self:
        if abs(self.text + self.clean + self.scan + self.photo - 1.0) > 1e-9:
            raise ValueError("доли text + clean + scan + photo должны давать 1")
        return self

    def share(self, kind: Kind) -> float:
        return float(getattr(self, kind))


class FeedbackOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    data_dir: Path = Path("data/service")  # база и загрузки сервиса P14
    # Где ещё лежит синтетика проекта (кроме датасета P6): совпавшие с ней файлы — не реальные.
    synthetic_dirs: list[Path] = Field(
        default_factory=lambda: [Path("data/synth"), Path("data/demo")]
    )
    # Файлы, которых нет среди синтетики, могут быть реальными документами (правило data/real:
    # не отправлять во внешние сервисы). По умолчанию они в архив для Google Drive не идут.
    include_unknown: bool = False


class SftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "sft_v1"
    seed: int = 42
    dataset: Path = Path("data/datasets/v1")  # датасет P6
    out: Path = Path("data/finetune/sft_v1")
    n_train: int = Field(default=1200, ge=1)  # примеров (страниц или текстов), не документов
    n_val: int = Field(default=100, ge=0)
    mix: Mix = Mix(text=0.25, clean=0.25, scan=0.25, photo=0.25)
    # Как в конвейере извлечения (configs/extract.yaml): обучение должно видеть ровно те же
    # промпты, размер картинки и вид текста, что модель получит при работе.
    vision_prompt: str = "vision_v2"
    text_prompt: str = "text_v3"
    max_image_side: int = 1280
    max_text_chars: int = 12000
    text_tables: Literal["grid", "labeled", "labeled_v2"] = "grid"
    feedback: FeedbackOptions = FeedbackOptions()
    archive: bool = True  # упаковать в <out>.zip для загрузки в Google Drive
    # Параметры обучения едут в архиве (train_config.yaml): ноутбук Colab читает их оттуда.
    train_config: Path = Path("configs/finetune/train_v1.yaml")


def load_config(path: Path) -> SftConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SftConfig.model_validate(data)
