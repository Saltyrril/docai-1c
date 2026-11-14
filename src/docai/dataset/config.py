"""Конфиг сборки датасета (YAML → Pydantic)."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from docai.augment.levels import Level
from docai.augment.raster import DPI_CHOICES


class SplitShares(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: float = Field(gt=0)
    val: float = Field(ge=0)
    test: float = Field(gt=0)

    @model_validator(mode="after")
    def _sum_is_one(self) -> Self:
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("доли train + val + test должны давать 1")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "docai-1c-synth"
    version: str = "v1"
    seed: int = 42
    n_documents: int = Field(default=3000, ge=1)
    error_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    out: Path = Path("data/datasets/v1")
    levels: list[Level] = Field(default_factory=lambda: list(Level), min_length=1)
    dpi: list[int] = Field(default_factory=lambda: [200], min_length=1)
    split: SplitShares = SplitShares(train=0.8, val=0.1, test=0.1)
    workers: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        bad = [d for d in self.dpi if d not in DPI_CHOICES]
        if bad:
            raise ValueError(f"dpi должен быть из {DPI_CHOICES}, получено {bad}")
        if len(set(self.levels)) != len(self.levels) or len(set(self.dpi)) != len(self.dpi):
            raise ValueError("levels и dpi не должны повторяться")
        return self


def load_config(path: Path) -> DatasetConfig:
    return DatasetConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
