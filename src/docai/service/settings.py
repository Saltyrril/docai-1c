"""Настройки HTTP-сервиса. Значения по умолчанию подходят для работы на ноутбуке."""

from pathlib import Path

from pydantic import BaseModel, Field

from docai.extract.config import ExtractConfig

# Сервис слушает только петлевой адрес: документы и результаты не должны уходить в сеть (D53).
HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class ServiceSettings(BaseModel):
    # Рабочая папка: SQLite, загруженные файлы и картинки страниц. data/ не коммитится.
    data_dir: Path = Path("data/service")
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    # Сколько заданий может ждать в очереди; сверх этого — ответ 429.
    max_queued: int = 50
    # Картинки страниц для формы проверки: длинная сторона и число страниц.
    preview_max_side: int = 1600
    preview_max_pages: int = 30
    # Допустимые значения заголовка Host (защита от DNS rebinding со страниц в браузере).
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])
    # Настройки конвейера извлечения (как в configs/extract.yaml).
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    # Способ Б: обмен с 1С через папки inbox/outbox (D54). None — <data_dir>/exchange.
    exchange_enabled: bool = True
    exchange_root: Path | None = None
    exchange_poll_s: float = 0.5

    @property
    def exchange_dir(self) -> Path:
        return self.exchange_root or self.data_dir / "exchange"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "service.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def job_dir(self, job_id: str) -> Path:
        return self.uploads_dir / job_id
