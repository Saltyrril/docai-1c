"""Очередь с одним воркером: GPU одна, задания выполняются строго по одному.

Очередь хранится в SQLite (`Store`), воркер — один поток. Пока он занят, остальные
задания ждут; после перезапуска сервиса прерванное задание начинается заново.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from docai.extract import ExtractionResult, Mode, extract
from docai.extract.config import ExtractConfig
from docai.extract.ocr import OcrError
from docai.extract.ollama import OllamaError
from docai.extract.router import InputKind, RoutingError
from docai.service.preview import render_previews
from docai.service.settings import ServiceSettings
from docai.service.store import JobRow, Store

log = logging.getLogger("docai.service")

# Функция извлечения: файл и режим → результат. В тестах вместо неё подставляется заглушка.
Extractor = Callable[[Path, Mode], ExtractionResult]


def make_extractor(config: ExtractConfig) -> Extractor:
    """Настоящее извлечение: конвейер из `docai.extract` с настройками сервиса."""

    def run(path: Path, mode: Mode) -> ExtractionResult:
        return extract([path], mode, config)

    return run


class Worker:
    def __init__(self, store: Store, extractor: Extractor, settings: ServiceSettings) -> None:
        self._store = store
        self._extractor = extractor
        self._settings = settings
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        requeued = self._store.requeue_running()
        if requeued:
            log.warning("после перезапуска возвращено в очередь заданий: %d", requeued)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="docai-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def notify(self) -> None:
        """Сообщить воркеру, что появилось новое задание."""
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                row = self._store.claim_next()
            except Exception:
                log.exception("не удалось взять задание из очереди")
                row = None
            if row is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            self._process(row)

    def _process(self, row: JobRow) -> None:
        job_dir = self._settings.job_dir(row.id)
        source = job_dir / row.stored_name
        log.info("задание %s: старт (%s, режим %s)", row.id, row.source_kind, row.mode)
        try:
            result = self._extractor(source, Mode(row.mode))
            pages = render_previews(
                source,
                InputKind(row.source_kind),
                job_dir,
                self._settings.preview_max_side,
                self._settings.preview_max_pages,
            )
            self._store.finish_job(row.id, result.model_dump_json(), pages)
            log.info("задание %s: готово", row.id)
        except (RoutingError, OcrError, OllamaError) as exc:
            log.warning("задание %s: ошибка окружения или входа: %s", row.id, exc)
            self._store.fail_job(row.id, str(exc))
        except Exception:
            log.exception("задание %s: внутренняя ошибка", row.id)
            self._store.fail_job(row.id, "внутренняя ошибка обработки, подробности — в журнале")
