"""Способ Б: обмен с 1С через папки, если HTTP недоступен (P15, D54).

1С кладёт файл в `inbox`, сервис отвечает файлом `<ид>.json` в `outbox`:

- `inbox/<ид>.json` с телом `{"method": "health"}` → `outbox/<ид>.json`, как `GET /v1/health`;
- `inbox/<ид>.pdf|.png|.jpg|.jpeg|.xml` → файл ставится в очередь как `POST /v1/extract`
  (режим auto); когда задание выполнено или завершилось ошибкой, в `outbox/<ид>.json`
  появляется то же, что `GET /v1/jobs/{id}`.

Ошибка (неизвестный метод, не тот тип файла, переполнена очередь) — `{"detail": "текст"}`.

Чтобы сервис не взял недописанный файл, 1С пишет его под расширением `.tmp` и затем
переименовывает: файлы с другими расширениями сервис не трогает. Ответ сервис пишет так же
(`.tmp`, затем переименование), поэтому 1С никогда не читает половину ответа.
Связь «ид запроса → задание» хранится в `pending/<ид>.txt` и переживает перезапуск сервиса.
"""

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger("docai.service")

# Какие файлы в inbox считаются запросами. Остальные (в том числе .tmp) пропускаются.
REQUEST_SUFFIX = ".json"
DOCUMENT_SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".xml"})
# Ид запроса становится именем файла ответа, поэтому допускаются только безопасные символы.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_REQUEST_BYTES = 64 * 1024
_FINAL_STATUSES = frozenset({"done", "failed"})


class ExchangeError(Exception):
    """Запрос нельзя выполнить; текст уходит в ответ как `detail`."""


# Поставить файл в очередь: путь и исходное имя → id задания. Ошибка — ExchangeError.
Enqueue = Callable[[Path, str], str]
# Состояние задания по id (как `GET /v1/jobs/{id}`); None — задания нет.
JobLookup = Callable[[str], BaseModel | None]


class FileExchange:
    def __init__(
        self,
        root: Path,
        *,
        health: Callable[[], BaseModel],
        enqueue: Enqueue,
        job_view: JobLookup,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.root = root
        self.inbox = root / "inbox"
        self.outbox = root / "outbox"
        self.pending = root / "pending"
        self.rejected = root / "rejected"
        self._health = health
        self._enqueue = enqueue
        self._job_view = job_view
        self._interval = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        for folder in (self.inbox, self.outbox, self.pending, self.rejected):
            folder.mkdir(parents=True, exist_ok=True)
        log.info("обмен файлами (способ Б): %s", self.root.resolve())
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="docai-exchange", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                log.exception("обмен файлами: сбой при просмотре папок")
            self._stop.wait(self._interval)

    def poll_once(self) -> None:
        """Один проход: принять новые запросы и отдать ответы по готовым заданиям."""
        for path in self._incoming():
            self._take(path)
        for marker in sorted(self.pending.glob("*.txt")):
            self._check_pending(marker)

    def _incoming(self) -> list[Path]:
        accepted = DOCUMENT_SUFFIXES | {REQUEST_SUFFIX}
        files = [p for p in self.inbox.iterdir() if p.is_file() and p.suffix.lower() in accepted]
        # Порядок поступления: сначала старые.
        return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))

    def _take(self, path: Path) -> None:
        request_id = path.stem
        if not _REQUEST_ID.match(request_id):
            # Ответить некуда: имя ответа строится из ида. Файл убирается в rejected/.
            log.warning("обмен файлами: недопустимое имя %r, файл перенесён в rejected", path.name)
            os.replace(path, self.rejected / path.name)
            return
        try:
            if path.suffix.lower() == REQUEST_SUFFIX:
                self._write(request_id, self._answer_request(path))
            else:
                job_id = self._enqueue(path, path.name)
                _write_atomic(self.pending / f"{request_id}.txt", job_id.encode("ascii"))
                log.info("обмен файлами: %s → задание %s", path.name, job_id)
        except ExchangeError as exc:
            self._write(request_id, json.dumps({"detail": str(exc)}, ensure_ascii=False))
        except PermissionError:
            # Файл ещё открыт другим процессом (например, 1С дописывает) — возьмём в следующий раз.
            return
        path.unlink(missing_ok=True)

    def _answer_request(self, path: Path) -> str:
        raw = path.read_bytes()
        if len(raw) > _MAX_REQUEST_BYTES:
            raise ExchangeError("запрос больше 64 КБ")
        try:
            # utf-8-sig: ЗаписьТекста в 1С пишет UTF-8 с BOM.
            body = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExchangeError(f"запрос не JSON в UTF-8: {exc}") from exc
        method = body.get("method") if isinstance(body, dict) else None
        if method == "health":
            return self._health().model_dump_json()
        raise ExchangeError(f"неизвестный метод {method!r}; поддерживается: health")

    def _check_pending(self, marker: Path) -> None:
        request_id = marker.stem
        job_id = marker.read_text(encoding="ascii").strip()
        view = self._job_view(job_id)
        if view is None:
            answer = json.dumps({"detail": "задание не найдено"}, ensure_ascii=False)
        elif getattr(view, "status", None) in _FINAL_STATUSES:
            answer = view.model_dump_json()
        else:
            return
        self._write(request_id, answer)
        marker.unlink(missing_ok=True)

    def _write(self, request_id: str, text: str) -> None:
        _write_atomic(self.outbox / f"{request_id}.json", text.encode("utf-8"))


def _write_atomic(path: Path, data: bytes) -> None:
    """Записать во временный файл и переименовать: читатель не увидит половину файла."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
