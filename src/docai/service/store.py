"""Хранилище заданий и правок в SQLite.

Одно соединение на весь сервис, доступ под замком: заданий мало, запросы короткие.
Очередь — это строки со статусом `queued`, порядок — по rowid. После перезапуска сервиса
задания, прерванные на середине (`running`), возвращаются в очередь.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    filename      TEXT NOT NULL,
    stored_name   TEXT NOT NULL,
    source_kind   TEXT NOT NULL,
    mode          TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT,
    result_json   TEXT,
    preview_pages INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id),
    created_at       TEXT NOT NULL,
    author           TEXT,
    comment          TEXT,
    changed          INTEGER NOT NULL,
    total            INTEGER NOT NULL,
    corrections_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_job ON feedback(job_id);
"""


def now_iso() -> str:
    """Текущий момент в UTC, ISO 8601 с секундами."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class JobRow:
    id: str
    status: str
    filename: str
    stored_name: str
    source_kind: str
    mode: str
    size_bytes: int
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    result_json: str | None
    preview_pages: int


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # isolation_level=None — автокоммит; транзакции открываются явно (BEGIN IMMEDIATE).
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> JobRow | None:
        return None if row is None else JobRow(**dict(row))

    def create_job(
        self,
        job_id: str,
        *,
        filename: str,
        stored_name: str,
        source_kind: str,
        mode: str,
        size_bytes: int,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, status, filename, stored_name, source_kind, mode,"
                " size_bytes, created_at) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)",
                (job_id, filename, stored_name, source_kind, mode, size_bytes, now_iso()),
            )

    def get_job(self, job_id: str) -> JobRow | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row(row)

    def count(self, status: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,))
            return int(row.fetchone()[0])

    def queue_ahead(self, job_id: str) -> int | None:
        """Сколько заданий впереди: выполняющееся плюс ждущие раньше. None, если не в очереди."""
        with self._lock:
            row = self._db.execute(
                "SELECT rowid FROM jobs WHERE id = ? AND status = 'queued'", (job_id,)
            ).fetchone()
            if row is None:
                return None
            before = self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND rowid < ?", (row[0],)
            ).fetchone()[0]
            running = self._db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
            return int(before) + int(running.fetchone()[0])

    def claim_next(self) -> JobRow | None:
        """Взять самое раннее ожидающее задание и пометить его выполняющимся."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT * FROM jobs WHERE status = 'queued' ORDER BY rowid LIMIT 1"
                ).fetchone()
                if row is not None:
                    self._db.execute(
                        "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                        (now_iso(), row["id"]),
                    )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return None if row is None else self.get_job(row["id"])

    def requeue_running(self) -> int:
        """После перезапуска: прерванные задания снова в очередь (в прежнем порядке)."""
        with self._lock:
            cursor = self._db.execute(
                "UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'"
            )
            return cursor.rowcount

    def finish_job(self, job_id: str, result_json: str, preview_pages: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = 'done', finished_at = ?, result_json = ?,"
                " preview_pages = ?, error = NULL WHERE id = ?",
                (now_iso(), result_json, preview_pages, job_id),
            )

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = 'failed', finished_at = ?, error = ? WHERE id = ?",
                (now_iso(), error, job_id),
            )

    def add_feedback(
        self,
        job_id: str,
        *,
        author: str | None,
        comment: str | None,
        corrections: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Сохранить правки человека; вернуть (id записи, число изменённых значений)."""
        changed = sum(1 for item in corrections if item["changed"])
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO feedback (job_id, created_at, author, comment, changed, total,"
                " corrections_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    now_iso(),
                    author,
                    comment,
                    changed,
                    len(corrections),
                    json.dumps(corrections, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid or 0), changed

    def feedback_count(self, job_id: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) FROM feedback WHERE job_id = ?", (job_id,))
            return int(row.fetchone()[0])
