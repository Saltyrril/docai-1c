"""Правки человека из `POST /v1/jobs/{id}/feedback` (сервис P14) как примеры для SFT.

Правило отбора (D77):
- XML из ЭДО разбирается без модели — такие правки модели ничему не учат;
- файл, совпавший байт в байт с файлом синтетики (датасет P6, `data/synth`, `data/demo`), —
  синтетика: у неё есть точный эталон, он уже в датасете (а правки на синтетике —
  проверочные, бывают и неверные);
- остальные файлы могут быть реальными документами. По правилу проекта реальные документы
  не отправляются во внешние сервисы, а архив уходит в Google Drive, поэтому такие правки
  берутся только при `feedback.include_unknown: true` — это решение владельца.

Берётся последняя правка задания; значения — то, что сервис отдал 1С, с заменой на
исправленные. Видимыми считаются все непустые поля (разметки страницы у таких файлов нет),
поэтому берутся только одностраничные файлы.
"""

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docai.schemas.documents import parse_document
from docai.service.views import flat_values
from docai.validation.normalize import parse_date

FeedbackStatus = str  # xml | synthetic | unknown | no_file | not_done | multipage
_INDEX = re.compile(r"^(\w+)\[(\d+)\]$")


@dataclass
class FeedbackJob:
    job_id: str
    filename: str
    source_kind: str
    path: Path
    changed: int
    values: dict[str, str | None]
    status: FeedbackStatus = "unknown"
    matched: str | None = None  # файл датасета, с которым совпал загруженный


@dataclass
class FeedbackReport:
    jobs: list[FeedbackJob] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for job in self.jobs:
            out[job.status] += 1
        return dict(sorted(out.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "jobs_with_feedback": len(self.jobs),
            "by_status": self.counts(),
            "changed_fields_total": sum(job.changed for job in self.jobs),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SynthIndex:
    """Файлы синтетики по размеру; хэш считается только у совпавших по размеру.

    Папки — датасет P6 и другие места, куда проект кладёт синтетику (P3/P4 `data/synth`,
    демонстрация P17 `data/demo`): PDF датасета и PDF этих папок рисуются отдельно и байт
    в байт не совпадают (Chromium пишет в PDF дату), поэтому нужны все.
    """

    def __init__(self, roots: list[Path]) -> None:
        self._by_size: dict[int, list[Path]] = defaultdict(list)
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg") and path.is_file():
                    self._by_size[path.stat().st_size].append(path)

    def match(self, path: Path) -> Path | None:
        candidates = self._by_size.get(path.stat().st_size)
        if not candidates:
            return None
        digest = _sha256(path)
        return next((c for c in candidates if _sha256(c) == digest), None)


def _latest_feedback(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT f.job_id, f.changed, f.corrections_json, j.filename, j.stored_name,
               j.source_kind, j.status, j.result_json
        FROM feedback f JOIN jobs j ON j.id = f.job_id
        WHERE f.id IN (SELECT MAX(id) FROM feedback GROUP BY job_id)
        ORDER BY f.id
        """
    ).fetchall()


def corrected_values(result_json: str | None, corrections: list[dict[str, Any]]) -> dict:
    """Значения, отданные 1С, с заменой на исправленные человеком."""
    # Берётся только документ: метаданные результата у старых заданий могли быть другими.
    document = json.loads(result_json).get("document") if result_json else None
    values: dict[str, str | None] = (
        flat_values(parse_document(document)) if document is not None else {}
    )
    for item in corrections:
        after = item.get("after")
        values[item["field"]] = after.strip() if isinstance(after, str) and after.strip() else None
    return values


def collect_feedback(data_dir: Path, synthetic_dirs: list[Path]) -> FeedbackReport:
    """Все задания с правками и их статус для SFT. Базу сервиса открывает только для чтения."""
    report = FeedbackReport()
    db_path = data_dir / "service.db"
    if not db_path.is_file():
        return report
    db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = _latest_feedback(db)
    finally:
        db.close()
    index: SynthIndex | None = None
    for row in rows:
        path = data_dir / "uploads" / row["job_id"] / row["stored_name"]
        job = FeedbackJob(
            job_id=row["job_id"],
            filename=row["filename"],
            source_kind=row["source_kind"],
            path=path,
            changed=int(row["changed"]),
            values=corrected_values(row["result_json"], json.loads(row["corrections_json"])),
        )
        report.jobs.append(job)
        if row["source_kind"] == "xml":
            job.status = "xml"
        elif row["status"] != "done":
            job.status = "not_done"
        elif not path.is_file():
            job.status = "no_file"
        else:
            index = index or SynthIndex(synthetic_dirs)
            matched = index.match(path)
            if matched is not None:
                job.status = "synthetic"
                job.matched = matched.as_posix()
    return report


def nest(values: dict[str, str | None]) -> dict[str, Any]:
    """`{"seller.inn": "…", "lines[0].price": "…"}` → вложенный словарь документа."""
    out: dict[str, Any] = {}
    for path, value in values.items():
        node: Any = out
        parts = path.split(".")
        for depth, part in enumerate(parts):
            last = depth == len(parts) - 1
            match = _INDEX.match(part)
            if match:
                key, idx = match.group(1), int(match.group(2))
                items = node.setdefault(key, [])
                while len(items) <= idx:
                    items.append({})
                if last:
                    items[idx] = value
                else:
                    node = items[idx]
            elif last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
    return out


def printed_view(values: dict[str, str | None]) -> tuple[dict[str, Any], dict[str, str]]:
    """Эталон и «видимые» поля для `draft_target`: все непустые поля; даты — ДД.ММ.ГГГГ."""
    visible: dict[str, str] = {}
    for path, value in values.items():
        if value is None or path == "doc_type":
            continue
        if path in ("date", "basis.date"):
            value = parse_date(value).strftime("%d.%m.%Y")
        visible[path] = value
    return nest(values), visible
