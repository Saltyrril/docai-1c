"""Проверка подключённой в Ollama модели: отвечает ли она валидным JSON на запросы конвейера.

Берутся документы val датасета P6 (test не трогаем): по PDF с текстовым слоем (путь text)
и картинки уровней clean, scan, photo (путь vision). Валидный JSON — каждый ответ модели
разобран в черновик без ошибок и документ собран.
"""

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docai.extract.config import ExtractConfig
from docai.extract.ollama import urlopen
from docai.extract.pipeline import extract
from docai.extract.router import Mode

CHECK_KINDS = ("text", "text", "clean", "scan", "photo")


@dataclass
class CheckRow:
    file: str
    route: str
    valid_json: bool
    doc_type_ok: bool
    number_ok: bool
    date_ok: bool
    seconds: float | None
    errors: list[str]


def model_info(config: ExtractConfig) -> dict[str, Any]:
    """`/api/show` модели: есть ли она и умеет ли картинки."""
    url = config.ollama_url.rstrip("/") + "/api/show"
    body = json.dumps({"model": config.model}).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def pick_items(dataset: Path, kinds: tuple[str, ...] = CHECK_KINDS) -> list[tuple[str, Path]]:
    """Одностраничные документы val P6 разных типов: (вид, файл)."""
    rows = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["split"] == "val" and r["pages"] == 1 and not r["has_errors"]]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["doc_type"], []).append(row)
    types = sorted(by_type)
    items: list[tuple[str, Path]] = []
    used: set[str] = set()
    for i, kind in enumerate(kinds):
        level = "clean" if kind == "text" else kind
        # Типы по кругу: каждый следующий документ — по возможности другого типа.
        order = types[i % len(types) :] + types[: i % len(types)]
        row = next(
            (
                r
                for doc_type in order
                for r in by_type[doc_type]
                if r["level"] == level and r["doc_id"] not in used
            ),
            None,
        )
        if row is None:
            raise ValueError(f"в val датасета {dataset} не хватает документов уровня {level}")
        used.add(row["doc_id"])
        path = dataset / (row["pdf"] if kind == "text" else row["images"][0])
        items.append((kind, path))
    return items


def check_model(config: ExtractConfig, dataset: Path) -> list[CheckRow]:
    rows = []
    for kind, path in pick_items(dataset):
        mode = Mode.TEXT if kind == "text" else Mode.VISION
        result = extract([path], mode, config)
        truth = json.loads(
            (dataset / "json" / f"{path.stem.split('_p1_')[0]}.json").read_text(encoding="utf-8")
        )
        doc = result.document or {}
        llm_errors = [e for e in result.errors if "JSON" in e or "схем" in e or "обрезан" in e]
        rows.append(
            CheckRow(
                file=path.name,
                route=f"{mode.value} ({kind})",
                valid_json=result.document is not None and not llm_errors,
                doc_type_ok=doc.get("doc_type") == truth["doc_type"],
                number_ok=doc.get("number") == truth["number"],
                date_ok=doc.get("date") == truth["date"],
                seconds=result.meta.timings_s.get("total"),
                errors=result.errors,
            )
        )
    return rows


def rows_json(rows: list[CheckRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]
