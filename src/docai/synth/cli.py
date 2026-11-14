"""Команда `docai synth generate`: серия синтетических документов в JSON."""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from docai.synth.generator import DEFAULT_ERROR_RATE, SynthDocument, generate_documents

app = typer.Typer(help="Синтетические документы: только данные (эталонная разметка).")
console = Console()

MANIFEST_NAME = "_manifest.jsonl"


def write_documents(documents: list[SynthDocument], out: Path, seed: int) -> Path:
    """Записать по JSON на документ и манифест `_manifest.jsonl` (метаданные генерации).

    Файл документа содержит только поля схемы, то есть это эталонная разметка.
    Сведения о намеренных ошибках лежат в манифесте, чтобы не попасть в разметку.
    """
    out.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    for item in documents:
        payload = item.document.model_dump(mode="json")
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        (out / item.file_name).write_text(text, encoding="utf-8", newline="\n")
        record = {
            "file": item.file_name,
            "index": item.index,
            "seed": seed,
            "doc_type": item.document.doc_type.value,
            "has_errors": item.has_errors,
            "error_kinds": list(item.error_kinds),
            "expected_error_codes": list(item.expected_codes),
        }
        manifest_lines.append(json.dumps(record, ensure_ascii=False))
    manifest = out / MANIFEST_NAME
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n")
    return manifest


@app.command()
def generate(
    n: Annotated[int, typer.Option("--n", min=1, help="Сколько документов создать.")] = 100,
    seed: Annotated[int, typer.Option(help="Seed: тот же seed даёт те же документы.")] = 42,
    out: Annotated[Path, typer.Option(help="Папка для JSON-файлов.")] = Path("data/synth/json"),
    error_rate: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Доля документов с намеренной ошибкой (0 — без них)."),
    ] = DEFAULT_ERROR_RATE,
) -> None:
    """Сгенерировать n документов и записать по одному JSON на документ."""
    started = time.perf_counter()
    documents = generate_documents(n, seed, error_rate)
    manifest = write_documents(documents, out, seed)
    elapsed = time.perf_counter() - started

    by_type = Counter(d.document.doc_type.value for d in documents)
    broken = sum(d.has_errors for d in documents)
    table = Table(title=f"Создано документов: {n}, seed {seed}, {elapsed:.1f} с")
    table.add_column("Тип")
    table.add_column("Штук", justify="right")
    for doc_type, count in sorted(by_type.items()):
        table.add_row(doc_type, str(count))
    console.print(table)
    console.print(f"С намеренной ошибкой: {broken}. Папка: {out}. Манифест: {manifest.name}")
