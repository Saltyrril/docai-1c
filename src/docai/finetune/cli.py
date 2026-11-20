"""Команды `docai finetune build` и `docai finetune check-model`."""

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from docai.extract.config import load_config as load_extract_config
from docai.extract.ollama import OllamaError
from docai.finetune.build import SftBuildError, build_sft_dataset
from docai.finetune.check import check_model, model_info, rows_json
from docai.finetune.config import load_config

app = typer.Typer(help="Дообучение Qwen3.5-2B: SFT-датасет для Colab и проверка модели (P21).")
console = Console()


@app.command()
def build(
    config: Annotated[Path, typer.Option(help="YAML-конфиг сборки.")] = Path(
        "configs/finetune/sft_v1.yaml"
    ),
    force: Annotated[bool, typer.Option(help="Удалить непустую папку и собрать заново.")] = False,
) -> None:
    """Собрать SFT-датасет и архив для Google Drive: `uv run docai finetune build`."""
    cfg = load_config(config)
    started = time.perf_counter()

    def progress(stage: str) -> None:
        console.print(f"[{time.perf_counter() - started:7.1f} с] {stage}")

    try:
        result = build_sft_dataset(cfg, force=force, progress=progress)
    except SftBuildError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    for split, counts in result.counts.items():
        console.print(
            f"{split}: {counts['examples']} примеров из {counts['documents']} документов; "
            f"виды {counts['by_kind']}"
        )
    fb = result.feedback
    if fb.get("enabled"):
        console.print(
            f"/feedback: заданий с правками {fb['jobs_with_feedback']}, {fb['by_status']}"
        )
        unknown = fb["by_status"].get("unknown", 0)
        if unknown and not fb["include_unknown"]:
            console.print(
                f"[yellow]{unknown} правок к файлам не из синтетики не взяты: это могут быть "
                "реальные документы (feedback.include_unknown в конфиге — решение "
                "владельца).[/yellow]"
            )
    if result.archive is not None:
        size = result.archive.stat().st_size / 2**20
        console.print(f"Архив для Google Drive: {result.archive} ({size:.0f} МиБ)")


@app.command(name="check-model")
def check_model_command(
    model: Annotated[str, typer.Option(help="Имя модели в Ollama.")] = "docai-qwen35-2b-ft",
    config: Annotated[Path, typer.Option(help="Конфиг извлечения.")] = Path("configs/extract.yaml"),
    dataset: Annotated[Path, typer.Option(help="Датасет P6 (берётся часть val).")] = Path(
        "data/datasets/v1"
    ),
    out: Annotated[Path | None, typer.Option(help="Куда записать результат JSON.")] = None,
) -> None:
    """Прогнать 5 документов val через модель и проверить, что ответы — валидный JSON."""
    cfg = load_extract_config(config).model_copy(update={"model": model})
    try:
        info = model_info(cfg)
    except OSError as exc:
        console.print(f"[red]Ollama не отвечает или нет модели {model}: {exc}[/red]")
        raise typer.Exit(2) from exc
    capabilities = info.get("capabilities") or []
    console.print(f"Модель {model}: {', '.join(capabilities)}")
    if "vision" not in capabilities:
        console.print("[red]У модели нет vision: проектор (mmproj) не подключился.[/red]")
        raise typer.Exit(1)
    try:
        rows = check_model(cfg, dataset)
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    table = Table("файл", "путь", "JSON", "тип", "номер", "дата", "с")
    mark = {True: "да", False: "[red]нет[/red]"}
    for row in rows:
        table.add_row(
            row.file,
            row.route,
            mark[row.valid_json],
            mark[row.doc_type_ok],
            mark[row.number_ok],
            mark[row.date_ok],
            f"{row.seconds:.1f}" if row.seconds is not None else "—",
        )
    console.print(table)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model, "capabilities": capabilities, "rows": rows_json(rows)}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    valid = sum(row.valid_json for row in rows)
    console.print(f"Валидный JSON: {valid} из {len(rows)}")
    if valid != len(rows):
        raise typer.Exit(1)
