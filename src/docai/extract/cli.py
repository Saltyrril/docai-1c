"""Команда `docai extract`: файл → JSON документа по нашей схеме."""

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from docai.extract.config import load_config
from docai.extract.ocr import OcrError
from docai.extract.ollama import OllamaError
from docai.extract.pipeline import extract as run_extract
from docai.extract.router import Mode, RoutingError

console = Console(stderr=True)


def extract(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="PDF, картинка или XML; несколько картинок — страницы одного документа."
        ),
    ],
    mode: Annotated[Mode, typer.Option(help="Путь извлечения; auto выбирает сам.")] = Mode.AUTO,
    out: Annotated[
        Path | None, typer.Option(help="Куда записать JSON; по умолчанию — в stdout.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option(help="YAML с настройками (см. configs/extract.yaml).")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Модель Ollama вместо указанной в конфиге.")
    ] = None,
    cross_check: Annotated[
        Mode | None,
        typer.Option(
            "--cross-check",
            help="Второй путь для сверки значений (text|vision|ocr); время вырастает вдвое.",
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Без журнала этапов.")] = False,
) -> None:
    """Извлечь поля документа: XML ФНС — парсером, PDF с текстом — по тексту, иначе — vision.

    Журнал этапов (время, токены, пиковая VRAM) пишется в stderr, JSON — в stdout или --out.
    """
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    settings = load_config(config)
    if model:
        settings = settings.model_copy(update={"model": model})
    try:
        result = run_extract(paths, mode, settings, cross_check)
    except (RoutingError, OcrError, OllamaError) as exc:
        console.print(f"[red]Ошибка:[/red] {exc}", markup=True, highlight=False)
        raise typer.Exit(2) from exc

    text = result.model_dump_json(indent=2) + "\n"
    if out is None:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        console.print(f"JSON: {out}")
    for warning in result.warnings:
        console.print(f"[yellow]предупреждение:[/yellow] {warning}", highlight=False)
    for error in result.errors:
        console.print(f"[red]ошибка:[/red] {error}", highlight=False)
    report = result.postprocess
    if report is not None:
        console.print(f"статус: [bold]{report.status_ru}[/bold] ({'; '.join(report.reasons)})")
        for fix in report.corrections:
            if fix.kind == "autofix":
                console.print(f"[cyan]исправлено:[/cyan] {fix.reason}", highlight=False)
        for item in report.review_fields[:15]:
            console.print(
                f"[yellow]проверить:[/yellow] {item.field} = {item.value} "
                f"(уверенность {item.confidence:.2f})",
                highlight=False,
            )
    if not result.valid:
        raise typer.Exit(1)
