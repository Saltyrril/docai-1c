"""Команды RAG-конвейера (P19): индексация и вопрос.

uv run docai rag index                # нарезка, BM25, векторы на CPU (продолжает после остановки)
uv run docai rag index --rebuild      # собрать заново (корпус или модель поменялись)
uv run docai ask "Какой срок оплаты по договору с ООО «Вектор»?"
uv run docai ask "..." --json         # ответ целиком, как его отдаёт POST /v1/rag/ask
"""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from docai.rag.config import load_rag_config

console = Console()

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="YAML конвейера (по умолчанию configs/rag/pipeline.yaml)."),
]


def index_command(
    config_path: ConfigOption = None,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Удалить индекс и собрать заново.")
    ] = False,
    stop_after: Annotated[
        int | None,
        typer.Option(help="Остановиться после стольких новых векторов (проверка продолжения)."),
    ] = None,
) -> None:
    """Построить индекс RAG: куски по пунктам, BM25, векторы Qwen3-Embedding (CPU), реестр.

    Векторы пишутся пачками: после Ctrl+C или сбоя повторный запуск продолжит с места
    остановки.
    """
    from docai.rag.index import IndexMismatchError, build_index
    from docai.rag.models import Qwen3Embedder

    config = load_rag_config(config_path)
    logging.basicConfig(level=logging.WARNING)
    progress = Progress(
        TextColumn("[bold]Векторы[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("осталось"),
        TimeRemainingColumn(),
        console=console,
    )
    task = None

    def on_start(total: int, already: int) -> None:
        nonlocal task
        if already:
            console.print(f"Уже посчитано {already} из {total} — продолжаю.")
        task = progress.add_task("embed", total=total, completed=already)
        progress.start()

    def on_progress(_done: int, batch: int) -> None:
        if task is not None:
            progress.advance(task, batch)

    def make_embedder() -> Qwen3Embedder:
        console.print(f"Загружаю {config.embedding.model} на CPU…")
        return Qwen3Embedder(config.embedding)

    try:
        report = build_index(
            config,
            make_embedder,
            rebuild=rebuild,
            on_start=on_start,
            on_progress=on_progress,
            stop_after=stop_after,
        )
    except IndexMismatchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except KeyboardInterrupt:
        progress.stop()
        console.print(
            "[yellow]Остановлено. Готовые пачки сохранены; продолжить: "
            "uv run docai rag index[/yellow]"
        )
        raise typer.Exit(130) from None
    finally:
        progress.stop()
    state = "готов" if report.complete else "не завершён"
    console.print(
        f"Индекс {state}: {report.documents} документов, {report.chunks} кусков, "
        f"реестр {report.registry_rows} строк; векторов сейчас {report.embedded_now}, "
        f"было {report.already_embedded}; {report.seconds} с. Папка: {config.index_dir}"
    )


def ask_command(
    question: Annotated[str, typer.Argument(help="Вопрос по договорам, регламентам или реестру.")],
    config_path: ConfigOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Ответ целиком в JSON.")] = False,
    debug: Annotated[
        bool, typer.Option(help="Показать кандидатов поиска и контекст модели.")
    ] = False,
) -> None:
    """Задать вопрос: ответ со ссылками на пункты документов или по реестру (SQL)."""
    from docai.extract.ollama import OllamaError
    from docai.rag.index import IndexNotReadyError
    from docai.rag.pipeline import RagPipeline

    config = load_rag_config(config_path)
    logging.basicConfig(level=logging.WARNING)
    try:
        pipeline = RagPipeline(config)
        result = pipeline.ask(question, debug=debug)
    except IndexNotReadyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc
    if as_json:
        sys.stdout.write(json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n")
        return
    console.print(f"[bold]{result.answer}[/bold]")
    if result.reason and not result.found:
        console.print(f"[dim]({result.reason})[/dim]")
    for c in result.citations:
        quote = c.quote if len(c.quote) <= 300 else c.quote[:297] + "…"
        console.print(
            f"[cyan][{c.n}][/cyan] {c.document} — [dim]{c.fragment_id}[/dim]\n    {quote}"
        )
    if result.sql:
        console.print(f"[dim]SQL: {result.sql}[/dim]")
    if debug:
        console.print_json(json.dumps(result.filters, ensure_ascii=False))
        for row in result.candidates or []:
            console.print(row)
    console.print(f"[dim]маршрут {result.route}, {result.timings.get('total_s')} с[/dim]")
