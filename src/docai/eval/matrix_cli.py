"""Команды `docai eval-matrix`: plan, check-vram, run, report."""

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from docai.eval.matrix import Cell, load_matrix_config

app = typer.Typer(
    help="Матрица экспериментов P12: модель × путь × качество входа.", no_args_is_help=True
)
console = Console(stderr=True)
ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="configs/eval/p12_matrix.yaml")]


@app.command()
def plan(config: ConfigOpt) -> None:
    """Сколько элементов в каждой ячейке, сколько готово и сколько времени осталось."""
    from docai.eval.matrix import progress_table

    cfg = load_matrix_config(config)
    table = Table(title=f"{cfg.name}: ход прогона")
    for column in ("Ячейка", "Элементов", "Готово", "с/элемент", "Осталось, ч"):
        table.add_column(column, justify="right" if column != "Ячейка" else "left")
    total = 0.0
    for row in progress_table(cfg):
        if "skipped" in row:
            table.add_row(row["cell"], "—", "—", "—", f"пропущена: {row['skipped']}")
            continue
        mark = "" if row["measured"] else " (оценка)"
        table.add_row(
            row["cell"], str(row["items"]), str(row["done"]),
            f"{row['seconds_per_item']}{mark}", str(row["remaining_hours"]),
        )  # fmt: skip
        total += row["remaining_hours"]
    Console().print(table)
    Console().print(f"Всего осталось ≈ {total:.1f} ч непрерывной работы от сети.")


@app.command("check-vram")
def check_vram_command(config: ConfigOpt) -> None:
    """Загрузить модели с require_full_vram и записать, помещаются ли они в VRAM целиком."""
    from docai.eval.matrix import check_vram

    cfg = load_matrix_config(config)
    try:
        result = check_vram(cfg)
    except RuntimeError as exc:
        console.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(2) from exc
    for model_id, entry in result.items():
        verdict = "помещается" if entry["fits"] else "НЕ помещается"
        Console().print(
            f"{model_id} ({entry['tag']}): {verdict}; в VRAM {entry['vram_share']:.0%} "
            f"из {entry['size_bytes'] / 2**30:.2f} ГиБ, контекст {entry['num_ctx']}"
        )
        if entry["reason"]:
            Console().print(f"  причина: {entry['reason']}")


@app.command()
def run(
    config: ConfigOpt,
    docs: Annotated[
        int | None, typer.Option(help="Считать только первые N документов (по порядку выборки).")
    ] = None,
    cell: Annotated[
        list[str] | None,
        typer.Option(help="Только эти ячейки (<модель>__<путь>), можно повторять."),
    ] = None,
) -> None:
    """Прогнать матрицу партиями. Ctrl+C безопасен: готовые ответы лежат в кэше."""
    from docai.eval.matrix import run_matrix

    cfg = load_matrix_config(config)
    started = time.monotonic()

    def on_pass(k: int, n: int, end: int) -> None:
        console.print(f"[bold]Партия {k}/{n}[/bold]: документы до №{end}")

    def progress(i: int, n: int, c: Cell, item, pred, cached: bool) -> None:
        state = "кэш" if cached else f"{pred.wall_s:.0f} с"
        if pred.exception:
            state += f", исключение: {pred.exception}"
        elapsed = (time.monotonic() - started) / 3600
        console.print(f"  {c.id} [{i}/{n}] {item.item_id}: {state} (прошло {elapsed:.1f} ч)")

    try:
        fresh = run_matrix(
            cfg, docs=docs, only=cell, progress=progress, on_pass=on_pass,
            notice=lambda t: console.print(f"[yellow]Питание:[/yellow] {t}"),
        )  # fmt: skip
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.print("Новых ответов: " + json.dumps(fresh, ensure_ascii=False))


@app.command()
def report(
    config: ConfigOpt,
    docs: Annotated[int | None, typer.Option(help="Не больше N документов в таблице.")] = None,
) -> None:
    """Свести готовое в reports/<дата>_<имя>/: таблица, графики, summary.json."""
    from docai.eval.matrix_report import aggregate, write_matrix_report

    cfg = load_matrix_config(config)
    try:
        data = aggregate(cfg, docs=docs)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(2) from exc
    out = write_matrix_report(cfg, data)
    console.print(f"Документов: {data['documents']}. Отчёт: {out / 'report.md'}")
