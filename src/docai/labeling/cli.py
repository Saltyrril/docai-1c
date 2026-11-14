"""Команды `docai label` (интерфейс разметки) и `docai label anonymize` (обезличивание)."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from docai.labeling.pseudonymize import DEFAULT_OUT, load_salt, pseudonymize_dir
from docai.labeling.store import DEFAULT_ROOT

app = typer.Typer(
    help="Разметка реальных документов: интерфейс и обезличивание для отчётов.",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def label(
    ctx: typer.Context,
    root: Annotated[
        Path, typer.Option(help="Папка с данными: files/ (исходники) и labels/ (эталоны).")
    ] = DEFAULT_ROOT,
    port: Annotated[int, typer.Option(min=1024, max=65535, help="Порт на 127.0.0.1.")] = 7860,
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Не открывать браузер.")] = False,
) -> None:
    """Без подкоманды открывает интерфейс разметки: `uv run docai label`."""
    if ctx.invoked_subcommand is not None:
        return
    from docai.labeling.app import run  # gradio тяжёлый: импортируем только здесь

    console.print(f"Интерфейс разметки: http://127.0.0.1:{port}  (остановить: Ctrl+C)")
    run(root, port, open_browser=not no_browser)


@app.command()
def anonymize(
    in_dir: Annotated[Path, typer.Option("--in", help="Папка с эталонами.")] = DEFAULT_ROOT
    / "labels",
    out: Annotated[Path, typer.Option(help="Куда писать обезличенные JSON.")] = DEFAULT_ROOT
    / DEFAULT_OUT,
    root: Annotated[Path, typer.Option(help="Где лежит секретная соль (.pseudo_salt).")] = (
        DEFAULT_ROOT
    ),
    force: Annotated[bool, typer.Option(help="Перезаписать существующие файлы.")] = False,
) -> None:
    """Заменить наименования, ИНН, КПП, адреса и банковские реквизиты на псевдонимы.

    Результат по умолчанию лежит в `data/real/anon` (не коммитится). В отчёты и на GitHub
    его можно переносить только после просмотра глазами.
    """
    if not in_dir.is_dir():
        console.print(f"[red]Нет папки {in_dir}[/red]")
        raise typer.Exit(1)
    written, problems = pseudonymize_dir(in_dir, out, load_salt(root), force=force)
    console.print(f"Обезличено документов: {len(written)} → {out}")
    for problem in problems:
        console.print(f"[red]{problem}[/red]")
    if problems:
        raise typer.Exit(1)
