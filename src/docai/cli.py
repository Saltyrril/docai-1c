"""Командная строка проекта: `uv run docai ...`."""

import typer
from rich.console import Console

from docai import __version__
from docai.synth.cli import app as synth_app

app = typer.Typer(help="docai-1c: извлечение реквизитов, проверка, 1С и RAG.", no_args_is_help=True)
console = Console()
app.add_typer(synth_app, name="synth")


@app.callback()
def main() -> None:
    """Точка входа. Подкоманды добавляются по мере появления модулей."""


@app.command()
def version() -> None:
    """Показать версию пакета."""
    console.print(f"docai-1c {__version__}")


if __name__ == "__main__":
    app()
