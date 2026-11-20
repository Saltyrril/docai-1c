"""Командная строка проекта: `uv run docai ...`."""

import typer
from rich.console import Console

from docai import __version__
from docai.augment.cli import augment
from docai.dataset.cli import app as dataset_app
from docai.edo.cli import app as edo_app
from docai.eval.cli import eval_command, eval_compare_command, eval_errors_command
from docai.eval.matrix_cli import app as matrix_app
from docai.extract.cli import extract
from docai.labeling.cli import app as label_app
from docai.rag.ask_cli import ask_command, index_command
from docai.rag.cli import app as rag_app
from docai.rag.eval_cli import eval_command as rag_eval_command
from docai.render.cli import render
from docai.service.cli import serve
from docai.synth.cli import app as synth_app

app = typer.Typer(help="docai-1c: извлечение реквизитов, проверка, 1С и RAG.", no_args_is_help=True)
console = Console()
app.add_typer(synth_app, name="synth")
app.add_typer(dataset_app, name="dataset")
app.add_typer(edo_app, name="edo")
app.add_typer(label_app, name="label")
app.add_typer(matrix_app, name="eval-matrix")
app.add_typer(rag_app, name="rag")
rag_app.command(name="index")(index_command)
rag_app.command(name="eval")(rag_eval_command)
app.command(name="render")(render)
app.command(name="augment")(augment)
app.command(name="extract")(extract)
app.command(name="serve")(serve)
app.command(name="eval")(eval_command)
app.command(name="eval-compare")(eval_compare_command)
app.command(name="eval-errors")(eval_errors_command)
app.command(name="ask")(ask_command)


@app.callback()
def main() -> None:
    """Точка входа. Подкоманды добавляются по мере появления модулей."""


@app.command()
def version() -> None:
    """Показать версию пакета."""
    console.print(f"docai-1c {__version__}")


if __name__ == "__main__":
    app()
