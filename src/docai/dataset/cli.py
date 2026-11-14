"""Команда `docai dataset build`."""

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from docai.dataset.build import BuildError, build_dataset
from docai.dataset.config import load_config

app = typer.Typer(help="Сборка датасета (документы, PDF, картинки, разбиение, манифест).")
console = Console()


@app.command()
def build(
    config: Annotated[Path, typer.Option(help="YAML-конфиг сборки.")] = Path(
        "configs/dataset_v1.yaml"
    ),
    out: Annotated[
        Path | None, typer.Option(help="Папка результата вместо `out` из конфига.")
    ] = None,
    n: Annotated[
        int | None, typer.Option("--n", min=1, help="Число документов вместо конфига.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Seed вместо конфига.")] = None,
    force: Annotated[
        bool, typer.Option(help="Удалить непустую папку результата и собрать заново.")
    ] = False,
) -> None:
    """Собрать датасет: `uv run docai dataset build --config configs/dataset_v1.yaml`."""
    cfg = load_config(config)
    overrides = {"out": out, "n_documents": n, "seed": seed}
    cfg = cfg.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    started = time.perf_counter()

    def progress(stage: str, done: int, total: int) -> None:
        console.print(f"[{time.perf_counter() - started:7.1f} с] {stage}: {done}/{total}")

    try:
        result = build_dataset(cfg, force=force, progress=progress)
    except BuildError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"Готово: {result.documents} документов, {result.rows} строк манифеста, "
        f"{result.images} картинок. Манифест: {result.manifest}"
    )


@app.command()
def card(
    dataset: Annotated[Path, typer.Option(help="Папка собранного датасета.")] = Path(
        "data/datasets/v1"
    ),
    out: Annotated[Path, typer.Option(help="Куда записать карточку.")] = Path("docs/DATASET.md"),
) -> None:
    """Посчитать статистику по манифесту и записать карточку датасета."""
    from docai.dataset.card import write_card

    write_card(dataset, out)
    console.print(f"Карточка записана: {out}")
