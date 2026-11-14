"""Команда `docai augment`: PDF → «сканы» и «фото» с пересчитанной разметкой полей."""

import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from docai.augment.levels import Level
from docai.augment.pipeline import process_pdf
from docai.augment.raster import DPI_CHOICES

console = Console()


class LevelChoice(StrEnum):
    CLEAN = "clean"
    SCAN = "scan"
    PHOTO = "photo"
    ALL = "all"


def augment(
    in_path: Annotated[
        Path,
        typer.Option(
            "--in", help="PDF или папка с PDF (рядом `<имя>.layout.json` из `docai render`)."
        ),
    ] = Path("data/synth/pdf"),
    out: Annotated[Path, typer.Option(help="Папка для картинок и *.layout.json.")] = Path(
        "data/synth/scans"
    ),
    level: Annotated[
        list[LevelChoice] | None,
        typer.Option(help="clean, scan, photo или all; можно повторять. По умолчанию scan."),
    ] = None,
    dpi: Annotated[
        list[int] | None,
        typer.Option(
            help=f"Разрешение растеризации из {DPI_CHOICES}; можно повторять. По умолчанию 200."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed искажений.")] = 42,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Обработать только первые N PDF.")
    ] = None,
    stem: Annotated[
        list[str] | None, typer.Option(help="Только PDF с этим именем файла (без .pdf).")
    ] = None,
) -> None:
    """Растеризовать PDF и исказить. Результат — только картинка (текстового слоя нет)."""
    chosen = level or [LevelChoice.SCAN]
    levels = (
        list(Level)
        if LevelChoice.ALL in chosen
        else [Level(item.value) for item in dict.fromkeys(chosen)]
    )
    dpis = list(dict.fromkeys(dpi or [200]))
    bad = [value for value in dpis if value not in DPI_CHOICES]
    if bad:
        console.print(f"[red]Разрешение должно быть одним из {DPI_CHOICES}, получено: {bad}.[/red]")
        raise typer.Exit(2)

    files = [in_path] if in_path.is_file() else sorted(in_path.glob("*.pdf"))
    if stem:
        files = [p for p in files if p.stem in set(stem)]
    if limit:
        files = files[:limit]
    if not files:
        console.print(f"[red]По пути {in_path} нет PDF.[/red]")
        raise typer.Exit(1)

    started = time.perf_counter()
    stats: Counter[str] = Counter()
    without_layout = 0
    for path in files:
        if not path.with_name(path.stem + ".layout.json").exists():
            without_layout += 1
        for item in process_pdf(path, out, levels, dpis, seed):
            stats[f"{item.level.value} {item.dpi} dpi"] += 1

    elapsed = time.perf_counter() - started
    console.print(
        f"Готово: {sum(stats.values())} картинок из {len(files)} PDF, {elapsed:.1f} с. Папка: {out}"
    )
    for name, count in sorted(stats.items()):
        console.print(f"  {name}: {count}")
    if without_layout:
        console.print(
            f"[yellow]У {without_layout} PDF нет .layout.json: "
            "картинки записаны без полей.[/yellow]"
        )
