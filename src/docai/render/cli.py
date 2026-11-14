"""Команда `docai render`: JSON документов → PDF и разметка расположения полей."""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from docai.render.engine import VARIANTS, Renderer, RenderError, choose_look
from docai.schemas.documents import parse_document

console = Console()
LAYOUT_SUFFIX = ".layout.json"


def render(
    in_dir: Annotated[
        Path, typer.Option("--in", help="Папка с JSON документов (из `docai synth generate`).")
    ] = Path("data/synth/json"),
    out: Annotated[Path, typer.Option(help="Папка для PDF и *.layout.json.")] = Path(
        "data/synth/pdf"
    ),
    seed: Annotated[int, typer.Option(help="Seed выбора вёрстки и заглушек.")] = 42,
    variant: Annotated[
        int | None, typer.Option(help=f"Принудительно один вариант вёрстки из {VARIANTS}.")
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Обработать только первые N.")] = None,
    stem: Annotated[
        list[str] | None, typer.Option(help="Только документы с этим именем файла (без .json).")
    ] = None,
) -> None:
    """Отрисовать JSON-документы в PDF. Рядом с каждым PDF — `<имя>.layout.json` с bbox полей."""
    files = sorted(p for p in in_dir.glob("*.json") if not p.name.startswith("_"))
    if stem:
        files = [p for p in files if p.stem in set(stem)]
    if limit:
        files = files[:limit]
    if not files:
        console.print(f"[red]В папке {in_dir} нет JSON документов.[/red]")
        raise typer.Exit(1)
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    stats: Counter[str] = Counter()
    pages_total = 0
    failed: list[str] = []
    with Renderer() as renderer:
        for path in files:
            doc = parse_document(path.read_text(encoding="utf-8"))
            chosen, decor = choose_look(doc.doc_type, path.stem, seed, variant)
            try:
                result = renderer.render(doc, variant=chosen, decor=decor, source=path.name)
            except RenderError as exc:
                failed.append(str(exc))
                continue
            (out / f"{path.stem}.pdf").write_bytes(result.pdf)
            layout_text = json.dumps(result.layout, ensure_ascii=False, indent=1)
            (out / f"{path.stem}{LAYOUT_SUFFIX}").write_text(
                layout_text + "\n", encoding="utf-8", newline="\n"
            )
            stats[f"{doc.doc_type.value} v{chosen}"] += 1
            pages_total += result.layout["page_count"]

    elapsed = time.perf_counter() - started
    done = sum(stats.values())
    console.print(f"Готово: {done} PDF, {pages_total} стр., {elapsed:.1f} с. Папка: {out}")
    for name, count in sorted(stats.items()):
        console.print(f"  {name}: {count}")
    if failed:
        console.print(f"[red]Не удалось ({len(failed)}):[/red]")
        for line in failed[:10]:
            console.print(f"  {line}")
        raise typer.Exit(1)
