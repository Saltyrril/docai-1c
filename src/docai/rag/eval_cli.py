"""Команда `docai rag eval`: оценка RAG и абляции (P20).

uv run docai rag eval --stage retrieval   # ранги найденных пунктов по четырём вариантам поиска
uv run docai rag eval --stage answers     # ответы конвейера по каждой конфигурации абляции
uv run docai rag eval --stage report      # метрики, интервалы, report.md и results.json
uv run docai rag eval                     # все три этапа
uv run docai rag eval --set manual        # то же на 30 вопросах владельца
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

STAGES = ("retrieval", "answers", "report")


def eval_command(
    config_path: Annotated[Path, typer.Option("--config", help="YAML оценки.")] = Path(
        "configs/rag/eval_p20.yaml"
    ),
    stage: Annotated[
        str, typer.Option(help="retrieval | answers | report | all (по умолчанию все).")
    ] = "all",
    qset: Annotated[
        str, typer.Option("--set", help="p18 — 200 вопросов P18; manual — вопросы владельца.")
    ] = "p18",
) -> None:
    """Оценить RAG: поиск (recall, MRR, nDCG), ответы, цитаты, «не нашёл», абляции."""
    from docai.rag.ablation import load_settings, run_all

    if stage != "all" and stage not in STAGES:
        raise typer.BadParameter(f"этап: {', '.join(STAGES)} или all")
    if qset not in ("p18", "manual"):
        raise typer.BadParameter("набор: p18 или manual")
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings(config_path)
    stages = STAGES if stage == "all" else (stage,)
    run = [s for s in stages if s != "report"]
    if run:
        run_all(settings, run, qset, say=console.print)
    if "report" in stages:
        from docai.rag.ablation_report import write_report

        path = write_report(settings, qset, config_path)
        console.print(f"Отчёт: {path}")
