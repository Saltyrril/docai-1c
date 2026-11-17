"""Команды `docai eval` и `docai eval-compare`."""

import json
import logging
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from docai.eval.config import EvalConfig, load_eval_config
from docai.eval.predict import Prediction
from docai.eval.sample import EvalItem

console = Console(stderr=True)


def eval_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="configs/eval/<имя>.yaml")],
    predict_only: Annotated[
        bool,
        typer.Option("--predict-only", help="Только получить ответы модели (кэш), без отчёта."),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Журнал этапов извлечения.")
    ] = False,
) -> None:
    """Оценить извлечение на выборке: метрики с 95% интервалами и отчёт в reports/<дата>_<имя>/."""
    from docai.eval.predict import run_predictions
    from docai.eval.report import write_report
    from docai.eval.run import evaluate
    from docai.eval.sample import select_items

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    cfg = load_eval_config(config)
    started = time.monotonic()

    def progress(i: int, n: int, item: EvalItem, pred: Prediction, cached: bool) -> None:
        state = "кэш" if cached else f"{pred.wall_s:.0f} с"
        if pred.exception:
            state += f", исключение: {pred.exception}"
        elapsed = (time.monotonic() - started) / 60
        console.print(f"[{i}/{n}] {item.item_id}: {state} (прошло {elapsed:.0f} мин)")

    try:
        if predict_only:
            items = select_items(cfg)
            run_predictions(
                items,
                cfg.extract.build(),
                cfg.extract.mode,
                cfg.cache_dir,
                cross_check=cfg.extract.cross_check,
                progress=progress,
            )
            console.print(f"Ответы модели: {len(items)} в {cfg.cache_dir}")
            return
        result, _, records = evaluate(cfg, progress=progress)
        out = write_report(result, records, cfg)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(2) from exc
    from docai.eval.report import fmt_metric

    o = result["overall"]
    console.print(
        f"Без правок: {fmt_metric('doc_ok', o['doc_ok'])}, "
        f"ячейки F1: {fmt_metric('cells_f1', o['cells_f1'])}, ECE: {fmt_metric('ece', o['ece'])}"
    )
    console.print(f"Отчёт: {out / 'report.md'}")


def eval_compare_command(
    a: Annotated[Path, typer.Argument(help="metrics.json конфигурации A (базовой)")],
    b: Annotated[Path, typer.Argument(help="metrics.json конфигурации B")],
) -> None:
    """Парное сравнение двух прогонов на общих документах: бутстреп разности и Макнемар."""
    from docai.eval.run import compare_runs

    first = json.loads(a.read_text(encoding="utf-8"))
    second = json.loads(b.read_text(encoding="utf-8"))
    cfg = EvalConfig.model_validate(second["config"])
    comp = compare_runs(second, first, cfg)
    if not comp["common_documents"]:
        console.print("[red]Общих документов нет.[/red]")
        raise typer.Exit(1)
    table = Table(title=f"B − A на {comp['common_documents']} общих документах")
    for column in ("Метрика", "A", "B", "B − A", "95% ДИ", "p"):
        table.add_column(column)
    for name, est in comp["metrics"].items():
        if est["a"] is None:
            continue
        table.add_row(
            name,
            f"{est['a']:.4f}",
            f"{est['b']:.4f}",
            f"{est['diff']:+.4f}",
            f"[{est['low']:+.4f}; {est['high']:+.4f}]" if est["low"] is not None else "—",
            f"{est['p_value']:.3f}" if est["p_value"] is not None else "—",
        )
    Console().print(table)
    mc = comp["mcnemar_doc_ok"]
    Console().print(
        f"Макнемар (без правок): только A {mc['a_only']}, только B {mc['b_only']}, "
        f"p = {mc['p_value']:.4f} ({mc['method']})"
    )
