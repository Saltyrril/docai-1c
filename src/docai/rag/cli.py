"""Команды `docai rag ...`: сборка корпуса и вопросов, выборка для ручной проверки,
проверка вопросов владельца.

    uv run docai rag build --config configs/rag/corpus_v1.yaml
    uv run docai rag review --config configs/rag/corpus_v1.yaml
    uv run docai rag check-manual --file configs/rag/questions_manual.yaml
"""

import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from docai.rag.corpus import Corpus, build_corpus, write_corpus
from docai.rag.manual import check_manual
from docai.rag.questions import CATEGORIES, generate_questions, run_sql

app = typer.Typer(help="RAG: корпус договоров и регламентов, вопросы с эталоном.")
console = Console()

DEFAULT_CONFIG = Path("configs/rag/corpus_v1.yaml")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_all(config: dict[str, Any]) -> tuple[Corpus, list[dict[str, Any]]]:
    corpus = build_corpus(config["n_contracts"], config["n_registry"], config["seed"])
    questions = [q.to_json() for q in generate_questions(corpus, config["seed"])]
    return corpus, questions


@app.command()
def build(
    config_path: Annotated[Path, typer.Option("--config", help="YAML-конфиг корпуса.")] = (
        DEFAULT_CONFIG
    ),
    out: Annotated[
        Path | None, typer.Option(help="Папка вывода (по умолчанию из конфига).")
    ] = None,
) -> None:
    """Сгенерировать корпус, реестр и 200 вопросов в папку `out` (data/rag/)."""
    config = load_config(config_path)
    started = time.perf_counter()
    corpus, questions = build_all(config)
    target = out or Path(config["out"])
    info = write_corpus(corpus, target, questions)
    info["config"] = config
    elapsed = time.perf_counter() - started

    table = Table(title=f"Корпус {info['version']}, seed {info['seed']}, {elapsed:.1f} с")
    table.add_column("Что")
    table.add_column("Штук", justify="right")
    for kind, count in info["documents"].items():
        table.add_row(f"документов: {kind}", str(count))
    table.add_row("фрагментов", str(info["fragments"]))
    table.add_row("строк реестра", str(info["registry_rows"]))
    for category, count in info["questions"].items():
        table.add_row(f"вопросов: {CATEGORIES[category]}", str(count))
    console.print(table)
    console.print(f"Папка: {target}. Наша организация: {info['our_company']['name']}")


def _tokens(text: str) -> set[str]:
    """Грубые основы слов (первые 5 букв) для оценки лексического пересечения."""
    words = re.findall(r"[а-яёa-z]{4,}", text.lower().replace("ё", "е"))
    return {w[:5] for w in words}


def overlap_by_category(questions: list[dict[str, Any]]) -> dict[str, float]:
    """Доля основ вопроса, встречающихся в источнике (чем меньше, тем труднее поиск по словам)."""
    values: dict[str, list[float]] = defaultdict(list)
    for q in questions:
        if not q["sources"]:
            continue
        qt = _tokens(q["question"])
        st = set().union(*(_tokens(s["quote"]) for s in q["sources"]))
        if qt:
            values[q["category"]].append(len(qt & st) / len(qt))
    return {k: round(mean(v), 3) for k, v in values.items()}


def stratified_sample(questions: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n случайных вопросов, по возможности поровну из каждой категории."""
    rng = random.Random(f"docai-rag:review:{seed}")
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)
    for items in by_cat.values():
        rng.shuffle(items)
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(by_cat.values()):
        for category in CATEGORIES:
            if by_cat.get(category) and len(picked) < n:
                picked.append(by_cat[category].pop())
    return sorted(picked, key=lambda q: q["id"])


def review_markdown(sample: list[dict[str, Any]], corpus: Corpus) -> str:
    lines = [
        "# Ручная проверка эталона: 20 случайных вопросов",
        "",
        "Для каждого вопроса сверьте эталонный ответ с цитатой источника (или с SQL по реестру) "
        "и отметьте результат. Цитата — дословный фрагмент документа корпуса; путь к документу — "
        "`data/rag/corpus/...`.",
        "",
    ]
    for q in sample:
        lines += [
            f"## {q['id']} · {q['category_name']}",
            "",
            f"**Вопрос.** {q['question']}",
            "",
            f"**Эталон.** {q['answer']}",
            "",
            f"Ключевые факты: {', '.join(f'`{k}`' for k in q['keys']) or '—'}",
            "",
        ]
        for s in q["sources"]:
            quote = s["quote"].replace("\n", "  \n> ")
            lines += [f"Источник `{s['fragment_id']}`:", "", f"> {quote}", ""]
        for s in q["superseded"]:
            quote = s["quote"].replace("\n", "  \n> ")
            lines += [f"Старая редакция `{s['fragment_id']}` (не должна попасть в ответ):", "",
                      f"> {quote}", ""]  # fmt: skip
        if q["sql"]:
            result = run_sql(corpus.registry, q["sql"])
            lines += ["SQL к реестру:", "", f"```sql\n{q['sql']}\n```", "",
                      f"Результат: `{result}`; строки реестра: {', '.join(q['registry_rows'][:12])}"
                      + (" …" if len(q["registry_rows"]) > 12 else ""), ""]  # fmt: skip
        if q.get("sql_detail"):
            detail = run_sql(corpus.registry, q["sql_detail"])
            lines += ["Детализация:", "", f"```sql\n{q['sql_detail']}\n```", "",
                      f"Результат: `{detail}`", ""]  # fmt: skip
        lines += ["- [ ] эталон верен", "- [ ] ошибка: ____", ""]
    return "\n".join(lines)


@app.command()
def review(
    config_path: Annotated[Path, typer.Option("--config", help="YAML-конфиг корпуса.")] = (
        DEFAULT_CONFIG
    ),
    out: Annotated[Path, typer.Option(help="Папка отчёта.")] = Path(
        "reports/2026-11-19_p18_rag_dataset"
    ),
) -> None:
    """Выборка вопросов для ручной проверки и сводка корпуса в папку отчёта."""
    config = load_config(config_path)
    corpus, questions = build_all(config)
    review_cfg = config.get("review", {})
    sample = stratified_sample(questions, review_cfg.get("n", 20), review_cfg.get("seed", 1))
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_sample.md").write_text(review_markdown(sample, corpus), encoding="utf-8",
                                          newline="\n")  # fmt: skip
    (out / "review_sample.json").write_text(
        json.dumps([q["id"] for q in sample], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    stats = corpus_stats(corpus, questions)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8", newline="\n")  # fmt: skip
    console.print(f"Выборка: {', '.join(q['id'] for q in sample)}. Папка: {out}")


def corpus_stats(corpus: Corpus, questions: list[dict[str, Any]]) -> dict[str, Any]:
    docs = corpus.documents
    contracts = [b.base for b in corpus.bundles]
    chars = [len(d.text) for d in docs]
    templates = Counter(q["template"].split(":")[0] for q in questions)
    return {
        "documents": dict(Counter(d.doc_kind for d in docs)),
        "contracts_by_kind": dict(Counter(c.kind.value for c in contracts)),
        "counterparties": len({c.them.inn for c in contracts}),
        "counterparties_with_several_contracts": sum(
            1 for k in Counter(c.them.inn for c in contracts).values() if k > 1
        ),
        "contracts_with_amendments": sum(1 for b in corpus.bundles if b.amendments),
        "amendment_changes": dict(
            Counter(f for b in corpus.bundles for a in b.amendments for f in a.changed)
        ),
        "fragments": sum(len(d.fragments) for d in docs),
        "chars_total": sum(chars),
        "chars_per_doc": {"min": min(chars), "mean": round(mean(chars)), "max": max(chars)},
        "registry_rows": len(corpus.registry),
        "registry_by_type": dict(Counter(r.doc_type for r in corpus.registry)),
        "registry_counterparties": len({r.seller_inn for r in corpus.registry}),
        "questions_by_category": dict(Counter(q["category"] for q in questions)),
        "questions_by_route": dict(Counter(q["route"] for q in questions)),
        "questions_from_regulations": sum(1 for q in questions if q["template"].startswith("reg")),
        "templates": dict(templates),
        "question_source_overlap": overlap_by_category(questions),
    }


@app.command("check-manual")
def check_manual_command(
    file: Annotated[Path, typer.Option(help="YAML с вопросами владельца.")] = Path(
        "configs/rag/questions_manual.yaml"
    ),
    config_path: Annotated[Path, typer.Option("--config", help="YAML-конфиг корпуса.")] = (
        DEFAULT_CONFIG
    ),
    out: Annotated[Path | None, typer.Option(help="Куда записать проверенные вопросы (JSONL).")] = (
        None
    ),
) -> None:
    """Проверить вопросы владельца: категории, цитаты есть в документах, id фрагментов."""
    config = load_config(config_path)
    corpus, _ = build_all(config)
    result = check_manual(file, corpus)
    for problem in result.problems:
        console.print(f"[red]{problem}[/red]")
    console.print(f"Заполнено вопросов: {len(result.questions)} из {result.total}; "
                  f"ошибок: {len(result.problems)}")  # fmt: skip
    target = out or Path(config["out"]) / "questions_manual.jsonl"
    if result.questions and not result.problems:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in result.questions),
            encoding="utf-8",
            newline="\n",
        )
        console.print(f"Записано: {target}")
    if result.problems:
        raise typer.Exit(1)
