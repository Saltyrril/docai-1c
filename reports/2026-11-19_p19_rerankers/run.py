# ruff: noqa: E501  — строки таблиц отчёта не режутся осмысленно
"""P19: сравнение реранкеров на CPU на 50 вопросах P18 — качество и скорость.

    uv run python reports/2026-11-19_p19_rerankers/run.py [--limit N]

Для каждого вопроса один раз строятся кандидаты: разбор вопроса → фильтр по договору или
контрагенту → BM25 (30) и векторы (30) → RRF → 20 лучших. Затем каждый реранкер переставляет
те же 20 кандидатов. Метрики — hit@k, recall@k, MRR по пересечению с источниками эталона;
скорость — время оценки 20 пар на вопрос (CPU, 6 потоков), без загрузки модели.
Пишет results.json и report.md рядом со скриптом.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from docai.rag.bm25 import rrf
from docai.rag.config import RerankerConfig, load_rag_config
from docai.rag.evaluate import (
    Span,
    fragment_spans,
    load_jsonl,
    ranking_metrics,
    source_spans,
    stratified,
)
from docai.rag.lemmas import lemmas
from docai.rag.models import Qwen3Embedder, make_reranker
from docai.rag.query import analyze
from docai.rag.search import RagIndex, Retriever

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def summarize(rows: list[dict[str, Any]], key: str, ks: list[int]) -> dict[str, Any]:
    names = ["mrr"] + [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks]
    overall = {n: mean([r[key][n] for r in rows]) for n in names}
    by_category: dict[str, dict[str, float]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    for category, items in groups.items():
        by_category[category] = {n: mean([r[key][n] for r in items]) for n in names}
    return {"overall": overall, "by_category": by_category}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Первые N вопросов (отладка).")
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    config = load_rag_config(ROOT / cfg["pipeline_config"])
    questions = load_jsonl(ROOT / cfg["questions"])
    spans = fragment_spans(ROOT / cfg["fragments"])
    sample = stratified(questions, cfg["per_category"], cfg["seed"])[: args.limit]
    ks: list[int] = cfg["ks"]
    n_candidates: int = cfg["candidates"]

    index = RagIndex(config.index_dir)
    embedder = Qwen3Embedder(config.embedding)
    retrieval = config.retrieval.model_copy(update={"rerank_candidates": n_candidates})
    retriever = Retriever(index, embedder, None, retrieval)
    today = date.fromisoformat(config.today)

    rows: list[dict[str, Any]] = []
    for q in sample:
        info = analyze(q["question"], index.matcher, today)
        allowed, filters = index.allowed(info)
        bm25 = [
            i for i, _ in index.bm25.search(lemmas(q["question"]), retrieval.bm25_top_k, allowed)
        ]
        vector = retriever.vector_search(q["question"], allowed, retrieval.vector_top_k)
        fused = [i for i, _ in rrf([bm25, vector], k=retrieval.rrf_k)][:n_candidates]
        chunks = [index.chunks[i] for i in fused]
        gold = source_spans(q, spans)
        old = source_spans(q, spans, "superseded")
        ranked = [Span(c.doc_id, c.start, c.end) for c in chunks]
        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "filters": filters,
                "candidates": [c.chunk_id for c in chunks],
                "gold": [s["fragment_id"] for s in q["sources"]],
                "superseded": [s["fragment_id"] for s in q.get("superseded") or []],
                "gold_in_pool": sum(1 for g in gold if any(s.overlaps(g) for s in ranked)),
                "rrf": ranking_metrics(ranked, gold, ks),
                "_chunks": chunks,
                "_gold": gold,
                "_old": old,
                "_question": q["question"],
            }
        )
        print(
            q["id"],
            "кандидатов",
            len(chunks),
            "источников в пуле",
            rows[-1]["gold_in_pool"],
            "из",
            len(gold),
            flush=True,
        )

    results: dict[str, Any] = {
        "config": cfg,
        "questions": [r["id"] for r in rows],
        "rrf": summarize(rows, "rrf", ks),
        "rerankers": {},
    }
    for spec in cfg["rerankers"]:
        rcfg = RerankerConfig(**{**config.reranker.model_dump(), **spec})
        started = time.perf_counter()
        reranker = make_reranker(rcfg, config.embedding.threads)
        assert reranker is not None
        load_s = time.perf_counter() - started
        reranker.score("прогрев", ["прогрев"])
        times: list[float] = []
        superseded_above = 0
        for r in rows:
            passages = [c.index_text() for c in r["_chunks"]]
            started = time.perf_counter()
            scores = reranker.score(r["_question"], passages)
            times.append(time.perf_counter() - started)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            ranked = [
                Span(r["_chunks"][i].doc_id, r["_chunks"][i].start, r["_chunks"][i].end)
                for i in order
            ]
            r[rcfg.kind] = ranking_metrics(ranked, r["_gold"], ks)
            r[f"{rcfg.kind}_top5"] = [r["_chunks"][i].chunk_id for i in order[:5]]
            if r["_old"]:
                first_gold = next(
                    (i for i, s in enumerate(ranked) if any(s.overlaps(g) for g in r["_gold"])), 99
                )
                first_old = next(
                    (i for i, s in enumerate(ranked) if any(s.overlaps(o) for o in r["_old"])), 99
                )
                superseded_above += first_old < first_gold
            print(rcfg.kind, r["id"], round(times[-1], 2), "с", r[rcfg.kind]["hit@1"], flush=True)
        summary = summarize(rows, rcfg.kind, ks)
        summary.update(
            model=rcfg.model,
            load_s=round(load_s, 1),
            seconds_per_question=mean(times),
            seconds_p95=round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
            ms_per_pair=round(1000 * statistics.fmean(times) / n_candidates, 1),
            superseded_ranked_above_source=superseded_above,
            amendment_questions=sum(1 for r in rows if r["_old"]),
        )
        results["rerankers"][rcfg.kind] = summary
        del reranker

    results["rows"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (HERE / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(results, ks)
    print(
        json.dumps({k: v["overall"] for k, v in results["rerankers"].items()}, ensure_ascii=False)
    )


def write_report(results: dict[str, Any], ks: list[int]) -> None:
    names = ["mrr"] + [f"hit@{k}" for k in ks] + ["recall@5"]
    lines = [
        "# P19: реранкеры на CPU — Qwen3-Reranker-0.6B против bge-reranker-v2-m3",
        "",
        f"Вопросов: {len(results['questions'])} (маршрут rag, seed {results['config']['seed']}); "
        f"кандидатов на вопрос: {results['config']['candidates']} (одни и те же для обоих). "
        "Попадание — пересечение с источником эталона P18. Скорость — CPU Ryzen 5 5600H, 6 потоков, "
        "float32, без загрузки модели.",
        "",
        "| Порядок | " + " | ".join(names) + " | с/вопрос | мс/пара | загрузка, с |",
        "|---|" + "---:|" * (len(names) + 3),
    ]
    row = results["rrf"]["overall"]
    lines.append(
        "| RRF без реранкера | " + " | ".join(f"{row[n]:.3f}" for n in names) + " | — | — | — |"
    )
    for s in results["rerankers"].values():
        o = s["overall"]
        lines.append(
            f"| {s['model']} | "
            + " | ".join(f"{o[n]:.3f}" for n in names)
            + f" | {s['seconds_per_question']:.2f} | {s['ms_per_pair']} | {s['load_s']} |"
        )
    lines += ["", "## По категориям (hit@1 / recall@5)", ""]
    categories = list(results["rrf"]["by_category"])
    lines.append("| Категория | RRF | " + " | ".join(results["rerankers"]) + " |")
    lines.append("|---|---:|" + "---:|" * len(results["rerankers"]))
    for c in categories:
        cells = [results["rrf"]["by_category"][c]]
        cells += [s["by_category"][c] for s in results["rerankers"].values()]
        lines.append(
            f"| {c} | "
            + " | ".join(f"{x['hit@1']:.2f} / {x['recall@5']:.2f}" for x in cells)
            + " |"
        )
    lines += ["", "## Доп. соглашения", ""]
    for kind, s in results["rerankers"].items():
        lines.append(
            f"- {kind}: старая редакция выше действующей в {s['superseded_ranked_above_source']} "
            f"из {s['amendment_questions']} вопросов"
        )
    in_pool = sum(1 for r in results["rows"] if r["gold_in_pool"])
    lines += [
        "",
        f"Источник есть среди 20 кандидатов в {in_pool} из {len(results['rows'])} вопросов "
        "(потолок для любого реранкера).",
        "",
    ]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
