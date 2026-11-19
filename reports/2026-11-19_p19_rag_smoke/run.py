# ruff: noqa: E501  — строки таблиц отчёта не режутся осмысленно
"""P19: приёмка RAG-конвейера на 20 вопросах P18 — цитаты, время ответа, пик VRAM.

    uv run python reports/2026-11-19_p19_rag_smoke/run.py [--model qwen3.5:2b-q4_K_M] [--tag q4]

Перед замером модели прогреваются (эмбеддинги и реранкер на CPU, LLM в Ollama), как
в сервисе с `--rag-warm-up`; время холодного старта пишется отдельно. VRAM — вся видеокарта
по nvidia-smi (как в P0/P9). Пишет results[_tag].json и report[_tag].md рядом со скриптом.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from docai.extract.vram import VramMonitor
from docai.rag.config import load_rag_config
from docai.rag.evaluate import fragment_spans, judge, load_jsonl, stratified
from docai.rag.generate import ask_llm
from docai.rag.pipeline import RagPipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Другая модель Ollama (сравнение).")
    parser.add_argument("--tag", default="", help="Суффикс файлов результата.")
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Независимая выборка: seed holdout_seed, без вопросов основной выборки.",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    config = load_rag_config(ROOT / cfg["pipeline_config"])
    if args.model:
        config.llm.model = args.model
    questions = load_jsonl(ROOT / cfg["questions"])
    spans = fragment_spans(ROOT / cfg["fragments"])
    exclude_path = ROOT / cfg["exclude_from"]
    exclude = (
        json.loads(exclude_path.read_text("utf-8"))["questions"] if exclude_path.is_file() else []
    )
    sample = stratified(questions, cfg["per_category"], cfg["seed"], exclude=exclude)
    if args.holdout:
        # Основная выборка разбиралась при отладке промпта и поиска; эта — нет.
        exclude = [*exclude, *(q["id"] for q in sample)]
        sample = stratified(questions, cfg["per_category"], cfg["holdout_seed"], exclude=exclude)
        args.tag = args.tag or "holdout"

    started = time.perf_counter()
    pipeline = RagPipeline(config)
    pipeline.warm_up()
    cpu_warm_s = time.perf_counter() - started
    started = time.perf_counter()
    ask_llm(config.llm, 'Ответь JSON {"ok": true}', {"type": "object"})
    llm_warm_s = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    poll_s = 0.25
    with VramMonitor(poll_s) as vram:
        for q in sample:
            result = pipeline.ask(q["question"], debug=True).model_dump()
            verdict = judge(q, result, spans)
            rows.append({"question": q, "result": result, "verdict": verdict})
            print(
                q["id"],
                q["category"],
                "OK " if verdict["citations_ok"] else "BAD",
                result["timings"]["total_s"],
                "с |",
                result["answer"][:150],
                flush=True,
            )
    stats = vram.stats()

    times = [r["result"]["timings"]["total_s"] for r in rows]
    ok = sum(1 for r in rows if r["verdict"]["citations_ok"])
    summary = {
        "model": config.llm.model,
        "reranker": config.reranker.model if config.reranker.kind != "none" else None,
        "questions": len(rows),
        "citations_ok": ok,
        "keys_ok": sum(1 for r in rows if r["verdict"].get("keys_ok")),
        "keys_total": sum(1 for r in rows if r["verdict"].get("keys_ok") is not None),
        "max_s": max(times),
        "mean_s": round(sum(times) / len(times), 2),
        "over_limit": sum(1 for t in times if t >= cfg["max_seconds"]),
        "vram_peak_gib": stats.peak_gib,
        "vram_baseline_mib": stats.baseline_mib,
        "vram_poll_s": poll_s,
        "cold_start_cpu_s": round(cpu_warm_s, 1),
        "cold_start_llm_s": round(llm_warm_s, 1),
    }
    summary["passed"] = bool(
        ok == len(rows)
        and summary["max_s"] < cfg["max_seconds"]
        and stats.peak_gib is not None
        and stats.peak_gib <= cfg["max_vram_gib"]
    )
    suffix = f"_{args.tag}" if args.tag else ""
    out = {"config": cfg, "summary": summary, "rows": rows}
    (HERE / f"results{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(out, HERE / f"report{suffix}.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_report(out: dict[str, Any], path: Path) -> None:
    s = out["summary"]
    lines = [
        "# P19: RAG-конвейер на 20 вопросах P18",
        "",
        f"Модель {s['model']} (Ollama, GPU), реранкер {s['reranker']} (CPU), эмбеддинги — CPU.",
        "",
        "| Критерий | Результат | Порог |",
        "|---|---|---|",
        f"| Правильные цитаты (или «не нашёл» / верный SQL) | {s['citations_ok']} из {s['questions']} | все |",
        f"| Время ответа, макс. | {s['max_s']:.1f} с (среднее {s['mean_s']:.1f} с) | < 15 с |",
        f"| Пик VRAM (вся видеокарта) | {s['vram_peak_gib']} ГиБ | ≤ 3,8 ГиБ |",
        f"| Ключевые факты в ответе (rag) | {s['keys_ok']} из {s['keys_total']} | справочно |",
        "",
        f"Холодный старт (не входит в замер): модели CPU {s['cold_start_cpu_s']} с, "
        f"LLM в Ollama {s['cold_start_llm_s']} с.",
        "",
        "| id | категория | маршрут | цитаты | время, с | ответ | цитаты (id) |",
        "|---|---|---|---|---:|---|---|",
    ]
    for r in out["rows"]:
        q, res, v = r["question"], r["result"], r["verdict"]
        cited = ", ".join(c["fragment_id"] for c in res["citations"]) or (
            "SQL" if res["route"] == "sql" else "—"
        )
        answer = res["answer"].replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {q['id']} | {q['category']} | {res['route']} | {'да' if v['citations_ok'] else 'нет'} "
            f"| {res['timings']['total_s']:.1f} | {answer} | {cited} |"
        )
    lines += ["", "Эталонные источники:", ""]
    for r in out["rows"]:
        q = r["question"]
        gold = ", ".join(x["fragment_id"] for x in q["sources"]) or (q.get("sql") or "нет ответа")
        lines.append(f"- {q['id']}: {q['question']} — {gold}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
