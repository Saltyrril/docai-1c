"""P9: прогон конвейера `docai.extract` на 10 документах для каждого пути.

Запуск (нужны Ollama с qwen3.5:2b-q8_0, nvidia-smi; для OCR — Tesseract с rus):
    uv run python reports/2026-11-14_p9_extract_smoke/run.py [--routes xml text vision ocr]

Пишет results.json и report.md рядом со скриптом; полные результаты — в work_dir.
Пути, не указанные в --routes, берутся из прошлого results.json.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from docai.edo import upd_to_xml
from docai.extract import ExtractConfig, Mode, extract, load_config
from docai.extract.ocr import OcrError
from docai.schemas.documents import Upd, parse_document

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TYPES = ("upd", "bill", "act", "torg12", "vat_invoice")


def select(manifest: list[dict], split: str, level: str, per_type: int) -> list[dict]:
    """По каждому типу: первый одностраничный и первый двухстраничный документ сплита."""
    rows = sorted(
        (r for r in manifest if r["split"] == split and r["level"] == level),
        key=lambda r: r["doc_id"],
    )
    chosen = []
    for doc_type in TYPES:
        of_type = [r for r in rows if r["doc_type"] == doc_type]
        single = [r for r in of_type if r["pages"] == 1]
        multi = [r for r in of_type if r["pages"] > 1]
        picked = single[:1] + (multi[:1] if multi else single[1:2])
        chosen.extend(picked[:per_type])
    return chosen


def run_one(
    route: str, doc_id: str, paths: list[Path], mode: Mode, config: ExtractConfig, out_dir: Path
) -> dict[str, Any]:
    result = extract(paths, mode, config)
    text = result.model_dump_json(indent=2) + "\n"
    (out_dir / f"{doc_id}.json").write_text(text, encoding="utf-8", newline="\n")
    meta = result.meta
    schema_ok = False
    if result.document is not None:
        parse_document(result.document)  # повторная проверка JSON по схеме
        schema_ok = True
    document = result.document or {}
    return {
        "route": route,
        "doc_id": doc_id,
        "inputs": [p.name for p in paths],
        "actual_route": meta.route,
        "valid": result.valid,
        "schema_ok": schema_ok,
        "doc_type": document.get("doc_type"),
        "pages": meta.pages,
        "lines": len(document.get("lines", [])),
        "warnings": len(result.warnings),
        "errors": result.errors,
        "prompt_tokens": meta.tokens.prompt,
        "completion_tokens": meta.tokens.completion,
        "llm_calls": len(meta.llm_calls),
        "timings_s": meta.timings_s,
        "peak_vram_mib": meta.vram.peak_mib if meta.vram else None,
        "baseline_vram_mib": meta.vram.baseline_mib if meta.vram else None,
    }


def summarize(rows: list[dict], limit_gib: float) -> dict[str, Any]:
    by_route: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)
    summary = {}
    for route, items in by_route.items():
        peaks = [r["peak_vram_mib"] for r in items if r["peak_vram_mib"] is not None]
        totals = [r["timings_s"]["total"] for r in items]
        summary[route] = {
            "documents": len(items),
            "valid": sum(r["valid"] for r in items),
            "schema_ok": sum(r["schema_ok"] for r in items),
            "with_errors": sum(bool(r["errors"]) for r in items),
            "pages": sum(r["pages"] or 0 for r in items),
            "llm_calls": sum(r["llm_calls"] for r in items),
            "prompt_tokens": sum(r["prompt_tokens"] for r in items),
            "completion_tokens": sum(r["completion_tokens"] for r in items),
            "time_total_s": round(sum(totals), 1),
            "time_mean_s": round(sum(totals) / len(totals), 1),
            "time_max_s": round(max(totals), 1),
            "peak_vram_mib": max(peaks) if peaks else None,
            "vram_ok": all(p / 1024 <= limit_gib for p in peaks) if peaks else None,
        }
    return summary


def _peak(mib: int | None) -> str:
    return "—" if mib is None else f"{mib} ({mib / 1024:.2f} ГиБ)"


def write_report(
    cfg: dict, summary: dict, rows: list[dict], skipped: dict[str, str], path: Path
) -> None:
    lines = [
        "# P9: конвейер извлечения на 10 документах каждого пути",
        "",
        "Проверяется работоспособность, не качество (качество — P11). Критерии: JSON валиден",
        "по схеме `Document` на всех 10 документах каждого пути; пиковая VRAM ≤ 3,8 ГиБ.",
        "",
        f"Запуск: `{cfg['run']}`. Конфиг извлечения: `{cfg['extract_config']}`.",
        "",
        "| Путь | Документов | Валидный JSON | С ошибками частей | Страниц | Вызовов LLM "
        "| Токены вход → выход | Время всего / среднее / макс, с | Пик VRAM, МиБ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for route, s in summary.items():
        lines.append(
            f"| {route} | {s['documents']} | {s['schema_ok']} | {s['with_errors']} | {s['pages']} "
            f"| {s['llm_calls']} | {s['prompt_tokens']} → {s['completion_tokens']} "
            f"| {s['time_total_s']} / {s['time_mean_s']} / {s['time_max_s']} "
            f"| {_peak(s['peak_vram_mib'])} |"
        )
    for route, reason in skipped.items():
        lines.append(f"| {route} | не запускался: {reason} | | | | | | | |")
    lines += [
        "",
        "## По документам",
        "",
        "| Путь | Документ | Страниц | Валиден | doc_type | Строк | Время, с | LLM, с "
        "| Пик VRAM, МиБ | Ошибки |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        errors = "; ".join(r["errors"]) or "—"
        lines.append(
            f"| {r['route']} | {r['doc_id']} | {r['pages'] or '—'} "
            f"| {'да' if r['schema_ok'] else 'нет'} | {r['doc_type'] or '—'} | {r['lines']} "
            f"| {r['timings_s']['total']} | {r['timings_s'].get('llm', 0)} "
            f"| {r['peak_vram_mib'] or '—'} | {errors} |"
        )
    notes = HERE / "notes.md"  # наблюдения, написанные вручную после прогона
    if notes.exists():
        lines += ["", notes.read_text(encoding="utf-8").strip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _xml_inputs(manifest: list[dict], cfg: dict, dataset: Path, work: Path) -> list[Path]:
    """Первые 10 УПД сплита → XML формата ФНС."""
    upds = sorted(
        (r for r in manifest if r["split"] == cfg["split"] and r["doc_type"] == "upd"),
        key=lambda r: r["doc_id"],
    )
    ids = list(dict.fromkeys(r["doc_id"] for r in upds))[:10]
    xml_dir = work / "xml_inputs"
    xml_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for doc_id in ids:
        upd = parse_document((dataset / "json" / f"{doc_id}.json").read_text(encoding="utf-8"))
        assert isinstance(upd, Upd)
        path = xml_dir / f"{doc_id}.xml"
        path.write_bytes(upd_to_xml(upd))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    config = load_config(ROOT / cfg["extract_config"])
    dataset = ROOT / cfg["dataset"]
    manifest_text = (dataset / "manifest.jsonl").read_text(encoding="utf-8")
    manifest = [json.loads(line) for line in manifest_text.splitlines()]
    docs = select(manifest, cfg["split"], cfg["image_level"], cfg["per_type"])
    work = ROOT / cfg["work_dir"]
    routes = args.routes or cfg["routes"]

    results_path = HERE / "results.json"
    previous = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    rows = [r for r in previous.get("rows", []) if r["route"] not in routes]
    skipped = {k: v for k, v in previous.get("skipped", {}).items() if k not in routes}

    for route in routes:
        out_dir = work / route
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if route == "xml":
                for path in _xml_inputs(manifest, cfg, dataset, work):
                    rows.append(run_one(route, path.stem, [path], Mode.AUTO, config, out_dir))
                continue
            for doc in docs:
                if route == "text":
                    paths, mode = [dataset / doc["pdf"]], Mode.AUTO
                else:
                    paths = [dataset / p for p in doc["images"]]
                    mode = Mode.AUTO if route == "vision" else Mode.OCR
                rows.append(run_one(route, doc["doc_id"], paths, mode, config, out_dir))
        except OcrError as exc:
            print(f"{route}: {exc}", file=sys.stderr)
            skipped[route] = str(exc).splitlines()[0]

    order = {name: i for i, name in enumerate(cfg["routes"])}
    rows.sort(key=lambda r: order.get(r["route"], 99))
    summary = summarize(rows, config.vram_limit_gib)
    summary = dict(sorted(summary.items(), key=lambda kv: order.get(kv[0], 99)))
    results = {
        "config": cfg,
        "extract_config": config.model_dump(),
        "documents": [d["doc_id"] for d in docs],
        "summary": summary,
        "skipped": skipped,
        "rows": rows,
    }
    results_text = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    results_path.write_text(results_text, encoding="utf-8", newline="\n")
    write_report(cfg, summary, rows, skipped, HERE / "report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
