"""P10: сколько намеренных ошибок ловят проверки, ложные срабатывания, калибровка уверенности.

Запуск (Ollama не нужна; нужны датасет v1 в data/datasets/v1 и прогон P9 в data/experiments):
    uv run python reports/2026-11-16_p10_validation/run.py

Пишет results.json и report.md рядом со скриптом. Содержимое реальных документов не используется.
"""
# ruff: noqa: E501  (длинные строки отчёта)

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from docai.eval import CORRUPTIONS, evaluate_clean, evaluate_corruptions
from docai.eval.catch import CorruptionStats
from docai.extract.pdftext import pdf_pages_text
from docai.postprocess import DocStatus, PostprocessConfig, postprocess
from docai.schemas.documents import parse_document
from docai.schemas.fields import FIELD_SPEC, get_field_value
from docai.validation import validate_document
from docai.validation.normalize import normalize_for_compare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = ROOT / "data" / "datasets" / "v1"
SMOKE = ROOT / "data" / "experiments" / "p9_extract_smoke"
ROUTES = ("text", "vision", "ocr")


def load_docs(rows: list[dict], n: int, seed: int) -> tuple[list[str], list]:
    ids = sorted({r["doc_id"] for r in rows if r["split"] == "test" and not r["has_errors"]})
    random.Random(seed).shuffle(ids)
    ids = sorted(ids[:n])
    docs = [parse_document((DATASET / "json" / f"{i}.json").read_bytes()) for i in ids]
    return ids, docs


def group_rate(stats: list[CorruptionStats], group: str, attr: str) -> tuple[int, int, float]:
    items = [s for s in stats if s.group == group]
    applied = sum(s.applied for s in items)
    hit = sum(getattr(s, attr) for s in items)
    return hit, applied, hit / applied if applied else float("nan")


def corruption_rows(stats: list[CorruptionStats]) -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "group": s.group,
            "title": s.title_ru,
            "applied": s.applied,
            "caught": s.caught,
            "caught_error": s.caught_error,
            "fixed_back": s.fixed_back,
            "flagged": s.flagged,
            "silent": s.silent,
            "wrong_fix": s.wrong_fix,
        }
        for s in stats
    ]


def p3_corrupted(rows: list[dict], cfg: PostprocessConfig) -> dict[str, Any]:
    """Документы датасета с намеренными ошибками генератора P3 (по манифесту)."""
    bad = sorted({r["doc_id"]: r["error_kinds"] for r in rows if r["has_errors"]}.items())
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for doc_id, kinds in bad:
        doc = parse_document((DATASET / "json" / f"{doc_id}.json").read_bytes())
        text = "\n".join(pdf_pages_text(DATASET / "pdf" / f"{doc_id}.pdf"))
        report = postprocess(doc, route="text", texts=[("text_layer", text)], config=cfg).report
        stat = by_kind[kinds[0]]
        stat["documents"] += 1
        stat["caught"] += bool(report.issues_before)
        stat["fixed"] += any(c.kind == "autofix" for c in report.corrections)
        stat["still_errors"] += bool(report.issues_after)
        stat["upload"] += report.status is DocStatus.UPLOAD
    return {k: dict(v) for k, v in by_kind.items()}


def compare_to_truth(doc_json: dict, truth, report) -> list[dict[str, Any]]:
    """Для каждого поля эталона: верно ли прочитано и какая уверенность поставлена."""
    got = parse_document(doc_json)
    conf = {f.field: f.confidence for f in report.fields}
    out = []
    for spec in FIELD_SPEC:
        if not spec.applies_to(truth.doc_type):
            continue
        t_val, g_val = get_field_value(truth, spec), get_field_value(got, spec)
        if spec.is_line_field:
            size = max(len(t_val), len(g_val))
            t_val = list(t_val) + [None] * (size - len(t_val))
            g_val = list(g_val) + [None] * (size - len(g_val))
            pairs = [
                (spec.name.replace("lines[]", f"lines[{i}]"), t, g)
                for i, (t, g) in enumerate(zip(t_val, g_val, strict=True))
            ]
        else:
            pairs = [(spec.name, t_val, g_val)]
        for name, t, g in pairs:
            nt = normalize_for_compare(spec.norm, t)
            if nt is None:
                continue
            out.append(
                {
                    "field": name,
                    "correct": nt == normalize_for_compare(spec.norm, g),
                    "confidence": conf.get(name),
                }
            )
    return out


def real_outputs(cfg: PostprocessConfig) -> dict[str, Any]:
    """Настоящие ответы модели из P9 (30 документов) против эталона: калибровка уверенности."""
    per_route: dict[str, list[dict]] = defaultdict(list)
    docs_info: dict[str, list[dict]] = defaultdict(list)
    alt_docs: dict[str, dict[str, Any]] = {r: {} for r in ROUTES}
    for route in ROUTES:
        for path in sorted((SMOKE / route).glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["document"]:
                alt_docs[route][path.stem] = parse_document(result["document"])
    for route in ROUTES:
        for doc_id, got in alt_docs[route].items():
            truth = parse_document((DATASET / "json" / f"{doc_id}.json").read_bytes())
            texts = []
            if route == "text":
                text = "\n".join(pdf_pages_text(DATASET / "pdf" / f"{doc_id}.pdf"))
                texts = [("text_layer", text)]
            others = [alt_docs[r][doc_id] for r in ROUTES if r != route and doc_id in alt_docs[r]]
            variants = {"alone": [], "with_other_paths": others}
            for variant, alternates in variants.items():
                report = postprocess(
                    got, route=route, texts=texts, alternates=alternates, config=cfg
                )
                fields = compare_to_truth(
                    report.document.model_dump(mode="json"), truth, report.report
                )
                key = f"{route}/{variant}"
                per_route[key] += fields
                docs_info[key].append(
                    {
                        "doc": doc_id,
                        "status": report.report.status.value,
                        "wrong": sum(not f["correct"] for f in fields),
                        "fixes": sum(c.kind == "autofix" for c in report.report.corrections),
                        "raw_issues": len(validate_document(got)),
                    }
                )
    return {
        key: summarize_calibration(fields, docs_info[key], cfg) for key, fields in per_route.items()
    }


def summarize_calibration(fields: list[dict], docs: list[dict], cfg: PostprocessConfig) -> dict:
    t = cfg.review_threshold
    scored = [f for f in fields if f["confidence"] is not None]
    wrong = [f for f in fields if not f["correct"]]
    high = [f for f in scored if f["confidence"] >= t]
    wrong_flagged = [f for f in wrong if f["confidence"] is not None and f["confidence"] < t]
    brier = sum((f["confidence"] - f["correct"]) ** 2 for f in scored) / len(scored)
    bins = []
    for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        cell = [f for f in scored if lo <= f["confidence"] < hi]
        if cell:
            bins.append(
                {
                    "range": f"{lo:.2f}–{min(hi, 1):.2f}",
                    "fields": len(cell),
                    "mean_confidence": round(sum(f["confidence"] for f in cell) / len(cell), 3),
                    "share_correct": round(sum(f["correct"] for f in cell) / len(cell), 3),
                }
            )
    statuses: dict[str, int] = defaultdict(int)
    for d in docs:
        statuses[d["status"]] += 1
    uploaded_bad = [d for d in docs if d["status"] == "upload" and d["wrong"]]
    return {
        "documents": len(docs),
        "fields": len(fields),
        "wrong_fields": len(wrong),
        "share_correct": round(1 - len(wrong) / len(fields), 3),
        "threshold": t,
        "precision_above_threshold": round(sum(f["correct"] for f in high) / len(high), 3)
        if high
        else None,
        "wrong_flagged_share": round(len(wrong_flagged) / len(wrong), 3) if wrong else None,
        "brier": round(brier, 4),
        "bins": bins,
        "statuses": dict(statuses),
        "uploaded_with_wrong_fields": len(uploaded_bad),
        "autofixes": sum(d["fixes"] for d in docs),
        "docs_with_raw_issues": sum(bool(d["raw_issues"]) for d in docs),
    }


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def write_report(res: dict[str, Any], path: Path) -> None:
    a, b = res["no_text"], res["text_layer"]
    lines = [
        "# P10: постобработка и уверенность — измерения",
        "",
        f"Документов датасета v1 (сплит test, без намеренных ошибок): **{res['n_docs']}**, "
        f"seed отбора {res['seed']}. Порчи — `docai.eval.corruptions` "
        f"({sum(c.group == 'rule' for c in CORRUPTIONS)} видов «для правил», "
        f"{sum(c.group == 'outside' for c in CORRUPTIONS)} видов «вне правил»). "
        "Формулы и пороги — D41–D43.",
        "",
        "## Критерии готовности",
        "",
        "| Критерий | Результат | Порог |",
        "|---|---|---|",
        f"| Проверки ловят порчи для правил (замечание проверок) | **{pct(a['rule_caught_rate'])}** "
        f"({a['rule_caught']} из {a['rule_applied']}) | ≥ 95% |",
        f"| Документы с ошибками генератора P3 (по манифесту), замечание проверок | "
        f"**{pct(res['p3']['caught_rate'])}** ({res['p3']['caught']} из {res['p3']['documents']}) | ≥ 95% |",
        f"| Ложные срабатывания проверок на чистых документах | "
        f"**{pct(a['clean_false_alarm_rate'])}** ({a['clean_false_alarm']} из {a['clean_documents']}) | ≤ 2% |",
        "",
        "## Что ловят проверки: по видам порчи",
        "",
        "`Поймано` — проверки дали замечание (ошибку или предупреждение). Здесь текста-источника "
        "нет: измеряются только правила `docai.validation`. `Исправлено` — после автоисправлений "
        "документ равен верному. `Неверных исправлений` — автоисправление сработало, "
        "а документ остался неверным.",
        "",
        "| Порча | Применена | Поймано | Из них ошибкой | Исправлено | Неверных исправлений |",
        "|---|---|---|---|---|---|",
    ]
    for group, title in (("rule", "Для правил"), ("outside", "Вне правил")):
        lines.append(f"| **{title}** | | | | | |")
        for row in a["rows"]:
            if row["group"] != group:
                continue
            n = row["applied"]
            lines.append(
                f"| {row['title']} | {n} | {pct(row['caught'] / n)} | {pct(row['caught_error'] / n)} "
                f"| {pct(row['fixed_back'] / n)} | {row['wrong_fix']} |"
            )
    lines += [
        "",
        f"Итого «для правил»: поймано {pct(a['rule_caught_rate'])}, из них замечанием уровня «ошибка» "
        f"{pct(a['rule_caught_error_rate'])}; исправлено автоматически {pct(a['rule_fixed_rate'])}; "
        f"неверных исправлений {a['rule_wrong_fix']}.",
        f"Итого «вне правил»: поймано правилами {pct(a['outside_caught_rate'])} — по смыслу "
        "(опечатка в наименовании, номер документа, КПП той же формы) правила их не видят.",
        "",
        "## Статус документа, когда есть текст-источник (текстовый слой PDF)",
        "",
        f"Чистые документы: «можно загружать» {b['clean_statuses'].get('upload', 0)} из "
        f"{b['clean_documents']}, «нужна проверка» {b['clean_statuses'].get('review', 0)}, "
        f"«отклонить» {b['clean_statuses'].get('reject', 0)}.",
        "",
        "`Не загружен` — статус не «можно загружать». `Молча загружен` — «можно загружать», "
        "хотя документ неверен (самый вредный исход).",
        "",
        "| Порча | Применена | Не загружен или исправлен | Молча загружен |",
        "|---|---|---|---|",
    ]
    for group, title in (("rule", "Для правил"), ("outside", "Вне правил")):
        lines.append(f"| **{title}** | | | |")
        for row in b["rows"]:
            if row["group"] != group:
                continue
            n = row["applied"]
            lines.append(
                f"| {row['title']} | {n} | {pct((n - row['silent']) / n)} | {row['silent']} ({pct(row['silent'] / n)}) |"
            )
    p3 = res["p3"]
    lines += [
        "",
        "## Ошибки генератора P3 (датасет v1)",
        "",
        f"Документов с намеренной ошибкой: {p3['documents']}. Замечание проверок: "
        f"{p3['caught']}; исправлено автоматически: {p3['fixed']}; после исправлений остались "
        f"ошибки: {p3['still_errors']}; статус «можно загружать»: {p3['upload']}.",
        "",
        "| Вид ошибки | Документов | Поймано | Исправлено | Остались ошибки |",
        "|---|---|---|---|---|",
    ]
    for kind, row in sorted(p3["by_kind"].items()):
        lines.append(
            f"| {kind} | {row['documents']} | {row['caught']} | {row['fixed']} | {row['still_errors']} |"
        )
    lines += [
        "",
        "## Уверенность на настоящих ответах модели (прогон P9, 30 документов)",
        "",
        "Каждое поле сравнено с эталоном. «Верно» — нормализованное значение равно эталонному. "
        f"Порог «нужна проверка»: {res['calibration']['text/alone']['threshold']}. "
        "**Приоры p₀ взяты из этих же 30 документов (D42), поэтому калибровка здесь оптимистична; "
        "честная проверка — на тестовом сплите в P13.**",
        "",
        "| Путь / вариант | Документов | Полей | Верно | Верных среди уверенность ≥ порога | "
        "Неверных поймано (уверенность < порога) | Brier | Статусы | Загружены с ошибкой |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in res["calibration"].items():
        st = ", ".join(f"{k} {v}" for k, v in sorted(c["statuses"].items()))
        lines.append(
            f"| {key} | {c['documents']} | {c['fields']} | {pct(c['share_correct'])} | "
            f"{pct(c['precision_above_threshold'])} | {pct(c['wrong_flagged_share'])} | "
            f"{c['brier']} | {st} | {c['uploaded_with_wrong_fields']} |"
        )
    lines += ["", "Таблица калибровки (доля верных полей в корзине уверенности):", ""]
    for key, c in res["calibration"].items():
        cells = "; ".join(
            f"{b_['range']}: уверенность {b_['mean_confidence']}, верно {b_['share_correct']} "
            f"({b_['fields']})"
            for b_ in c["bins"]
        )
        lines.append(f"- `{key}` — {cells}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    cfg = PostprocessConfig()
    n_docs, seed = 250, 2026
    rows = [
        json.loads(line)
        for line in (DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids, docs = load_docs(rows, n_docs, seed)
    text_of = {
        id(doc): "\n".join(pdf_pages_text(DATASET / "pdf" / f"{doc_id}.pdf"))
        for doc_id, doc in zip(ids, docs, strict=True)
    }

    def texts_for(doc: Any) -> list[tuple[str, str]]:
        return [("text_layer", text_of[id(doc)])]

    # А. Только правила: текста-источника нет.
    clean = evaluate_clean(docs, config=cfg)
    stats = evaluate_corruptions(docs, config=cfg)
    rule_hit, rule_n, rule_rate = group_rate(stats, "rule", "caught")
    err_hit, _, err_rate = group_rate(stats, "rule", "caught_error")
    fix_hit, _, fix_rate = group_rate(stats, "rule", "fixed_back")
    wf, _, _ = group_rate(stats, "rule", "wrong_fix")
    _, _, out_rate = group_rate(stats, "outside", "caught")
    no_text = {
        "clean_documents": clean.documents,
        "clean_false_alarm": clean.false_alarm,
        "clean_false_alarm_rate": clean.false_alarm_rate,
        "rule_applied": rule_n,
        "rule_caught": rule_hit,
        "rule_caught_rate": rule_rate,
        "rule_caught_error_rate": err_rate,
        "rule_fixed_rate": fix_rate,
        "rule_wrong_fix": wf,
        "outside_caught_rate": out_rate,
        "rows": corruption_rows(stats),
    }
    assert err_hit <= rule_hit and fix_hit <= rule_n

    # Б. С текстовым слоем PDF как источником.
    clean_t = evaluate_clean(docs, texts_for=texts_for, config=cfg)
    stats_t = evaluate_corruptions(docs, texts_for=texts_for, config=cfg)
    with_text = {
        "clean_documents": clean_t.documents,
        "clean_statuses": dict(clean_t.statuses),
        "rows": corruption_rows(stats_t),
    }

    # В. Ошибки генератора P3 из манифеста.
    p3_kinds = p3_corrupted(rows, cfg)
    p3_total = {
        key: sum(v[key] for v in p3_kinds.values())
        for key in ("documents", "caught", "fixed", "still_errors", "upload")
    }
    p3 = {
        **p3_total,
        "caught_rate": p3_total["caught"] / p3_total["documents"],
        "by_kind": p3_kinds,
    }

    # Г. Настоящие ответы модели.
    calibration = real_outputs(cfg)

    results = {
        "n_docs": n_docs,
        "seed": seed,
        "postprocess_config": cfg.model_dump(),
        "no_text": no_text,
        "text_layer": with_text,
        "p3": p3,
        "calibration": calibration,
    }
    (HERE / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (HERE / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": "uv run python reports/2026-11-16_p10_validation/run.py",
                "dataset": "data/datasets/v1 (сплит test)",
                "n_docs": n_docs,
                "seed": seed,
                "p9_outputs": "data/experiments/p9_extract_smoke",
                "postprocess": cfg.model_dump(),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    write_report(results, HERE / "report.md")
    print(json.dumps({k: results[k] for k in ("p3",)}, ensure_ascii=False, indent=1)[:1500])
    print(
        "rule caught",
        rule_rate,
        "clean FA",
        clean.false_alarm_rate,
        "clean statuses",
        dict(clean_t.statuses),
    )
    for key, c in calibration.items():
        print(
            key,
            c["documents"],
            c["share_correct"],
            c["precision_above_threshold"],
            c["wrong_flagged_share"],
            c["brier"],
            c["statuses"],
            c["uploaded_with_wrong_fields"],
        )


if __name__ == "__main__":
    main()
