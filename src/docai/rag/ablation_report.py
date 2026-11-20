"""Отчёт оценки RAG (P20): таблицы метрик с интервалами, абляции, категории, ошибки.

Читает кэши `ablation.py` и ничего не запускает: пересчёт метрик и интервалов не требует ни
индекса, ни моделей. Пишет в `report_dir`: `report.md` (для человека), `results.json`
(метрики и счётчики по вопросам), `errors.md` (неверные ответы основной конфигурации с
причиной), `config.yaml` (копия настроек оценки).
"""

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from docai.eval.stats import Estimate, McNemar, PairedEstimate, ratio
from docai.rag.ablation import (
    AblationSpec,
    Cache,
    EvalSettings,
    answers_cache_path,
    config_for,
    fingerprint,
    has_gold,
    retrieval_cache_path,
    router_agreement,
    tuned_question_ids,
)
from docai.rag.config import load_rag_config
from docai.rag.evaluate import Span, fragment_spans, load_jsonl
from docai.rag.questions import CATEGORIES
from docai.rag.scoring import (
    ANSWER_METRICS,
    RETRIEVAL_METRICS,
    answer_row,
    checkable_keys,
    compare,
    covered_at,
    estimate,
    mcnemar_on,
    retrieval_row,
    spans_of,
)

Row = dict[str, float]
FULL = "full"


class IncompleteRunError(RuntimeError):
    """В кэше нет части ответов: сначала нужно запустить этапы retrieval и answers."""


# --- Загрузка -------------------------------------------------------------------------------


def _span(item: Sequence[Any]) -> Span:
    return Span(item[1], int(item[2]), int(item[3]))


def load_retrieval(
    settings: EvalSettings, qset: str, questions: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """id вопроса → запись кэша поиска (варианты с рангами)."""
    base = load_rag_config(settings.pipeline_config)
    fp = fingerprint(config_for(base, AblationSpec(name="x", label="x", run=[], reranker=True)))
    cache = Cache(retrieval_cache_path(settings, qset), fp)
    out: dict[str, dict[str, Any]] = {}
    missing = []
    for q in questions:
        if not has_gold(q):
            continue
        row = cache.get(q)
        if row is None:
            missing.append(q["id"])
        else:
            out[q["id"]] = row
    if missing:
        raise IncompleteRunError(
            f"нет рангов поиска для {len(missing)} вопросов: запустите --stage retrieval"
        )
    return out


def load_answers(
    settings: EvalSettings, qset: str, questions: Sequence[dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """конфигурация → id вопроса → ответ; недостающие маршруты берутся из `reuse_from`."""
    base = load_rag_config(settings.pipeline_config)
    own: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in settings.ablations:
        cache = Cache(
            answers_cache_path(settings, qset, spec.name), fingerprint(config_for(base, spec))
        )
        own[spec.name] = {}
        for q in questions:
            if q["route"] in spec.run:
                row = cache.get(q)
                if row is None:
                    raise IncompleteRunError(
                        f"нет ответа {spec.name}/{q['id']}: запустите --stage answers"
                    )
                own[spec.name][q["id"]] = row
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in settings.ablations:
        rows = dict(own[spec.name])
        if spec.reuse_from:
            for q in questions:
                if q["route"] not in spec.run:
                    rows[q["id"]] = own[spec.reuse_from][q["id"]]
        merged[spec.name] = rows
    return merged


# --- Подсчёт --------------------------------------------------------------------------------


def _context_recall(gold: Sequence[Span], context: Sequence[Span]) -> float:
    covered = covered_at(context, gold, len(context))
    return sum(covered) / len(gold) if gold else 0.0


def retrieval_rows(
    questions: Sequence[dict[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    spans: Mapping[str, Span],
    key: str,
    ks: Sequence[int],
) -> dict[str, Row]:
    rows: dict[str, Row] = {}
    for q in questions:
        if not has_gold(q):
            continue
        gold = spans_of(q, spans)
        entry = cache[q["id"]]
        variant = entry["variants"].get(key)
        ranked = [_span(x) for x in variant["ranked"]] if variant else []
        context = [_span(x) for x in variant["context"]] if variant else []
        row = retrieval_row(ranked, gold, ks)
        row["ctx_recall"] = _context_recall(gold, context)
        rows[q["id"]] = row
    return rows


def answer_rows(
    questions: Sequence[dict[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    spans: Mapping[str, Span],
    fragment_text: Mapping[str, str],
) -> dict[str, Row]:
    rows = {}
    for q in questions:
        gold = [fragment_text[s["fragment_id"]] for s in q.get("sources") or []]
        rows[q["id"]] = answer_row(q, answers[q["id"]], spans, gold)
    return rows


def pick(rows: Mapping[str, Row], ids: Sequence[str]) -> list[Row]:
    return [rows[i] for i in ids if i in rows]


# --- Форматирование -------------------------------------------------------------------------


def _num(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(e: Estimate) -> str:
    if e.value != e.value:
        return "—"
    return f"{_num(100 * e.value, 1)}% [{_num(100 * e.low, 1)}; {_num(100 * e.high, 1)}]"


def dec(e: Estimate, digits: int = 3) -> str:
    if e.value != e.value:
        return "—"
    return f"{_num(e.value, digits)} [{_num(e.low, digits)}; {_num(e.high, digits)}]"


def _p(p: float) -> str:
    """«p < 0,001» или «p = 0,014»."""
    return "p < 0,001" if p < 0.001 else f"p = {_num(p, 3)}"


def delta(e: PairedEstimate, as_pct: bool, digits: int = 3) -> str:
    scale = 100 if as_pct else 1
    unit = " п.п." if as_pct else ""
    d = digits if not as_pct else 1
    sign = "+" if e.diff >= 0 else "−"
    return (
        f"{sign}{_num(abs(e.diff) * scale, d)}{unit} "
        f"[{_num(e.low * scale, d)}; {_num(e.high * scale, d)}], {_p(e.p_value)}"
    )


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str = "") -> str:
    sep = ["---"] + [
        "---:" if (align[i : i + 1] == "r") else "---" for i in range(len(headers) - 1)
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def count_cell(rows: Sequence[Row], num: str, den: str) -> str:
    n = sum(r.get(den, 0.0) for r in rows)
    k = sum(r.get(num, 0.0) for r in rows)
    return f"{int(k)} из {int(n)}" if n else "—"


# --- Отчёт ----------------------------------------------------------------------------------


def write_report(settings: EvalSettings, qset: str, config_path: Path) -> Path:
    questions_path = settings.questions if qset == "p18" else settings.manual_questions
    questions = load_jsonl(questions_path) if questions_path.is_file() else []
    out_dir = settings.report_dir if qset == "p18" else settings.report_dir / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not questions:
        text = (
            "# Вопросы владельца\n\nФайл вопросов пуст или не создан: "
            f"`{questions_path}`. Заполните `configs/rag/questions_manual.yaml`, выполните "
            "`uv run docai rag check-manual` и повторите `uv run docai rag eval --set manual`.\n"
        )
        (out_dir / "report.md").write_text(text, encoding="utf-8")
        return out_dir / "report.md"

    spans = fragment_spans(settings.fragments)
    fragment_text = {f["fragment_id"]: f["text"] for f in load_jsonl(settings.fragments)}
    retrieval = load_retrieval(settings, qset, questions)
    answers = load_answers(settings, qset, questions)
    seed, boots = settings.seed, settings.n_resamples
    ids_all = [q["id"] for q in questions]
    by_id = {q["id"]: q for q in questions}
    gold_ids = [i for i in ids_all if has_gold(by_id[i])]

    variants = {}
    for spec in settings.ablations:
        variants.setdefault(spec.retrieval_key, spec)
    ret_rows = {
        key: retrieval_rows(questions, retrieval, spans, key, settings.ks) for key in variants
    }
    ans_rows = {
        spec.name: answer_rows(questions, answers[spec.name], spans, fragment_text)
        for spec in settings.ablations
    }
    # Названия конфигураций поиска в таблице поиска — по первой конфигурации с таким поиском.
    ret_label = {key: spec.label for key, spec in variants.items()}

    tuned = tuned_question_ids(settings.tuned_on) if qset == "p18" else set()
    untouched = [i for i in ids_all if i not in tuned and by_id[i]["route"] != "table"]
    tuned_ids = [i for i in ids_all if i in tuned and by_id[i]["route"] != "table"]
    table_ids = [i for i in ids_all if by_id[i]["route"] == "table"]

    retrieval_metrics = {**RETRIEVAL_METRICS, "ctx_recall": ratio("ctx_recall", "n")}

    result: dict[str, Any] = {
        "name": settings.name,
        "set": qset,
        "seed": seed,
        "n_resamples": boots,
        "questions": len(questions),
        "questions_with_gold": len(gold_ids),
        "router_agreement": router_agreement(questions),
        "retrieval": {},
        "answers": {},
        "paired": {},
        "categories": {},
        "subsets": {},
    }

    # --- поиск ---
    ret_est = {}
    for key in variants:
        ret_est[key] = estimate(
            pick(ret_rows[key], gold_ids), retrieval_metrics, n_resamples=boots, seed=seed
        )
        result["retrieval"][key] = {m: e.as_dict() for m, e in ret_est[key].items()}
    chain_ret = [
        ("bm25", "hybrid", "гибрид − только BM25"),
        ("vector", "hybrid", "гибрид − только векторы"),
        ("hybrid", "hybrid+rerank", "гибрид + реранкер − гибрид"),
        ("vector", "vector+rerank", "векторы + реранкер − только векторы"),
        ("vector+rerank", "hybrid+rerank", "гибрид + реранкер − векторы + реранкер"),
    ]
    ret_paired = {}
    for a, b, label in chain_ret:
        if a in variants and b in variants:
            ret_paired[label] = compare(
                pick(ret_rows[a], gold_ids),
                pick(ret_rows[b], gold_ids),
                retrieval_metrics,
                n_resamples=boots,
                seed=seed,
            )
            result["paired"][f"retrieval: {label}"] = {
                m: e.as_dict() for m, e in ret_paired[label].items()
            }

    # --- ответы ---
    ans_est = {}
    for spec in settings.ablations:
        ans_est[spec.name] = estimate(
            pick(ans_rows[spec.name], ids_all), ANSWER_METRICS, n_resamples=boots, seed=seed
        )
        result["answers"][spec.name] = {m: e.as_dict() for m, e in ans_est[spec.name].items()}
    answer_chain = [
        ("bm25", "hybrid", "гибрид − только BM25"),
        ("vector", "hybrid", "гибрид − только векторы"),
        ("hybrid", FULL, "гибрид + реранкер − гибрид"),
        ("vector", "vector_rerank", "векторы + реранкер − только векторы"),
        ("vector_rerank", FULL, "гибрид + реранкер − векторы + реранкер"),
        ("no_router", FULL, "с маршрутизатором SQL − без него"),
    ]
    ans_paired: dict[str, tuple[dict[str, PairedEstimate], McNemar]] = {}
    names = {s.name for s in settings.ablations}
    for a, b, label in answer_chain:
        if a in names and b in names:
            ra, rb = pick(ans_rows[a], ids_all), pick(ans_rows[b], ids_all)
            ans_paired[label] = (
                compare(ra, rb, ANSWER_METRICS, n_resamples=boots, seed=seed),
                mcnemar_on(ra, rb, "correct_a", only="answerable"),
            )
            result["paired"][f"answers: {label}"] = {
                **{m: e.as_dict() for m, e in ans_paired[label][0].items()},
                "mcnemar_correct": ans_paired[label][1].as_dict(),
            }

    # --- по категориям (основная конфигурация) ---
    cat_lines = []
    for cat, cat_name in CATEGORIES.items():
        cat_ids = [i for i in ids_all if by_id[i]["category"] == cat]
        if not cat_ids:
            continue
        cat_gold = [i for i in cat_ids if has_gold(by_id[i])]
        entry: dict[str, Any] = {"n": len(cat_ids)}
        r_est = (
            estimate(
                pick(ret_rows["hybrid+rerank"], cat_gold),
                retrieval_metrics,
                n_resamples=boots,
                seed=seed,
            )
            if cat_gold and "hybrid+rerank" in variants
            else None
        )
        a_est = estimate(
            pick(ans_rows[FULL], cat_ids), ANSWER_METRICS, n_resamples=boots, seed=seed
        )
        entry["answers"] = {m: e.as_dict() for m, e in a_est.items()}
        if r_est:
            entry["retrieval"] = {m: e.as_dict() for m, e in r_est.items()}
        result["categories"][cat] = entry
        cat_lines.append((cat, cat_name, cat_ids, cat_gold, r_est, a_est))

    # --- подмножества ---
    for label, ids in (
        ("все", ids_all),
        ("без затронутых", untouched),
        ("затронутые", tuned_ids),
        ("реестр", table_ids),
    ):
        if not ids:
            continue
        sub_gold = [i for i in ids if has_gold(by_id[i])]
        sub: dict[str, Any] = {"questions": len(ids), "with_gold": len(sub_gold)}
        sub["answers"] = {
            m: e.as_dict()
            for m, e in estimate(
                pick(ans_rows[FULL], ids), ANSWER_METRICS, n_resamples=boots, seed=seed
            ).items()
        }
        if sub_gold and "hybrid+rerank" in variants:
            sub["retrieval"] = {
                m: e.as_dict()
                for m, e in estimate(
                    pick(ret_rows["hybrid+rerank"], sub_gold),
                    retrieval_metrics,
                    n_resamples=boots,
                    seed=seed,
                ).items()
            }
        result["subsets"][label] = sub

    # --- проверка самой проверки ---
    full_keys = partial = none_keys = 0
    for i in gold_ids:
        q = by_id[i]
        texts = [fragment_text[x["fragment_id"]] for x in q["sources"]]
        found = len(checkable_keys(q, texts))
        if found == len(q["keys"]):
            full_keys += 1
        elif found:
            partial += 1
        else:
            none_keys += 1
    result["checker"] = {
        "all_keys_checkable": full_keys,
        "some_keys_checkable": partial,
        "none_checkable": none_keys,
        "of": len(gold_ids),
    }

    # --- причины ошибок основной конфигурации ---
    diagnosis, error_items = diagnose(questions, answers[FULL], retrieval, spans)
    result["error_causes"] = dict(diagnosis)

    result["rows"] = {
        "retrieval": {k: v for k, v in ret_rows.items()},
        "answers": ans_rows,
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(config_path, out_dir / "config.yaml")

    text = render(
        settings, qset, questions, variants, ret_label, ret_est, ret_paired, ans_est, ans_paired,
        cat_lines, ans_rows, ret_rows, result, tuned, untouched, tuned_ids, diagnosis,
        retrieval, answers,
    )  # fmt: skip
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    (out_dir / "errors.md").write_text(render_errors(error_items), encoding="utf-8")
    return out_dir / "report.md"


# --- Причины ошибок -------------------------------------------------------------------------


def diagnose(
    questions: Sequence[dict[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    retrieval: Mapping[str, Mapping[str, Any]],
    spans: Mapping[str, Span],
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Причина каждого неверного ответа основной конфигурации (по вопросам rag и table).

    Для rag: эталонный пункт не попал в контекст модели (поиск), попал, но модель отказалась
    или ответила неверно (генерация). Для table — запрос неверен или не построен.
    """
    causes: Counter[str] = Counter()
    items: list[dict[str, Any]] = []
    for q in questions:
        result = answers[q["id"]]
        row = answer_row(q, result, spans)
        if q["route"] == "none":
            if not row.get("refusal_ok"):
                causes["без ответа: ответ вместо отказа"] += 1
                items.append({"q": q, "result": result, "cause": "ответ вместо отказа"})
            continue
        if row.get("correct_a"):
            continue
        if q["route"] == "table":
            cause = "реестр: неверное число" if result.get("found") else "реестр: нет результата"
        else:
            gold = spans_of(q, spans)
            context = [_span(x) for x in retrieval[q["id"]]["variants"]["hybrid+rerank"]["context"]]
            in_context = all(covered_at(context, gold, len(context)))
            any_in = any(covered_at(context, gold, len(context)))
            if not result.get("found"):
                cause = (
                    "отказ при эталоне в контексте" if any_in else "отказ: эталона нет в контексте"
                )
            elif in_context:
                cause = "неверный ответ при всех эталонных пунктах в контексте"
            else:
                cause = "неверный ответ: эталонных пунктов не хватает в контексте"
        causes[cause] += 1
        items.append({"q": q, "result": result, "cause": cause})
    return causes, items


def render_errors(items: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Неверные ответы основной конфигурации (гибрид + реранкер, с маршрутизатором)",
        "",
        "Ответ и цитаты — синтетический корпус P18, реальных документов здесь нет.",
        "",
    ]
    for item in items:
        q, r = item["q"], item["result"]
        cites = ", ".join(c["fragment_id"] for c in r.get("citations") or []) or "—"
        gold = ", ".join(s["fragment_id"] for s in q.get("sources") or []) or "—"
        lines += [
            f"## {q['id']} · {CATEGORIES.get(q['category'], q['category'])} · {item['cause']}",
            "",
            f"- Вопрос: {q['question']}",
            f"- Эталон: {q['answer']}",
            f"- Ключевые факты: {q.get('keys') or q.get('expected_value')}",
            f"- Ответ: {r.get('answer')}",
            f"- Цитаты: {cites}; эталонные источники: {gold}",
            "",
        ]
    return "\n".join(lines)


# --- Разметка отчёта ------------------------------------------------------------------------


def render(
    settings: EvalSettings,
    qset: str,
    questions: Sequence[dict[str, Any]],
    variants: Mapping[str, AblationSpec],
    ret_label: Mapping[str, str],
    ret_est: Mapping[str, Mapping[str, Estimate]],
    ret_paired: Mapping[str, Mapping[str, PairedEstimate]],
    ans_est: Mapping[str, Mapping[str, Estimate]],
    ans_paired: Mapping[str, tuple[Mapping[str, PairedEstimate], McNemar]],
    cat_lines: Sequence[tuple[Any, ...]],
    ans_rows: Mapping[str, Mapping[str, Row]],
    ret_rows: Mapping[str, Mapping[str, Row]],
    result: Mapping[str, Any],
    tuned: set[str],
    untouched: Sequence[str],
    tuned_ids: Sequence[str],
    diagnosis: Counter[str],
    retrieval: Mapping[str, Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    by_id = {q["id"]: q for q in questions}
    n_ans = sum(1 for q in questions if q["route"] != "none")
    n_none = sum(1 for q in questions if q["route"] == "none")
    n_gold = result["questions_with_gold"]
    title = (
        "оценка RAG на 200 вопросах P18" if qset == "p18" else "оценка RAG на вопросах владельца"
    )
    L: list[str] = [f"# P20: {title}", ""]
    L += [
        f"Вопросов: {len(questions)} (с ответом {n_ans}, без ответа {n_none}); "
        f"для оценки поиска — {n_gold} вопросов с эталонными источниками (маршрут rag). "
        f"Единица выборки — вопрос; интервалы — 95% процентильный бутстреп, "
        f"{settings.n_resamples} повторов, seed {settings.seed}; сравнения — парный бутстреп "
        "на тех же вопросах (p — двусторонний), для правильности ответа ещё тест Макнемара "
        "(D44–D45). Корпус и вопросы — синтетические (P18), модели — как в P19.",
        "",
    ]

    # 1. Поиск
    L += ["## 1. Поиск (абляции)", ""]
    rows = []
    for key in variants:
        e = ret_est[key]
        rows.append(
            [ret_label[key], pct(e["recall@5"]), pct(e["recall@10"]), dec(e["mrr"]),
             dec(e["ndcg@10"]), pct(e["ctx_recall"])]
        )  # fmt: skip
    L += [
        md_table(
            ["Поиск", "recall@5", "recall@10", "MRR", "nDCG@10", "Эталон в контексте модели"],
            rows,
        ),
        "",
        "recall@k — доля эталонных пунктов среди первых k найденных (пересечение смещений, D62), "
        "в среднем по вопросам; MRR — по первому найденному эталонному пункту; nDCG@10 — "
        "бинарная релевантность, каждый эталонный пункт засчитывается один раз; «в контексте» — "
        "доля эталонных пунктов среди пунктов, которые уходят в модель (первые 5 + пункты "
        "доп. соглашений). Фильтр по договору и контрагенту из вопроса включён во всех строках.",
        "",
        "Разность между соседними строками:",
        "",
    ]
    prow = []
    for label, est in ret_paired.items():
        prow.append(
            [label, delta(est["recall@5"], True), delta(est["mrr"], False),
             delta(est["ndcg@10"], False)]
        )  # fmt: skip
    L += [md_table(["Сравнение", "recall@5", "MRR", "nDCG@10"], prow), ""]

    # 2. Ответы
    L += ["## 2. Ответы (абляции)", ""]
    rows = []
    for spec in settings.ablations:
        e = ans_est[spec.name]
        rows.append(
            [spec.label, pct(e["correct"]), pct(e["false_refusal"]), pct(e["refusal_ok"]),
             pct(e["cite_precision"]), pct(e["supported"]), pct(e["verified"]), pct(e["span_ok"])]
        )  # fmt: skip
    L += [
        md_table(
            [
                "Конфигурация",
                "Правильных ответов",
                "Ложных «не нашёл»",
                "Верных «не нашёл»",
                "Цитата содержит факт",
                "Факты ответа есть в цитатах",
                "Верно и подтверждено",
                "Цитата = эталонный пункт (P19)",
            ],
            rows,
        ),
        "",
        f"Знаменатели: правильность и ложные «не нашёл» — {n_ans} вопросов с ответом "
        "(rag + реестр); "
        f"верные «не нашёл» — {n_none} вопросов без ответа; «цитата содержит факт» — по числу "
        "цитат; «факты в цитатах» — по ответам rag с цитатами; «верно и подтверждено» и "
        f"«цитата = эталон» — по ответам rag ({n_gold} вопросов). Правильность проверяется "
        "программой: ключевые факты эталона (числа, даты, названия) найдены в тексте ответа "
        "после нормализации (rag) или эталонное число есть в тексте (реестр); LLM-судьи нет. "
        "«Цитата содержит факт» — хотя бы один ключевой факт лежит в **тексте** процитированного "
        "пункта, а не просто пункт пересекается с эталоном.",
        "",
        "Разность между соседними строками (доля правильных ответов):",
        "",
    ]
    prow = []
    for label, (est, mc) in ans_paired.items():
        prow.append(
            [label, delta(est["correct"], True), f"{mc.b} : {mc.c} ({_p(mc.p_value)})",
             delta(est["refusal_ok"], True), delta(est["false_refusal"], True)]
        )  # fmt: skip
    L += [
        md_table(
            [
                "Сравнение",
                "Правильность, Δ",
                "Макнемар (только A : только B)",
                "Верные «не нашёл», Δ",
                "Ложные «не нашёл», Δ",
            ],
            prow,
        ),
        "",
    ]
    ra = result["router_agreement"]
    L += [
        "Что прогонялось. Агрегаты по реестру идут через SQL и не зависят от поиска, поэтому их "
        "результат (прогон «гибрид + реранкер») один и тот же во всех строках с маршрутизатором. "
        "Без маршрутизатора заново прогонялись только вопросы реестра (они уходят в RAG); "
        "остальные вопросы идут тем же путём, потому что правила маршрутизатора отправляют их в "
        f"RAG — расхождений правил с эталоном на {ra['questions']} вопросах: "
        f"{len(ra['disagree'])}.",
        "",
    ]

    # 3. Категории
    L += ["## 3. По категориям вопросов (гибрид + реранкер, конвейер P19)", ""]
    rows = []
    for _cat, cat_name, cat_ids, _cat_gold, r_est, a_est in cat_lines:
        is_none = by_id[cat_ids[0]]["route"] == "none"
        rows.append(
            [
                cat_name,
                str(len(cat_ids)),
                pct(r_est["recall@5"]) if r_est else "—",
                dec(r_est["mrr"]) if r_est else "—",
                pct(r_est["ctx_recall"]) if r_est else "—",
                pct(a_est["refusal_ok"]) if is_none else pct(a_est["correct"]),
                "—" if is_none else pct(a_est["false_refusal"]),
                pct(a_est["verified"]) if r_est else "—",
            ]
        )
    L += [
        md_table(
            [
                "Категория",
                "Вопросов",
                "recall@5",
                "MRR",
                "Эталон в контексте",
                "Правильных ответов*",
                "Ложных «не нашёл»",
                "Верно и подтверждено",
            ],
            rows,
        ),
        "",
        "\\* для категории «ответа нет в корпусе» — доля верных «не нашёл».",
        "",
        "Правильность по конфигурациям и категориям (верных из вопросов):",
        "",
    ]
    head = ["Категория"] + [s.label for s in settings.ablations]
    rows = []
    for _cat, cat_name, cat_ids, *_ in cat_lines:
        is_none = by_id[cat_ids[0]]["route"] == "none"
        cells = [cat_name]
        for spec in settings.ablations:
            sub = pick(ans_rows[spec.name], cat_ids)
            cells.append(count_cell(sub, "refusal_ok", "no_answer") if is_none
                         else count_cell(sub, "correct_a", "answerable"))  # fmt: skip
        rows.append(cells)
    L += [md_table(head, rows), ""]

    # 4. Подмножества
    L += ["## 4. Устойчивость: вопросы, по которым настраивался P19", ""]
    sub_rows = []
    for label, sub in result["subsets"].items():
        e = sub["answers"]
        r = sub.get("retrieval")
        sub_rows.append(
            [
                {"все": "Все вопросы", "без затронутых": "Rag-вопросы без затронутых настройкой",
                 "затронутые": "Rag-вопросы, затронутые настройкой",
                 "реестр": "Агрегаты по реестру (все затронуты)"}[label],
                str(sub["questions"]),
                _fmt(r["recall@5"], True) if r else "—",
                _fmt(r["mrr"], False) if r else "—",
                _fmt(e["correct"], True),
                _fmt(e["false_refusal"], True),
            ]
        )  # fmt: skip
    L += [
        md_table(
            [
                "Подмножество",
                "Вопросов",
                "recall@5",
                "MRR",
                "Правильных ответов",
                "Ложных «не нашёл»",
            ],
            sub_rows,
        ),
        "",
        f"«Затронутые настройкой» — {len(tuned_ids)} вопросов по документам, на которых в P19 "
        "выбирали реранкер и разбирали ошибки (сравнение реранкеров, основная выборка "
        "приёмки). Подсказки текста-в-SQL настраивались на всех вопросах реестра, правила "
        "маршрутизатора — на всех 200, поэтому реестр считается затронутым целиком. "
        "Если «без затронутых» заметно ниже «затронутых» — оценка по всем вопросам оптимистична.",
        "",
    ]

    # 5. Причины ошибок
    L += ["## 5. Почему ответы неверны (основная конфигурация)", ""]
    total_wrong = sum(diagnosis.values())
    L += [f"Всего неверных: {total_wrong}.", ""]
    L += [md_table(["Причина", "Вопросов"], [[k, str(v)] for k, v in diagnosis.most_common()]), ""]
    L += ["Разбор каждого неверного ответа — `errors.md`.", ""]

    # 6. Надёжность проверки
    chk = result["checker"]
    L += [
        "## 6. Надёжность проверки",
        "",
        "- Правильность ответа проверяется по всем ключевым фактам эталона. Метрики цитат — только "
        "по фактам, которые дословно лежат в тексте эталонных пунктов (число прописью и номер "
        f"договора из вводной части там не найти). Из {chk['of']} вопросов по документам все "
        f"ключевые факты проверяемы у {chk['all_keys_checkable']}, часть — у "
        f"{chk['some_keys_checkable']}, ни одного — у {chk['none_checkable']} (такие вопросы в "
        "метриках цитат не участвуют).",
        "- Правильность по ключевым фактам не отличает ответ, где все факты названы, но вывод "
        "перепутан или добавлено лишнее расхождение (в сравнении двух договоров): такой ответ "
        "засчитывается, если назван договор-победитель. Оценка в этой категории — верхняя.",
        "- Старая редакция условия рядом с новой не наказывается (ключевые факты — новые "
        "значения): ответ «было 30, стало 20» верен.",
        "",
    ]
    if settings.audit_notes:
        L += ["Ручной аудит проверки (прогон основной конфигурации):", ""]
        L += [f"- {note}" for note in settings.audit_notes]
        L += [""]
    return "\n".join(L) + "\n"


def _fmt(d: Mapping[str, float | None], as_pct: bool) -> str:
    v, lo, hi = d["value"], d["low"], d["high"]
    if v is None or lo is None or hi is None:
        return "—"
    if as_pct:
        return f"{_num(100 * v, 1)}% [{_num(100 * lo, 1)}; {_num(100 * hi, 1)}]"
    return f"{_num(v, 3)} [{_num(lo, 3)}; {_num(hi, 3)}]"
