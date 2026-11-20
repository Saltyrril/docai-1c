"""Метрики оценки RAG (P20): поиск, правильность ответов, точность цитат, «не нашёл».

Всё, что можно проверить без LLM-судьи, проверяется программой:
- поиск: recall@k, MRR, nDCG@10 — по пересечению найденного пункта с фрагментом-источником
  эталона (`Span.overlaps`, D62), а не по совпадению id;
- правильность ответа: ключевые факты эталона (`keys`: числа, даты, названия) найдены в тексте
  ответа после нормализации (`facts.keys_found`, `prepare`) — и в вопросах по документам, и в
  агрегатах по реестру;
- точность цитат: ключевой факт лежит в **тексте** процитированного пункта (а не только
  пересекаются смещения с эталонным пунктом, как в `evaluate.judge`);
- «не нашёл»: верный отказ на вопросах без ответа и ложный отказ на остальных.

Единица выборки для бутстрепа — вопрос (D45): метрика пересчитывается по суммам счётчиков
выбранных вопросов. Счётчики — одна строка на вопрос (`dict[str, float]`).
"""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from docai.eval.stats import (
    Estimate,
    McNemar,
    MetricFn,
    PairedEstimate,
    bootstrap,
    mcnemar,
    paired_bootstrap,
    ratio,
)
from docai.rag.evaluate import Span, judge, source_spans
from docai.rag.facts import keys_found

# --- Поиск ----------------------------------------------------------------------------------


def covered_at(ranked: Sequence[Span], gold: Sequence[Span], k: int) -> list[bool]:
    """Для каждого эталонного фрагмента: пересекается ли он с одним из первых k найденных."""
    top = ranked[:k]
    return [any(s.overlaps(g) for s in top) for g in gold]


def ndcg_at(ranked: Sequence[Span], gold: Sequence[Span], k: int) -> float:
    """nDCG@k с бинарной релевантностью и «кредитом» за эталонный фрагмент один раз.

    Найденный пункт приносит 1, если он пересекается с ещё не покрытым эталонным фрагментом
    (если два пункта задевают один фрагмент — вторая находка не даёт ничего, иначе nDCG
    мог бы превысить 1). Идеальный порядок — min(число эталонных, k) единиц подряд.
    """
    if not gold:
        return 0.0
    left = list(gold)
    dcg = 0.0
    for rank, span in enumerate(ranked[:k], start=1):
        hit = next((g for g in left if span.overlaps(g)), None)
        if hit is not None:
            left.remove(hit)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def retrieval_row(
    ranked: Sequence[Span], gold: Sequence[Span], ks: Sequence[int] = (5, 10)
) -> dict[str, float]:
    """Счётчики одного вопроса: recall@k, hit@k, MRR, nDCG@10 (по одному вопросу = один вклад)."""
    first = next((i for i, s in enumerate(ranked, 1) if any(s.overlaps(g) for g in gold)), None)
    row: dict[str, float] = {"n": 1.0, "mrr": 1.0 / first if first else 0.0}
    for k in ks:
        covered = covered_at(ranked, gold, k)
        row[f"recall@{k}"] = sum(covered) / len(gold) if gold else 0.0
        row[f"hit@{k}"] = 1.0 if any(covered) else 0.0
    row["ndcg@10"] = ndcg_at(ranked, gold, 10)
    return row


RETRIEVAL_METRICS: dict[str, MetricFn] = {
    "recall@5": ratio("recall@5", "n"),
    "recall@10": ratio("recall@10", "n"),
    "mrr": ratio("mrr", "n"),
    "ndcg@10": ratio("ndcg@10", "n"),
    "hit@5": ratio("hit@5", "n"),
}

# --- Ответы ---------------------------------------------------------------------------------

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}  # fmt: skip
_WORD_DATE = re.compile(
    r"[«\"]?(\d{1,2})[»\"]?\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})(?:\s*(?:года|г\.?))?",
    re.IGNORECASE,
)
# «3 (три) процента», «30 (тридцать) дней»: слово в скобках после числа — повтор числа.
_SPELLED = re.compile(r"(?<=\d)\s*\([а-яёА-ЯЁ\s\-]+\)")


def prepare(text: str) -> str:
    """Привести текст к виду, в котором ищутся ключевые факты (даты и числа прописью).

    «31 декабря 2025 г.» → «31.12.2025»; «3 (три) процента» → «3 процента». Ключевые факты
    эталона записаны цифрами (`31.12.2025`, `3%`), а ответ и текст договора вправе писать
    иначе — без этого верный ответ засчитывался бы неверным.
    """
    text = _SPELLED.sub("", text)

    def date(m: re.Match[str]) -> str:
        return f"{int(m.group(1)):02d}.{_MONTHS[m.group(2).lower()]:02d}.{m.group(3)}"

    return _WORD_DATE.sub(date, text)


def _bool(value: Any) -> float:
    return 1.0 if value else 0.0


def checkable_keys(question: Mapping[str, Any], gold_texts: Sequence[str]) -> list[str]:
    """Ключевые факты, которые можно проверить по тексту цитат.

    Факт проверяем, если он дословно лежит в тексте эталонных пунктов вопроса: число, записанное
    в документе прописью («пять сотых процента»), и номер договора из вводной части (он не в
    пункте с условием) в тексте цитаты искать бессмысленно — такие факты отбрасываются. Ответ
    они по-прежнему проверяют (`correct_a`).
    """
    union = prepare("\n".join(gold_texts))
    return [k for k in map(str, question.get("keys") or []) if keys_found(union, [k])]


def answer_row(
    question: Mapping[str, Any],
    result: Mapping[str, Any],
    spans: Mapping[str, Span],
    gold_texts: Sequence[str] = (),
) -> dict[str, float]:
    """Счётчики одного вопроса по ответу системы (`AskResult` в виде словаря).

    Все значения — 0/1 или число цитат; знаменатель метрики — соответствующий счётчик:
      answerable      — у вопроса есть ответ в корпусе (rag и table);
      correct_a       — (answerable) ответ правильный: ключевые факты эталона есть в тексте
                        (rag и реестр);
      false_refusal   — (answerable) система ответила «не нашёл»;
      no_answer       — вопрос без ответа в корпусе;
      refusal_ok      — (no_answer) верный отказ: «не нашёл» и цитат нет;
      refused_early   — (no_answer) отказ без модели: контрагента или договора из вопроса
                        нет в корпусе (`reason` конвейера);
      rag_answerable  — answerable с маршрутом rag: к ним относятся метрики цитат;
      span_ok         — цитаты правильные по P19 (`evaluate.judge`): пересекаются с эталонным
                        пунктом, все из документов вопроса;
      checkable       — (rag) у вопроса есть проверяемые по тексту ключевые факты
                        (`checkable_keys`); метрики цитат считаются только по таким вопросам;
      verified        — (checkable) ответ правильный и проверяемые факты лежат в цитатах;
      has_cites       — (checkable) ответ найден и у него есть цитаты (знаменатель supported);
      supported       — проверяемые факты лежат в текстах процитированных пунктов;
      cited, cited_fact — число цитат и тех из них, в тексте которых есть проверяемый факт;
      sql_ok          — (table) эталонное число есть в строках результата запроса.
    """
    route = question["route"]
    found = bool(result.get("found"))
    citations = result.get("citations") or []
    answer = str(result.get("answer") or "")
    reason = str(result.get("reason") or "")
    row: dict[str, float] = {"n": 1.0}
    if route == "none":
        refused = not found and not citations
        row["no_answer"] = 1.0
        row["refusal_ok"] = _bool(refused)
        early = "нет контрагента" in reason or "нет договора" in reason
        row["refused_early"] = _bool(refused and early)
        return row
    row["answerable"] = 1.0
    row["false_refusal"] = _bool(not found)
    verdict = judge(dict(question), dict(result), dict(spans))
    keys = [str(k) for k in question.get("keys") or []]
    correct = found and keys_found(prepare(answer), keys)
    if route == "table":
        row["correct_a"] = _bool(correct)
        row["sql_ok"] = _bool(verdict["citations_ok"])
        return row
    row["rag_answerable"] = 1.0
    row["correct_a"] = _bool(correct)
    row["span_ok"] = _bool(verdict["citations_ok"])
    facts = checkable_keys(question, gold_texts)
    if not facts:
        return row
    texts = [str(c.get("quote") or "") for c in citations] if found else []
    supported = bool(texts) and keys_found(prepare("\n".join(texts)), facts)
    row["checkable"] = 1.0
    row["has_cites"] = _bool(bool(texts))
    row["supported"] = _bool(supported)
    row["verified"] = _bool(correct and supported)
    row["cited"] = float(len(texts))
    row["cited_fact"] = float(
        sum(1 for t in texts if any(keys_found(prepare(t), [key]) for key in facts))
    )
    return row


ANSWER_METRICS: dict[str, MetricFn] = {
    "correct": ratio("correct_a", "answerable"),
    "false_refusal": ratio("false_refusal", "answerable"),
    "refusal_ok": ratio("refusal_ok", "no_answer"),
    "cite_precision": ratio("cited_fact", "cited"),
    "supported": ratio("supported", "has_cites"),
    "verified": ratio("verified", "checkable"),
    "span_ok": ratio("span_ok", "rag_answerable"),
}


# --- Статистика -----------------------------------------------------------------------------


def to_matrix(rows: Sequence[Mapping[str, float]], keys: Sequence[str]) -> np.ndarray:
    return np.array([[float(r.get(k, 0.0)) for k in keys] for r in rows], dtype=float).reshape(
        len(rows), len(keys)
    )


def all_keys(rows: Sequence[Mapping[str, float]]) -> list[str]:
    return sorted({k for r in rows for k in r})


def estimate(
    rows: Sequence[Mapping[str, float]],
    metrics: Mapping[str, MetricFn],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Estimate]:
    """Точечная оценка и 95% интервал бутстрепа по вопросам."""
    keys = all_keys(rows)
    return bootstrap(to_matrix(rows, keys), keys, metrics, n_resamples=n_resamples, seed=seed)


def compare(
    rows_a: Sequence[Mapping[str, float]],
    rows_b: Sequence[Mapping[str, float]],
    metrics: Mapping[str, MetricFn],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, PairedEstimate]:
    """Парный бутстреп разности B − A на одних и тех же вопросах."""
    keys = sorted(set(all_keys(rows_a)) | set(all_keys(rows_b)))
    return paired_bootstrap(
        to_matrix(rows_a, keys),
        to_matrix(rows_b, keys),
        keys,
        metrics,
        n_resamples=n_resamples,
        seed=seed,
    )


def mcnemar_on(
    rows_a: Sequence[Mapping[str, float]],
    rows_b: Sequence[Mapping[str, float]],
    key: str,
    only: str | None = None,
) -> McNemar:
    """Макнемар для бинарного исхода вопроса (например, `correct_a`) в двух конфигурациях.

    only — счётчик, который должен быть ненулевым, чтобы вопрос участвовал (`answerable`).
    """
    pairs = [(a, b) for a, b in zip(rows_a, rows_b, strict=True) if only is None or a.get(only)]
    return mcnemar(
        [bool(a.get(key, 0.0)) for a, _ in pairs], [bool(b.get(key, 0.0)) for _, b in pairs]
    )


def spans_of(question: Mapping[str, Any], spans: Mapping[str, Span]) -> list[Span]:
    return source_spans(dict(question), dict(spans))
