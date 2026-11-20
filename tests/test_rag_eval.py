"""Оценка RAG (P20): метрики поиска, правильность ответов, цитаты, переключатели абляций."""

import math
from pathlib import Path

import pytest

from docai.rag.ablation import AblationSpec, Cache, config_for, fingerprint, question_hash
from docai.rag.config import RagConfig
from docai.rag.evaluate import Span
from docai.rag.facts import keys_found
from docai.rag.generate import NOT_FOUND
from docai.rag.index import build_index
from docai.rag.pipeline import RagPipeline
from docai.rag.scoring import (
    ANSWER_METRICS,
    RETRIEVAL_METRICS,
    answer_row,
    checkable_keys,
    compare,
    estimate,
    mcnemar_on,
    ndcg_at,
    prepare,
    retrieval_row,
)
from test_rag_pipeline import FakeEmbedder, FakeReranker, ScriptedLlm, write_mini_corpus

A = Span("K-1", 0, 100)
B = Span("K-1", 100, 200)
C = Span("K-2", 0, 100)
X = Span("K-9", 0, 100)


# --- Поиск ----------------------------------------------------------------------------------


def test_ndcg_perfect_and_order() -> None:
    assert ndcg_at([A, B], [A, B], 10) == pytest.approx(1.0)
    # Один из двух эталонных пунктов на первом месте, второй — на третьем.
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at([A, X, B], [A, B], 10) == pytest.approx(expected)
    assert ndcg_at([X, X, X], [A], 10) == 0.0
    assert ndcg_at([], [A], 10) == 0.0


def test_ndcg_counts_each_gold_once() -> None:
    # Два найденных пункта задевают один эталонный фрагмент: второй не даёт прироста.
    inside_a = Span("K-1", 10, 20)
    assert ndcg_at([A, inside_a], [A], 10) == pytest.approx(1.0)
    assert ndcg_at([A, inside_a, B], [A, B], 10) < 1.0


def test_ndcg_cutoff_and_ideal_limited_by_k() -> None:
    # Эталонных пунктов больше k: идеал — k единиц подряд.
    gold = [Span("K-1", i * 10, i * 10 + 5) for i in range(5)]
    assert ndcg_at(gold[:2], gold, 2) == pytest.approx(1.0)


def test_retrieval_row_recall_mrr() -> None:
    row = retrieval_row([X, A, X, X, X, B], [A, B, C], ks=(5, 10))
    assert row["mrr"] == pytest.approx(0.5)
    assert row["recall@5"] == pytest.approx(1 / 3)
    assert row["recall@10"] == pytest.approx(2 / 3)
    assert row["hit@5"] == 1.0
    missed = retrieval_row([X], [A], ks=(5, 10))
    assert missed["mrr"] == 0.0 and missed["recall@10"] == 0.0 and missed["ndcg@10"] == 0.0


# --- Ответы ---------------------------------------------------------------------------------


QUESTION = {
    "id": "Q1",
    "category": "multi_condition",
    "route": "rag",
    "question": "?",
    "keys": ["10", "0,1%|0,1 процент"],
    "sources": [{"fragment_id": "K-1#2.2", "doc_id": "K-1", "quote": "x"}],
    "doc_ids": ["K-1"],
}
SPANS = {"K-1#2.2": Span("K-1", 0, 100)}
GOLD = ["Срок оплаты 10 дней. Неустойка 0,1% в день."]


def cite(doc: str, start: int, end: int, quote: str, n: int = 1) -> dict[str, object]:
    return {"n": n, "doc_id": doc, "start": start, "end": end, "quote": quote, "fragment_id": "x"}


def test_answer_row_correct_and_supported() -> None:
    result = {
        "found": True,
        "answer": "Оплата в течение 10 дней, неустойка 0,1% [1].",
        "citations": [cite("K-1", 0, 100, "Срок оплаты 10 дней. Неустойка 0,1% в день.")],
    }
    row = answer_row(QUESTION, result, SPANS, GOLD)
    assert row["correct_a"] == 1.0 and row["supported"] == 1.0 and row["verified"] == 1.0
    assert row["span_ok"] == 1.0 and row["cited"] == 1.0 and row["cited_fact"] == 1.0


def test_answer_row_right_answer_wrong_citation() -> None:
    result = {
        "found": True,
        "answer": "Оплата в течение 10 дней, неустойка 0,1%.",
        "citations": [cite("K-1", 500, 600, "Адрес склада: г. Москва.")],
    }
    row = answer_row(QUESTION, result, SPANS, GOLD)
    assert row["correct_a"] == 1.0
    assert row["supported"] == 0.0 and row["verified"] == 0.0
    assert row["cited_fact"] == 0.0 and row["span_ok"] == 0.0


def test_prepare_dates_and_spelled_numbers() -> None:
    assert prepare("до 31 декабря 2025 года") == "до 31.12.2025"
    assert prepare("с «5» марта 2024 г. по 1 апреля 2024") == "с 05.03.2024 по 01.04.2024"
    assert (
        prepare("удержание 3 (три) процента, 30 (тридцать) дней") == "удержание 3 процента, 30 дней"
    )
    keys = ["31.12.2025", "3%|3 процент"]
    answer = "Действует до 31 декабря 2025 г., удержание 3 (три) процента."
    assert not keys_found(answer, keys) and keys_found(prepare(answer), keys)


def test_checkable_keys_drop_facts_absent_from_gold_text() -> None:
    q = {"keys": ["10", "2025/125", "0,1%|0,1 процент"]}
    assert checkable_keys(q, ["10 дней, 0,1% в день"]) == ["10", "0,1%|0,1 процент"]
    assert checkable_keys(q, ["десять дней"]) == []
    # Без проверяемых фактов метрики цитат у вопроса не считаются, а правильность — да.
    row = answer_row(
        QUESTION, {"found": True, "answer": "10 дней, 0,1%", "citations": []}, SPANS, []
    )
    assert row["correct_a"] == 1.0 and "checkable" not in row


def test_answer_row_refusals_and_wrong_answers() -> None:
    refused = answer_row(
        QUESTION, {"found": False, "answer": NOT_FOUND, "citations": []}, SPANS, GOLD
    )
    assert refused["false_refusal"] == 1.0 and refused["correct_a"] == 0.0
    assert refused["has_cites"] == 0.0
    wrong = answer_row(
        QUESTION,
        {"found": True, "answer": "Оплата в течение 30 дней.", "citations": []},
        SPANS,
        GOLD,
    )
    assert wrong["false_refusal"] == 0.0 and wrong["correct_a"] == 0.0


def test_answer_row_no_answer_questions() -> None:
    q = {"id": "Q2", "category": "no_answer", "route": "none", "question": "?", "keys": []}
    early = answer_row(
        q,
        {"found": False, "citations": [], "reason": "в документах нет контрагента ООО «Икс»"},
        {},
    )
    assert (
        early["no_answer"] == 1.0 and early["refusal_ok"] == 1.0 and early["refused_early"] == 1.0
    )
    late = answer_row(q, {"found": False, "citations": [], "reason": "в найденных ..."}, {})
    assert late["refusal_ok"] == 1.0 and late["refused_early"] == 0.0
    answered = answer_row(
        q, {"found": True, "answer": "30 дней", "citations": [cite("K-1", 0, 1, "x")]}, {}
    )
    assert answered["refusal_ok"] == 0.0 and "answerable" not in answered


def test_answer_row_registry_keys() -> None:
    q = {
        "id": "Q3", "category": "registry_aggregate", "route": "table", "question": "?",
        "expected_value": "2026-10-21", "keys": ["21.10.2026", "557/Ю"],
    }  # fmt: skip
    sql = {"route": "sql", "found": True, "sql_rows": [["557/Ю", "2026-10-21"]]}
    text = "Последний документ — № 557/Ю от 21 октября 2026 года."
    ok = answer_row(q, {**sql, "answer": text}, {})
    assert ok["correct_a"] == 1.0 and ok["sql_ok"] == 1.0
    # Номер документа в ответе пропущен — ответ неполный.
    partial = answer_row(q, {**sql, "answer": "Получен 21 октября 2026 года."}, {})
    assert partial["correct_a"] == 0.0
    via_rag = answer_row(q, {"route": "rag", "found": True, "answer": text, "citations": []}, {})
    assert via_rag["correct_a"] == 1.0 and via_rag["sql_ok"] == 0.0
    wrong = answer_row(q, {"route": "rag", "found": False, "answer": NOT_FOUND}, {})
    assert wrong["correct_a"] == 0.0 and wrong["false_refusal"] == 1.0


# --- Статистика -----------------------------------------------------------------------------


def test_estimate_interval_and_paired_comparison() -> None:
    rows_a = [retrieval_row([X], [A]) for _ in range(30)] + [
        retrieval_row([A], [A]) for _ in range(10)
    ]
    rows_b = [retrieval_row([A], [A]) for _ in range(40)]
    est = estimate(rows_a, RETRIEVAL_METRICS, n_resamples=300, seed=1)
    assert est["recall@5"].value == pytest.approx(0.25)
    assert est["recall@5"].low <= 0.25 <= est["recall@5"].high
    again = estimate(rows_a, RETRIEVAL_METRICS, n_resamples=300, seed=1)
    assert est["mrr"].low == again["mrr"].low  # seed фиксирует интервал
    diff = compare(rows_a, rows_b, RETRIEVAL_METRICS, n_resamples=300, seed=1)["recall@5"]
    assert diff.diff == pytest.approx(0.75) and diff.low > 0 and diff.p_value < 0.05


def test_answer_metrics_and_mcnemar() -> None:
    right = {"n": 1.0, "answerable": 1.0, "correct_a": 1.0}
    wrong = {"n": 1.0, "answerable": 1.0, "correct_a": 0.0}
    refused = {"n": 1.0, "no_answer": 1.0, "refusal_ok": 1.0}
    rows = [right, right, wrong, refused]
    est = estimate(rows, ANSWER_METRICS, n_resamples=100, seed=1)
    assert est["correct"].value == pytest.approx(2 / 3)
    assert est["refusal_ok"].value == 1.0
    other = [right, wrong, wrong, refused]
    mc = mcnemar_on(rows, other, "correct_a", only="answerable")
    assert (mc.b, mc.c) == (1, 0)


@pytest.fixture()
def mini(tmp_path: Path) -> RagConfig:
    config = write_mini_corpus(tmp_path)
    build_index(config, FakeEmbedder)
    return config


# --- Переключатели абляции ------------------------------------------------------------------


def test_config_for_ablation() -> None:
    base = RagConfig()
    spec = AblationSpec(name="b", label="b", mode="bm25", reranker=False, router=False, run=["rag"])
    config = config_for(base, spec)
    assert config.retrieval.mode == "bm25" and config.reranker.kind == "none"
    assert config.use_router is False
    assert base.retrieval.mode == "hybrid" and base.use_router  # исходный конфиг цел


def test_cache_respects_fingerprint_and_question(tmp_path: Path, mini: RagConfig) -> None:
    q = {"id": "Q1", "question": "Вопрос?"}
    path = tmp_path / "c.jsonl"
    Cache(path, "fp1").put(q, {"answer": "a"})
    assert Cache(path, "fp1").get(q)["answer"] == "a"  # type: ignore[index]
    assert Cache(path, "fp2").get(q) is None  # другой отпечаток — запись не используется
    assert Cache(path, "fp1").get({"id": "Q1", "question": "Другой?"}) is None
    assert question_hash(q) != question_hash({"id": "Q1", "question": "Другой?"})
    spec = AblationSpec(name="x", label="x", run=["rag"], reranker=True)
    other = AblationSpec(name="y", label="y", mode="bm25", run=["rag"])
    assert fingerprint(config_for(mini, spec)) != fingerprint(config_for(mini, other))


def make(config: RagConfig, spec: AblationSpec, llm: ScriptedLlm):
    return RagPipeline(
        config_for(config, spec),
        embedder=FakeEmbedder(),
        reranker=FakeReranker() if spec.reranker else None,
        load_reranker=False,
        llm=llm,
    )


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_retrieval_modes_use_only_their_search(mini: RagConfig, mode: str) -> None:
    spec = AblationSpec(name=mode, label=mode, mode=mode, run=["rag"])  # type: ignore[arg-type]
    result = make(mini, spec, ScriptedLlm()).ask("Какая арендная плата у Орлова?", debug=True)
    assert result.found and result.reranker is None
    assert result.candidates
    for c in result.candidates:
        if mode == "bm25":
            assert c["bm25_rank"] is not None and c["vector_rank"] is None
        elif mode == "vector":
            assert c["vector_rank"] is not None and c["bm25_rank"] is None
    if mode == "hybrid":
        assert any(c["bm25_rank"] is not None for c in result.candidates)
        assert any(c["vector_rank"] is not None for c in result.candidates)


def test_router_switch_sends_registry_question_to_rag(mini: RagConfig) -> None:
    question = "Сколько УПД получено от ООО «Ракурс-Комплект» за 2025 год?"
    on = AblationSpec(name="on", label="on", run=["table"], reranker=True)
    off = AblationSpec(name="off", label="off", run=["table"], reranker=True, router=False)
    sql = "SELECT COUNT(*) FROM registry WHERE doc_type = 'upd'"
    assert make(mini, on, ScriptedLlm(sql=sql)).ask(question).route == "sql"
    assert make(mini, off, ScriptedLlm(sql=sql)).ask(question).route == "rag"
