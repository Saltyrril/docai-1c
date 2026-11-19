"""Тесты корпуса и вопросов для RAG (P18)."""

import json
import re
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docai.cli import app
from docai.rag import facts as F
from docai.rag.cli import stratified_sample
from docai.rag.corpus import Corpus, build_corpus
from docai.rag.facts import keys_found, normalize_answer
from docai.rag.manual import check_manual
from docai.rag.questions import QUOTAS, Question, generate_questions, run_sql
from docai.rag.terms import TODAY, Kind, pick_our_company
from docai.rag.words import (
    days_gen,
    days_nom,
    number_words,
    number_words_genitive,
    plural,
    rubles_text,
)

SEED = 42


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return build_corpus(300, 900, SEED)


@pytest.fixture(scope="module")
def questions(corpus: Corpus) -> list[Question]:
    return generate_questions(corpus, SEED)


# --- Числа и слова -----------------------------------------------------------------------


def test_number_words() -> None:
    assert number_words(1_250_000) == "один миллион двести пятьдесят тысяч"
    assert number_words(21_001) == "двадцать одна тысяча один"
    assert number_words(2_000, feminine=True) == "две тысячи"
    assert number_words(611_900) == "шестьсот одиннадцать тысяч девятьсот"
    assert number_words_genitive(45) == "сорока пяти"
    assert number_words_genitive(14) == "четырнадцати"
    assert number_words_genitive(90) == "девяноста"


def test_plural_and_days() -> None:
    assert [plural(n, ("день", "дня", "дней")) for n in (1, 2, 5, 11, 21, 104)] == [
        "день", "дня", "дней", "дней", "день", "дня",
    ]  # fmt: skip
    assert days_nom(3, "working") == "3 рабочих дня"
    assert days_nom(30) == "30 календарных дней"
    assert days_nom(1, "banking") == "1 банковский день"
    assert days_gen(3, "working") == "3 рабочих дней"
    assert days_gen(21, "working") == "21 рабочего дня"
    assert days_gen(10, "working", words=True) == "10 (десяти) рабочих дней"


def test_rubles_text() -> None:
    assert rubles_text(Decimal("85000"), 0) == "85 000,00 руб."
    assert rubles_text(Decimal("85000"), 1) == ("85 000 (восемьдесят пять тысяч) рублей 00 копеек")
    assert rubles_text(Decimal("1001"), 2) == "1 001 рубль"


def test_normalize_and_keys() -> None:
    assert normalize_answer("85 000,00 руб.") == "85000,00 руб."
    assert normalize_answer("0.1 %") == "0,1%"
    assert keys_found("Неустойка 0,1 % в день", ["0,1%|0,1 процент"])
    assert keys_found("не более 10 процентов", ["10%|10 процент"])
    assert not keys_found("30 дней", ["45"])


# --- Корпус ------------------------------------------------------------------------------


def test_corpus_composition(corpus: Corpus) -> None:
    kinds = Counter(b.base.kind for b in corpus.bundles)
    assert kinds == {Kind.SUPPLY: 110, Kind.SERVICES: 90, Kind.LEASE: 45, Kind.WORKS: 55}
    assert 5 <= len(corpus.regulations) <= 10
    with_amendments = [b for b in corpus.bundles if b.amendments]
    assert len(with_amendments) >= 60
    # У части контрагентов несколько договоров — иначе ссылка «договор с X» была бы однозначной.
    per_cp = Counter(b.base.them.inn for b in corpus.bundles)
    assert sum(1 for k in per_cp.values() if k > 1) >= 40
    assert all(b.base.them.inn != corpus.our_inn for b in corpus.bundles)


def test_fragments_match_text(corpus: Corpus) -> None:
    ids: set[str] = set()
    for doc in corpus.documents:
        for f in doc.fragments:
            assert doc.text[f.start : f.end] == f.text
            assert f.fragment_id not in ids
            ids.add(f.fragment_id)
        assert "г.." not in doc.text and "руб.." not in doc.text


def test_each_fact_has_one_clause(corpus: Corpus) -> None:
    for b in corpus.bundles:
        for fact in b.base_facts():
            if fact == F.SUBJECT:
                continue
            holders = [f for f in b.doc.fragments if fact in f.facts]
            assert len(holders) == 1, (b.contract_id, fact)


def test_amendments_change_conditions(corpus: Corpus) -> None:
    for b in corpus.bundles:
        previous = b.base.date
        for a in b.amendments:
            assert a.signed > previous and a.effective <= TODAY
            previous = a.signed
            for fact in a.changed:
                frag = a.doc.fact_fragment(fact)
                assert frag is not None
                # Новое значение есть в тексте соглашения (номер счёта, срок, сумма...).
                for key in F.contract_facts(a.after)[fact].keys:
                    digits = [k for k in key.split("|") if re.fullmatch(r"[\d.,]+", k)]
                    if digits and not re.fullmatch(r"\d\d\.\d\d\.\d{4}", digits[0]):
                        assert any(d in normalize_answer(frag.text) for d in digits), (
                            a.doc.doc_id, fact, key)  # fmt: skip


def test_phrasing_varies(corpus: Corpus) -> None:
    """Одно и то же условие сформулировано по-разному (не одна шаблонная фраза)."""
    for fact in (F.PAYMENT, F.PENALTY_US, F.VALIDITY, F.ACCEPTANCE, F.CLAIMS, F.COURT):
        openings = Counter()
        for b in corpus.bundles:
            frag = b.doc.fact_fragment(fact)
            if frag is not None:
                body = frag.text.split(". ", 1)[1]
                openings[" ".join(body.split()[:3])] += 1
        assert len(openings) >= 3, (fact, openings)
        assert openings.most_common(1)[0][1] / sum(openings.values()) < 0.6, fact


def test_registry(corpus: Corpus) -> None:
    our = pick_our_company()
    contracts = {b.contract_id: b.effective for b in corpus.bundles}
    assert len(corpus.registry) == 900
    for row in corpus.registry:
        c = contracts[row.contract_id]
        assert row.buyer_inn == our.inn and row.seller_inn == c.them.inn
        assert c.is_active(date.fromisoformat(row.date))
        assert date.fromisoformat(row.date) <= TODAY
        assert row.amount_with_vat == row.amount_without_vat + row.vat_amount
    assert len({r.seller_inn for r in corpus.registry}) >= 50


def test_corpus_is_deterministic(corpus: Corpus) -> None:
    again = build_corpus(300, 900, SEED)
    assert [d.text for d in again.documents] == [d.text for d in corpus.documents]
    assert [r.as_csv() for r in again.registry] == [r.as_csv() for r in corpus.registry]


# --- Вопросы -----------------------------------------------------------------------------


def test_question_quotas(questions: list[Question]) -> None:
    assert len(questions) == 200
    counts = Counter(q.category for q in questions)
    assert counts == {k: sum(v) for k, v in QUOTAS.items()}
    assert len({q.question for q in questions}) == 200
    assert [q.qid for q in questions] == [f"Q{i:03d}" for i in range(1, 201)]


def test_sources_exist_and_quote_fragment(corpus: Corpus, questions: list[Question]) -> None:
    index = corpus.fragment_index()
    for q in questions:
        for s in q.sources + q.superseded:
            frag = index[s["fragment_id"]]
            assert s["quote"] == frag.text
            assert s["doc_id"] == frag.doc_id
        if q.category == "no_answer":
            assert not q.sources and q.route == "none" and not q.keys
        elif q.category == "registry_aggregate":
            assert q.route == "table" and q.sql and q.registry_rows
        else:
            assert q.sources and q.route == "rag"


def test_gold_answers_contain_keys(questions: list[Question]) -> None:
    for q in questions:
        assert keys_found(q.answer, q.keys), (q.qid, q.answer, q.keys)


def test_numeric_keys_are_in_sources(questions: list[Question]) -> None:
    """Числа из эталона есть в цитатах: ответ и напечатанный текст не разошлись."""
    for q in questions:
        if not q.sources:
            continue
        text = normalize_answer(" ".join(s["quote"] for s in q.sources))
        # В сравнении последний ключ — номер договора-«победителя», его в цитате нет.
        keys = q.keys[:-1] if q.template.startswith("compare:") else q.keys
        for key in keys:
            for alt in key.split("|"):
                if re.fullmatch(r"\d[\d,]*", alt):
                    assert alt in text, (q.qid, alt)


def test_amendment_questions(corpus: Corpus, questions: list[Question]) -> None:
    for q in (x for x in questions if x.category == "amendment_override"):
        assert q.superseded and q.wrong_answers
        assert all("-DS" in s["doc_id"] or s["doc_id"] == "REG-07" for s in q.sources)
        assert not keys_found(" ".join(q.wrong_answers), q.keys) or q.template.startswith("reg")
        # Дата «действует с …» из эталона подтверждена пунктом о вступлении в силу.
        assert "действует с" in q.answer
        assert any("вступает в силу" in s["quote"] for s in q.sources), q.qid


def test_simple_facts_are_not_amended(corpus: Corpus, questions: list[Question]) -> None:
    for q in questions:
        if q.category in ("simple_fact", "paraphrase", "multi_condition"):
            assert all("-DS" not in s["doc_id"] for s in q.sources)


def test_registry_sql_matches_expected(corpus: Corpus, questions: list[Question]) -> None:
    for q in (x for x in questions if x.category == "registry_aggregate"):
        assert q.sql is not None
        (value, *_), *_ = run_sql(corpus.registry, q.sql)
        if isinstance(value, float):
            assert f"{value:.2f}" == q.expected_value, q.qid
        else:
            assert str(value) == q.expected_value, q.qid
        if q.sql_detail:
            detail = run_sql(corpus.registry, q.sql_detail)
            assert sum(k for _, k in detail) == int(q.expected_value or 0), q.qid
            for name, k in detail:
                assert f"{name} — {k}" in q.answer, (q.qid, name, k)


def test_no_answer_is_really_absent(corpus: Corpus, questions: list[Question]) -> None:
    """Чего нет — того нет во всём корпусе, а не только в одном документе."""
    text = " ".join(d.text for d in corpus.documents).lower()
    for q in (x for x in questions if x.category == "no_answer"):
        if q.template == "unknown_number":
            number = re.search(r"№ (\S+) с ", q.question)
            assert number and f"№ {number.group(1)}" not in text, q.qid
        if q.template == "unknown_counterparty":
            name = re.search(r"(ООО «[^»]+»)", q.question)
            assert name and name.group(1).lower() not in text, q.qid
    for stem in ("страхов", "электронной почт", "e-mail", "фио", "телефон", "представительск"):
        assert stem not in text, stem


def test_questions_deterministic(corpus: Corpus, questions: list[Question]) -> None:
    again = generate_questions(corpus, SEED)
    assert [q.to_json() for q in again] == [q.to_json() for q in questions]


def test_stratified_sample(questions: list[Question]) -> None:
    data = [q.to_json() for q in questions]
    sample = stratified_sample(data, 20, 2026)
    assert len(sample) == 20 and len({q["id"] for q in sample}) == 20
    assert len({q["category"] for q in sample}) == 7
    assert sample == stratified_sample(data, 20, 2026)


# --- Вопросы владельца -------------------------------------------------------------------


def test_manual_template_is_valid(corpus: Corpus) -> None:
    result = check_manual(Path("configs/rag/questions_manual.yaml"), corpus)
    assert result.total == 30 and not result.problems and not result.questions


def test_manual_check_finds_fragments_and_errors(corpus: Corpus, tmp_path: Path) -> None:
    contract = corpus.bundles[0].doc
    fragment = contract.fact_fragment(F.PAYMENT)
    assert fragment is not None
    quote = fragment.text.split(". ", 1)[1][:60]
    path = tmp_path / "manual.yaml"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "M01",
                        "category": "simple_fact",
                        "question": "Срок оплаты?",
                        "answer": "…",
                        "sources": [{"doc_id": contract.doc_id, "quote": quote}],
                    },
                    {
                        "id": "M02",
                        "category": "simple_fact",
                        "question": "Ещё?",
                        "answer": "…",
                        "sources": [{"doc_id": contract.doc_id, "quote": "такой фразы нет"}],
                    },
                    {"id": "M03", "category": "какая-то", "question": "?", "answer": "…"},
                    {"id": "M04", "category": "no_answer", "question": "Нет?", "answer": "Нет."},
                    {"id": "M05", "category": "simple_fact", "question": "", "answer": ""},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = check_manual(path, corpus)
    assert result.total == 5
    assert [q["id"] for q in result.questions] == ["M01", "M02", "M04"]
    assert result.questions[0]["sources"][0]["fragment_id"] == fragment.fragment_id
    assert any("M02" in p and "не найдена" in p for p in result.problems)
    assert any("M03" in p and "категория" in p for p in result.problems)
    assert not any("M04" in p for p in result.problems)


# --- CLI ---------------------------------------------------------------------------------


def test_cli_build(tmp_path: Path) -> None:
    config = tmp_path / "rag.yaml"
    config.write_text(
        f"seed: {SEED}\nn_contracts: 300\nn_registry: 900\nout: {tmp_path.as_posix()}/rag\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["rag", "build", "--config", str(config)])
    assert result.exit_code == 0, result.output
    out = tmp_path / "rag"
    info = json.loads((out / "build_info.json").read_text(encoding="utf-8"))
    assert info["documents"]["contract"] == 300
    assert sum(info["questions"].values()) == 200
    lines = (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    first = json.loads((out / "fragments.jsonl").read_text(encoding="utf-8").splitlines()[0])
    doc_text = (out / "corpus" / "contracts" / f"{first['doc_id']}.md").read_text(encoding="utf-8")
    assert doc_text[first["start"] : first["end"]] == first["text"]
    assert (out / "registry.csv").read_text(encoding="utf-8-sig").startswith("doc_id;doc_type")
