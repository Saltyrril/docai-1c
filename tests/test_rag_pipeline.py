"""Тесты RAG-конвейера (P19): нарезка, BM25, разбор вопроса, маршрутизатор, SQL, индекс
с продолжением, ответ с цитатами, endpoint. Вместо моделей — заглушки: torch, веса
Hugging Face и Ollama не нужны. Chroma — настоящая, во временной папке.
"""

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docai.extract.ollama import OllamaError
from docai.rag.bm25 import Bm25Index, rrf
from docai.rag.chunking import chunk_corpus, chunk_document, parse_date
from docai.rag.config import RagConfig, load_rag_config
from docai.rag.corpus import Corpus, build_corpus, write_corpus
from docai.rag.evaluate import Span, judge, ranking_metrics, stratified, value_found
from docai.rag.generate import (
    NOT_FOUND,
    attribute,
    build_prompt,
    parse_answer,
    split_sentences,
)
from docai.rag.index import (
    IndexMismatchError,
    IndexNotReadyError,
    IndexPaths,
    build_index,
)
from docai.rag.lemmas import lemmas
from docai.rag.pipeline import RagPipeline
from docai.rag.query import NameMatcher, analyze, find_period
from docai.rag.questions import generate_questions
from docai.rag.router import route_question
from docai.rag.search import Hit, RagIndex
from docai.rag.text_sql import (
    SqlRejectedError,
    SqlResult,
    aggregate_hint,
    build_hints,
    doc_types,
    hint_problems,
    numbers_supported,
    run_select,
    validate_sql,
)
from docai.service.app import create_app
from docai.service.settings import ServiceSettings

TODAY = date(2026, 11, 19)
OURS = ["АО «Тайга-Строй»"]
OUR_INNS = ["7704037472"]

CONTRACT_1 = """# ДОГОВОР ПОСТАВКИ ТОВАРА № П-668/23

г. Краснодар, 31 января 2023 года

АО «Тайга-Строй» (далее — «Покупатель») в лице генерального директора, с одной стороны, и ООО «Ракурс-Комплект», именуемое в дальнейшем «Поставщик», с другой стороны, заключили настоящий Договор о нижеследующем:

## 1. Предмет Договора

1.1. Поставщик обязуется поставлять Покупателю строительные материалы, а Покупатель обязуется принимать и оплачивать Товар.

## 2. Цена и порядок расчётов

2.1. Цена Договора составляет 3 050 000 рублей с учётом НДС.

2.2. Покупатель оплачивает Товар в течение 30 календарных дней с даты поставки.

## 3. Санкции

3.1. При задержке оплаты Покупатель выплачивает неустойку 0,05 процента от неоплаченной суммы за каждый день просрочки.

## 4. Реквизиты и подписи Сторон

**Поставщик:** ООО «Ракурс-Комплект»
ИНН 1841477442, КПП 184144043
р/с 40702810107390690330 в АО «Таёжный Эксперт Банк»

**Покупатель:** АО «Тайга-Строй»
ИНН 7704037472, КПП 770443110
"""

AMENDMENT_1 = """# ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1

к договору поставки № П-668/23 от 31.01.2023

г. Краснодар, 11.09.2024

АО «Тайга-Строй» (далее — «Покупатель») и ООО «Ракурс-Комплект» (далее — «Поставщик») заключили настоящее дополнительное соглашение к Договору № П-668/23 от 31.01.2023 о нижеследующем:

1. Срок оплаты, установленный пунктом 2.2 Договора, сократить с 30 до 20 календарных дней.

2. Соглашение вступает в силу с «1» октября 2024 г.

3. Остальные условия Договора, не затронутые настоящим Соглашением, остаются неизменными.
"""

CONTRACT_2 = """# ДОГОВОР АРЕНДЫ № А-114/24

г. Самара, «22» февраля 2024 г.

Индивидуальный предприниматель Орлов Мефодий Жоресович (далее — «Арендодатель») и АО «Тайга-Строй» (далее — «Арендатор») заключили настоящий Договор:

## 1. Арендная плата

1.1. Арендная плата составляет 120 000,00 руб. в месяц и вносится не позднее 10-го числа текущего месяца.

1.2. Коммунальные услуги в арендную плату не входят и оплачиваются отдельно.

## 2. Реквизиты и подписи Сторон

**Арендодатель:** ИП Орлов Мефодий Жоресович
ИНН 542787372908

**Арендатор:** АО «Тайга-Строй»
ИНН 7704037472, КПП 770443110
"""

REGULATION = """# ПОРЯДОК СОГЛАСОВАНИЯ И ПРОВЕДЕНИЯ ОПЛАТ ПОСТАВЩИКАМ

Утверждён приказом генерального директора № 15 от 20.01.2025

## 1. Согласование

1.1. Заявки на сумму до 300 000 рублей согласуют руководитель подразделения и главный бухгалтер.

1.2. Платёжные дни — вторник и четверг.
"""

REGISTRY = """﻿doc_id;doc_type;doc_type_name;number;date;seller_name;seller_inn;buyer_name;buyer_inn;contract_id;contract_number;contract_date;amount_without_vat;vat_amount;amount_with_vat;vat_rates;lines;p3_source
R-1;upd;УПД;101;2025-03-01;ООО «Ракурс-Комплект»;1841477442;АО «Тайга-Строй»;7704037472;K-0001;П-668/23;2023-01-31;1000.00;200.00;1200.00;20%;1;x
R-2;upd;УПД;102;2025-05-10;ООО «Ракурс-Комплект»;1841477442;АО «Тайга-Строй»;7704037472;K-0001;П-668/23;2023-01-31;2000.00;400.00;2400.00;20%;2;x
R-3;bill;Счёт на оплату;5;2026-02-01;ООО «Ракурс-Комплект»;1841477442;АО «Тайга-Строй»;7704037472;K-0001;П-668/23;2023-01-31;500.00;110.00;610.00;22%;1;x
"""


def write_mini_corpus(root: Path) -> RagConfig:
    contracts = root / "corpus" / "contracts"
    regulations = root / "corpus" / "regulations"
    contracts.mkdir(parents=True)
    regulations.mkdir(parents=True)
    for name, text in [
        ("K-0001.md", CONTRACT_1),
        ("K-0001-DS1.md", AMENDMENT_1),
        ("K-0002.md", CONTRACT_2),
    ]:
        (contracts / name).write_text(text, encoding="utf-8", newline="\n")
    (regulations / "REG-01.md").write_text(REGULATION, encoding="utf-8", newline="\n")
    (root / "registry.csv").write_text(REGISTRY, encoding="utf-8", newline="\n")
    return RagConfig(
        corpus_dir=root / "corpus",
        registry_csv=root / "registry.csv",
        index_dir=root / "index",
    )


class FakeEmbedder:
    """Мешок лемм, хэшированный в 64 измерения: похожие тексты — близкие векторы."""

    name = "fake"

    def __init__(self) -> None:
        self.documents = 0

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * 64
        for lemma in lemmas(text):
            v[int(hashlib.md5(lemma.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeReranker:
    name = "fake-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(lemmas(query))
        return [len(q & set(lemmas(p))) / (len(q) or 1) for p in passages]


class ScriptedLlm:
    """LLM-заглушка: отвечает по первому фрагменту, где есть слово-признак."""

    def __init__(self, sql: str | None = None) -> None:
        self.prompts: list[str] = []
        self.sql = sql

    def __call__(self, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], None]:
        self.prompts.append(prompt)
        if "sql" in schema["properties"]:
            return {"sql": self.sql or ""}, None
        if "found" not in schema["properties"]:
            return {"answer": "Получено 2 документа."}, None
        blocks = re.findall(r"^\[(\d+)\] (.+)\n(.+)$", prompt, flags=re.MULTILINE)
        for n, heading, text in blocks:
            question = prompt.split("Вопрос:")[-1].lower()
            if "соглашение" in heading.lower() and "срок оплаты" in question:
                return {"found": True, "answer": f"Срок оплаты — 20 календарных дней [{n}]."}, None
            if "арендная плата" in text.lower() and "аренд" in question:
                return {"found": True, "answer": f"120 000,00 руб. в месяц [{n}]."}, None
        return {"found": False, "answer": NOT_FOUND}, None


@pytest.fixture()
def mini(tmp_path: Path) -> RagConfig:
    config = write_mini_corpus(tmp_path)
    build_index(config, FakeEmbedder)
    return config


# --- Нарезка ------------------------------------------------------------------------------


def test_parse_date_formats() -> None:
    assert parse_date("г. Краснодар, 31 января 2023 года") == date(2023, 1, 31)
    assert parse_date("г. Самара, «22» февраля 2024 г.") == date(2024, 2, 22)
    assert parse_date("г. Москва, 01.07.2025") == date(2025, 7, 1)
    assert parse_date("без даты") is None


def test_chunk_document_structure_and_offsets() -> None:
    meta, chunks = chunk_document(
        CONTRACT_1, "K-0001", "contracts/K-0001.md", our_names=OURS, our_inns=OUR_INNS
    )
    labels = [c.label for c in chunks]
    assert labels == ["preamble", "1.1", "2.1", "2.2", "3.1", "req-1", "req-2"]
    for c in chunks:
        assert CONTRACT_1[c.start : c.end] == c.text
    assert chunks[3].section == "Цена и порядок расчётов"
    assert meta.doc_kind == "contract"
    assert meta.contract_kind == "supply"
    assert meta.number == "П-668/23"
    assert meta.doc_date == date(2023, 1, 31)
    assert meta.counterparty == "ООО «Ракурс-Комплект»"
    assert meta.counterparty_inn == "1841477442"
    heading = chunks[3].heading()
    assert "№ П-668/23 от 31.01.2023" in heading and "п. 2.2" in heading
    assert chunks[3].metadata()["contract_date"] == 20230131


def test_chunk_amendment_links_and_entrepreneur(tmp_path: Path) -> None:
    config = write_mini_corpus(tmp_path)
    metas, chunks = chunk_corpus(config.corpus_dir, our_names=OURS, our_inns=OUR_INNS)
    ds = metas["K-0001-DS1"]
    assert ds.doc_kind == "amendment"
    assert (ds.parent_number, ds.parent_date, ds.parent_doc_id) == (
        "П-668/23",
        date(2023, 1, 31),
        "K-0001",
    )
    assert ds.doc_date == date(2024, 9, 11)
    assert ds.effective_date == date(2024, 10, 1)
    assert "действует с 01.10.2024" in ds.caption()
    by_id = {c.chunk_id: c for c in chunks}
    assert by_id["K-0001-DS1#1"].amends == ("2.2",)
    assert not by_id["K-0001-DS1#1"].boilerplate
    assert by_id["K-0001-DS1#2"].boilerplate and by_id["K-0001-DS1#3"].boilerplate
    lease = metas["K-0002"]
    assert lease.counterparty == "ИП Орлов Мефодий Жоресович"
    assert lease.counterparty_inn == "542787372908"
    assert lease.doc_date == date(2024, 2, 22)
    assert metas["REG-01"].doc_kind == "regulation"
    assert metas["REG-01"].doc_date == date(2025, 1, 20)


def test_long_clause_is_split_by_sentences() -> None:
    text = (
        "# ДОГОВОР № 1\n\nг. Москва, 01.02.2024\n\n## 1. Раздел\n\n1.1. "
        + " ".join(f"Предложение номер {i} довольно длинное." for i in range(40))
        + "\n"
    )
    _, chunks = chunk_document(text, "K-9", "x.md", max_chars=300)
    parts = [c for c in chunks if c.label.startswith("1.1")]
    assert len(parts) > 1 and all(len(c.text) <= 300 for c in parts)
    for c in parts:
        assert text[c.start : c.end] == c.text


@pytest.fixture(scope="module")
def p18(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Corpus]:
    corpus = build_corpus(300, 900, 42)
    root = tmp_path_factory.mktemp("p18")
    write_corpus(corpus, root, [])
    return root, corpus


def test_chunks_match_p18_fragments(p18: tuple[Path, Corpus]) -> None:
    """Нарезка по структуре даёт ровно пункты эталона P18 и те же метаданные документов."""
    root, corpus = p18
    metas, chunks = chunk_corpus(root / "corpus", our_names=OURS, our_inns=OUR_INNS)
    fragments = {f.fragment_id: f for doc in corpus.documents for f in doc.fragments}
    assert {c.chunk_id: (c.start, c.end) for c in chunks} == {
        k: (f.start, f.end) for k, f in fragments.items()
    }
    documents = [json.loads(x) for x in (root / "documents.jsonl").read_text("utf-8").splitlines()]
    for doc in documents:
        meta = metas[doc["doc_id"]]
        assert meta.counterparty == doc.get("counterparty")
        assert meta.counterparty_inn == doc.get("counterparty_inn")
        if doc["doc_kind"] == "contract":
            assert (meta.number, meta.contract_kind) == (doc["number"], doc["contract_kind"])
            assert meta.doc_date and meta.doc_date.isoformat() == doc["date"]
        if doc["doc_kind"] == "amendment":
            assert meta.parent_doc_id == doc["parent_id"]
            assert meta.effective_date and meta.effective_date.isoformat() == doc["effective"]


# --- Леммы, BM25, RRF ---------------------------------------------------------------------


def test_lemmas_and_compound_words() -> None:
    result = lemmas("Сроки оплаты по договору с «Балтика-Торг»")
    assert "срок" in result and "оплата" in result and "договор" in result
    assert "балтика-торг" in result and "балтика" in result and "торг" in result
    assert "по" not in result  # служебное слово


def test_bm25_ranks_matching_document_first() -> None:
    docs = [
        lemmas(t) for t in ["срок оплаты товара", "арендная плата", "плата за аренду помещения"]
    ]
    index = Bm25Index(docs)
    assert index.search(lemmas("арендная плата"), 3)[0][0] == 1
    assert index.search(lemmas("арендная плата"), 3, allowed={2})[0][0] == 2
    assert index.search(lemmas("неизвестное слово"), 3) == []


def test_rrf_combines_rankings() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [item for item, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


# --- Разбор вопроса -----------------------------------------------------------------------


def test_name_matcher_inflection_and_longest_match() -> None:
    matcher = NameMatcher(
        [
            "ООО «Маяк»",
            "ООО «Маяк-Сервис»",
            "ООО «Вектор»",
            "АО «Ладога»",
            "ООО «Ладога»",
            "ИП Потапова Любовь Леонидовна",
        ]
    )
    assert matcher.find("Какой срок оплаты по Вектору?") == ["ООО «Вектор»"]
    assert matcher.find("Сколько УПД прислал ООО «Маяк-Сервис»?") == ["ООО «Маяк-Сервис»"]
    assert matcher.find("Что прислал нам «Маяк» в мае?") == ["ООО «Маяк»"]
    assert matcher.find("Неустойка по договору с АО «Ладога»") == ["АО «Ладога»"]
    assert matcher.find("НДС от ИП Потапова Любовь Леонидовна") == ["ИП Потапова Любовь Леонидовна"]
    assert matcher.find("какой вектор развития") == []  # со строчной буквы — не название


def test_analyze_contract_refs_and_unknown_orgs() -> None:
    matcher = NameMatcher(["ООО «Ракурс-Комплект»"])
    info = analyze(
        "Какая неустойка по договору поставки № П-668/23 от 31.01.2023 с ООО «Ракурс-Комплект»?",
        matcher,
        TODAY,
    )
    assert info.counterparties == ["ООО «Ракурс-Комплект»"]
    assert [(c.number, c.date) for c in info.contracts] == [("П-668/23", date(2023, 1, 31))]
    assert not info.unknown_orgs
    unknown = analyze("Срок оплаты по договору с ООО «Оникс-Логистик»?", matcher, TODAY)
    assert unknown.unknown_orgs == ["ООО «Оникс-Логистик»"] and not unknown.counterparties


@pytest.mark.parametrize(
    ("text", "period"),
    [
        ("за 2025 год", (date(2025, 1, 1), date(2025, 12, 31))),
        ("в 2026 году", (date(2026, 1, 1), date(2026, 12, 31))),
        ("за IV квартал 2025 года", (date(2025, 10, 1), date(2025, 12, 31))),
        ("за 1 квартал 2026", (date(2026, 1, 1), date(2026, 3, 31))),
        ("в мае 2026 года", (date(2026, 5, 1), date(2026, 5, 31))),
        ("в феврале 2024", (date(2024, 2, 1), date(2024, 2, 29))),
        ("с 01.03.2025 по 15.04.2025", (date(2025, 3, 1), date(2025, 4, 15))),
        ("в этом году", (date(2026, 1, 1), TODAY)),
        ("по договору от 01.02.2024", None),
    ],
)
def test_find_period(text: str, period: tuple[date, date] | None) -> None:
    assert find_period(text, TODAY) == period


# --- Маршрутизатор ------------------------------------------------------------------------


def test_router_examples() -> None:
    assert route_question("Сколько УПД получено от ООО «Кедр-Транс» за 2026 год?") == "sql"
    assert route_question("Когда получен последний документ от ООО «Маяк»?") == "sql"
    assert route_question("На какую общую сумму ООО «Балтика» выставил нам счетов?") == "sql"
    assert route_question("Сколько дней у нас на подписание актов от «Успех-Снаб»?") == "rag"
    assert route_question("Сколько счетов можно оплатить без договора?") == "rag"
    assert route_question("Какая неустойка за просрочку оплаты?") == "rag"


def test_router_on_p18_questions(p18: tuple[Path, Corpus]) -> None:
    """Все 200 вопросов P18: агрегаты по реестру → sql, остальные (и «нет ответа») → rag."""
    _, corpus = p18
    questions = generate_questions(corpus, 42)
    wrong = [
        q.question
        for q in questions
        if (route_question(q.question) == "sql") != (q.route == "table")
    ]
    assert wrong == []


# --- SQL ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE registry",
        "SELECT 1; DELETE FROM registry",
        "PRAGMA table_info(registry)",
        "SELECT * FROM registry -- комментарий",
        "UPDATE registry SET number = '1'",
        "ATTACH DATABASE 'x.db' AS x",
        "",
    ],
)
def test_validate_sql_rejects(sql: str) -> None:
    with pytest.raises(SqlRejectedError):
        validate_sql(sql)


def test_validate_sql_accepts_select_and_cleans() -> None:
    assert validate_sql("```sql\nSELECT COUNT(*) FROM registry;\n```") == (
        "SELECT COUNT(*) FROM registry"
    )
    # Слово «delete» внутри строки — не оператор.
    assert validate_sql("SELECT * FROM registry WHERE number = 'delete; me'")


@pytest.mark.parametrize(
    ("question", "kinds", "compute"),
    [
        (
            "Сколько накладных ТОРГ-12 получено от АО «Сибирь-Опт» за 2026 год?",
            ["torg12"],
            "COUNT(*)",
        ),
        (
            "На какую общую сумму с НДС получено отгрузочных документов (УПД, ТОРГ-12, акты)?",
            ["upd", "torg12", "act"],
            "ROUND(SUM(amount_with_vat), 2)",
        ),
        (
            "На какую общую сумму ООО «Балтика» выставил нам счетов на оплату?",
            ["bill"],
            "ROUND(SUM(amount_with_vat), 2)",
        ),
        (
            "Сколько НДС предъявил нам ООО «Старт» по счетам-фактурам и УПД?",
            ["vat_invoice", "upd"],
            "ROUND(SUM(vat_amount), 2)",
        ),
        (
            "Какой самый крупный по сумме документ получен от АО «Заря»?",
            [],
            "doc_type_name, number, date, amount_with_vat … ORDER BY amount_with_vat DESC LIMIT 1",
        ),
        (
            "Сколько разных контрагентов прислали нам документы за IV квартал 2025 года?",
            [],
            "COUNT(DISTINCT seller_inn)",
        ),
        (
            "От какого контрагента получено больше всего документов за 2025 год?",
            [],
            "seller_name, COUNT(*) AS n … GROUP BY seller_inn ORDER BY n DESC LIMIT 1",
        ),
    ],
)
def test_sql_hints_from_question(question: str, kinds: list[str], compute: str) -> None:
    assert doc_types(question) == kinds
    assert aggregate_hint(question) == compute


def test_hint_problems_catch_ignored_hints() -> None:
    matcher = NameMatcher(["ООО «Ракурс-Комплект»"])
    question = "Сколько УПД получено от ООО «Ракурс-Комплект» за 2025 год?"
    hints = build_hints(
        analyze(question, matcher, TODAY), {"ООО «Ракурс-Комплект»": "1841477442"}, question
    )
    good = (
        "SELECT COUNT(*) FROM registry WHERE seller_inn = '1841477442' AND doc_type = 'upd' "
        "AND date BETWEEN '2025-01-01' AND '2025-12-31'"
    )
    assert hint_problems(good, hints) == []
    copied = "SELECT COUNT(*) FROM registry WHERE seller_inn = '1234567890' AND doc_type = 'act'"
    problems = hint_problems(copied, hints)
    assert any("'1841477442'" in p for p in problems)
    assert any("виды документов" in p for p in problems)
    summed = good.replace("COUNT(*)", "ROUND(SUM(vat_amount), 2)")
    assert hint_problems(summed, hints) == ["нужно вычислить count(*)"]
    no_period = build_hints(analyze("Сколько УПД всего?", matcher, TODAY), {}, "Сколько УПД всего?")
    extra = "SELECT COUNT(*) FROM registry WHERE doc_type = 'upd' AND date >= '2026-11-19'"
    assert hint_problems(extra, no_period) == ["лишнее условие на date: в вопросе период не указан"]


def test_sql_route_retries_when_hints_ignored(mini: RagConfig) -> None:
    class TwoStepLlm(ScriptedLlm):
        def __call__(self, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], None]:
            if "sql" in schema["properties"]:
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return {"sql": "SELECT COUNT(*) FROM registry WHERE doc_type = 'upd'"}, None
                return {
                    "sql": "SELECT COUNT(*) FROM registry WHERE seller_inn = '1841477442' "
                    "AND doc_type = 'upd'"
                }, None
            return super().__call__(prompt, schema)

    llm = TwoStepLlm()
    result = make_pipeline(mini, llm).ask("Сколько УПД получено от ООО «Ракурс-Комплект»?")
    assert len(llm.prompts) == 3 and "Замечания" in llm.prompts[1]  # 2 запроса SQL + ответ
    assert "'1841477442'" in (result.sql or "") and result.sql_rows == [["2"]]


def test_run_select_read_only_and_limits(mini: RagConfig) -> None:
    db = IndexPaths(mini.index_dir).registry
    result = run_select(db, "SELECT COUNT(*), ROUND(SUM(amount_with_vat), 2) FROM registry")
    assert result.rows == [[3, 4210.0]]
    assert run_select(db, "SELECT * FROM registry", max_rows=2).truncated
    with pytest.raises(SqlRejectedError):
        run_select(
            db,
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
            "SELECT COUNT(*) FROM n",
            max_vm_steps=100_000,
        )
    # База открыта только на чтение: даже без проверки текста запись не пройдёт.
    con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM registry")
    con.close()


def test_numbers_supported() -> None:
    result = SqlResult("SELECT 1", ["n", "d"], [[2, "2025-05-10"]])
    assert numbers_supported("Получено 2 документа, последний от 10.05.2025.", "", result)
    assert not numbers_supported("Получено 3 документа.", "", result)


# --- Индекс и продолжение -----------------------------------------------------------------


def test_index_resumes_after_stop(tmp_path: Path) -> None:
    config = write_mini_corpus(tmp_path)
    first = FakeEmbedder()
    report = build_index(config, lambda: first, write_batch=4, stop_after=4)
    assert not report.complete and report.embedded_now == 4
    with pytest.raises(IndexNotReadyError):
        RagIndex(config.index_dir)
    second = FakeEmbedder()
    report = build_index(config, lambda: second, write_batch=4)
    assert report.complete
    assert report.already_embedded == 4
    assert second.documents == report.chunks - 4  # уже посчитанное не пересчитывается
    third = FakeEmbedder()
    report = build_index(config, lambda: third)
    assert third.documents == 0 and report.embedded_now == 0
    manifest = json.loads(IndexPaths(config.index_dir).manifest.read_text("utf-8"))
    assert manifest["complete"] and manifest["embedded"] == manifest["chunks"]


def test_index_detects_changed_corpus(mini: RagConfig) -> None:
    path = mini.corpus_dir / "contracts" / "K-0002.md"
    path.write_text(CONTRACT_2.replace("120 000,00", "130 000,00"), encoding="utf-8")
    with pytest.raises(IndexMismatchError):
        build_index(mini, FakeEmbedder)
    report = build_index(mini, FakeEmbedder, rebuild=True)
    assert report.complete and report.already_embedded == 0


# --- Ответ --------------------------------------------------------------------------------


def make_pipeline(config: RagConfig, llm: ScriptedLlm) -> RagPipeline:
    return RagPipeline(config, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm)


def test_split_sentences_keeps_abbreviations() -> None:
    text = "Адрес — г. Воронеж, алл. Привольная, д. 77 [1]. Площадь 192,2 м²."
    assert split_sentences(text) == [
        "Адрес — г. Воронеж, алл. Привольная, д. 77 [1].",
        "Площадь 192,2 м².",
    ]


def test_attribute_comparison_uses_both_fragments(mini: RagConfig) -> None:
    index = RagIndex(mini.index_dir)
    hits = [Hit(index.chunks[index.position[i]]) for i in ("K-0001#2.1", "K-0002#1.1")]
    refs = attribute("Цена 3 050 000 рублей против 120 000,00 руб. в месяц.", hits)
    assert refs == [1, 2]


def test_parse_answer_maps_refs_to_fragments(mini: RagConfig) -> None:
    index = RagIndex(mini.index_dir)
    hits = [
        Hit(index.chunks[index.position["K-0001#2.2"]]),
        Hit(index.chunks[index.position["K-0001#3.1"]]),
    ]
    parsed = parse_answer({"found": True, "answer": "Срок — 30 дней [1]. Неустойка [2][7]."}, hits)
    assert parsed.found
    assert [c.fragment_id for c in parsed.citations] == ["K-0001#2.2", "K-0001#3.1"]
    assert "[7]" not in parsed.answer
    # Фраза без ссылки сверяется с фрагментами: «30» есть только в п. 2.2.
    matched = parse_answer({"found": True, "answer": "Срок — 30 календарных дней."}, hits)
    assert matched.found and matched.answer == "Срок — 30 календарных дней [1]."
    assert [(c.fragment_id, c.source) for c in matched.citations] == [("K-0001#2.2", "matched")]
    unrelated = parse_answer({"found": True, "answer": "Погода сегодня хорошая."}, hits)
    assert not unrelated.found and unrelated.answer == NOT_FOUND
    refused = parse_answer({"found": False, "answer": "Срок — 30 дней [1]."}, hits)
    assert not refused.found and refused.citations == []
    prompt = build_prompt("Какой срок?", hits)
    assert "[1] Договор поставки № П-668/23" in prompt and "[2]" in prompt


def test_rag_answer_uses_current_amendment(mini: RagConfig) -> None:
    llm = ScriptedLlm()
    pipeline = make_pipeline(mini, llm)
    result = pipeline.ask(
        "Какой срок оплаты по договору поставки № П-668/23 с ООО «Ракурс-Комплект»?", debug=True
    )
    assert result.route == "rag" and result.found
    assert result.citations[0].fragment_id == "K-0001-DS1#1"
    assert result.citations[0].quote.startswith("1. Срок оплаты")
    assert "[1]" in result.answer or "[" in result.answer
    # Поиск ограничен договором из вопроса и его соглашением.
    assert result.filters["contracts"] == ["K-0001"]
    context = result.filters["context"]
    assert any(c["chunk_id"] == "K-0001-DS1#1" for c in context)
    assert all(c["chunk_id"].startswith("K-0001") for c in context)


def test_rag_answer_for_entrepreneur_filter(mini: RagConfig) -> None:
    result = make_pipeline(mini, ScriptedLlm()).ask("Какая арендная плата у Орлова?")
    assert result.found
    assert result.filters["counterparties"] == ["ИП Орлов Мефодий Жоресович"]
    assert result.citations[0].fragment_id == "K-0002#1.1"


def test_rag_not_found_cases(mini: RagConfig) -> None:
    llm = ScriptedLlm()
    pipeline = make_pipeline(mini, llm)
    unknown = pipeline.ask("Какой срок оплаты по договору с ООО «Оникс-Логистик»?")
    assert not unknown.found and unknown.answer == NOT_FOUND
    assert "Оникс-Логистик" in (unknown.reason or "")
    missing = pipeline.ask("Каков срок действия договора № 999-П с ООО «Ракурс-Комплект»?")
    assert not missing.found and "№ 999-П" in (missing.reason or "")
    assert llm.prompts == []  # до модели дело не дошло
    nothing = pipeline.ask("Сколько дней отпуска положено сотрудникам?")
    assert not nothing.found and nothing.citations == []


def test_sql_route_end_to_end(mini: RagConfig) -> None:
    llm = ScriptedLlm(
        sql="SELECT COUNT(*) FROM registry WHERE seller_inn = '1841477442' AND doc_type = 'upd' "
        "AND date BETWEEN '2025-01-01' AND '2025-12-31'"
    )
    result = make_pipeline(mini, llm).ask(
        "Сколько УПД получено от ООО «Ракурс-Комплект» за 2025 год?"
    )
    assert result.route == "sql" and result.found
    assert result.sql_rows == [["2"]]
    assert result.answer == "Получено 2 документа."
    sql_prompt = llm.prompts[0]
    assert "seller_inn = '1841477442'" in sql_prompt
    assert "date BETWEEN '2025-01-01' AND '2025-12-31'" in sql_prompt


def test_sql_route_rejects_writes(mini: RagConfig) -> None:
    llm = ScriptedLlm(sql="DELETE FROM registry")
    result = make_pipeline(mini, llm).ask("Сколько УПД получено от ООО «Ракурс-Комплект»?")
    assert not result.found and result.sql is None
    assert "проверку" in result.answer
    assert len(llm.prompts) == 2  # вторая попытка с текстом ошибки
    assert "только SELECT" in llm.prompts[1]
    rows = run_select(IndexPaths(mini.index_dir).registry, "SELECT COUNT(*) FROM registry")
    assert rows.rows == [[3]]


# --- Сервис -------------------------------------------------------------------------------


def test_rag_endpoint(mini: RagConfig, tmp_path: Path) -> None:
    settings = ServiceSettings(data_dir=tmp_path / "service", exchange_enabled=False)
    app = create_app(settings, rag_factory=lambda: make_pipeline(mini, ScriptedLlm()))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/v1/rag/ask", json={"question": "Какая арендная плата у Орлова?"})
        assert response.status_code == 200
        body = response.json()
        assert body["found"] and body["citations"][0]["fragment_id"] == "K-0002#1.1"
        assert client.post("/v1/rag/ask", json={"question": ""}).status_code == 422


def test_rag_endpoint_errors(tmp_path: Path) -> None:
    config = write_mini_corpus(tmp_path)  # индекс не собран

    def broken_llm(prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], None]:
        raise OllamaError("Ollama недоступна")

    settings = ServiceSettings(data_dir=tmp_path / "service", exchange_enabled=False)
    app = create_app(settings, rag_factory=lambda: RagPipeline(config))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/v1/rag/ask", json={"question": "Срок оплаты?"})
        assert response.status_code == 503 and "индекс" in response.json()["detail"]
    build_index(config, FakeEmbedder)
    app = create_app(
        settings,
        rag_factory=lambda: RagPipeline(
            config, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=broken_llm
        ),
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/v1/rag/ask", json={"question": "Какая арендная плата у Орлова?"})
        assert response.status_code == 503


def test_pipeline_config_file_matches_defaults() -> None:
    """configs/rag/pipeline.yaml совпадает с умолчаниями RagConfig."""
    assert load_rag_config(Path("configs/rag/pipeline.yaml")) == RagConfig()


# --- Оценка -------------------------------------------------------------------------------


def test_ranking_metrics_and_overlap() -> None:
    gold = [Span("K-1", 100, 200), Span("K-2", 0, 50)]
    ranked = [Span("K-3", 0, 10), Span("K-1", 150, 160), Span("K-2", 50, 60)]
    m = ranking_metrics(ranked, gold, [1, 3])
    assert m["mrr"] == 0.5 and m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["recall@3"] == 0.5  # K-2 [50, 60) не пересекается с [0, 50)


def test_judge_routes() -> None:
    spans = {"K-1#2.2": Span("K-1", 10, 20), "K-1#2.1": Span("K-1", 0, 9)}
    question = {
        "route": "rag",
        "sources": [{"fragment_id": "K-1#2.2"}],
        "superseded": [{"fragment_id": "K-1#2.1"}],
        "doc_ids": ["K-1"],
        "keys": ["20 календарных дней"],
    }

    def answer(*cited: tuple[str, int, int], found: bool = True) -> dict[str, Any]:
        return {
            "route": "rag",
            "found": found,
            "answer": "Срок — 20 календарных дней [1].",
            "citations": [{"doc_id": d, "start": a, "end": b} for d, a, b in cited],
        }

    assert judge(question, answer(("K-1", 10, 20)), spans)["citations_ok"]
    # Старая редакция рядом с действующей допустима и считается отдельно; одна она — ошибка.
    both = judge(question, answer(("K-1", 10, 20), ("K-1", 0, 9)), spans)
    assert both["citations_ok"] and both["cited_superseded"] == 1
    assert not judge(question, answer(("K-1", 0, 9)), spans)["citations_ok"]
    assert not judge(question, answer(("K-9", 10, 20)), spans)["citations_ok"]
    none_q = {"route": "none", "sources": []}
    assert judge(none_q, {"route": "rag", "found": False, "citations": []}, spans)["citations_ok"]
    table_q = {"route": "table", "expected_value": 1540417.62}
    sql_answer = {"route": "sql", "sql_rows": [[1540417.62]], "citations": []}
    assert judge(table_q, sql_answer, spans)["citations_ok"]
    assert value_found("2026-10-21", [["Акт", "5462", "2026-10-21", 1.0]])
    assert not value_found("6", [["7"]])


def test_stratified_is_deterministic() -> None:
    questions = [{"id": f"Q{i}", "category": "a" if i % 2 else "b"} for i in range(20)]
    first = stratified(questions, {"a": 3, "b": 2}, seed=1)
    assert first == stratified(questions, {"a": 3, "b": 2}, seed=1)
    assert len(first) == 5
    assert not {q["id"] for q in stratified(questions, {"a": 3}, 1, exclude=["Q1"])} & {"Q1"}
