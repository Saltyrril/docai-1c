"""Сборка корпуса RAG: договоры, доп. соглашения, регламенты, реестр — и запись на диск.

Структура папки (по умолчанию `data/rag/`):

    corpus/contracts/K-0001.md, K-0001-DS1.md ...   договоры и доп. соглашения
    corpus/regulations/REG-01.md ...                 регламенты
    documents.jsonl      один документ корпуса — одна строка (метаданные, путь)
    fragments.jsonl      фрагменты: id, документ, метка пункта, смещения, текст
    registry.csv         реестр первичных документов (для вопросов-агрегатов)
    questions.jsonl      вопросы с эталонными ответами и источниками
    _truth/contracts.jsonl  условия договоров (исходные и действующие) — только для отладки
    build_info.json      seed, версия, число документов, контрольные суммы
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from docai.rag.amendments import Amendment, generate_amendments
from docai.rag.contract_text import Style, render_contract
from docai.rag.document import CorpusDoc, Fragment
from docai.rag.facts import Fact, contract_facts
from docai.rag.registry import RegistryRow, build_registry, write_registry
from docai.rag.regulations import build_regulations
from docai.rag.terms import Contract, generate_contracts, pick_our_company

CORPUS_VERSION = "rag-v1"


@dataclass(slots=True)
class ContractBundle:
    """Договор, его текст, доп. соглашения и действующие условия."""

    base: Contract
    doc: CorpusDoc
    amendments: list[Amendment]

    @property
    def effective(self) -> Contract:
        return self.amendments[-1].after if self.amendments else self.base

    @property
    def contract_id(self) -> str:
        return self.base.contract_id

    def facts(self) -> dict[str, Fact]:
        return contract_facts(self.effective)

    def base_facts(self) -> dict[str, Fact]:
        return contract_facts(self.base)

    def history(self, fact: str) -> list[tuple[Fragment, Fact, str]]:
        """Редакции факта по порядку: (фрагмент, значение, «договор» или «ДС № k от …»)."""
        result: list[tuple[Fragment, Fact, str]] = []
        fragment = self.doc.fact_fragment(fact)
        base = self.base_facts().get(fact)
        if fragment is not None and base is not None:
            result.append((fragment, base, "договор"))
        for amendment in self.amendments:
            if fact in amendment.changed:
                frag = amendment.doc.fact_fragment(fact)
                assert frag is not None
                value = contract_facts(amendment.after)[fact]
                result.append(
                    (
                        frag,
                        value,
                        f"доп. соглашение № {amendment.number} от "
                        f"{amendment.signed.strftime('%d.%m.%Y')}",
                    )
                )
        return result

    def changed(self, fact: str) -> bool:
        """Изменил ли какой-нибудь ДС этот факт (и итог отличается от исходного)."""
        history = self.history(fact)
        return len(history) > 1 and history[-1][1].answer != history[0][1].answer

    def touched(self, fact: str) -> bool:
        return any(fact in a.changed for a in self.amendments)

    def source(self, fact: str) -> Fragment | None:
        history = self.history(fact)
        return history[-1][0] if history else None

    @property
    def docs(self) -> list[CorpusDoc]:
        return [self.doc] + [a.doc for a in self.amendments]


@dataclass(slots=True)
class Corpus:
    seed: int
    bundles: list[ContractBundle]
    regulations: list[CorpusDoc]
    registry: list[RegistryRow]
    our_name: str
    our_inn: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def documents(self) -> list[CorpusDoc]:
        docs: list[CorpusDoc] = []
        for bundle in self.bundles:
            docs += bundle.docs
        return docs + self.regulations

    def fragment_index(self) -> dict[str, Fragment]:
        return {f.fragment_id: f for d in self.documents for f in d.fragments}

    def bundle(self, contract_id: str) -> ContractBundle:
        for b in self.bundles:
            if b.contract_id == contract_id:
                return b
        raise KeyError(contract_id)


def build_corpus(n_contracts: int, n_registry: int, seed: int) -> Corpus:
    contracts = generate_contracts(n_contracts, seed)
    bundles: list[ContractBundle] = []
    for c in contracts:
        style = Style.random(random.Random(f"docai-rag:text:{seed}:{c.contract_id}"))
        doc = render_contract(c, style)
        amendments = generate_amendments(c, doc, style, seed)
        bundles.append(ContractBundle(c, doc, amendments))
    registry = build_registry([b.effective for b in bundles], n_registry, seed)
    our = pick_our_company()
    return Corpus(seed, bundles, build_regulations(), registry, our.name, our.inn)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(type(value))


def _contract_truth(c: Contract) -> dict[str, Any]:
    data = asdict(c)
    data["our"] = c.our.name
    data["them"] = {"name": c.them.name, "inn": c.them.inn, "kpp": c.them.kpp}
    data["them_bank"] = {"name": c.them_bank.name, "bik": c.them_bank.bik}
    data["weight"] = round(c.weight, 4)
    return data


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(r, ensure_ascii=False, default=_json_default) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_path(doc: CorpusDoc) -> str:
    folder = "regulations" if doc.doc_kind == "regulation" else "contracts"
    return f"corpus/{folder}/{doc.doc_id}.md"


def write_corpus(corpus: Corpus, out: Path, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Записать корпус и вопросы; вернуть сводку (она же build_info.json)."""
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("corpus/contracts", "corpus/regulations", "_truth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    fragments: list[dict[str, Any]] = []
    for doc in corpus.documents:
        rel = doc_path(doc)
        (out / rel).write_text(doc.text, encoding="utf-8", newline="\n")
        documents.append(
            {
                "doc_id": doc.doc_id,
                "doc_kind": doc.doc_kind,
                "title": doc.header[0],
                "path": rel,
                "chars": len(doc.text),
                "fragments": len(doc.fragments),
                **doc.meta,
            }
        )
        fragments += [f.to_json() for f in doc.fragments]
    _write_jsonl(out / "documents.jsonl", documents)
    _write_jsonl(out / "fragments.jsonl", fragments)
    write_registry(corpus.registry, out / "registry.csv")
    _write_jsonl(out / "questions.jsonl", questions)
    truth = [
        {
            "contract_id": b.contract_id,
            "base": _contract_truth(b.base),
            "effective": _contract_truth(b.effective),
            "amendments": [a.doc.doc_id for a in b.amendments],
        }
        for b in corpus.bundles
    ]
    _write_jsonl(out / "_truth" / "contracts.jsonl", truth)

    by_kind: dict[str, int] = {}
    for doc in corpus.documents:
        by_kind[doc.doc_kind] = by_kind.get(doc.doc_kind, 0) + 1
    by_category: dict[str, int] = {}
    for q in questions:
        by_category[q["category"]] = by_category.get(q["category"], 0) + 1
    info = {
        "version": CORPUS_VERSION,
        "seed": corpus.seed,
        "our_company": {"name": corpus.our_name, "inn": corpus.our_inn},
        "documents": by_kind,
        "fragments": len(fragments),
        "corpus_chars": sum(d["chars"] for d in documents),
        "registry_rows": len(corpus.registry),
        "questions": by_category,
        "sha256": {
            name: _sha256(out / name)
            for name in ("documents.jsonl", "fragments.jsonl", "registry.csv", "questions.jsonl")
        },
    }
    (out / "build_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return info
