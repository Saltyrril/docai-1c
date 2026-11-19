"""Реестр полученных первичных документов: таблица для вопросов-агрегатов.

Строки берутся из генератора P3 (`generate_document`) со своим seed и без намеренных
ошибок. В P3 продавец и покупатель случайны, а в реестре нужна одна организация —
наша, — и документы должны ссылаться на наши договоры. Поэтому у документа P3
сохраняются тип, номер, дата, строки и суммы, а продавцом становится контрагент
подходящего договора, покупателем — наша организация, основанием — этот договор (D61).
"""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from pathlib import Path

from docai.rag.terms import TODAY, Contract, Kind
from docai.schemas.common import DocType
from docai.synth.catalog import get_catalog
from docai.synth.generator import generate_document

DOC_TYPE_NAMES = {
    DocType.UPD: "УПД",
    DocType.BILL: "Счёт на оплату",
    DocType.ACT: "Акт",
    DocType.TORG12: "ТОРГ-12",
    DocType.VAT_INVOICE: "Счёт-фактура",
}
COLUMNS = (
    "doc_id",
    "doc_type",
    "doc_type_name",
    "number",
    "date",
    "seller_name",
    "seller_inn",
    "buyer_name",
    "buyer_inn",
    "contract_id",
    "contract_number",
    "contract_date",
    "amount_without_vat",
    "vat_amount",
    "amount_with_vat",
    "vat_rates",
    "lines",
    "p3_source",
)


@dataclass(frozen=True, slots=True)
class RegistryRow:
    doc_id: str
    doc_type: str
    doc_type_name: str
    number: str
    date: str  # ISO
    seller_name: str
    seller_inn: str
    buyer_name: str
    buyer_inn: str
    contract_id: str
    contract_number: str
    contract_date: str
    amount_without_vat: Decimal
    vat_amount: Decimal  # 0, если «без НДС»
    amount_with_vat: Decimal
    vat_rates: str
    lines: int
    p3_source: str  # seed:номер документа в генераторе P3

    def as_csv(self) -> list[str]:
        return [
            str(getattr(self, name)) if not isinstance(getattr(self, name), Decimal)
            else f"{getattr(self, name):.2f}"
            for name in COLUMNS
        ]  # fmt: skip


@cache
def _catalog_categories() -> dict[str, str]:
    return {item.name: item.category for item in get_catalog()}


def _category_of_lines(names: list[str]) -> str:
    """Самая частая категория строк (при равенстве — по алфавиту, чтобы не зависеть от хэшей)."""
    counts = Counter(_catalog_categories().get(name, "") for name in names)
    return min(counts, key=lambda cat: (-counts[cat], cat))


def build_registry(contracts: list[Contract], n_rows: int, seed: int) -> list[RegistryRow]:
    """n_rows документов, привязанных к действующим на дату документа договорам.

    Документ P3 с товарами идёт к договору поставки той же категории каталога, с
    услугами — к договору услуг той же категории. Если подходящего договора на дату
    нет, документ пропускается. Документы с датой позже «сегодня» (TODAY) не берутся.
    Чаще документы приходят по договорам с большим весом (`Contract.weight`).
    """
    rng = random.Random(f"docai-rag:registry:{seed}")
    by_category: dict[str, list[Contract]] = {}
    for c in contracts:
        if c.kind in (Kind.SUPPLY, Kind.SERVICES):
            by_category.setdefault(c.category, []).append(c)
    rows: list[RegistryRow] = []
    index = 0
    while len(rows) < n_rows:
        index += 1
        if index > n_rows * 20:
            raise RuntimeError("не удалось набрать реестр: слишком мало подходящих договоров")
        doc = generate_document(seed, index, error_rate=0.0).document
        if doc.date > TODAY:
            continue
        category = _category_of_lines([line.name for line in doc.lines])
        candidates = [c for c in by_category.get(category, []) if c.is_active(doc.date)]
        if not candidates:
            continue
        contract = rng.choices(candidates, weights=[c.weight for c in candidates])[0]
        rates = sorted({line.vat_rate.value for line in doc.lines if line.vat_rate})
        totals = doc.totals
        assert totals.amount_without_vat is not None and totals.amount_with_vat is not None
        rows.append(
            RegistryRow(
                doc_id=f"R-{len(rows) + 1:05d}",
                doc_type=doc.doc_type.value,
                doc_type_name=DOC_TYPE_NAMES[doc.doc_type],
                number=doc.number,
                date=doc.date.isoformat(),
                seller_name=contract.them.name,
                seller_inn=contract.them.inn,
                buyer_name=contract.our.name,
                buyer_inn=contract.our.inn,
                contract_id=contract.contract_id,
                contract_number=contract.number,
                contract_date=contract.date.isoformat(),
                amount_without_vat=totals.amount_without_vat,
                vat_amount=totals.vat_amount or Decimal(0),
                amount_with_vat=totals.amount_with_vat,
                vat_rates=";".join(rates),
                lines=len(doc.lines),
                p3_source=f"{seed}:{index}",
            )
        )
    return rows


def write_registry(rows: list[RegistryRow], path: Path) -> None:
    """CSV в UTF-8 с BOM (открывается в Excel без настройки), разделитель «;»."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";", lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(row.as_csv())


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))
