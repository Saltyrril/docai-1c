"""Генератор синтетических документов: только данные (эталонная разметка), без картинок.

Документ строится из справочников и каталога так, чтобы пройти все проверки
`docai.validation`. В режиме «намеренные ошибки» часть документов портится:
неверная арифметика или невалидный ИНН. Каждый документ получает собственный
генератор случайных чисел из (seed, номер), поэтому документ №i не зависит от `n`.
"""

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from docai.schemas.common import BaseDocument, DocType, VatRate
from docai.schemas.documents import parse_document
from docai.synth.catalog import (
    GOODS_CATEGORIES,
    SERVICE_CATEGORIES,
    CatalogItem,
    items_by_category,
)
from docai.synth.directory import Entity, get_directory
from docai.validation import Severity, ValidationIssue, has_errors, validate_document
from docai.validation.vat import is_rate_in_force

DATE_FROM = date(2025, 1, 1)
DATE_TO = date(2026, 12, 31)
DEFAULT_ERROR_RATE = 0.05
MAX_LINES = 25

_CENT = Decimal("0.01")

_DOC_TYPE_WEIGHTS = {
    DocType.UPD: 30,
    DocType.BILL: 20,
    DocType.ACT: 15,
    DocType.TORG12: 20,
    DocType.VAT_INVOICE: 15,
}
_CATEGORY_WEIGHTS = {
    "канцтовары": 20,
    "стройматериалы": 22,
    "IT-услуги": 26,
    "продукты": 12,
    "транспорт": 20,
}
# (минимум, максимум, вес): большинство документов короткие, но бывают и на 25 строк.
_LINE_COUNT_BUCKETS = ((1, 3, 45), (4, 8, 35), (9, 15, 15), (16, 25, 5))


class SynthError(RuntimeError):
    """Ошибка самого генератора (документ должен был пройти проверки, но не прошёл)."""


@dataclass(frozen=True, slots=True)
class SynthDocument:
    """Сгенерированный документ и сведения о нём (для манифеста; в JSON документа их нет)."""

    index: int
    document: BaseDocument
    error_kinds: tuple[str, ...] = ()
    expected_codes: tuple[str, ...] = ()
    issues: tuple[ValidationIssue, ...] = field(default=(), repr=False)

    @property
    def has_errors(self) -> bool:
        return bool(self.error_kinds)

    @property
    def file_name(self) -> str:
        return f"{self.index:05d}_{self.document.doc_type.value}.json"


def money(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def main_vat_rate(on: date) -> VatRate:
    """Основная ставка на дату: 20% до 2026 года, 22% с 01.01.2026."""
    return VatRate.R22 if on >= date(2026, 1, 1) else VatRate.R20


# --- Номера -----------------------------------------------------------------------------

_LETTERS = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯ"
_PREFIXES = ("УТ", "ЦБ", "ТН", "АБ", "ИН", "РТ", "КС", "ПР")


def make_doc_number(rng: random.Random, year: int = 2026) -> str:
    """Номер в одном из форматов: «123», «А-0045», «УТ-000123», «123/А», «15/2026»."""
    kind = rng.choices(range(5), weights=(40, 20, 20, 10, 10))[0]
    if kind == 0:
        return str(rng.randint(1, 9999))
    if kind == 1:
        return f"{rng.choice(_LETTERS)}-{rng.randint(1, 9999):04d}"
    if kind == 2:
        return f"{rng.choice(_PREFIXES)}-{rng.randint(1, 999999):06d}"
    if kind == 3:
        return f"{rng.randint(1, 999)}/{rng.choice(_LETTERS)}"
    return f"{rng.randint(1, 999)}/{year}"


def make_contract_number(rng: random.Random) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        return str(rng.randint(1, 999))
    if kind == 1:
        return f"{rng.randint(1, 99)}/{rng.choice((24, 25, 26))}"
    if kind == 2:
        return f"Д-{rng.randint(1, 9999):04d}"
    return f"{rng.randint(1, 999)}-{rng.choice(('П', 'У', 'Д'))}/{rng.choice((24, 25, 26))}"


# --- Стороны, основание, банк -----------------------------------------------------------


def _party(entity: Entity) -> dict[str, Any]:
    return {"name": entity.name, "inn": entity.inn, "kpp": entity.kpp, "address": entity.address}


def _bank(entity: Entity) -> dict[str, Any]:
    bank = entity.bank
    return {
        "bank_name": bank.name,
        "bik": bank.bik,
        "corr_account": bank.corr_account,
        "account": entity.account,
    }


def _basis(rng: random.Random, doc_type: DocType, doc_date: date) -> dict[str, Any] | None:
    if rng.random() < 0.15:
        return None
    if doc_type is DocType.ACT:
        name = rng.choice(("Договор оказания услуг", "Договор подряда", "Договор"))
    elif doc_type is DocType.TORG12:
        name = rng.choice(("Договор поставки", "Договор купли-продажи", "Договор"))
    else:
        name = rng.choice(("Договор", "Договор поставки", "Договор оказания услуг"))
    start = max(date(2023, 1, 1), doc_date - timedelta(days=900))
    contract_date = start + timedelta(days=rng.randint(0, (doc_date - start).days))
    return {"name": name, "number": make_contract_number(rng), "date": contract_date}


# --- Ставки НДС и строки ----------------------------------------------------------------


def _pick_category(rng: random.Random, doc_type: DocType) -> str:
    names = {
        DocType.ACT: SERVICE_CATEGORIES,
        DocType.TORG12: GOODS_CATEGORIES,
    }.get(doc_type, tuple(_CATEGORY_WEIGHTS))
    return rng.choices(names, weights=[_CATEGORY_WEIGHTS[c] for c in names])[0]


def _special_rate(rng: random.Random, on: date) -> VatRate | None:
    """Ставка для продавца без НДС по общей ставке (УСН): «без НДС», 5% или 7%."""
    rate = rng.choices((VatRate.NO_VAT, VatRate.R5, VatRate.R7), weights=(70, 15, 15))[0]
    return rate if is_rate_in_force(rate, on) else VatRate.NO_VAT


def _quantity(rng: random.Random, item: CatalogItem) -> Decimal:
    top = max(1, item.qty_max)
    if item.integer_qty:
        # Малые количества встречаются чаще больших.
        return Decimal(max(1, int(top * rng.random() ** 3)))
    tenths = max(1, int(top * 10 * rng.random() ** 3))
    return Decimal(tenths) / 10 if rng.random() < 0.7 else Decimal(max(1, tenths // 10))


def _line_count(rng: random.Random) -> int:
    low, high, _ = rng.choices(_LINE_COUNT_BUCKETS, weights=[w for *_, w in _LINE_COUNT_BUCKETS])[0]
    return rng.randint(low, high)


def _build_lines(
    rng: random.Random, doc_type: DocType, doc_date: date, seller: Entity
) -> list[dict[str, Any]]:
    main = main_vat_rate(doc_date)
    uses_main_vat = rng.random() < (0.5 if seller.is_individual else 0.88)
    special = None if uses_main_vat else _special_rate(rng, doc_date)

    categories = [_pick_category(rng, doc_type)]
    if rng.random() < 0.15:
        categories.append(_pick_category(rng, doc_type))
    pool: list[CatalogItem] = []
    for category in dict.fromkeys(categories):
        pool += items_by_category()[category]
    count = min(_line_count(rng), len(pool))
    items = rng.sample(pool, count)

    lines: list[dict[str, Any]] = []
    for item in items:
        rate = special or (VatRate.R10 if item.category == "продукты" else main)
        price = money(item.price * Decimal(str(round(rng.uniform(0.9, 1.1), 3))))
        quantity = _quantity(rng, item)
        without = money(quantity * price)
        vat = money(without * rate.fraction)
        lines.append(
            {
                "name": item.name,
                "unit_name": item.unit_name,
                "unit_code": item.unit_code,
                "quantity": quantity,
                "price": price,
                "amount_without_vat": without,
                "vat_rate": rate,
                # При «без НДС» в документе стоит прочерк.
                "vat_amount": None if rate is VatRate.NO_VAT else vat,
                "amount_with_vat": without + vat,
            }
        )
    return lines


def _totals(lines: list[dict[str, Any]]) -> dict[str, Decimal | None]:
    without = sum((line["amount_without_vat"] for line in lines), Decimal(0))
    with_vat = sum((line["amount_with_vat"] for line in lines), Decimal(0))
    all_no_vat = all(line["vat_rate"] is VatRate.NO_VAT for line in lines)
    return {
        "amount_without_vat": without,
        "vat_amount": None if all_no_vat else with_vat - without,
        "amount_with_vat": with_vat,
    }


# --- Намеренные ошибки ------------------------------------------------------------------


def _corrupt_line_amount(rng: random.Random, data: dict[str, Any]) -> str:
    line = rng.choice(data["lines"])
    line["amount_without_vat"] += Decimal(rng.randint(1, 500)) + Decimal("0.17")
    return "line_amount"


def _corrupt_line_vat(rng: random.Random, data: dict[str, Any]) -> str:
    candidates = [ln for ln in data["lines"] if ln["vat_amount"] is not None]
    if not candidates:
        return _corrupt_line_amount(rng, data)
    candidates[rng.randrange(len(candidates))]["vat_amount"] += Decimal(rng.randint(1, 90))
    return "line_vat"


def _corrupt_total(rng: random.Random, data: dict[str, Any]) -> str:
    key = rng.choice(("amount_without_vat", "amount_with_vat"))
    data["totals"][key] += Decimal(rng.choice((-1, 1)) * rng.randint(1, 1000))
    return "total"


def _corrupt_inn(rng: random.Random, data: dict[str, Any]) -> str:
    party = data["seller" if rng.random() < 0.5 else "buyer"]
    inn = party["inn"]
    # Меняем последнюю цифру: контрольные цифры однозначно определяются остальными.
    party["inn"] = inn[:-1] + str((int(inn[-1]) + rng.randint(1, 9)) % 10)
    return "invalid_inn"


_CORRUPTIONS = (
    ("line_amount", _corrupt_line_amount),
    ("line_vat", _corrupt_line_vat),
    ("total", _corrupt_total),
    ("invalid_inn", _corrupt_inn),
)


# --- Сборка документа -------------------------------------------------------------------


def _build_data(rng: random.Random) -> dict[str, Any]:
    directory = get_directory()
    doc_type = rng.choices(list(_DOC_TYPE_WEIGHTS), weights=list(_DOC_TYPE_WEIGHTS.values()))[0]
    doc_date = DATE_FROM + timedelta(days=rng.randint(0, (DATE_TO - DATE_FROM).days))
    seller, buyer = rng.sample(directory.entities, 2)
    lines = _build_lines(rng, doc_type, doc_date, seller)

    data: dict[str, Any] = {
        "doc_type": doc_type,
        "number": make_doc_number(rng, doc_date.year),
        "date": doc_date,
        "seller": _party(seller),
        "buyer": _party(buyer),
        "lines": lines,
        "totals": _totals(lines),
    }
    if doc_type is DocType.UPD:
        data["upd_status"] = rng.choices((1, 2), weights=(60, 40))[0]
    if doc_type is DocType.BILL or (
        doc_type in (DocType.ACT, DocType.TORG12) and rng.random() < 0.6
    ):
        data["seller_bank"] = _bank(seller)
    if doc_type is not DocType.VAT_INVOICE:
        data["basis"] = _basis(rng, doc_type, doc_date)
    return data


def generate_document(
    seed: int, index: int, error_rate: float = DEFAULT_ERROR_RATE
) -> SynthDocument:
    """Документ №index при заданном seed. Не зависит от других документов серии."""
    rng = random.Random(f"docai-synth:{seed}:{index}")
    data = _build_data(rng)
    kinds: tuple[str, ...] = ()
    if rng.random() < error_rate:
        _, corrupt = rng.choice(_CORRUPTIONS)
        kinds = (corrupt(rng, data),)

    document = parse_document(data)
    issues = tuple(validate_document(document))
    if kinds:
        if not has_errors(list(issues)):
            raise SynthError(f"документ {index}: ошибка «{kinds[0]}» не обнаружена проверками")
        codes = tuple(sorted({i.code.value for i in issues if i.severity is Severity.ERROR}))
    else:
        if issues:
            details = "; ".join(i.message for i in issues)
            raise SynthError(f"документ {index} должен быть корректным, но: {details}")
        codes = ()
    return SynthDocument(index, document, kinds, codes, issues)


def generate_documents(
    n: int, seed: int, error_rate: float = DEFAULT_ERROR_RATE
) -> list[SynthDocument]:
    """Серия из n документов (индексы с 1)."""
    return [generate_document(seed, i, error_rate) for i in range(1, n + 1)]
