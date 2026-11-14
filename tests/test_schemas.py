"""Схемы документов: выбор класса по типу, типы полей, ограничения."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from docai.schemas import (
    DOCUMENT_CLASSES,
    Act,
    Bill,
    DocType,
    Torg12,
    Upd,
    VatInvoice,
    VatRate,
    parse_document,
)
from factories import ALL_DOC_TYPES, doc_data


@pytest.mark.parametrize(
    ("doc_type", "cls"),
    [("upd", Upd), ("bill", Bill), ("act", Act), ("torg12", Torg12), ("vat_invoice", VatInvoice)],
)
def test_parse_picks_class_by_doc_type(doc_type: str, cls: type) -> None:
    doc = parse_document(doc_data(doc_type))
    assert type(doc) is cls
    assert DOCUMENT_CLASSES[DocType(doc_type)] is cls


def test_money_is_decimal_and_date_is_date() -> None:
    doc = parse_document(doc_data("bill"))
    assert doc.totals.amount_with_vat == Decimal("2561.99")
    assert isinstance(doc.lines[0].price, Decimal)
    assert doc.date == date(2026, 11, 14)
    assert doc.lines[0].vat_rate is VatRate.R22


@pytest.mark.parametrize("doc_type", ALL_DOC_TYPES)
def test_json_round_trip(doc_type: str) -> None:
    doc = parse_document(doc_data(doc_type))
    again = parse_document(doc.model_dump_json())
    assert again == doc


def test_upd_status_only_1_or_2() -> None:
    assert parse_document(doc_data("upd", upd_status=2)).upd_status == 2
    with pytest.raises(ValidationError):
        parse_document(doc_data("upd", upd_status=3))


def test_bill_requires_bank_details() -> None:
    data = doc_data("bill")
    del data["seller_bank"]
    with pytest.raises(ValidationError):
        parse_document(data)


def test_vat_invoice_has_no_bank_or_basis() -> None:
    with pytest.raises(ValidationError):
        parse_document(doc_data("vat_invoice", basis={"number": "1"}))
    with pytest.raises(ValidationError):
        parse_document(doc_data("upd", seller_bank={"bik": "044525225"}))


def test_unknown_doc_type_and_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_document({**doc_data("upd"), "doc_type": "receipt"})
    with pytest.raises(ValidationError):
        parse_document(doc_data("upd", comment="лишнее поле"))


def test_strings_are_stripped() -> None:
    doc = parse_document(doc_data("act", number="  45  "))
    assert doc.number == "45"


def test_doc_type_titles() -> None:
    assert DocType.UPD.title_ru == "УПД"
    assert {t.title_ru for t in DocType} >= {"счёт-фактура", "счёт на оплату"}


@pytest.mark.parametrize(
    ("rate", "percent", "fraction", "calculated"),
    [
        (VatRate.R22, Decimal(22), Decimal("0.22"), False),
        (VatRate.R5, Decimal(5), Decimal("0.05"), False),
        (VatRate.R0, Decimal(0), Decimal(0), False),
        (VatRate.R22_122, Decimal(22), Decimal(22) / Decimal(122), True),
        (VatRate.R10_110, Decimal(10), Decimal(10) / Decimal(110), True),
        (VatRate.NO_VAT, None, Decimal(0), False),
    ],
)
def test_vat_rate_properties(
    rate: VatRate, percent: Decimal | None, fraction: Decimal, calculated: bool
) -> None:
    assert rate.percent == percent
    assert rate.fraction == fraction
    assert rate.is_calculated is calculated
