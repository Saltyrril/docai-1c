"""Полная проверка документа: заполненность, стороны, банк, ставки, арифметика."""

from datetime import date

import pytest

from docai.schemas import parse_document
from docai.validation import IssueCode, Severity, has_errors, validate_document
from factories import ALL_DOC_TYPES, INDIVIDUAL_INN, SBER, doc_data, line


def codes(issues: list) -> list[IssueCode]:
    return [item.code for item in issues]


@pytest.mark.parametrize("doc_type", ALL_DOC_TYPES)
def test_correct_documents_have_no_issues(doc_type: str) -> None:
    assert validate_document(parse_document(doc_data(doc_type))) == []


def test_individual_seller_without_kpp() -> None:
    seller = {"name": "ИП Иванов И. И.", "inn": INDIVIDUAL_INN, "address": "Тверь"}
    assert validate_document(parse_document(doc_data("upd", seller=seller))) == []


def test_individual_with_kpp_is_error() -> None:
    seller = {"name": "ИП Иванов", "inn": INDIVIDUAL_INN, "kpp": "773601001", "address": "Тверь"}
    issues = validate_document(parse_document(doc_data("upd", seller=seller)))
    assert codes(issues) == [IssueCode.KPP_FOR_INDIVIDUAL]
    assert issues[0].field == "seller.kpp"


def test_organization_without_kpp_is_warning() -> None:
    buyer = {**SBER, "kpp": None}
    issues = validate_document(parse_document(doc_data("act", buyer=buyer)))
    assert codes(issues) == [IssueCode.KPP_MISSING]
    assert issues[0].severity is Severity.WARNING
    assert not has_errors(issues)


def test_broken_requisites_are_reported_with_fields() -> None:
    data = doc_data("bill")
    data["seller"]["inn"] = "7707083894"
    data["buyer"]["kpp"] = "77130100"
    data["seller_bank"]["account"] = "40702810038000000001"
    data["seller_bank"]["corr_account"] = "30101810400000000226"
    issues = validate_document(parse_document(data))
    found = {(item.code, item.field) for item in issues}
    assert found == {
        (IssueCode.INN_CHECKSUM, "seller.inn"),
        (IssueCode.KPP_FORMAT, "buyer.kpp"),
        (IssueCode.ACCOUNT_KEY, "seller_bank.account"),
        (IssueCode.CORR_ACCOUNT_BIK_MISMATCH, "seller_bank.corr_account"),
    }
    assert has_errors(issues)
    assert all(item.message for item in issues)


def test_bad_bik_reported_once() -> None:
    data = doc_data("bill")
    data["seller_bank"]["bik"] = "144525225"
    assert codes(validate_document(parse_document(data))) == [IssueCode.BIK_COUNTRY]


def test_missing_required_fields() -> None:
    data = doc_data("bill")
    data["seller_bank"]["bik"] = None
    data["seller"]["inn"] = "  "
    issues = validate_document(parse_document(data))
    missing = {item.field for item in issues if item.code is IssueCode.FIELD_MISSING}
    assert missing == {"seller_bank.bik", "seller.inn"}


def test_missing_line_fields_in_upd() -> None:
    data = doc_data("upd")
    data["lines"][1]["amount_with_vat"] = None
    data["lines"][1]["quantity"] = None
    issues = validate_document(parse_document(data))
    missing = {item.field for item in issues if item.code is IssueCode.FIELD_MISSING}
    assert missing == {"lines[1].amount_with_vat", "lines[1].quantity"}


def test_vat_invoice_requires_addresses() -> None:
    data = doc_data("vat_invoice")
    data["buyer"]["address"] = None
    issues = validate_document(parse_document(data))
    assert [(i.code, i.field) for i in issues] == [(IssueCode.FIELD_MISSING, "buyer.address")]


def test_old_rate_in_2026_document() -> None:
    data = doc_data("upd", date=date(2026, 1, 1))
    data["lines"] = [line("1", "1000.00", "1000.00", "20%", "200.00", "1200.00")] * 2
    data["totals"] = {
        "amount_without_vat": "2000.00",
        "vat_amount": "400.00",
        "amount_with_vat": "2400.00",
    }
    issues = validate_document(parse_document(data))
    # Одна ошибка на ставку, а не на каждую строку.
    assert codes(issues) == [IssueCode.VAT_RATE_NOT_IN_FORCE]
    assert issues[0].field == "lines[0].vat_rate"


def test_arithmetic_errors_flow_through() -> None:
    data = doc_data("torg12")
    data["lines"][0]["amount_without_vat"] = "2000.10"
    issues = validate_document(parse_document(data))
    assert IssueCode.LINE_AMOUNT_MISMATCH in codes(issues)
    assert IssueCode.TOTAL_WITHOUT_VAT_MISMATCH in codes(issues)


def test_issue_str() -> None:
    data = doc_data("upd")
    data["seller"]["inn"] = "7707083894"
    issue = validate_document(parse_document(data))[0]
    assert str(issue).startswith("error: INN_CHECKSUM [seller.inn]: ")


def test_partial_bank_details_in_act() -> None:
    # В акте реквизиты необязательны: без БИК ключ счёта сверить не с чем.
    bank = {"bank_name": "Банк", "account": "40702810938000000001"}
    assert validate_document(parse_document(doc_data("act", seller_bank=bank))) == []
    bank = {"bik": "044525225", "corr_account": "30101810400000000225"}
    assert validate_document(parse_document(doc_data("torg12", seller_bank=bank))) == []
