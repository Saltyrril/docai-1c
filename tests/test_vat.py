"""Ставки НДС по дате документа: границы 2019, 2025, 2026 годов."""

from datetime import date

import pytest

from docai.schemas import DocType, VatRate
from docai.validation import IssueCode, Severity, allowed_vat_rates, validate_vat_rate
from docai.validation.vat import RATE_PERIODS

LAST_DAY_20 = date(2025, 12, 31)
FIRST_DAY_22 = date(2026, 1, 1)


def codes(issues: list) -> list[IssueCode]:
    return [item.code for item in issues]


@pytest.mark.parametrize("rate", [VatRate.R20, VatRate.R20_120])
def test_20_valid_on_31_12_2025(rate: VatRate) -> None:
    assert validate_vat_rate(rate, LAST_DAY_20, DocType.UPD) == []


@pytest.mark.parametrize("rate", [VatRate.R20, VatRate.R20_120])
def test_20_invalid_on_01_01_2026(rate: VatRate) -> None:
    issues = validate_vat_rate(rate, FIRST_DAY_22, DocType.UPD, "lines[0].vat_rate")
    assert codes(issues) == [IssueCode.VAT_RATE_NOT_IN_FORCE]
    assert issues[0].severity is Severity.ERROR
    assert issues[0].field == "lines[0].vat_rate"
    assert "31.12.2025" in issues[0].message
    assert "01.01.2026" in issues[0].message


@pytest.mark.parametrize("rate", [VatRate.R22, VatRate.R22_122])
def test_22_valid_from_01_01_2026(rate: VatRate) -> None:
    assert validate_vat_rate(rate, FIRST_DAY_22, DocType.VAT_INVOICE) == []
    assert validate_vat_rate(rate, date(2026, 11, 14), DocType.VAT_INVOICE) == []


@pytest.mark.parametrize("doc_type", [DocType.UPD, DocType.VAT_INVOICE, DocType.ACT, None])
def test_22_invalid_on_31_12_2025(doc_type: DocType | None) -> None:
    issues = validate_vat_rate(VatRate.R22, LAST_DAY_20, doc_type)
    assert codes(issues) == [IssueCode.VAT_RATE_BEFORE_START]
    assert issues[0].severity is Severity.ERROR


def test_22_in_bill_of_2025_is_only_warning() -> None:
    # Счёт на оплату в декабре 2025 могли выставить по 22% под отгрузку 2026 года.
    issues = validate_vat_rate(VatRate.R22, LAST_DAY_20, DocType.BILL)
    assert codes(issues) == [IssueCode.VAT_RATE_BEFORE_START]
    assert issues[0].severity is Severity.WARNING


@pytest.mark.parametrize("rate", [VatRate.R5, VatRate.R7, VatRate.R5_105, VatRate.R7_107])
def test_usn_rates_5_and_7(rate: VatRate) -> None:
    assert validate_vat_rate(rate, date(2025, 1, 1)) == []
    assert validate_vat_rate(rate, FIRST_DAY_22) == []
    assert codes(validate_vat_rate(rate, date(2024, 12, 31))) == [IssueCode.VAT_RATE_BEFORE_START]


def test_18_until_2018() -> None:
    assert validate_vat_rate(VatRate.R18, date(2018, 12, 31)) == []
    assert codes(validate_vat_rate(VatRate.R18, date(2019, 1, 1))) == [
        IssueCode.VAT_RATE_NOT_IN_FORCE
    ]
    assert codes(validate_vat_rate(VatRate.R20, date(2018, 12, 31))) == [
        IssueCode.VAT_RATE_BEFORE_START
    ]


@pytest.mark.parametrize("rate", [VatRate.NO_VAT, VatRate.R0, VatRate.R10, VatRate.R10_110])
@pytest.mark.parametrize("day", [date(2004, 1, 1), LAST_DAY_20, FIRST_DAY_22, date(2030, 6, 1)])
def test_always_valid_rates(rate: VatRate, day: date) -> None:
    assert validate_vat_rate(rate, day) == []


def test_dates_before_2004_not_supported() -> None:
    issues = validate_vat_rate(VatRate.R20, date(2003, 12, 31))
    assert codes(issues) == [IssueCode.VAT_DATE_UNSUPPORTED]
    assert issues[0].severity is Severity.WARNING


def test_allowed_rates_on_boundaries() -> None:
    before = set(allowed_vat_rates(LAST_DAY_20))
    after = set(allowed_vat_rates(FIRST_DAY_22))
    assert VatRate.R20 in before and VatRate.R22 not in before
    assert VatRate.R22 in after and VatRate.R20 not in after
    assert {VatRate.R5, VatRate.R7, VatRate.R10, VatRate.R0, VatRate.NO_VAT} <= before & after


def test_every_rate_has_period() -> None:
    assert set(RATE_PERIODS) == set(VatRate)
