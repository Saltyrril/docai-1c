"""Арифметика строк и итогов, округления, допуски."""

from decimal import Decimal

import pytest

from docai.schemas import LineItem, Totals, VatRate
from docai.validation import IssueCode, validate_line, validate_totals
from docai.validation.arithmetic import expected_vat, format_money
from factories import line


def make_line(*args: str | None) -> LineItem:
    return LineItem.model_validate(line(*args))  # type: ignore[arg-type]


def totals(without: str | None, vat: str | None, with_vat: str | None) -> Totals:
    return Totals(
        amount_without_vat=None if without is None else Decimal(without),
        vat_amount=None if vat is None else Decimal(vat),
        amount_with_vat=None if with_vat is None else Decimal(with_vat),
    )


def codes(issues: list) -> list[IssueCode]:
    return [item.code for item in issues]


# --- Строка --------------------------------------------------------------------


def test_correct_line() -> None:
    assert validate_line(make_line("2", "1000.00", "2000.00", "22%", "440.00", "2440.00"), 0) == []


@pytest.mark.parametrize(
    ("amount", "ok"),
    [("99.99", True), ("100.00", True), ("99.98", True), ("100.01", False), ("99.97", False)],
)
def test_quantity_times_price_tolerance(amount: str, ok: bool) -> None:
    # 3 × 33.33 = 99.99; допуск 0,01 включительно.
    item = make_line("3", "33.33", amount, None, None, None)
    assert (validate_line(item, 0) == []) is ok


def test_fractional_quantity_rounding() -> None:
    # 1,235 кг × 99,99 = 123,48765 → в документе 123,49 (округление до копеек).
    assert validate_line(make_line("1.235", "99.99", "123.49", None, None, None), 0) == []
    issues = validate_line(make_line("1.235", "99.99", "123.50", None, None, None), 4)
    assert codes(issues) == [IssueCode.LINE_AMOUNT_MISMATCH]
    assert issues[0].field == "lines[4].amount_without_vat"
    assert issues[0].message.startswith("Строка 5:")


def test_price_including_vat_is_accepted() -> None:
    # Цена 1220 с НДС: количество × цена совпадает с суммой с НДС.
    assert validate_line(make_line("2", "1220.00", "2000.00", "22%", "440.00", "2440.00"), 0) == []


def test_amount_mismatch_against_amount_with_vat_only() -> None:
    issues = validate_line(make_line("2", "1220.00", None, "22%", None, "2450.00"), 0)
    assert codes(issues) == [IssueCode.LINE_AMOUNT_MISMATCH]
    assert issues[0].field == "lines[0].amount_with_vat"


@pytest.mark.parametrize(("vat", "ok"), [("22.00", True), ("21.99", True), ("21.98", False)])
def test_line_vat_tolerance(vat: str, ok: bool) -> None:
    # 99,99 × 22% = 21,9978.
    with_vat = str(Decimal("99.99") + Decimal(vat))
    issues = validate_line(make_line("3", "33.33", "99.99", "22%", vat, with_vat), 0)
    assert (codes(issues) == []) is ok
    if not ok:
        assert codes(issues) == [IssueCode.LINE_VAT_MISMATCH]


@pytest.mark.parametrize(
    ("rate", "vat", "with_vat"),
    [
        ("5%", "50.00", "1050.00"),
        ("7%", "70.00", "1070.00"),
        ("10%", "100.00", "1100.00"),
        ("20%", "200.00", "1200.00"),
        ("0%", "0.00", "1000.00"),
    ],
)
def test_percent_rates(rate: str, vat: str, with_vat: str) -> None:
    assert validate_line(make_line("1", "1000.00", "1000.00", rate, vat, with_vat), 0) == []


def test_usn_7_rounding() -> None:
    # 333,33 × 7% = 23,3331 → 23,33.
    assert validate_line(make_line("1", "333.33", "333.33", "7%", "23.33", "356.66"), 0) == []
    issues = validate_line(make_line("1", "333.33", "333.33", "7%", "23.35", "356.68"), 0)
    assert codes(issues) == [IssueCode.LINE_VAT_MISMATCH]


def test_wrong_rate_in_line() -> None:
    issues = validate_line(make_line("1", "1000.00", "1000.00", "22%", "200.00", "1200.00"), 0)
    assert codes(issues) == [IssueCode.LINE_VAT_MISMATCH]


@pytest.mark.parametrize(("rate", "vat"), [("22/122", "220.00"), ("5/105", "52.38")])
def test_calculated_rate_from_amount_with_vat(rate: str, vat: str) -> None:
    # Аванс: 1220 × 22/122 = 220; 1100 × 5/105 = 52,38.
    with_vat = "1220.00" if rate == "22/122" else "1100.00"
    assert validate_line(make_line("1", with_vat, None, rate, vat, with_vat), 0) == []


def test_percent_rate_without_net_amount() -> None:
    assert validate_line(make_line("1", "1220.00", None, "22%", "220.00", "1220.00"), 0) == []


def test_no_vat_line() -> None:
    assert validate_line(make_line("1", "500.00", "500.00", "без НДС", None, "500.00"), 0) == []
    assert validate_line(make_line("1", "500.00", "500.00", "без НДС", "0", "500.00"), 0) == []
    issues = validate_line(make_line("1", "500.00", "500.00", "без НДС", "90.00", "590.00"), 2)
    assert IssueCode.LINE_VAT_FOR_NO_VAT in codes(issues)


def test_line_sum_mismatch() -> None:
    issues = validate_line(make_line("2", "1000.00", "2000.00", "22%", "440.00", "2440.01"), 0)
    assert codes(issues) == [IssueCode.LINE_SUM_MISMATCH]


def test_line_without_numbers_has_nothing_to_check() -> None:
    assert validate_line(LineItem(name="Услуга"), 0) == []


def test_expected_vat_without_base() -> None:
    assert expected_vat(VatRate.R22, None, None) is None
    assert expected_vat(VatRate.NO_VAT, None, None) == 0
    assert expected_vat(VatRate.TAX_AGENT, None, None) == 0


def test_tax_agent_behaves_like_no_vat() -> None:
    agent = "НДС исчисляется налоговым агентом"
    assert validate_line(make_line("1", "500.00", "500.00", agent, None, "500.00"), 0) == []
    issues = validate_line(make_line("1", "500.00", "500.00", agent, "90.00", "590.00"), 0)
    assert IssueCode.LINE_VAT_FOR_NO_VAT in codes(issues)


def test_calculated_rate_in_percent_notation() -> None:
    # 18,03% — то же, что 22/122: НДС выделяется из суммы с налогом.
    assert expected_vat(VatRate.R18_03, None, Decimal("122.00")) == Decimal("22.00")
    line = make_line("1", "122.00", None, "18,03%", "22.00", "122.00")
    assert validate_line(line, 0) == []


def test_format_money() -> None:
    assert format_money(Decimal("12200.5")) == "12 200,50"


# --- Итоги ---------------------------------------------------------------------

LINES = [
    make_line("2", "1000.00", "2000.00", "22%", "440.00", "2440.00"),
    make_line("3", "33.33", "99.99", "22%", "22.00", "121.99"),
]


def test_correct_totals() -> None:
    assert validate_totals(LINES, totals("2099.99", "462.00", "2561.99")) == []


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("2100.00", "462.00", "2561.99"), [IssueCode.TOTAL_WITHOUT_VAT_MISMATCH,
                                            IssueCode.TOTAL_SUM_MISMATCH]),
        (("2099.99", "462.01", "2561.99"), [IssueCode.TOTAL_VAT_MISMATCH,
                                            IssueCode.TOTAL_SUM_MISMATCH]),
        (("2099.99", "462.00", "2562.00"), [IssueCode.TOTAL_WITH_VAT_MISMATCH,
                                            IssueCode.TOTAL_SUM_MISMATCH]),
    ],
)  # fmt: skip
def test_totals_mismatch(values: tuple[str, str, str], expected: list[IssueCode]) -> None:
    assert codes(validate_totals(LINES, totals(*values))) == expected


def test_partial_totals_are_checked_where_possible() -> None:
    assert validate_totals(LINES, totals(None, None, "2561.99")) == []
    assert codes(validate_totals(LINES, totals(None, None, "2561.00"))) == [
        IssueCode.TOTAL_WITH_VAT_MISMATCH
    ]


def test_total_vat_tolerance_grows_with_lines() -> None:
    # 4 строки по 0,25: НДС строки 0,055 → 0,06; итого 0,24 при 1,00 × 22% = 0,22.
    four = [make_line("1", "0.25", "0.25", "22%", "0.06", "0.31") for _ in range(4)]
    assert validate_totals(four, totals("1.00", "0.24", "1.24")) == []
    # Та же разница 0,02 на одной строке — уже ошибка.
    one = [make_line("1", "1.00", "1.00", "22%", None, None)]
    assert codes(validate_totals(one, totals("1.00", "0.24", "1.24"))) == [
        IssueCode.TOTAL_VAT_RATE_MISMATCH
    ]


def test_total_vat_rate_without_line_vat() -> None:
    # Как в счёте на оплату: НДС только в итогах.
    lines = [make_line("10", "100.00", "1000.00", "22%", None, None)]
    assert validate_totals(lines, totals("1000.00", "220.00", "1220.00")) == []
    assert codes(validate_totals(lines, totals("1000.00", "200.00", "1200.00"))) == [
        IssueCode.TOTAL_VAT_RATE_MISMATCH
    ]


def test_no_vat_document() -> None:
    lines = [make_line("1", "500.00", "500.00", "без НДС", None, "500.00")]
    assert validate_totals(lines, totals("500.00", None, "500.00")) == []
    issues = validate_totals(lines, totals("500.00", "90.00", "590.00"))
    assert set(codes(issues)) >= {IssueCode.TOTAL_VAT_MISMATCH, IssueCode.TOTAL_VAT_RATE_MISMATCH}


def test_mixed_rates_skip_rate_check() -> None:
    lines = [
        make_line("1", "1000.00", "1000.00", "22%", "220.00", "1220.00"),
        make_line("1", "1000.00", "1000.00", "10%", "100.00", "1100.00"),
    ]
    assert validate_totals(lines, totals("2000.00", "320.00", "2320.00")) == []


def test_empty_lines() -> None:
    issues = validate_totals([], totals("0", "0", "0"))
    assert codes(issues) == [IssueCode.LINES_EMPTY]
