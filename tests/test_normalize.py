"""Нормализаторы сумм, дат, реквизитов, ставок и значений для сравнения."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from docai.schemas import DocType, NormKind, VatRate
from docai.validation import normalize_for_compare, parse_date, parse_money, parse_vat_rate
from docai.validation.normalize import (
    normalize_doc_number,
    normalize_inn,
    normalize_kpp,
    normalize_okei,
    normalize_text,
    parse_decimal,
    quantize_money,
    split_inn_kpp,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12 200,00", "12200.00"),
        ("12200.00", "12200.00"),
        ("12 200 руб. 00 коп.", "12200.00"),
        ("12 200 руб. 5 коп.", "12200.05"),
        ("12200 р. 50 к.", "12200.50"),
        ("12 200 рублей 99 копеек", "12200.99"),
        ("-1 500 руб. 10 коп.", "-1500.10"),
        ("12 200,00", "12200.00"),
        ("12 200,00 ₽", "12200.00"),
        ("12 200,00 руб.", "12200.00"),
        ("1 234 567,89", "1234567.89"),
        ("1.234.567,89", "1234567.89"),
        ("1,234,567.89", "1234567.89"),
        ("1 234 567", "1234567"),
        ("1.234.567", "1234567"),
        ("1,234,567", "1234567"),
        ("12 200-00", "12200.00"),
        ("-1 500,00", "-1500.00"),
        ("−1 500,00", "-1500.00"),
        ("0,5", "0.5"),
        ("100", "100"),
    ],
)
def test_parse_money(text: str, expected: str) -> None:
    assert parse_money(text) == Decimal(expected)


def test_parse_money_non_strings() -> None:
    assert parse_money(12200) == Decimal(12200)
    assert parse_money(Decimal("1.50")) == Decimal("1.50")
    assert parse_money(0.1) == Decimal("0.1")


@pytest.mark.parametrize("text", ["", "abc", "12,20,3.4,5", "1.2.3,4,5", "--5", "12 200 коп."])
def test_parse_money_errors(text: str) -> None:
    with pytest.raises(ValueError, match="не удалось разобрать"):
        parse_money(text)


def test_parse_decimal_rejects_bool() -> None:
    with pytest.raises(ValueError):
        parse_decimal(True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0.005", "0.01"), ("0.004", "0.00"), ("-0.005", "-0.01"), ("21.9978", "22.00")],
)
def test_quantize_money_half_up(value: str, expected: str) -> None:
    assert quantize_money(Decimal(value)) == Decimal(expected)


@pytest.mark.parametrize(
    "text",
    [
        "14.11.2026",
        "14.11.26",
        "14/11/2026",
        "14-11-2026",
        "2026-11-14",
        "14 ноября 2026 г.",
        "14 ноября 2026 года",
        "«14» ноября 2026 г.",
        '"14" ноября 2026г.',
        "14 нояб. 2026",
        "14 Ноябрь 2026",
        "  14  НОЯБРЯ  2026 г ",
    ],
)
def test_parse_date(text: str) -> None:
    assert parse_date(text) == date(2026, 11, 14)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 мая 2026 г.", date(2026, 5, 1)),
        ("31 декабря 2025 г.", date(2025, 12, 31)),
        ("1 января 2026 г.", date(2026, 1, 1)),
        ("3 марта 2025", date(2025, 3, 3)),
        ("5 сентября 2024", date(2024, 9, 5)),
        ("05.02.2026", date(2026, 2, 5)),
        ("1 февраля 2026", date(2026, 2, 1)),
        ("1 апреля 2026", date(2026, 4, 1)),
        ("1 июня 2026", date(2026, 6, 1)),
        ("1 июля 2026", date(2026, 7, 1)),
        ("1 августа 2026", date(2026, 8, 1)),
        ("1 октября 2026", date(2026, 10, 1)),
    ],
)
def test_parse_date_months(text: str, expected: date) -> None:
    assert parse_date(text) == expected


def test_parse_date_objects() -> None:
    assert parse_date(date(2026, 1, 1)) == date(2026, 1, 1)
    assert parse_date(datetime(2026, 1, 1, 12, 30)) == date(2026, 1, 1)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("31.02.2026", "несуществующая"),
        ("30 февраля 2026 г.", "несуществующая"),
        ("14 брюмера 2026", "неизвестный месяц"),
        ("завтра", "не удалось"),
        ("2026.11.14", "не удалось"),
    ],
)
def test_parse_date_errors(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_date(text)


@pytest.mark.parametrize("text", ["7707 083 893", "ИНН 7707083893", "ИНН: 7707-083-893"])
def test_normalize_inn(text: str) -> None:
    assert normalize_inn(text) == "7707083893"


@pytest.mark.parametrize("text", ["7736 01 001", "КПП 773601001", " 773601001 "])
def test_normalize_kpp(text: str) -> None:
    assert normalize_kpp(text) == "773601001"


def test_normalize_kpp_cyrillic_letters_to_latin() -> None:
    assert normalize_kpp("7736 ав 001") == "7736AB001"


def test_split_inn_kpp() -> None:
    assert split_inn_kpp("ИНН/КПП 7707083893/773601001") == ("7707083893", "773601001")
    assert split_inn_kpp("7707083893 / 773601001") == ("7707083893", "773601001")
    assert split_inn_kpp("500100732259") == ("500100732259", None)


def test_normalize_okei_and_text() -> None:
    assert normalize_okei("796") == "796"
    assert normalize_okei("6") == "006"
    assert normalize_okei("шт") == ""
    assert normalize_text("ООО  «Ёлка»") == normalize_text('ооо "елка"') == "ооо елка"
    assert normalize_doc_number("№ 12 / 3") == "12/3"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("22%", VatRate.R22),
        ("22 %", VatRate.R22),
        ("НДС 22%", VatRate.R22),
        ("20.00%", VatRate.R20),
        ("10", VatRate.R10),
        ("0%", VatRate.R0),
        ("0.22", VatRate.R22),
        ("5%", VatRate.R5),
        ("7 %", VatRate.R7),
        ("22/122", VatRate.R22_122),
        ("22 / 122", VatRate.R22_122),
        ("9,09%", VatRate.R9_09),
        ("16,67 %", VatRate.R16_67),
        ("18.03%", VatRate.R18_03),
        ("Без НДС", VatRate.NO_VAT),
        ("без налога (НДС)", VatRate.NO_VAT),
        ("НДС исчисляется налоговым агентом", VatRate.TAX_AGENT),
        (VatRate.R10_110, VatRate.R10_110),
    ],
)
def test_parse_vat_rate(text: str | VatRate, expected: VatRate) -> None:
    assert parse_vat_rate(text) is expected


@pytest.mark.parametrize("text", ["15%", "абв", "22/120"])
def test_parse_vat_rate_errors(text: str) -> None:
    with pytest.raises(ValueError, match="ставка"):
        parse_vat_rate(text)


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        (NormKind.TEXT, " ООО «Ромашка» ", "ооо ромашка"),
        (NormKind.DOC_NUMBER, "№ 123", "123"),
        (NormKind.DATE, "14 ноября 2026 г.", date(2026, 11, 14)),
        (NormKind.MONEY, "12 200 руб. 00 коп.", Decimal("12200.00")),
        (NormKind.MONEY, "12200", Decimal("12200.00")),
        (NormKind.QUANTITY, "1 000,500", Decimal("1000.5")),
        (NormKind.INN, "7707 083 893", "7707083893"),
        (NormKind.KPP, "7736 01 001", "773601001"),
        (NormKind.DIGITS, "0445 25 225", "044525225"),
        (NormKind.OKEI, "6", "006"),
        (NormKind.VAT_RATE, "НДС 22 %", VatRate.R22),
        (NormKind.EXACT, DocType.UPD, "upd"),
        (NormKind.EXACT, 1, "1"),
        (NormKind.MONEY, None, None),
        (NormKind.TEXT, "   ", None),
        (NormKind.MONEY, "сто рублей", "сто рублей"),  # мусор остаётся строкой
        (NormKind.DATE, "вчера", "вчера"),
    ],
)
def test_normalize_for_compare(kind: NormKind, value: object, expected: object) -> None:
    assert normalize_for_compare(kind, value) == expected


def test_normalized_values_compare_equal() -> None:
    a = normalize_for_compare(NormKind.MONEY, "12 200,00")
    b = normalize_for_compare(NormKind.MONEY, "12200 руб. 00 коп.")
    assert a == b
    assert normalize_for_compare(NormKind.QUANTITY, "10") == normalize_for_compare(
        NormKind.QUANTITY, "10,000"
    )
