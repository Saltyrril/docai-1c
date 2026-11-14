"""ИНН, КПП, БИК, расчётный и корреспондентский счёт.

Публичные реквизиты (ИНН и корр. счета банков, ЕКС казначейства) должны проходить:
так проверяется, что алгоритмы совпадают с настоящими.
"""

import pytest

from docai.validation import (
    IssueCode,
    account_key,
    inn_check_digits,
    is_valid_inn,
    validate_account,
    validate_bik,
    validate_corr_account,
    validate_inn,
    validate_kpp,
)
from factories import INDIVIDUAL_INN, SELLER_BANK


def codes(issues: list) -> list[IssueCode]:
    return [item.code for item in issues]


# --- ИНН ---------------------------------------------------------------------

PUBLIC_INN = [
    "7707083893",  # ПАО Сбербанк
    "7710140679",  # АО «ТБанк»
    "7702070139",  # Банк ВТБ (ПАО)
    "7728168971",  # АО «Альфа-Банк»
    "7707329152",  # ФНС России
]


@pytest.mark.parametrize("inn", [*PUBLIC_INN, INDIVIDUAL_INN])
def test_valid_inn(inn: str) -> None:
    assert validate_inn(inn) == []
    assert is_valid_inn(inn)


@pytest.mark.parametrize(
    "inn",
    [
        "7707083894",  # испорчена контрольная цифра
        "7707083839",  # переставлены две последние цифры
        "7707038893",  # переставлены цифры в середине
        "500100732258",  # испорчена 12-я цифра
        "500100732269",  # испорчена 11-я цифра
        "500100723259",  # перестановка в середине ИНН ИП
    ],
)
def test_broken_inn_checksum(inn: str) -> None:
    issues = validate_inn(inn, "seller.inn")
    assert codes(issues) == [IssueCode.INN_CHECKSUM]
    assert issues[0].field == "seller.inn"
    assert "контрольные" in issues[0].message


@pytest.mark.parametrize("inn", ["770708389", "77070838931", "5001007322590", ""])
def test_inn_length(inn: str) -> None:
    expected = IssueCode.INN_LENGTH if inn else IssueCode.INN_FORMAT
    assert codes(validate_inn(inn)) == [expected]


@pytest.mark.parametrize("inn", ["7707 083893", "77070838a3", "７７０７０８３８９３"])
def test_inn_format(inn: str) -> None:
    assert codes(validate_inn(inn)) == [IssueCode.INN_FORMAT]


def test_inn_check_digits_remainder_10() -> None:
    # Сумма 131, остаток от деления на 11 равен 10 → контрольная цифра 0.
    assert inn_check_digits("119111111") == "0"
    assert validate_inn("1191111110") == []
    assert inn_check_digits("770708389") == "3"
    assert inn_check_digits("5001007322") == "59"
    with pytest.raises(ValueError):
        inn_check_digits("12345")


# --- КПП ---------------------------------------------------------------------


@pytest.mark.parametrize("kpp", ["773601001", "771301001", "7736AB001", "77360Z001"])
def test_valid_kpp(kpp: str) -> None:
    assert validate_kpp(kpp) == []


@pytest.mark.parametrize("kpp", ["77360100", "7736010011", "7736аб001", "7736ab001", "A73601001"])
def test_invalid_kpp(kpp: str) -> None:
    assert codes(validate_kpp(kpp)) == [IssueCode.KPP_FORMAT]


# --- БИК ---------------------------------------------------------------------


@pytest.mark.parametrize("bik", ["044525225", "044525974", "004525988"])
def test_valid_bik(bik: str) -> None:
    assert validate_bik(bik) == []


@pytest.mark.parametrize(
    ("bik", "code"),
    [
        ("04452522", IssueCode.BIK_FORMAT),
        ("0445252250", IssueCode.BIK_FORMAT),
        ("04452522A", IssueCode.BIK_FORMAT),
        ("144525225", IssueCode.BIK_COUNTRY),
        ("054525225", IssueCode.BIK_COUNTRY),
    ],
)
def test_invalid_bik(bik: str, code: IssueCode) -> None:
    assert codes(validate_bik(bik)) == [code]


# --- Корреспондентский счёт ----------------------------------------------------

PUBLIC_BANKS = [
    ("044525225", "30101810400000000225"),  # Сбербанк
    ("044525974", "30101810145250000974"),  # ТБанк
    ("044525187", "30101810700000000187"),  # ВТБ
    ("044525593", "30101810200000000593"),  # Альфа-Банк
]


@pytest.mark.parametrize(("bik", "corr"), PUBLIC_BANKS)
def test_public_corr_accounts_valid(bik: str, corr: str) -> None:
    assert validate_corr_account(corr, bik) == []


def test_treasury_single_account_valid() -> None:
    # ЕКС УФК по г. Москве и БИК ТОФК — публичные реквизиты.
    assert validate_corr_account("40102810545370000003", "004525988") == []
    assert codes(validate_corr_account("40102810645370000003", "004525988")) == [
        IssueCode.CORR_ACCOUNT_KEY
    ]
    assert codes(validate_corr_account("30101810400000000225", "004525988")) == [
        IssueCode.CORR_ACCOUNT_PREFIX
    ]


@pytest.mark.parametrize(
    ("corr", "bik", "code"),
    [
        ("30101810500000000225", "044525225", IssueCode.CORR_ACCOUNT_KEY),
        ("30101810400000000226", "044525225", IssueCode.CORR_ACCOUNT_BIK_MISMATCH),
        ("30101810400000000225", "044525974", IssueCode.CORR_ACCOUNT_BIK_MISMATCH),
        ("40702810400000000225", "044525225", IssueCode.CORR_ACCOUNT_PREFIX),
        ("3010181040000000022", "044525225", IssueCode.CORR_ACCOUNT_FORMAT),
        ("3010181040000000022X", "044525225", IssueCode.CORR_ACCOUNT_FORMAT),
    ],
)
def test_invalid_corr_account(corr: str, bik: str, code: IssueCode) -> None:
    assert codes(validate_corr_account(corr, bik)) == [code]


def test_corr_account_with_broken_bik_checks_format_only() -> None:
    assert validate_corr_account("30101810400000000225", "12345") == []


# --- Расчётный счёт -------------------------------------------------------------


def test_seller_account_valid() -> None:
    assert validate_account(SELLER_BANK["account"], SELLER_BANK["bik"]) == []


def test_account_key_function() -> None:
    account = SELLER_BANK["account"]
    assert account_key(account, SELLER_BANK["bik"]) == account[8]
    with pytest.raises(ValueError):
        account_key("123", SELLER_BANK["bik"])


@pytest.mark.parametrize("position", [0, 5, 8, 12, 19])
def test_any_single_digit_error_breaks_key(position: int) -> None:
    account = SELLER_BANK["account"]
    broken_digit = str((int(account[position]) + 1) % 10)
    broken = account[:position] + broken_digit + account[position + 1 :]
    issues = validate_account(broken, SELLER_BANK["bik"], "seller_bank.account")
    assert codes(issues) == [IssueCode.ACCOUNT_KEY]
    assert issues[0].field == "seller_bank.account"


def test_account_does_not_match_other_bank() -> None:
    assert codes(validate_account(SELLER_BANK["account"], "044525974")) == [IssueCode.ACCOUNT_KEY]


@pytest.mark.parametrize(
    "account", ["4070281093800000000", "407028109380000000011", "4070281093800000000Z"]
)
def test_account_format(account: str) -> None:
    assert codes(validate_account(account, "044525225")) == [IssueCode.ACCOUNT_FORMAT]


def test_account_with_invalid_bik_skips_key() -> None:
    assert validate_account(SELLER_BANK["account"], "") == []
    assert validate_account(SELLER_BANK["account"], "144525225") == []


def test_treasury_account_skips_key() -> None:
    assert validate_account("03100643000000017300", "004525988") == []


def test_account_in_bank_of_russia_uses_rkc_number() -> None:
    bik = "044525000"  # условный номер 000 — подразделение Банка России
    body = "40101810000000010001"
    key = account_key(body, bik, in_credit_institution=False)
    account = body[:8] + key + body[9:]
    assert validate_account(account, bik) == []


def test_clearing_currency_letter() -> None:
    # Буква в 6-м разряде заменяется цифрой (А→0, ..., Х→9) перед расчётом.
    body = "40702К10038000000001"
    key = account_key(body, "044525225")
    account = body[:8] + key + body[9:]
    assert validate_account(account, "044525225") == []
    digits_version = account[:5] + "5" + account[6:]  # К → 5
    assert validate_account(digits_version, "044525225") == []
