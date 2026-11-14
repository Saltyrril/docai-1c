"""Проверка реквизитов: ИНН, КПП, БИК, расчётный и корреспондентский счёт.

На вход — нормализованные строки (без пробелов, см. `docai.validation.normalize`).
Каждая функция возвращает список ошибок; пустой список — реквизит верен.
Источники правил — docs/DECISIONS.md (D8–D10).
"""

import re

from docai.validation.issues import IssueCode, ValidationIssue, issue

# Весовые коэффициенты контрольных чисел ИНН.
_INN10_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)

# КПП: NNNN (код налогового органа) PP (причина: цифра или латинская A–Z) XXX (номер).
_KPP_RE = re.compile(r"\d{4}[0-9A-Z]{2}\d{3}")

# Порядок расчёта контрольного ключа (ЦБ РФ, 08.09.1997 № 515): веса 7,1,3 по кругу
# для 23 разрядов: 3 разряда из БИК + 20 разрядов счёта.
_ACCOUNT_WEIGHTS = ((7, 1, 3) * 8)[:23]
# Буква клиринговой валюты в 6-м разряде счёта заменяется цифрой (п. 8 Порядка).
_CLEARING_LETTERS = dict(zip("АВСЕНКМРТХ", "0123456789", strict=True))
_ACCOUNT_RE = re.compile(r"\d{5}[\dАВСЕНКМРТХ]\d{14}")

# Казначейские счета ТОФК (с 2021 года) начинаются с «03»; ключ по Порядку № 515
# к ним не применяется (см. DECISIONS.md).
_TREASURY_ACCOUNT_PREFIX = "03"
_CORR_ACCOUNT_PREFIX = "30101"
_EKS_PREFIX = "40102"  # единый казначейский счёт: «корр. счёт» в реквизитах ТОФК


def _digits(value: str) -> list[int]:
    return [int(char) for char in value]


def _check_digit(digits: list[int], weights: tuple[int, ...]) -> int:
    return sum(d * w for d, w in zip(digits, weights, strict=False)) % 11 % 10


def inn_check_digits(body: str) -> str:
    """Контрольные цифры ИНН по его первым 9 (юрлицо) или 10 (физлицо, ИП) цифрам."""
    if not body.isdigit() or len(body) not in (9, 10):
        raise ValueError("нужны 9 или 10 цифр ИНН без контрольных")
    digits = _digits(body)
    if len(digits) == 9:
        return str(_check_digit(digits, _INN10_WEIGHTS))
    first = _check_digit(digits, _INN12_WEIGHTS_1)
    second = _check_digit([*digits, first], _INN12_WEIGHTS_2)
    return f"{first}{second}"


def validate_inn(value: str, field: str = "inn") -> list[ValidationIssue]:
    """ИНН: 10 цифр у юрлица, 12 у физлица и ИП; контрольные цифры."""
    if not value.isascii() or not value.isdigit():
        return [issue(IssueCode.INN_FORMAT, f"ИНН «{value}» должен состоять только из цифр", field)]
    if len(value) not in (10, 12):
        return [
            issue(
                IssueCode.INN_LENGTH,
                f"В ИНН {len(value)} цифр, а должно быть 10 (организация) или 12 (ИП, физлицо)",
                field,
            )
        ]
    expected = inn_check_digits(value[:-1] if len(value) == 10 else value[:-2])
    actual = value[-len(expected) :]
    if actual != expected:
        return [
            issue(
                IssueCode.INN_CHECKSUM,
                f"Неверные контрольные цифры ИНН {value}: {actual}, ожидалось {expected}",
                field,
            )
        ]
    return []


def is_valid_inn(value: str) -> bool:
    return not validate_inn(value)


def validate_kpp(value: str, field: str = "kpp") -> list[ValidationIssue]:
    """КПП: 9 символов NNNNPPXXX; в PP допустимы цифры и заглавные латинские буквы."""
    if len(value) != 9:
        return [issue(IssueCode.KPP_FORMAT, f"В КПП {len(value)} символов, а должно быть 9", field)]
    if not _KPP_RE.fullmatch(value):
        return [
            issue(
                IssueCode.KPP_FORMAT,
                f"КПП «{value}» не по формату: 4 цифры, 2 цифры или латинские буквы, 3 цифры",
                field,
            )
        ]
    return []


def validate_bik(value: str, field: str = "bik") -> list[ValidationIssue]:
    """БИК: 9 цифр; разряды 1–2 — «04» (Россия) или «00» (органы Федерального казначейства)."""
    if not value.isascii() or not value.isdigit() or len(value) != 9:
        return [issue(IssueCode.BIK_FORMAT, f"БИК «{value}» должен состоять из 9 цифр", field)]
    if value[:2] not in ("04", "00"):
        return [
            issue(
                IssueCode.BIK_COUNTRY,
                f"БИК {value} должен начинаться с 04 (банки России) или 00 (казначейство)",
                field,
            )
        ]
    return []


def _is_treasury_bik(bik: str) -> bool:
    return bik.startswith("00")


def _is_bank_of_russia_bik(bik: str) -> bool:
    """БИК подразделения Банка России: условный номер 000–002 в разрядах 7–9."""
    return bik.startswith("04") and bik[6:] in ("000", "001", "002")


def _key_prefix(bik: str, *, in_credit_institution: bool) -> str:
    """Три разряда перед номером счёта для расчёта ключа (п. 2 Порядка № 515)."""
    if in_credit_institution:
        return bik[6:9]  # условный номер кредитной организации
    return "0" + bik[4:6]  # условный номер РКЦ (подразделения Банка России)


def _key_sum_ok(prefix: str, account: str) -> bool:
    account = account[:5] + _CLEARING_LETTERS.get(account[5], account[5]) + account[6:]
    digits = _digits(prefix + account)
    return sum(d * w % 10 for d, w in zip(digits, _ACCOUNT_WEIGHTS, strict=True)) % 10 == 0


def account_key(account: str, bik: str, *, in_credit_institution: bool = True) -> str:
    """Контрольный ключ (9-й разряд) счёта: младший разряд (сумма × 3), п. 6 Порядка № 515."""
    if not _ACCOUNT_RE.fullmatch(account) or not (bik.isdigit() and len(bik) == 9):
        raise ValueError("нужны счёт из 20 разрядов и БИК из 9 цифр")
    prefix = _key_prefix(bik, in_credit_institution=in_credit_institution)
    zeroed = account[:8] + "0" + account[9:]
    zeroed = zeroed[:5] + _CLEARING_LETTERS.get(zeroed[5], zeroed[5]) + zeroed[6:]
    digits = _digits(prefix + zeroed)
    total = sum(d * w % 10 for d, w in zip(digits, _ACCOUNT_WEIGHTS, strict=True))
    return str(total % 10 * 3 % 10)


def _account_format_issue(
    value: str, code: IssueCode, title: str, field: str
) -> ValidationIssue | None:
    if not _ACCOUNT_RE.fullmatch(value):
        return issue(code, f"{title} «{value}» должен состоять из 20 цифр", field)
    return None


def validate_account(
    account: str, bik: str, field: str = "account", bik_field: str = "bik"
) -> list[ValidationIssue]:
    """Расчётный счёт: 20 разрядов и контрольный ключ вместе с БИК.

    Если БИК сам неверен, ключ не проверяется (ошибку БИК вернёт `validate_bik`).
    Для казначейских счетов (03…) ключ не проверяется.
    """
    problem = _account_format_issue(account, IssueCode.ACCOUNT_FORMAT, "Расчётный счёт", field)
    if problem:
        return [problem]
    if validate_bik(bik, bik_field) or account.startswith(_TREASURY_ACCOUNT_PREFIX):
        return []
    in_ko = not (_is_treasury_bik(bik) or _is_bank_of_russia_bik(bik))
    if not _key_sum_ok(_key_prefix(bik, in_credit_institution=in_ko), account):
        return [
            issue(
                IssueCode.ACCOUNT_KEY,
                f"Расчётный счёт {account} не сходится с БИК {bik}: неверный контрольный ключ "
                f"(9-я цифра {account[8]}, ожидалась "
                f"{account_key(account, bik, in_credit_institution=in_ko)})",
                field,
            )
        ]
    return []


def validate_corr_account(
    corr_account: str, bik: str, field: str = "corr_account", bik_field: str = "bik"
) -> list[ValidationIssue]:
    """Корреспондентский счёт банка: 30101…, последние 3 цифры = разряды 7–9 БИК, ключ.

    Для БИК казначейства (00…) в поле корр. счёта указывается единый казначейский
    счёт 40102…; он проверяется по ключу, без сверки последних цифр.
    """
    problem = _account_format_issue(
        corr_account, IssueCode.CORR_ACCOUNT_FORMAT, "Корреспондентский счёт", field
    )
    if problem:
        return [problem]
    if validate_bik(bik, bik_field):
        return []
    treasury = _is_treasury_bik(bik)
    expected_prefix = _EKS_PREFIX if treasury else _CORR_ACCOUNT_PREFIX
    if not corr_account.startswith(expected_prefix):
        return [
            issue(
                IssueCode.CORR_ACCOUNT_PREFIX,
                f"Корреспондентский счёт {corr_account} должен начинаться с {expected_prefix}",
                field,
            )
        ]
    if not treasury and corr_account[-3:] != bik[-3:]:
        return [
            issue(
                IssueCode.CORR_ACCOUNT_BIK_MISMATCH,
                f"Последние 3 цифры корр. счёта ({corr_account[-3:]}) не совпадают "
                f"с последними 3 цифрами БИК ({bik[-3:]})",
                field,
            )
        ]
    if not _key_sum_ok(_key_prefix(bik, in_credit_institution=False), corr_account):
        return [
            issue(
                IssueCode.CORR_ACCOUNT_KEY,
                f"Корреспондентский счёт {corr_account} не сходится с БИК {bik}: "
                "неверный контрольный ключ",
                field,
            )
        ]
    return []
