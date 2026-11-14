"""Проверки реквизитов: ИНН, КПП, БИК, расчётный счёт, суммы и НДС.

Все проверки возвращают список `ValidationIssue` (код, сообщение на русском, поле,
серьёзность) и не бросают исключений на неверных данных.
"""

from docai.validation.arithmetic import validate_line, validate_totals
from docai.validation.document import (
    validate_bank,
    validate_document,
    validate_party,
    validate_required,
    validate_vat_rates,
)
from docai.validation.issues import IssueCode, Severity, ValidationIssue, has_errors
from docai.validation.normalize import (
    normalize_for_compare,
    parse_date,
    parse_decimal,
    parse_money,
    parse_vat_rate,
)
from docai.validation.requisites import (
    account_key,
    inn_check_digits,
    is_valid_inn,
    validate_account,
    validate_bik,
    validate_corr_account,
    validate_inn,
    validate_kpp,
)
from docai.validation.vat import allowed_vat_rates, is_rate_in_force, validate_vat_rate

__all__ = [
    "IssueCode",
    "Severity",
    "ValidationIssue",
    "account_key",
    "allowed_vat_rates",
    "has_errors",
    "inn_check_digits",
    "is_rate_in_force",
    "is_valid_inn",
    "normalize_for_compare",
    "parse_date",
    "parse_decimal",
    "parse_money",
    "parse_vat_rate",
    "validate_account",
    "validate_bank",
    "validate_bik",
    "validate_corr_account",
    "validate_document",
    "validate_inn",
    "validate_kpp",
    "validate_line",
    "validate_party",
    "validate_required",
    "validate_totals",
    "validate_vat_rate",
    "validate_vat_rates",
]
