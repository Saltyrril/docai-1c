"""Найденная проверкой ошибка: код, сообщение на русском, поле, серьёзность."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Severity(StrEnum):
    ERROR = "error"  # документ неверен
    WARNING = "warning"  # подозрительно, но бывает в настоящих документах


class IssueCode(StrEnum):
    """Коды ошибок. Стабильны: по ним считаются метрики и пишутся тесты."""

    # Реквизиты
    INN_FORMAT = "INN_FORMAT"
    INN_LENGTH = "INN_LENGTH"
    INN_CHECKSUM = "INN_CHECKSUM"
    KPP_FORMAT = "KPP_FORMAT"
    KPP_FOR_INDIVIDUAL = "KPP_FOR_INDIVIDUAL"
    KPP_MISSING = "KPP_MISSING"
    BIK_FORMAT = "BIK_FORMAT"
    BIK_COUNTRY = "BIK_COUNTRY"
    ACCOUNT_FORMAT = "ACCOUNT_FORMAT"
    ACCOUNT_KEY = "ACCOUNT_KEY"
    CORR_ACCOUNT_FORMAT = "CORR_ACCOUNT_FORMAT"
    CORR_ACCOUNT_PREFIX = "CORR_ACCOUNT_PREFIX"
    CORR_ACCOUNT_BIK_MISMATCH = "CORR_ACCOUNT_BIK_MISMATCH"
    CORR_ACCOUNT_KEY = "CORR_ACCOUNT_KEY"
    # НДС
    VAT_RATE_NOT_IN_FORCE = "VAT_RATE_NOT_IN_FORCE"
    VAT_RATE_BEFORE_START = "VAT_RATE_BEFORE_START"
    VAT_DATE_UNSUPPORTED = "VAT_DATE_UNSUPPORTED"
    # Арифметика строк
    LINE_AMOUNT_MISMATCH = "LINE_AMOUNT_MISMATCH"
    LINE_VAT_MISMATCH = "LINE_VAT_MISMATCH"
    LINE_VAT_FOR_NO_VAT = "LINE_VAT_FOR_NO_VAT"
    LINE_SUM_MISMATCH = "LINE_SUM_MISMATCH"
    # Итоги
    LINES_EMPTY = "LINES_EMPTY"
    TOTAL_WITHOUT_VAT_MISMATCH = "TOTAL_WITHOUT_VAT_MISMATCH"
    TOTAL_VAT_MISMATCH = "TOTAL_VAT_MISMATCH"
    TOTAL_WITH_VAT_MISMATCH = "TOTAL_WITH_VAT_MISMATCH"
    TOTAL_SUM_MISMATCH = "TOTAL_SUM_MISMATCH"
    TOTAL_VAT_RATE_MISMATCH = "TOTAL_VAT_RATE_MISMATCH"
    # Заполненность
    FIELD_MISSING = "FIELD_MISSING"


class ValidationIssue(BaseModel):
    """Одна ошибка проверки. Проверки возвращают список таких записей."""

    model_config = ConfigDict(frozen=True)

    code: IssueCode
    message: str
    field: str | None = None
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        where = f" [{self.field}]" if self.field else ""
        return f"{self.severity.value}: {self.code.value}{where}: {self.message}"


def issue(
    code: IssueCode,
    message: str,
    field: str | None = None,
    severity: Severity = Severity.ERROR,
) -> ValidationIssue:
    """Короткий конструктор ошибки."""
    return ValidationIssue(code=code, message=message, field=field, severity=severity)


def has_errors(issues: list[ValidationIssue]) -> bool:
    """Есть ли среди найденного ошибки (а не только предупреждения)."""
    return any(item.severity is Severity.ERROR for item in issues)
