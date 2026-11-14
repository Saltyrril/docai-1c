"""Допустимость ставки НДС на дату документа (ст. 164 НК РФ).

Периоды действия ставок (источники — docs/DECISIONS.md, D11):
- 18% и 18/118 — с 01.01.2004 по 31.12.2018;
- 20% и 20/120 — с 01.01.2019 по 31.12.2025 (303-ФЗ);
- 22% и 22/122 — с 01.01.2026 (425-ФЗ);
- 5%, 7%, 5/105, 7/107 — с 01.01.2025, для плательщиков УСН (176-ФЗ);
- 0%, 10%, 10/110, «без НДС» — весь поддерживаемый период.

Поддерживаются даты с 01.01.2004: до этого основная ставка была 20%, а такие
документы проекту не нужны — по ним выдаётся предупреждение, а не ошибка.
"""

from datetime import date
from typing import NamedTuple

from docai.schemas.common import DocType, VatRate
from docai.validation.issues import IssueCode, Severity, ValidationIssue, issue

SUPPORTED_FROM = date(2004, 1, 1)


class RatePeriod(NamedTuple):
    start: date | None  # включительно; None — без нижней границы
    end: date | None  # включительно; None — действует по сей день


RATE_PERIODS: dict[VatRate, RatePeriod] = {
    VatRate.NO_VAT: RatePeriod(None, None),
    VatRate.R0: RatePeriod(None, None),
    VatRate.R10: RatePeriod(None, None),
    VatRate.R10_110: RatePeriod(None, None),
    VatRate.R18: RatePeriod(SUPPORTED_FROM, date(2018, 12, 31)),
    VatRate.R18_118: RatePeriod(SUPPORTED_FROM, date(2018, 12, 31)),
    VatRate.R20: RatePeriod(date(2019, 1, 1), date(2025, 12, 31)),
    VatRate.R20_120: RatePeriod(date(2019, 1, 1), date(2025, 12, 31)),
    VatRate.R22: RatePeriod(date(2026, 1, 1), None),
    VatRate.R22_122: RatePeriod(date(2026, 1, 1), None),
    VatRate.R5: RatePeriod(date(2025, 1, 1), None),
    VatRate.R7: RatePeriod(date(2025, 1, 1), None),
    VatRate.R5_105: RatePeriod(date(2025, 1, 1), None),
    VatRate.R7_107: RatePeriod(date(2025, 1, 1), None),
}

# Счёт на оплату не налоговый документ: в декабре 2025 его могли выставить по ставке
# 22% под отгрузку 2026 года. Поэтому для счёта «будущая» ставка — предупреждение.
_FUTURE_RATE_WARNING_TYPES = frozenset({DocType.BILL})


def _fmt(day: date) -> str:
    return day.strftime("%d.%m.%Y")


def is_rate_in_force(rate: VatRate, on: date) -> bool:
    """Действует ли ставка на дату."""
    period = RATE_PERIODS[rate]
    return (period.start is None or on >= period.start) and (period.end is None or on <= period.end)


def allowed_vat_rates(on: date) -> list[VatRate]:
    """Ставки, допустимые на дату, в порядке объявления `VatRate`."""
    return [rate for rate in VatRate if is_rate_in_force(rate, on)]


def validate_vat_rate(
    rate: VatRate,
    doc_date: date,
    doc_type: DocType | None = None,
    field: str = "vat_rate",
) -> list[ValidationIssue]:
    """Проверить, что ставка НДС действовала на дату документа."""
    if doc_date < SUPPORTED_FROM:
        return [
            issue(
                IssueCode.VAT_DATE_UNSUPPORTED,
                f"Ставки НДС для документов до {_fmt(SUPPORTED_FROM)} не проверяются",
                field,
                Severity.WARNING,
            )
        ]
    if is_rate_in_force(rate, doc_date):
        return []
    period = RATE_PERIODS[rate]
    if period.start is not None and doc_date < period.start:
        severity = Severity.WARNING if doc_type in _FUTURE_RATE_WARNING_TYPES else Severity.ERROR
        return [
            issue(
                IssueCode.VAT_RATE_BEFORE_START,
                f"Ставка НДС {rate.value} применяется с {_fmt(period.start)}, "
                f"а документ датирован {_fmt(doc_date)}",
                field,
                severity,
            )
        ]
    assert period.end is not None  # иначе ставка действовала бы
    return [
        issue(
            IssueCode.VAT_RATE_NOT_IN_FORCE,
            f"Ставка НДС {rate.value} действовала по {_fmt(period.end)}, "
            f"а документ датирован {_fmt(doc_date)}",
            field,
        )
    ]
