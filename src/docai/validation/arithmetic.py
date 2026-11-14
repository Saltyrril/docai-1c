"""Арифметика строк и итогов.

Правила (docs/DECISIONS.md, D12):
- количество × цена = сумма строки, допуск 0,01;
- НДС строки = база × ставка, допуск 0,01;
- сумма без НДС + НДС = сумма с НДС — точно (это сложение двух сумм в копейках);
- суммы по строкам = итоги — точно (1С считает итоги сложением строк);
- НДС итога = итог × ставка (если ставка одна), допуск 0,01 на каждую строку.

Цена может быть указана как без НДС, так и с НДС (в счетах на оплату часто
«цена включает НДС»), поэтому количество × цена сверяется с любой из двух сумм.
Проверка, которой не хватает данных (поле не заполнено), пропускается: о пропусках
сообщает проверка заполненности в `docai.validation.document`.
"""

from collections.abc import Sequence
from decimal import Decimal

from docai.schemas.common import LineItem, Totals, VatRate
from docai.validation.issues import IssueCode, ValidationIssue, issue

LINE_TOLERANCE = Decimal("0.01")
ZERO = Decimal(0)


def format_money(value: Decimal) -> str:
    """12200.5 → «12 200,50» (для сообщений)."""
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def _close(a: Decimal, b: Decimal, tolerance: Decimal) -> bool:
    return abs(a - b) <= tolerance


def expected_vat(
    rate: VatRate, amount_without_vat: Decimal | None, amount_with_vat: Decimal | None
) -> Decimal | None:
    """Точная (неокруглённая) сумма НДС по ставке; None, если базы нет.

    Процентная ставка берётся от суммы без НДС, расчётная (22/122) — от суммы с НДС.
    Если суммы без НДС нет, процентная ставка пересчитывается из суммы с НДС.
    """
    if rate is VatRate.NO_VAT:
        return ZERO
    if not rate.is_calculated and amount_without_vat is not None:
        return amount_without_vat * rate.fraction
    if amount_with_vat is None:
        return None
    percent = rate.percent
    assert percent is not None
    return amount_with_vat * percent / (100 + percent)


def _line_vat(line: LineItem) -> Decimal | None:
    """НДС строки для сложения: «без НДС» без суммы считается нулём."""
    if line.vat_amount is None and line.vat_rate is VatRate.NO_VAT:
        return ZERO
    return line.vat_amount


def validate_line(line: LineItem, index: int) -> list[ValidationIssue]:
    """Проверить арифметику одной строки (index — с нуля, в сообщениях — с единицы)."""
    prefix = f"lines[{index}]"
    title = f"Строка {index + 1}"
    issues: list[ValidationIssue] = []

    if line.quantity is not None and line.price is not None:
        product = line.quantity * line.price
        amounts = [a for a in (line.amount_without_vat, line.amount_with_vat) if a is not None]
        if amounts and not any(_close(product, a, LINE_TOLERANCE) for a in amounts):
            issues.append(
                issue(
                    IssueCode.LINE_AMOUNT_MISMATCH,
                    f"{title}: количество × цена = {format_money(product)}, "
                    f"а сумма {format_money(amounts[0])}",
                    f"{prefix}.amount_without_vat"
                    if line.amount_without_vat is not None
                    else f"{prefix}.amount_with_vat",
                )
            )

    rate = line.vat_rate
    if rate is VatRate.NO_VAT and line.vat_amount not in (None, ZERO):
        issues.append(
            issue(
                IssueCode.LINE_VAT_FOR_NO_VAT,
                f"{title}: ставка «без НДС», а сумма НДС {format_money(line.vat_amount)}",
                f"{prefix}.vat_amount",
            )
        )
    elif rate is not None and line.vat_amount is not None:
        expected = expected_vat(rate, line.amount_without_vat, line.amount_with_vat)
        if expected is not None and not _close(line.vat_amount, expected, LINE_TOLERANCE):
            issues.append(
                issue(
                    IssueCode.LINE_VAT_MISMATCH,
                    f"{title}: НДС {format_money(line.vat_amount)} не равен "
                    f"{format_money(expected)} по ставке {rate.value}",
                    f"{prefix}.vat_amount",
                )
            )

    vat = _line_vat(line)
    if None not in (line.amount_without_vat, vat, line.amount_with_vat):
        assert line.amount_without_vat is not None and vat is not None
        assert line.amount_with_vat is not None
        if line.amount_without_vat + vat != line.amount_with_vat:
            issues.append(
                issue(
                    IssueCode.LINE_SUM_MISMATCH,
                    f"{title}: сумма без НДС {format_money(line.amount_without_vat)} + "
                    f"НДС {format_money(vat)} ≠ сумма с НДС {format_money(line.amount_with_vat)}",
                    f"{prefix}.amount_with_vat",
                )
            )
    return issues


def _sum_or_none(values: Sequence[Decimal | None]) -> Decimal | None:
    if not values or any(v is None for v in values):
        return None
    return sum((v for v in values if v is not None), ZERO)


def validate_totals(lines: Sequence[LineItem], totals: Totals) -> list[ValidationIssue]:
    """Сверить итоги со строками и между собой."""
    if not lines:
        return [issue(IssueCode.LINES_EMPTY, "В документе нет ни одной строки", "lines")]
    issues: list[ValidationIssue] = []

    checks = (
        (IssueCode.TOTAL_WITHOUT_VAT_MISMATCH, "amount_without_vat", "без НДС",
         [line.amount_without_vat for line in lines]),
        (IssueCode.TOTAL_VAT_MISMATCH, "vat_amount", "НДС", [_line_vat(line) for line in lines]),
        (IssueCode.TOTAL_WITH_VAT_MISMATCH, "amount_with_vat", "с НДС",
         [line.amount_with_vat for line in lines]),
    )  # fmt: skip
    for code, attr, title, values in checks:
        total = getattr(totals, attr)
        lines_sum = _sum_or_none(values)
        if total is not None and lines_sum is not None and total != lines_sum:
            issues.append(
                issue(
                    code,
                    f"Итого {title} {format_money(total)}, а сумма по строкам "
                    f"{format_money(lines_sum)}",
                    f"totals.{attr}",
                )
            )

    rates = {line.vat_rate for line in lines}
    single_rate = rates.pop() if len(rates) == 1 else None
    total_vat = totals.vat_amount
    if total_vat is None and single_rate is VatRate.NO_VAT:
        total_vat = ZERO

    if None not in (totals.amount_without_vat, total_vat, totals.amount_with_vat):
        assert totals.amount_without_vat is not None and total_vat is not None
        assert totals.amount_with_vat is not None
        if totals.amount_without_vat + total_vat != totals.amount_with_vat:
            issues.append(
                issue(
                    IssueCode.TOTAL_SUM_MISMATCH,
                    f"Итого без НДС {format_money(totals.amount_without_vat)} + "
                    f"НДС {format_money(total_vat)} ≠ всего {format_money(totals.amount_with_vat)}",
                    "totals.amount_with_vat",
                )
            )

    if single_rate is not None and total_vat is not None:
        expected = expected_vat(single_rate, totals.amount_without_vat, totals.amount_with_vat)
        tolerance = LINE_TOLERANCE * len(lines)
        if expected is not None and not _close(total_vat, expected, tolerance):
            issues.append(
                issue(
                    IssueCode.TOTAL_VAT_RATE_MISMATCH,
                    f"Итого НДС {format_money(total_vat)} не равен {format_money(expected)} "
                    f"по ставке {single_rate.value} (допуск {format_money(tolerance)})",
                    "totals.vat_amount",
                )
            )
    return issues
