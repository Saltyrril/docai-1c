"""Автоисправления сумм — только там, где ответ однозначен (D41).

Идея: модель ошиблась в одном числе, а остальные числа документа это опровергают и
указывают единственное верное значение. Исправляется одно поле суммы за раз:

1. Строка. Перебираются исправления одного из полей «сумма без НДС», «НДС», «сумма
   с НДС» (значение выводится из соседних полей строки). Исправление берётся, только если
   оно единственное из всех возможных и после него в строке нет ни одной ошибки.
   Количество и цена не исправляются: при неверной «сумма = количество × цена» неизвестно,
   какое из трёх чисел неверно. Ставка НДС тоже: она может быть верной при неверной дате.
2. Подтверждение. Строка содержит ещё не меньше трёх других чисел, которые согласуются
   с исправлением, либо итог документа сходится после него. Если итог сходился ДО
   исправления, а после перестал, исправление отклоняется: итог подтверждает прочитанное.
3. Итоги. Если все строки верны, расходится ровно один итог, а итоги между собой
   (без НДС + НДС = с НДС) не сходятся и после замены на сумму по строкам сойдутся, итог
   заменяется. Расхождение всех итогов сразу означает пропущенную строку — не исправляется.
4. Если известен текст-источник и в нём дословно есть прочитанное значение, модель прочитала
   верно, а ошибка в самом документе: исправление отклоняется (документ уйдёт на проверку).
"""

from collections.abc import Sequence
from decimal import Decimal

from docai.postprocess.evidence import SourceText, verbatim
from docai.postprocess.models import Correction
from docai.schemas.common import BaseDocument, LineItem
from docai.schemas.fields import NormKind
from docai.validation import has_errors, validate_line
from docai.validation.arithmetic import ZERO, expected_vat, format_money
from docai.validation.normalize import quantize_money

_COLUMNS = {
    "amount_without_vat": "amount_without_vat",
    "vat_amount": "vat_amount",
    "amount_with_vat": "amount_with_vat",
}
_TITLES = {
    "amount_without_vat": "сумма без НДС",
    "vat_amount": "сумма НДС",
    "amount_with_vat": "сумма с НДС",
}


def _line_value(line: LineItem, attr: str) -> Decimal | None:
    """Значение для сложения: «без НДС» без суммы НДС считается нулём."""
    value = getattr(line, attr)
    if value is None and attr == "vat_amount" and line.vat_rate is not None:
        return ZERO if line.vat_rate.is_untaxed else None
    return value


def column_sum(lines: Sequence[LineItem], attr: str) -> Decimal | None:
    values = [_line_value(line, attr) for line in lines]
    if not values or any(v is None for v in values):
        return None
    return sum((v for v in values if v is not None), ZERO)


def _line_candidates(line: LineItem) -> list[tuple[str, Decimal]]:
    """Значения полей сумм, выводимые из других полей строки."""
    out: list[tuple[str, Decimal]] = []
    a, v, w = line.amount_without_vat, line.vat_amount, line.amount_with_vat
    if line.quantity is not None and line.price is not None:
        out.append(("amount_without_vat", quantize_money(line.quantity * line.price)))
    if w is not None and v is not None:
        out.append(("amount_without_vat", quantize_money(w - v)))
    if a is not None and line.vat_rate is not None and not line.vat_rate.is_untaxed:
        expected = expected_vat(line.vat_rate, a, w)
        if expected is not None:
            out.append(("vat_amount", quantize_money(expected)))
    if w is not None and a is not None:
        out.append(("vat_amount", quantize_money(w - a)))
    if a is not None and v is not None:
        out.append(("amount_with_vat", quantize_money(a + v)))
    return out


def _printed_in_source(value: Decimal | None, sources: list[SourceText]) -> bool:
    """Прочитанное значение дословно (и не случайно) есть в тексте: модель прочитала верно."""
    result = verbatim(NormKind.MONEY, value, sources)
    return result is not None and result == (True, True)


def _fmt(value: Decimal | None) -> str | None:
    return None if value is None else format_money(value)


def _fix_line(doc: BaseDocument, index: int, sources: list[SourceText]) -> Correction | None:
    line = doc.lines[index]
    if not has_errors(validate_line(line, index)):
        return None
    found: set[tuple[str, Decimal]] = set()
    for attr, value in _line_candidates(line):
        if value == getattr(line, attr) or value < 0:
            continue
        trial = line.model_copy(update={attr: value})
        if not has_errors(validate_line(trial, index)):
            found.add((attr, value))
    if len(found) != 1:
        return None
    attr, value = next(iter(found))
    old = getattr(line, attr)

    others = [
        getattr(line, name)
        for name in ("quantity", "price", *_COLUMNS)
        if name != attr and getattr(line, name) is not None
    ]
    before_sum = column_sum(doc.lines, attr)
    after_lines = [*doc.lines]
    after_lines[index] = line.model_copy(update={attr: value})
    after_sum = column_sum(after_lines, attr)
    total = getattr(doc.totals, attr)
    if total is not None and before_sum == total and after_sum != total:
        return None  # итог подтверждает прочитанное значение
    confirmed_by_total = total is not None and after_sum == total
    if len(others) < 3 and not confirmed_by_total:
        return None
    if _printed_in_source(old, sources):
        return None

    doc.lines[index] = after_lines[index]
    reason = (
        f"Строка {index + 1}: {_TITLES[attr]} {_fmt(old)} не сходится с остальными числами "
        f"строки; единственное значение, при котором строка верна, — {_fmt(value)}"
    )
    if confirmed_by_total:
        reason += "; итог по документу после этого сходится"
    return Correction(
        kind="autofix",
        field=f"lines[{index}].{attr}",
        before=None if old is None else str(old),
        after=str(value),
        rule="line_amount_single_field",
        reason=reason,
    )


def _fix_totals(doc: BaseDocument, sources: list[SourceText]) -> Correction | None:
    totals = doc.totals
    if not doc.lines or any(has_errors(validate_line(ln, i)) for i, ln in enumerate(doc.lines)):
        return None
    sums = {attr: column_sum(doc.lines, attr) for attr in _COLUMNS}
    current = {attr: getattr(totals, attr) for attr in _COLUMNS}
    if any(current[attr] is None or sums[attr] is None for attr in _COLUMNS):
        return None
    wrong = [attr for attr in _COLUMNS if current[attr] != sums[attr]]
    if len(wrong) != 1:
        return None  # ничего не расходится или расходится не один итог (пропущена строка?)
    attr = wrong[0]
    new = sums[attr]
    assert new is not None
    identity_before = (
        current["amount_without_vat"] + current["vat_amount"] == current["amount_with_vat"]  # type: ignore[operator]
    )
    fixed = {**current, attr: new}
    identity_after = fixed["amount_without_vat"] + fixed["vat_amount"] == fixed["amount_with_vat"]  # type: ignore[operator]
    if identity_before or not identity_after:
        return None
    if _printed_in_source(current[attr], sources):
        return None
    setattr(totals, attr, new)
    return Correction(
        kind="autofix",
        field=f"totals.{attr}",
        before=str(current[attr]),
        after=str(new),
        rule="total_equals_lines_sum",
        reason=(
            f"Итого: {_TITLES[attr]} {_fmt(current[attr])} не равно сумме по строкам "
            f"{_fmt(new)}, а остальные итоги и строки верны; после замены итоги сходятся"
        ),
    )


def autofix_document[D: BaseDocument](
    doc: D, sources: Sequence[SourceText] = ()
) -> tuple[D, list[Correction]]:
    """Копия документа с исправленными суммами и журнал исправлений."""
    result = doc.model_copy(deep=True)
    src = list(sources)
    out: list[Correction] = []
    for index in range(len(result.lines)):
        fix = _fix_line(result, index, src)
        if fix is not None:
            out.append(fix)
    total_fix = _fix_totals(result, src)
    if total_fix is not None:
        out.append(total_fix)
    return result, out
