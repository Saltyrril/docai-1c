"""Полная проверка документа: заполненность, реквизиты, ставки НДС, арифметика."""

from docai.schemas.common import BankDetails, BaseDocument, Party, VatRate
from docai.schemas.fields import FIELD_SPEC, get_field_value
from docai.validation.arithmetic import validate_line, validate_totals
from docai.validation.issues import IssueCode, Severity, ValidationIssue, issue
from docai.validation.requisites import (
    validate_account,
    validate_bik,
    validate_corr_account,
    validate_inn,
    validate_kpp,
)
from docai.validation.vat import validate_vat_rate


def _is_empty(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def validate_required(doc: BaseDocument) -> list[ValidationIssue]:
    """Обязательные для этого типа документа поля (по FIELD_SPEC) заполнены."""
    issues: list[ValidationIssue] = []
    for spec in FIELD_SPEC:
        if not spec.is_required(doc.doc_type):
            continue
        value = get_field_value(doc, spec)
        if spec.is_line_field:
            for index, item in enumerate(value):
                if _is_empty(item):
                    field = spec.name.replace("lines[]", f"lines[{index}]")
                    message = f"Строка {index + 1}: не заполнено поле «{spec.title_ru}»"
                    issues.append(issue(IssueCode.FIELD_MISSING, message, field))
        elif _is_empty(value):
            issues.append(
                issue(IssueCode.FIELD_MISSING, f"Не заполнено поле «{spec.title_ru}»", spec.name)
            )
    return issues


def validate_party(party: Party, role: str) -> list[ValidationIssue]:
    """ИНН и КПП стороны; КПП есть у организации и отсутствует у ИП."""
    issues: list[ValidationIssue] = []
    if party.inn:
        issues += validate_inn(party.inn, f"{role}.inn")
    if party.kpp:
        issues += validate_kpp(party.kpp, f"{role}.kpp")
    if party.inn and len(party.inn) == 12 and party.kpp:
        issues.append(
            issue(
                IssueCode.KPP_FOR_INDIVIDUAL,
                "У ИП (ИНН из 12 цифр) не бывает КПП",
                f"{role}.kpp",
            )
        )
    if party.inn and len(party.inn) == 10 and party.inn.isdigit() and not party.kpp:
        issues.append(
            issue(
                IssueCode.KPP_MISSING,
                "У организации (ИНН из 10 цифр) не указан КПП",
                f"{role}.kpp",
                Severity.WARNING,
            )
        )
    return issues


def validate_bank(bank: BankDetails, prefix: str = "seller_bank") -> list[ValidationIssue]:
    """БИК, расчётный и корреспондентский счёт (оба сверяются с БИК)."""
    issues: list[ValidationIssue] = []
    bik_field = f"{prefix}.bik"
    if bank.bik:
        issues += validate_bik(bank.bik, bik_field)
    if bank.account:
        issues += validate_account(bank.account, bank.bik or "", f"{prefix}.account", bik_field)
    if bank.corr_account:
        issues += validate_corr_account(
            bank.corr_account, bank.bik or "", f"{prefix}.corr_account", bik_field
        )
    return issues


def validate_vat_rates(doc: BaseDocument) -> list[ValidationIssue]:
    """Каждая ставка в строках действовала на дату документа (по разу на ставку)."""
    issues: list[ValidationIssue] = []
    seen: set[VatRate] = set()
    for index, line in enumerate(doc.lines):
        if line.vat_rate is None or line.vat_rate in seen:
            continue
        seen.add(line.vat_rate)
        issues += validate_vat_rate(
            line.vat_rate, doc.date, doc.doc_type, f"lines[{index}].vat_rate"
        )
    return issues


def validate_document(doc: BaseDocument) -> list[ValidationIssue]:
    """Все проверки документа. Пустой список — ошибок и предупреждений нет."""
    issues = validate_required(doc)
    issues += validate_party(doc.seller, "seller")
    issues += validate_party(doc.buyer, "buyer")
    bank = getattr(doc, "seller_bank", None)
    if bank is not None:
        issues += validate_bank(bank)
    issues += validate_vat_rates(doc)
    for index, line in enumerate(doc.lines):
        issues += validate_line(line, index)
    issues += validate_totals(doc.lines, doc.totals)
    return issues
