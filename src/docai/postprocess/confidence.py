"""Уверенность по каждому полю (0–1): формула в D42.

    уверенность = σ( logit(p₀) + Σᵢ ln LRᵢ ),   σ(x) = 1 / (1 + e^−x)

p₀ — доля верно прочитанных полей этой группы на этом пути извлечения (до проверок).
LRᵢ — отношение правдоподобия сигнала: во сколько раз он вероятнее при верном значении,
чем при неверном. Больше 1 — свидетельство «верно», меньше 1 — «неверно». Сигналы:
проверки (прошла / не прошла), дословное вхождение в текст, совпадение с другим путём.
Сигналы считаются независимыми (наивный байесовский вывод), поэтому «пройденные»
проверки ограничены общим потолком, а исправленные поля — потолком уверенности.
"""

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum

from docai.postprocess.autofix import column_sum
from docai.postprocess.config import PostprocessConfig
from docai.postprocess.evidence import SourceText, verbatim
from docai.postprocess.models import FieldConfidence
from docai.schemas.common import BaseDocument
from docai.schemas.fields import FIELD_SPEC, FieldSpec, NormKind, get_field_value
from docai.validation import IssueCode, Severity, ValidationIssue
from docai.validation.arithmetic import ZERO
from docai.validation.normalize import normalize_for_compare

_LINE_RE = re.compile(r"^lines\[(\d+)\]\.(\w+)$")
_AMOUNTS = ("amount_without_vat", "vat_amount", "amount_with_vat")
_C = IssueCode
_TOTAL_CODES = {
    "amount_without_vat": _C.TOTAL_WITHOUT_VAT_MISMATCH,
    "vat_amount": _C.TOTAL_VAT_MISMATCH,
    "amount_with_vat": _C.TOTAL_WITH_VAT_MISMATCH,
}
_KIND_BY_NORM = {
    NormKind.INN: "id",
    NormKind.KPP: "id",
    NormKind.DIGITS: "id",
    NormKind.OKEI: "code",
    NormKind.MONEY: "money",
    NormKind.QUANTITY: "money",
    NormKind.DATE: "date",
    NormKind.TEXT: "text",
    NormKind.DOC_NUMBER: "text",
    NormKind.EXACT: "enum",
    NormKind.VAT_RATE: "enum",
}


class Link(Enum):
    NAMED = "named"  # поле названо в ошибке
    INVOLVED = "involved"  # поле входит в сверяемое равенство


def kind_of(spec: FieldSpec) -> str:
    return _KIND_BY_NORM[spec.norm]


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _line_fields(index: int, *names: str) -> list[str]:
    return [f"lines[{index}].{name}" for name in names]


def involved_fields(issue: ValidationIssue, n_lines: int) -> list[str]:
    """Поля, которые участвуют в равенстве, нарушенном ошибкой, кроме названного в ней."""
    match = _LINE_RE.match(issue.field or "")
    index = int(match[1]) if match else None
    code = issue.code
    all_lines = range(n_lines)
    if index is not None:
        if code is _C.LINE_AMOUNT_MISMATCH:
            return _line_fields(index, "quantity", "price", *_AMOUNTS[::2])
        if code is _C.LINE_VAT_MISMATCH:
            return _line_fields(
                index, "amount_without_vat", "amount_with_vat", "vat_rate", "vat_amount"
            )
        if code in (_C.LINE_VAT_FOR_NO_VAT,):
            return _line_fields(index, "vat_rate", "vat_amount")
        if code is _C.LINE_SUM_MISMATCH:
            return _line_fields(index, *_AMOUNTS)
        if code in (_C.VAT_RATE_NOT_IN_FORCE, _C.VAT_RATE_BEFORE_START, _C.VAT_DATE_UNSUPPORTED):
            return ["date"]
    for attr, total_code in _TOTAL_CODES.items():
        if code is total_code:
            return [name for i in all_lines for name in _line_fields(i, attr)]
    if code is _C.TOTAL_SUM_MISMATCH:
        return [f"totals.{a}" for a in _AMOUNTS]
    if code is _C.TOTAL_VAT_RATE_MISMATCH:
        return [
            "totals.amount_without_vat",
            *(n for i in all_lines for n in _line_fields(i, "vat_rate")),
        ]
    if code in (_C.ACCOUNT_KEY, _C.ACCOUNT_FORMAT):
        return ["seller_bank.bik"]
    if code in (_C.CORR_ACCOUNT_KEY, _C.CORR_ACCOUNT_BIK_MISMATCH, _C.CORR_ACCOUNT_PREFIX):
        return ["seller_bank.bik"]
    if code in (_C.KPP_FOR_INDIVIDUAL, _C.KPP_MISSING) and issue.field:
        return [issue.field.replace(".kpp", ".inn")]
    return []


def _link_map(
    issues: Sequence[ValidationIssue], n_lines: int
) -> dict[str, list[tuple[Link, ValidationIssue]]]:
    links: dict[str, list[tuple[Link, ValidationIssue]]] = defaultdict(list)
    for item in issues:
        if item.field:
            links[item.field].append((Link.NAMED, item))
        for name in involved_fields(item, n_lines):
            links[name].append((Link.INVOLVED, item))
    return links


class _Facts:
    """Какие проверки сработали: нужны, чтобы начислить «пройденные» проверки полю."""

    def __init__(self, doc: BaseDocument, issues: Sequence[ValidationIssue]) -> None:
        self.doc = doc
        self.codes_by_line: dict[int, set[IssueCode]] = defaultdict(set)
        self.codes: set[IssueCode] = set()
        for item in issues:
            self.codes.add(item.code)
            match = _LINE_RE.match(item.field or "")
            if match:
                self.codes_by_line[int(match[1])].add(item.code)
        self.has_vat_issue = bool(
            self.codes
            & {_C.VAT_RATE_NOT_IN_FORCE, _C.VAT_RATE_BEFORE_START, _C.VAT_DATE_UNSUPPORTED}
        )

    def line_ok(self, index: int, code: IssueCode) -> bool:
        return code not in self.codes_by_line[index]

    def product(self, index: int) -> bool:
        ln = self.doc.lines[index]
        has_amount = ln.amount_without_vat is not None or ln.amount_with_vat is not None
        return (
            ln.quantity is not None
            and ln.price is not None
            and has_amount
            and self.line_ok(index, _C.LINE_AMOUNT_MISMATCH)
        )

    def vat_relation(self, index: int) -> bool:
        ln = self.doc.lines[index]
        base = ln.amount_without_vat is not None or ln.amount_with_vat is not None
        return (
            ln.vat_rate is not None
            and not ln.vat_rate.is_untaxed
            and ln.vat_amount is not None
            and base
            and self.line_ok(index, _C.LINE_VAT_MISMATCH)
        )

    def line_sum(self, index: int) -> bool:
        ln = self.doc.lines[index]
        vat_known = ln.vat_amount is not None or (
            ln.vat_rate is not None and ln.vat_rate.is_untaxed
        )
        return (
            ln.amount_without_vat is not None
            and ln.amount_with_vat is not None
            and vat_known
            and self.line_ok(index, _C.LINE_SUM_MISMATCH)
        )

    def column_sum(self, attr: str) -> bool:
        total = getattr(self.doc.totals, attr)
        return (
            total is not None
            and column_sum(self.doc.lines, attr) is not None
            and _TOTAL_CODES[attr] not in self.codes
        )

    def identity(self) -> bool:
        t = self.doc.totals
        return (
            t.amount_without_vat is not None
            and t.vat_amount is not None
            and t.amount_with_vat is not None
            and _C.TOTAL_SUM_MISMATCH not in self.codes
        )


def _validation_passes(
    name: str, doc: BaseDocument, facts: _Facts, cfg: PostprocessConfig
) -> list[tuple[str, float]]:
    """Пройденные проверки поля: (описание, LR). Только для полей без ошибок."""
    line = _LINE_RE.match(name)
    if line:
        i, attr = int(line[1]), line[2]
        out: list[tuple[str, float]] = []
        if attr in ("quantity", "price", "amount_without_vat") and facts.product(i):
            out.append(("количество × цена = сумма", cfg.lr_product))
        if attr in ("amount_without_vat", "vat_amount", "vat_rate"):
            ln = doc.lines[i]
            if facts.vat_relation(i):
                out.append(("НДС = база × ставка", cfg.lr_vat_rate))
            elif attr == "vat_rate" and ln.vat_rate is not None and ln.vat_rate.is_untaxed:
                out.append(("ставка без налога", cfg.lr_untaxed_rate))
        if attr in _AMOUNTS and facts.line_sum(i):
            out.append(("без НДС + НДС = с НДС", cfg.lr_line_sum))
        if attr in _AMOUNTS and facts.column_sum(attr):
            out.append(("сумма по строкам = итог", cfg.lr_column_sum))
        return out
    out = []
    match name:
        case "seller.inn" | "buyer.inn":
            inn = getattr(doc, name.split(".")[0]).inn or ""
            out.append(("контрольные цифры ИНН", cfg.lr_inn12 if len(inn) == 12 else cfg.lr_inn10))
        case "seller.kpp" | "buyer.kpp":
            out.append(("формат КПП", cfg.lr_kpp))
        case "seller_bank.bik":
            bank = getattr(doc, "seller_bank", None)
            coupled = bank is not None and (bank.account or bank.corr_account)
            out.append(("БИК и ключи счетов", cfg.lr_key if coupled else cfg.lr_bik_weak))
        case "seller_bank.account" | "seller_bank.corr_account":
            bank = getattr(doc, "seller_bank", None)
            if bank is not None and bank.bik:
                out.append(("ключ счёта по БИК", cfg.lr_key))
        case "date":
            if any(ln.vat_rate is not None for ln in doc.lines) and not facts.has_vat_issue:
                out.append(("ставки действовали на дату", cfg.lr_date_vat))
        case "totals.amount_without_vat" | "totals.vat_amount" | "totals.amount_with_vat":
            attr = name.split(".")[1]
            if facts.column_sum(attr):
                out.append(("итог = сумма по строкам", cfg.lr_total_sum))
            if facts.identity():
                out.append(("итоги согласованы", cfg.lr_total_identity))
    return out


def _field_instances(doc: BaseDocument):
    """(путь, спецификация, значение) для каждого поля документа, включая строки."""
    for spec in FIELD_SPEC:
        if not spec.applies_to(doc.doc_type):
            continue
        value = get_field_value(doc, spec)
        if spec.is_line_field:
            for index, item in enumerate(value):
                yield spec.name.replace("lines[]", f"lines[{index}]"), spec, item
        else:
            yield spec.name, spec, value


def _value_of(path: str, doc: BaseDocument) -> object:
    for name, _, value in _field_instances(doc):
        if name == path:
            return value
    return None


def _empty(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _same(spec: FieldSpec, a: object, b: object) -> bool | None:
    """Совпадают ли значения по нормализованному виду; None, если сравнить нечего."""
    if _empty(b):
        return None
    return normalize_for_compare(spec.norm, a) == normalize_for_compare(spec.norm, b)


def score_fields(
    doc: BaseDocument,
    issues: Sequence[ValidationIssue],
    *,
    route: str,
    sources: Sequence[SourceText] = (),
    alternates: Sequence[BaseDocument] = (),
    corrected: frozenset[str] = frozenset(),
    config: PostprocessConfig | None = None,
) -> list[FieldConfidence]:
    """Уверенность во всех полях документа; отсутствующее обязательное поле — 0."""
    cfg = config or PostprocessConfig()
    facts = _Facts(doc, issues)
    links = _link_map(issues, len(doc.lines))
    priors = cfg.priors.get(route) or cfg.priors["vision"]
    alt_values = [{n: v for n, _, v in _field_instances(alt)} for alt in alternates]
    alt_lines = [len(alt.lines) == len(doc.lines) for alt in alternates]
    out: list[FieldConfidence] = []

    for path, spec, value in _field_instances(doc):
        if _empty(value):
            if spec.is_required(doc.doc_type):
                out.append(
                    FieldConfidence(
                        field=path, value=None, confidence=0.0, reasons=["не найдено в документе"]
                    )
                )
            continue
        signals: dict[str, float] = defaultdict(lambda: 1.0)
        reasons: list[str] = []
        prior = priors[kind_of(spec)]
        log_odds = logit(min(max(prior, 0.001), 0.999))
        reasons.append(f"исходная вероятность {prior:.2f} ({route}, {kind_of(spec)})")

        failed = False
        for link, item in links.get(path, []):
            if item.severity is Severity.ERROR:
                lr = cfg.lr_error_named if link is Link.NAMED else cfg.lr_error_involved
                failed = True
            else:
                lr = cfg.lr_warning
            signals["validation"] *= lr
            reasons.append(f"{item.code.value}: {item.message}")
        if not failed:
            boost = 1.0
            for text, lr in _validation_passes(path, doc, facts, cfg):
                boost *= lr
                reasons.append(f"проверка пройдена: {text} (×{lr:g})")
            if boost > cfg.validation_boost_cap:
                reasons.append(f"потолок пройденных проверок ×{cfg.validation_boost_cap:g}")
                boost = cfg.validation_boost_cap
            signals["validation"] *= boost

        if sources:
            found = verbatim(spec.norm, value, list(sources))
            if found is not None:
                is_found, strong = found
                if is_found:
                    lr = cfg.lr_found_strong if strong else cfg.lr_found_weak
                    reasons.append(f"значение дословно есть в тексте (×{lr:g})")
                elif spec.norm is NormKind.OKEI:
                    # Код ОКЕИ во многих формах не печатается: его отсутствие в тексте не улика.
                    lr = 1.0
                elif not strong:
                    lr = cfg.lr_missing_weak
                    reasons.append(f"короткое значение не найдено в тексте (×{lr:g})")
                else:
                    layer = any(s.kind == "text_layer" for s in sources)
                    lr = cfg.lr_missing_text_layer if layer else cfg.lr_missing_ocr
                    reasons.append(f"значения нет в тексте (×{lr:g})")
                if lr != 1.0:
                    signals["verbatim"] *= lr

        for values, same_len in zip(alt_values, alt_lines, strict=True):
            if spec.is_line_field and not same_len:
                continue
            verdict = _same(spec, value, values.get(path))
            if verdict is None:
                continue
            lr = cfg.lr_cross_agree if verdict else cfg.lr_cross_disagree
            reasons.append(
                f"другой путь {'совпал' if verdict else 'дал другое значение'} (×{lr:g})"
            )
            signals["cross_path"] *= lr

        for lr in signals.values():
            log_odds += math.log(lr)
        confidence = sigmoid(log_odds)
        is_corrected = path in corrected
        if is_corrected:
            confidence = min(confidence, cfg.corrected_cap)
            reasons.append(f"исправлено автоматически, потолок {cfg.corrected_cap}")
        confidence = min(max(confidence, cfg.prob_min), cfg.prob_max)
        out.append(
            FieldConfidence(
                field=path,
                value=_show(value),
                confidence=round(confidence, 4),
                signals={k: round(v, 4) for k, v in signals.items() if v != 1.0},
                reasons=reasons,
                corrected=is_corrected,
            )
        )
    return out


def _show(value: object) -> str:
    return str(getattr(value, "value", value))


__all__ = ["ZERO", "Link", "involved_fields", "kind_of", "score_fields"]
