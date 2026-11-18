"""Анализ ошибок извлечения (P13): каждое неверное значение — к одной причине.

Вход — сравнение документа с эталоном (`DocComparison`, после постобработки), сам эталон
и текст, который видела модель (текстовый слой PDF или OCR). Каждое неверное значение шапки
или ячейки таблицы и каждое расхождение в числе строк получает одну причину; причины
проверяются в порядке списка `CAUSES`, первая подошедшая побеждает. Порядок важен: например,
ИНН, равный почтовому индексу из адреса, — это «значение из соседнего поля», а не «ошибка
в цифрах».

Для каждой причины считаются доля среди всех неверных значений, доля среди блокирующих
(обязательные поля — из-за них документ не загружается без правок) и «потолок эффекта» —
сколько непрошедших документов прошли бы, если бы исчезли все ошибки этой причины и только
они. Потолок — верхняя оценка пользы исправления, по нему выбираются изменения (D49).
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from docai.eval.compare import DocComparison, display, header_value, line_value, norm
from docai.eval.text import similarity
from docai.schemas.common import BaseDocument
from docai.schemas.fields import FIELDS_BY_NAME, NormKind, fields_for

# Причины: код → описание (порядок — порядок проверки и вывода).
CAUSES: dict[str, str] = {
    "doc_type": "неверный тип документа (и поля, которые от него зависят)",
    "rows_lost": "потеряны строки таблицы",
    "rows_extra": "лишние строки таблицы (разбита или повторена строка)",
    "prompt_leak": "значение из примера в промпте",
    "missing": "поле не найдено (null при значении в документе)",
    "hallucinated": "выдуманное значение (нет ни в документе, ни в эталоне поля)",
    "party_swap": "перепутаны продавец и покупатель",
    "decimal_lost": "потеряна дробная часть числа (десятичная запятая)",
    "column_shift": "значение из соседней графы той же строки",
    "neighbour_field": "значение из соседнего поля (индекс вместо ИНН, корр. счёт вместо р/с)",
    "date": "ошибка даты",
    "digits": "ошибка в цифрах (число есть в документе, но не то)",
    "text_postcode": "адрес без почтового индекса",
    "text_marks": "в значение попали пометки формы («(2а)», «ИНН», «КПП»)",
    "text_other": "неточная запись текста",
    "rows_unmatched": "строка таблицы не сопоставлена (искажена почти целиком)",
    "autofix": "постобработка испортила верный ответ модели",
    "not_extracted": "документ не собран",
    "other": "прочее",
}

# Значения из примера в промптах text_v1/v2 и vision_v1/v2 («ИНН/КПП 7707083893/773601001»).
PROMPT_EXAMPLE_VALUES = frozenset({"7707083893", "773601001"})

_DIGIT_NORMS = frozenset(
    {NormKind.INN, NormKind.KPP, NormKind.DIGITS, NormKind.OKEI, NormKind.DOC_NUMBER}
)
_NUMBER_NORMS = frozenset({NormKind.MONEY, NormKind.QUANTITY})
_NUMBER_RE = re.compile(r"\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
_POSTCODE_RE = re.compile(r"^\d{6},\s*")
_MARKS_RE = re.compile(r"\(\d+[абвaб]?\)|\bинн\b|\bкпп\b", re.IGNORECASE)


@dataclass(slots=True)
class ErrorItem:
    """Одно неверное значение (или одно расхождение числа строк)."""

    item_id: str
    field: str  # имя поля по FIELD_SPEC; для строк — "lines" (число строк)
    ref: str | None
    pred: str | None
    required: bool
    cause: str
    row: int | None = None  # номер строки эталона для ячеек


@dataclass(slots=True)
class SourceIndex:
    """Что есть в тексте, который видела модель: числа и цифровые последовательности."""

    numbers: frozenset[Decimal]
    digit_runs: frozenset[str]
    text: str

    @classmethod
    def build(cls, text: str) -> "SourceIndex":
        numbers = set()
        for match in _NUMBER_RE.finditer(text):
            raw = match.group().replace(" ", "").replace(" ", "").replace(",", ".")
            try:
                numbers.add(Decimal(raw))
            except InvalidOperation:
                continue
        runs = frozenset(re.findall(r"\d+", text))
        return cls(frozenset(numbers), runs, text.casefold())

    def has_number(self, value: str) -> bool:
        try:
            return Decimal(value) in self.numbers
        except InvalidOperation:
            return False

    def has_digits(self, value: str) -> bool:
        digits = re.sub(r"\D", "", value)
        return bool(digits) and any(digits in run for run in self.digit_runs)

    def has_text(self, value: str) -> bool:
        return value.casefold() in self.text


def _hamming_close(a: str, b: str, limit: int = 2) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) <= limit


def _doc_values(truth: BaseDocument) -> dict[str, str]:
    """Все непустые значения эталона: поле → нормализованное значение строкой."""
    out: dict[str, str] = {}
    for spec in fields_for(truth.doc_type):
        if spec.is_line_field:
            for i in range(len(truth.lines)):
                v = display(norm(spec, line_value(truth, i, spec)))
                if v is not None:
                    out[f"{spec.name}#{i}"] = v
        else:
            v = display(norm(spec, header_value(truth, spec)))
            if v is not None:
                out[spec.name] = v
    return out


def _counterpart(name: str) -> str | None:
    if name.startswith("seller."):
        return "buyer." + name.removeprefix("seller.")
    if name.startswith("buyer."):
        return "seller." + name.removeprefix("buyer.")
    return None


def _same(a: str | None, b: str | None) -> bool:
    """Равны как строки или как числа («25» и «25.00»)."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return False


def _is_fraction_lost(ref: str, pred: str) -> bool:
    try:
        r, p = Decimal(ref), Decimal(pred)
    except InvalidOperation:
        return False
    return r != r.to_integral_value() and p == r.to_integral_value(rounding="ROUND_DOWN")


def classify_value(
    name: str,
    ref: str | None,
    pred: str | None,
    *,
    source: SourceIndex,
    doc_values: dict[str, str],
    row_values: dict[str, str] | None = None,
    type_wrong: bool = False,
) -> str:
    """Причина одного неверного значения (`ref` и `pred` — нормализованные строки)."""
    spec = FIELDS_BY_NAME[name]
    if name in ("doc_type", "upd_status") or (type_wrong and pred is None):
        return "doc_type"
    if pred is not None and re.sub(r"\D", "", pred) in PROMPT_EXAMPLE_VALUES:
        return "prompt_leak"
    if pred is None:
        return "missing"
    other = _counterpart(name)
    if other is not None and doc_values.get(other) == pred:
        return "party_swap"
    if spec.norm in _NUMBER_NORMS and ref is not None and _is_fraction_lost(ref, pred):
        return "decimal_lost"
    if row_values is not None:
        # Номер строки («№ п/п») тоже графа той же строки, но в схеме его нет.
        row_number = row_values.get("#")
        if any(_same(v, pred) for k, v in row_values.items() if k not in (name, "#")) or (
            spec.norm in _NUMBER_NORMS and _same(row_number, pred) and not _same(ref, pred)
        ):
            return "column_shift"
    if spec.norm is NormKind.DATE:
        return "date"
    neighbours = [v for k, v in doc_values.items() if k != name and not k.startswith("lines[]")]
    if spec.norm in _DIGIT_NORMS | _NUMBER_NORMS and len(re.sub(r"\D", "", pred)) >= 4:
        digits = re.sub(r"\D", "", pred)
        if any(v == pred or digits in re.sub(r"\D", "", v) for v in neighbours):
            return "neighbour_field"
    if ref is None:
        # В эталоне пусто, а модель что-то вернула.
        found = source.has_text(pred) or (spec.norm in _NUMBER_NORMS and source.has_number(pred))
        return "other" if found else "hallucinated"
    if spec.norm in _NUMBER_NORMS:
        if not source.has_number(pred):
            return "hallucinated"
        return "digits"
    if spec.norm in _DIGIT_NORMS:
        if not source.has_digits(pred):
            return "digits" if _hamming_close(ref, pred) else "hallucinated"
        return "digits"
    if spec.norm is NormKind.TEXT:
        if ref in pred and any(v in pred for v in neighbours if len(v) >= 4 and v != ref):
            return "neighbour_field"
        if _POSTCODE_RE.match(ref) and _POSTCODE_RE.sub("", ref) == pred:
            return "text_postcode"
        if _MARKS_RE.search(pred) and not _MARKS_RE.search(ref):
            return "text_marks"
        if similarity(pred, ref) >= 0.5:
            return "text_other"
        return "other" if source.has_text(pred) else "hallucinated"
    return "other"


def _raw_values(raw: DocComparison | None) -> tuple[dict[str, str | None], dict[tuple, str | None]]:
    """Ответ модели до постобработки: поле шапки → значение, (строка, поле) → значение ячейки."""
    if raw is None:
        return {}, {}
    header = {f.field: f.pred for f in raw.header} | dict(raw.empty_ref)
    cells = {(p.pred_index, c.field): c.pred for p in raw.pairs for c in p.cells}
    return header, cells


def classify_document(
    item_id: str,
    cmp: DocComparison,
    truth: BaseDocument,
    source_text: str,
    raw: DocComparison | None = None,
) -> list[ErrorItem]:
    """Все неверные значения документа (после постобработки) с причинами.

    `raw` — сравнение ответа модели до постобработки. Если постобработка заменила значение
    (автоисправление суммы), причина ищется по значению модели: исправление выведено из него.
    Если модель ответила верно, а постобработка испортила — причина `autofix`.
    """
    source = SourceIndex.build(source_text)
    values = _doc_values(truth)
    header_values = {k: v for k, v in values.items() if "#" not in k}
    type_wrong = cmp.extracted and not cmp.type_ok
    raw_header, raw_cells = _raw_values(raw)
    out: list[ErrorItem] = []

    def add(name, ref, pred, required, row=None, row_values=None, raw_key=None) -> None:
        model_pred = pred
        if raw is not None:
            if raw_key is not None and raw_key in raw_cells:
                model_pred = raw_cells[raw_key]
            elif raw_key is None and name in raw_header:
                model_pred = raw_header[name]
        if model_pred != pred and _same(model_pred, ref):
            cause = "autofix"
        else:
            cause = classify_value(
                name,
                ref,
                model_pred,
                source=source,
                doc_values=header_values,
                row_values=row_values,
                type_wrong=type_wrong,
            )
        out.append(ErrorItem(item_id, name, ref, pred, required, cause, row))

    if not cmp.extracted:
        out.append(ErrorItem(item_id, "document", None, None, True, "not_extracted"))
        return out
    for f in cmp.header:
        if not f.correct:
            add(f.field, f.ref, f.pred, f.required)
    for name, pred in cmp.empty_ref:
        if pred is not None:
            add(name, None, pred, FIELDS_BY_NAME[name].is_required(truth.doc_type))
    for pair in cmp.pairs:
        row_values: dict[str, str | None] = {c.field: c.ref for c in pair.cells}
        row_values["#"] = str(pair.ref_index + 1)
        for c in pair.cells:
            if not c.correct:
                key = (pair.pred_index, c.field)
                add(c.field, c.ref, c.pred, c.required, pair.ref_index, row_values, key)
    if cmp.n_pred_rows != cmp.n_ref_rows:
        cause = "rows_lost" if cmp.n_pred_rows < cmp.n_ref_rows else "rows_extra"
        ref_rows, pred_rows = str(cmp.n_ref_rows), str(cmp.n_pred_rows)
        out.append(ErrorItem(item_id, "lines", ref_rows, pred_rows, True, cause))
    elif cmp.rows_matched < cmp.n_ref_rows:
        missing = str(cmp.n_ref_rows - cmp.rows_matched)
        out.append(ErrorItem(item_id, "lines", missing, None, True, "rows_unmatched"))
    return out


@dataclass(slots=True)
class CauseSummary:
    cause: str
    title: str
    errors: int  # неверных значений с этой причиной
    share_all: float  # доля среди всех неверных значений
    blocking: int  # из них в обязательных полях
    share_blocking: float  # доля среди всех блокирующих
    docs: int  # документов, где причина есть среди блокирующих
    docs_share_failed: float  # доля среди непрошедших документов
    ceiling_docs: int  # непрошедших документов, у которых все блокирующие — только эта причина
    examples: list[dict[str, Any]] = field(default_factory=list)


def summarize(
    errors: Sequence[ErrorItem], doc_ids: Iterable[str], ok_ids: Iterable[str], n_examples: int = 3
) -> dict[str, Any]:
    """Сводка по причинам: доли, потолок эффекта и примеры (по одному на документ)."""
    doc_ids = list(doc_ids)
    ok = set(ok_ids)
    failed = [d for d in doc_ids if d not in ok]
    blocking = [e for e in errors if e.required]
    by_doc: dict[str, set[str]] = defaultdict(set)
    for e in blocking:
        by_doc[e.item_id].add(e.cause)
    all_counts = Counter(e.cause for e in errors)
    block_counts = Counter(e.cause for e in blocking)
    out: list[CauseSummary] = []
    for cause, title in CAUSES.items():
        if not all_counts[cause]:
            continue
        docs = sum(cause in by_doc[d] for d in failed)
        ceiling = sum(by_doc[d] == {cause} for d in failed)
        examples: list[dict[str, Any]] = []
        seen: set[str] = set()
        for e in sorted(
            (e for e in errors if e.cause == cause), key=lambda e: (not e.required, e.item_id)
        ):
            if e.item_id in seen:
                continue
            seen.add(e.item_id)
            examples.append(asdict(e))
            if len(examples) == n_examples:
                break
        out.append(
            CauseSummary(
                cause=cause,
                title=title,
                errors=all_counts[cause],
                share_all=all_counts[cause] / len(errors) if errors else 0.0,
                blocking=block_counts[cause],
                share_blocking=block_counts[cause] / len(blocking) if blocking else 0.0,
                docs=docs,
                docs_share_failed=docs / len(failed) if failed else 0.0,
                ceiling_docs=ceiling,
                examples=examples,
            )
        )
    out.sort(key=lambda s: (-s.blocking, -s.errors))
    return {
        "documents": len(doc_ids),
        "ok": len(ok & set(doc_ids)),
        "failed": len(failed),
        "errors": len(errors),
        "blocking_errors": len(blocking),
        "causes": [asdict(s) for s in out],
    }


# ---- прогон анализа по кэшу и отчёт -------------------------------------------------------


def source_text(prediction: Any) -> str:
    """Текст, который видела модель (текстовый слой или OCR); для vision — пусто."""
    raw = getattr(prediction, "raw", None)
    if raw is None:
        return ""
    return "\n".join(text for _, text in raw.texts)


def analyze_records(records: Sequence[Any], predictions: dict[str, Any]) -> dict[str, Any]:
    """Причины ошибок по записям оценки (`DocRecord`) и кэшу ответов модели."""
    errors: list[ErrorItem] = []
    for r in records:
        if r.truth is None:
            continue
        text = source_text(predictions.get(r.item_id))
        errors += classify_document(r.item_id, r.comparison, r.truth, text, r.raw)
    summary = summarize(
        errors, [r.item_id for r in records], [r.item_id for r in records if r.comparison.ok]
    )
    by_doc: dict[str, set[str]] = defaultdict(set)
    for e in errors:
        if e.required:
            by_doc[e.item_id].add(e.cause)
    combos = Counter(" + ".join(sorted(by_doc[r.item_id])) for r in records if not r.comparison.ok)
    summary["cause_sets"] = [{"causes": k, "docs": v} for k, v in combos.most_common()]
    summary["by_doc_type"] = _by_doc_type(records, errors)
    return summary


def _by_doc_type(records: Sequence[Any], errors: Sequence[ErrorItem]) -> dict[str, Any]:
    types = {r.item_id: r.doc_type for r in records}
    out: dict[str, Any] = {}
    for doc_type in sorted(set(types.values())):
        ids = [r.item_id for r in records if r.doc_type == doc_type]
        ok = sum(r.comparison.ok for r in records if r.doc_type == doc_type)
        blocking = Counter(e.cause for e in errors if e.required and types[e.item_id] == doc_type)
        out[doc_type] = {"documents": len(ids), "ok": ok, "blocking": dict(blocking.most_common())}
    return out


def analyze_groups(
    records: Sequence[Any], predictions: dict[str, Any], groups: dict[str, set[str]]
) -> dict[str, int]:
    """Потолок эффекта для наборов причин: сколько непрошедших документов прошли бы,
    если бы исчезли все блокирующие ошибки этих причин."""
    by_doc: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.truth is None:
            continue
        text = source_text(predictions.get(r.item_id))
        for e in classify_document(r.item_id, r.comparison, r.truth, text, r.raw):
            if e.required:
                by_doc[r.item_id].add(e.cause)
    failed = [r.item_id for r in records if not r.comparison.ok]
    return {name: sum(by_doc[d] <= causes for d in failed) for name, causes in groups.items()}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def _value(v: Any) -> str:
    if v is None:
        return "null"
    text = str(v)
    return text if len(text) <= 60 else text[:57] + "…"


def render_errors_md(summary: dict[str, Any], title: str, intro: str = "") -> str:
    s = summary
    lines = [f"# {title}", ""]
    if intro:
        lines += [intro, ""]
    lines += [
        f"Документов: **{s['documents']}**, без правок: **{s['ok']}**, не прошли: "
        f"**{s['failed']}**. Неверных значений: {s['errors']}, из них в обязательных полях "
        f"(блокирующих): {s['blocking_errors']}.",
        "",
        "Причина назначается каждому неверному значению (поле шапки, ячейка таблицы, число "
        "строк) по правилам `docai.eval.errors` (D49). «Документов» — непрошедшие документы, "
        "где причина есть среди блокирующих; «потолок» — сколько из них прошли бы, если бы "
        "исчезла только эта причина.",
        "",
        "| Причина | Блокирующих | Доля блок. | Всех ошибок | Доля всех | Документов (доля "
        "непрошедших) | Потолок |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in s["causes"]:
        lines.append(
            f"| {c['title']} (`{c['cause']}`) | {c['blocking']} | {_pct(c['share_blocking'])} | "
            f"{c['errors']} | {_pct(c['share_all'])} | {c['docs']} ({_pct(c['docs_share_failed'])})"
            f" | {c['ceiling_docs']} |"
        )
    lines += ["", "## Примеры (по 3 на причину, сначала блокирующие)", ""]
    for c in s["causes"]:
        lines += [f"**{c['title']}** (`{c['cause']}`)", ""]
        lines += ["| Документ | Поле | Эталон | Ответ |", "|---|---|---|---|"]
        for e in c["examples"]:
            name = e["field"] if e["row"] is None else f"{e['field']} (строка {e['row'] + 1})"
            mark = "" if e["required"] else " (необяз.)"
            lines.append(
                f"| {e['item_id']} | `{name}`{mark} | {_value(e['ref'])} | {_value(e['pred'])} |"
            )
        lines.append("")
    lines += ["## Сочетания причин у непрошедших документов", "",
              "| Причины | Документов |", "|---|---|"]  # fmt: skip
    for row in s["cause_sets"][:15]:
        lines.append(f"| {row['causes']} | {row['docs']} |")
    lines += ["", "## По типам документов (блокирующие ошибки)", "",
              "| Тип | Док. | Без правок | Главные причины |", "|---|---|---|---|"]  # fmt: skip
    for doc_type, v in s["by_doc_type"].items():
        top = ", ".join(f"{k} {n}" for k, n in list(v["blocking"].items())[:4]) or "—"
        lines.append(f"| {doc_type} | {v['documents']} | {v['ok']} | {top} |")
    return "\n".join(lines) + "\n"
