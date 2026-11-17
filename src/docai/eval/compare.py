"""Сравнение одного предсказанного документа с эталоном по FIELD_SPEC (определения — D44).

Все значения сравниваются после `normalize_for_compare` (вид нормализации задан в FIELD_SPEC).
Набор полей берётся по типу ЭТАЛОНА: если модель ошиблась в типе, поля, которых у её
типа нет, считаются пустыми (то есть неверными).

Поля шапки:
- значение *верно*, если нормализованные значения равны (два пустых тоже равны);
- accuracy поля = верных / эталонных непустых; если эталон пуст, а модель что-то вернула,
  это *выдумка* (отдельная доля: выдумок / пустых в эталоне);
- ANLS — для полей с нормализацией TEXT и DOC_NUMBER.

Строки таблицы:
- строки предсказания и эталона сопоставляются венгерским алгоритмом по сходству строк:
  среднее по полям строки, где хотя бы одно значение непусто (текст — 1 − NL, остальное —
  1 при равенстве и 0 иначе); пара с сходством ниже порога (0,5) не считается сопоставленной;
- строка *найдена* — у неё есть пара; строка *верна* — есть пара и все её ячейки равны;
- ячейка: TP — непустая эталонная ячейка в сопоставленной паре с равным значением;
  precision = TP / непустых ячеек предсказания, recall = TP / непустых ячеек эталона.

Документ «можно загружать без правок» (главная бизнес-метрика): документ собран, все
обязательные для типа эталона поля шапки верны, строк столько же, сколько в эталоне,
каждая строка сопоставлена и все её обязательные ячейки верны.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any

from docai.eval.assign import linear_assignment
from docai.eval.text import anls_score, similarity
from docai.postprocess import FieldConfidence
from docai.schemas.common import BaseDocument
from docai.schemas.fields import FIELD_SPEC, FieldSpec, NormKind, fields_for
from docai.validation.normalize import normalize_for_compare

TEXT_NORMS = frozenset({NormKind.TEXT, NormKind.DOC_NUMBER})


def _get(obj: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def header_value(doc: BaseDocument | None, spec: FieldSpec) -> Any:
    if doc is None or not spec.applies_to(doc.doc_type):
        return None
    return _get(doc, spec.path)


def line_value(doc: BaseDocument | None, index: int, spec: FieldSpec) -> Any:
    if doc is None or not spec.applies_to(doc.doc_type) or index >= len(doc.lines):
        return None
    return _get(doc.lines[index], spec.path)


def norm(spec: FieldSpec, value: Any) -> Any:
    return normalize_for_compare(spec.norm, value)


def display(value: Any) -> str | None:
    """Нормализованное значение строкой (для отчёта)."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Decimal):
        return f"{value:f}"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@dataclass(slots=True)
class FieldResult:
    """Одно поле шапки."""

    field: str
    ref: str | None
    pred: str | None
    correct: bool
    required: bool
    anls: float | None = None  # только для текстовых полей с непустым эталоном


@dataclass(slots=True)
class CellResult:
    field: str
    ref: str | None
    pred: str | None
    correct: bool
    required: bool


@dataclass(slots=True)
class RowPair:
    ref_index: int
    pred_index: int
    similarity: float
    cells: list[CellResult]

    @property
    def exact(self) -> bool:
        return all(c.correct for c in self.cells)

    @property
    def required_ok(self) -> bool:
        return all(c.correct for c in self.cells if c.required)


@dataclass(slots=True)
class LineFieldCounts:
    ref: int = 0  # непустых эталонных ячеек
    pred: int = 0  # непустых ячеек предсказания
    correct: int = 0  # верных (в сопоставленных парах)
    anls: float = 0.0  # сумма ANLS по эталонным ячейкам (текстовые поля)


@dataclass(slots=True)
class ConfidencePoint:
    field: str
    confidence: float
    correct: bool


@dataclass(slots=True)
class DocComparison:
    doc_type: str
    pred_doc_type: str | None
    extracted: bool
    header: list[FieldResult]
    # Поля шапки, пустые в эталоне и применимые к его типу: (поле, что вернула модель).
    empty_ref: list[tuple[str, str | None]]
    n_ref_rows: int
    n_pred_rows: int
    pairs: list[RowPair]
    line_fields: dict[str, LineFieldCounts]
    required_errors: list[str]
    confidence: list[ConfidencePoint] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Можно загружать без правок: ни одной ошибки в обязательных полях."""
        return self.extracted and not self.required_errors

    @property
    def type_ok(self) -> bool:
        return self.pred_doc_type == self.doc_type

    @property
    def rows_matched(self) -> int:
        return len(self.pairs)

    @property
    def rows_exact(self) -> int:
        return sum(p.exact for p in self.pairs)

    @property
    def cells(self) -> LineFieldCounts:
        total = LineFieldCounts()
        for counts in self.line_fields.values():
            total.ref += counts.ref
            total.pred += counts.pred
            total.correct += counts.correct
        return total

    @property
    def header_correct(self) -> int:
        return sum(f.correct for f in self.header)

    @property
    def f1(self) -> float:
        """F1 по всем значениям документа (шапка и ячейки) — для поиска худших примеров."""
        cells = self.cells
        tp = self.header_correct + cells.correct
        n_ref = len(self.header) + cells.ref
        n_pred = sum(f.pred is not None for f in self.header) + cells.pred
        n_pred += sum(pred is not None for _, pred in self.empty_ref)
        return 2 * tp / (n_ref + n_pred) if n_ref + n_pred else 1.0


def _line_specs(doc_type: Any) -> list[FieldSpec]:
    return [s for s in fields_for(doc_type) if s.is_line_field]


def row_similarity(
    ref: BaseDocument, i: int, pred: BaseDocument, j: int, specs: list[FieldSpec]
) -> float:
    scores = []
    for spec in specs:
        r, p = norm(spec, line_value(ref, i, spec)), norm(spec, line_value(pred, j, spec))
        if r is None and p is None:
            continue
        if spec.norm in TEXT_NORMS:
            scores.append(similarity(display(p), display(r)))
        else:
            scores.append(1.0 if r == p else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def match_rows(
    ref: BaseDocument, pred: BaseDocument | None, threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Сопоставленные пары (строка эталона, строка предсказания, сходство) по венгерскому."""
    if pred is None or not ref.lines or not pred.lines:
        return []
    specs = _line_specs(ref.doc_type)
    sim = [
        [row_similarity(ref, i, pred, j, specs) for j in range(len(pred.lines))]
        for i in range(len(ref.lines))
    ]
    cost = [[1.0 - s for s in row] for row in sim]
    return [(i, j, sim[i][j]) for i, j in linear_assignment(cost) if sim[i][j] >= threshold]


def compare_documents(
    ref: BaseDocument,
    pred: BaseDocument | None,
    *,
    anls_threshold: float = 0.5,
    row_match_threshold: float = 0.5,
    confidences: list[FieldConfidence] | None = None,
) -> DocComparison:
    doc_type = ref.doc_type
    header: list[FieldResult] = []
    empty_ref: list[tuple[str, str | None]] = []
    required_errors: list[str] = []

    for spec in fields_for(doc_type):
        if spec.is_line_field:
            continue
        r, p = norm(spec, header_value(ref, spec)), norm(spec, header_value(pred, spec))
        correct = r == p
        required = spec.is_required(doc_type)
        if required and not correct:
            required_errors.append(spec.name)
        if r is None:
            empty_ref.append((spec.name, display(p)))
            continue
        score = None
        if spec.norm in TEXT_NORMS:
            score = anls_score(display(p), display(r), anls_threshold)
        header.append(FieldResult(spec.name, display(r), display(p), correct, required, score))

    specs = _line_specs(doc_type)
    line_fields = {s.name: LineFieldCounts() for s in specs}
    n_pred_rows = len(pred.lines) if pred is not None else 0
    for spec in specs:
        counts = line_fields[spec.name]
        counts.ref = sum(
            norm(spec, line_value(ref, i, spec)) is not None for i in range(len(ref.lines))
        )
        counts.pred = sum(
            norm(spec, line_value(pred, j, spec)) is not None for j in range(n_pred_rows)
        )

    pairs: list[RowPair] = []
    for i, j, sim in match_rows(ref, pred, row_match_threshold):
        cells = []
        for spec in specs:
            r, p = norm(spec, line_value(ref, i, spec)), norm(spec, line_value(pred, j, spec))
            correct = r == p
            cells.append(
                CellResult(spec.name, display(r), display(p), correct, spec.is_required(doc_type))
            )
            counts = line_fields[spec.name]
            if r is not None and correct:
                counts.correct += 1
            if r is not None and spec.norm in TEXT_NORMS:
                counts.anls += anls_score(display(p), display(r), anls_threshold)
        pairs.append(RowPair(i, j, round(sim, 4), cells))

    if pred is None:
        required_errors.append("документ не собран")
    elif len(ref.lines) != n_pred_rows:
        required_errors.append(f"строк {n_pred_rows} вместо {len(ref.lines)}")
    elif len(pairs) != len(ref.lines):
        required_errors.append(f"не сопоставлено строк: {len(ref.lines) - len(pairs)}")
    for pair in pairs:
        for cell in pair.cells:
            if cell.required and not cell.correct:
                required_errors.append(
                    f"lines[{pair.ref_index}].{cell.field.removeprefix('lines[].')}"
                )

    result = DocComparison(
        doc_type=str(doc_type),
        pred_doc_type=str(pred.doc_type) if pred is not None else None,
        extracted=pred is not None,
        header=header,
        empty_ref=empty_ref,
        n_ref_rows=len(ref.lines),
        n_pred_rows=n_pred_rows,
        pairs=pairs,
        line_fields=line_fields,
        required_errors=required_errors,
    )
    if confidences:
        result.confidence = confidence_points(ref, pred, pairs, confidences)
    return result


_SPECS = {s.name: s for s in FIELD_SPEC}


def confidence_points(
    ref: BaseDocument,
    pred: BaseDocument | None,
    pairs: list[RowPair],
    confidences: list[FieldConfidence],
) -> list[ConfidencePoint]:
    """Верно ли каждое поле, которому постобработка поставила уверенность.

    Поле строки предсказания `lines[j].x` верно, только если строка j сопоставлена со
    строкой эталона и значение в ней равно эталонному; несопоставленная строка — неверна.
    """
    ref_of = {p.pred_index: p.ref_index for p in pairs}
    out = []
    for fc in confidences:
        name = fc.field
        if name.startswith("lines["):
            index_text, _, attr = name[len("lines[") :].partition("].")
            spec = _SPECS.get(f"lines[].{attr}")
            if spec is None:
                continue
            j = int(index_text)
            i = ref_of.get(j)
            if i is None:
                correct = False
            else:
                correct = norm(spec, line_value(ref, i, spec)) == norm(
                    spec, line_value(pred, j, spec)
                )
        else:
            spec = _SPECS.get(name)
            if spec is None:
                continue
            correct = norm(spec, header_value(ref, spec)) == norm(spec, header_value(pred, spec))
        out.append(ConfidencePoint(name, fc.confidence, correct))
    return out
