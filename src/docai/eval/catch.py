"""Какую долю намеренных ошибок ловят проверки и сколько ложных срабатываний на чистых.

Определения (D43):
- ошибка *поймана проверками*, если в испорченном документе после нормализации есть хотя бы
  одно замечание `docai.validation` (ошибка или предупреждение; отдельно считается доля
  замечаний уровня «ошибка»);
- *не пропущена*, если итоговый статус не «можно загружать» или документ после
  автоисправлений равен верному (ошибку убрали);
- *молча загружена* — статус «можно загружать», а документ не равен верному: самый вредный исход;
- ложное срабатывание проверок на чистом документе — любое их замечание (даже предупреждение)
  или любое автоисправление. Статус «не можно загружать» у чистого документа считается
  отдельно: он зависит от того, есть ли текст-источник.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from docai.eval.corruptions import CORRUPTIONS, Corruption
from docai.postprocess import DocStatus, PostprocessConfig, SourceKind, postprocess
from docai.schemas.common import BaseDocument
from docai.schemas.documents import parse_document
from docai.validation import Severity

TextsFor = Callable[[BaseDocument], Sequence[tuple[SourceKind, str]]]


@dataclass(slots=True)
class CorruptionStats:
    name: str
    group: str
    title_ru: str
    applied: int = 0
    caught: int = 0  # проверки дали замечание (ошибку или предупреждение)
    caught_error: int = 0  # из них — хотя бы одна ошибка
    fixed_back: int = 0  # документ после исправлений равен верному
    flagged: int = 0  # статус не «можно загружать»
    silent: int = 0  # «можно загружать», но документ неверен
    wrong_fix: int = 0  # автоисправление, после которого документ всё равно неверен
    examples: list[str] = field(default_factory=list)

    @property
    def caught_rate(self) -> float:
        return self.caught / self.applied if self.applied else float("nan")

    @property
    def not_missed_rate(self) -> float:
        return (self.applied - self.silent) / self.applied if self.applied else float("nan")


@dataclass(slots=True)
class CleanStats:
    documents: int = 0
    with_issues: int = 0  # замечание проверок (ошибка или предупреждение)
    with_errors: int = 0
    with_fixes: int = 0
    not_upload: int = 0  # статус не «можно загружать»
    false_alarm: int = 0  # замечание проверок или автоисправление
    statuses: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarm / self.documents if self.documents else float("nan")


def same_document(a: BaseDocument, b: BaseDocument) -> bool:
    return a.model_dump(mode="json") == b.model_dump(mode="json")


def corrupt(doc: BaseDocument, corruption: Corruption, seed: str) -> BaseDocument | None:
    """Испорченная копия документа или None, если порча к нему неприменима."""
    rng = random.Random(f"{seed}:{corruption.name}")
    data: dict[str, Any] = doc.model_dump()
    if not corruption.apply(rng, data):
        return None
    return parse_document(data)


def evaluate_clean(
    docs: Sequence[BaseDocument],
    *,
    route: str = "text",
    texts_for: TextsFor | None = None,
    config: PostprocessConfig | None = None,
) -> CleanStats:
    stats = CleanStats()
    for doc in docs:
        texts = texts_for(doc) if texts_for else ()
        report = postprocess(doc, route=route, texts=texts, config=config).report
        stats.documents += 1
        issues = bool(report.issues_before)
        errors = any(i.severity is Severity.ERROR for i in report.issues_before)
        fixes = any(c.kind == "autofix" for c in report.corrections)
        not_upload = report.status is not DocStatus.UPLOAD
        stats.with_issues += issues
        stats.with_errors += errors
        stats.with_fixes += fixes
        stats.not_upload += not_upload
        stats.false_alarm += issues or fixes
        stats.statuses[report.status.value] += 1
    return stats


def evaluate_corruptions(
    docs: Sequence[BaseDocument],
    *,
    corruptions: Sequence[Corruption] = CORRUPTIONS,
    route: str = "text",
    texts_for: TextsFor | None = None,
    config: PostprocessConfig | None = None,
    seed: str = "p10",
    max_examples: int = 3,
) -> list[CorruptionStats]:
    """Испортить каждый документ каждой порчей и посмотреть, что скажут проверки и статус.

    texts_for получает ВЕРНЫЙ документ: источник текста — это печатный документ,
    а не его испорченное прочтение.
    """
    out = []
    for corruption in corruptions:
        stats = CorruptionStats(corruption.name, corruption.group, corruption.title_ru)
        for index, doc in enumerate(docs):
            bad = corrupt(doc, corruption, f"{seed}:{index}")
            if bad is None or same_document(bad, doc):
                continue
            texts = texts_for(doc) if texts_for else ()
            result = postprocess(bad, route=route, texts=texts, config=config)
            report = result.report
            assert result.document is not None
            stats.applied += 1
            caught = bool(report.issues_before)
            restored = same_document(result.document, doc)
            uploaded = report.status is DocStatus.UPLOAD
            stats.caught += caught
            stats.caught_error += any(i.severity is Severity.ERROR for i in report.issues_before)
            stats.fixed_back += restored
            stats.flagged += not uploaded
            if uploaded and not restored:
                stats.silent += 1
                if len(stats.examples) < max_examples:
                    stats.examples.append(f"документ №{index}")
            if any(c.kind == "autofix" for c in report.corrections) and not restored:
                stats.wrong_fix += 1
        out.append(stats)
    return out
