"""Документ корпуса как разделы и пункты; печать в markdown с разметкой фрагментов.

Фрагмент — это один пункт документа (или вводная часть, или блок реквизитов стороны).
У фрагмента есть устойчивый id `<doc_id>#<метка>` (например `K-0001#4.2`) и смещения
в тексте файла, поэтому эталон не зависит от того, как индекс P21 нарежет текст:
попадание считается по пересечению найденного куска с фрагментом.
"""

import re
from dataclasses import dataclass, field
from typing import Any

_DOUBLE_DOT = re.compile(r"(г|руб)\.\.")


def tidy(text: str) -> str:
    """«…2023 г..» → «…2023 г.»: сокращение в конце предложения."""
    return _DOUBLE_DOT.sub(lambda m: m.group(1) + ".", text)


@dataclass(slots=True)
class Clause:
    text: str
    facts: tuple[str, ...] = ()  # какие условия (ключи фактов) изложены в пункте
    label: str | None = None  # метка без нумерации (например «req-1» у реквизитов)


@dataclass(slots=True)
class Section:
    title: str | None
    clauses: list[Clause]
    numbered: bool = True  # нумеровать ли пункты «4.1.», «4.2.»


@dataclass(slots=True)
class Fragment:
    fragment_id: str
    doc_id: str
    label: str
    section: str | None
    text: str
    start: int
    end: int
    facts: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "doc_id": self.doc_id,
            "label": self.label,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(slots=True)
class CorpusDoc:
    """Документ корпуса: договор, доп. соглашение или регламент."""

    doc_id: str
    doc_kind: str  # contract | amendment | regulation
    header: list[str]  # заголовок и строки под ним (город, дата, «утверждён приказом...»)
    preamble: str | None
    sections: list[Section]
    meta: dict[str, Any] = field(default_factory=dict)
    # Заполняется при печати.
    text: str = ""
    fragments: list[Fragment] = field(default_factory=list)
    # Метка пункта по ключу факта: для ссылок «пункт 4.2 Договора» в доп. соглашениях.
    fact_labels: dict[str, str] = field(default_factory=dict)

    def render(self) -> None:
        """Собрать markdown и фрагменты (метки пунктов проставляются здесь)."""
        parts: list[str] = []
        fragments: list[Fragment] = []
        position = 0

        def emit(chunk: str) -> tuple[int, int]:
            nonlocal position
            start = position
            parts.append(chunk)
            position += len(chunk)
            return start, position

        for section in self.sections:
            for clause in section.clauses:
                clause.text = tidy(clause.text)
        if self.preamble:
            self.preamble = tidy(self.preamble)
        head = "# " + self.header[0]
        if len(self.header) > 1:
            head += "\n\n" + "\n\n".join(self.header[1:])
        if self.preamble:
            head += "\n\n" + self.preamble
        start, end = emit(head)
        fragments.append(self._fragment("preamble", None, head, start, end, ()))
        emit("\n")

        number = 0
        for section in self.sections:
            title = section.title
            if title is not None:
                number += 1
                emit(f"\n## {number}. {title}\n")
            for index, clause in enumerate(section.clauses, start=1):
                if clause.label is not None:
                    label, body = clause.label, clause.text
                elif section.numbered and title is not None:
                    label = f"{number}.{index}"
                    body = f"{label}. {clause.text}"
                else:
                    label = str(index)
                    body = f"{label}. {clause.text}"
                emit("\n")
                start, end = emit(body)
                emit("\n")
                fragments.append(self._fragment(label, title, body, start, end, clause.facts))
                for fact in clause.facts:
                    self.fact_labels.setdefault(fact, label)
        self.text = "".join(parts)
        self.fragments = fragments

    def _fragment(
        self,
        label: str,
        section: str | None,
        text: str,
        start: int,
        end: int,
        facts: tuple[str, ...],
    ) -> Fragment:
        return Fragment(
            f"{self.doc_id}#{label}", self.doc_id, label, section, text, start, end, facts
        )

    def fragment(self, label: str) -> Fragment:
        for item in self.fragments:
            if item.label == label:
                return item
        raise KeyError(f"{self.doc_id}: нет фрагмента {label}")

    def fact_fragment(self, fact: str) -> Fragment | None:
        """Фрагмент, где изложен факт (первый по порядку), или None."""
        for item in self.fragments:
            if fact in item.facts:
                return item
        return None
