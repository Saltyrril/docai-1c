"""Нарезка документов корпуса по структуре: раздел → пункт, а не окна по N символов.

Единица нарезки — пункт договора («2.2. Оплата производится …»), вводная часть
(заголовок, город и дата, стороны) и блок реквизитов каждой стороны. Метаданные
документа (вид, номер, дата, контрагент, для доп. соглашения — договор и дата вступления
в силу) читаются из самого текста, а не из файлов эталона: так же нарезался бы любой
договор, положенный в папку корпуса.

У каждого куска есть смещения в тексте файла (`text[start:end] == chunk.text`), поэтому
попадание сверяется с эталоном P18 по пересечению, а не по совпадению id (D62).
"""

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Метка пункта в начале абзаца: «2.2. », «1. », «3.1.2. ».
_CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\.\s")
_SECTION = re.compile(r"^##\s+(?:(\d+)\.\s+)?(.+?)\s*$")
_REQUISITES = re.compile(r"^\*\*([^*:]+):\*\*\s*(.*)")
_ORG = re.compile(
    r"(?:ООО|АО|ПАО|ЗАО|НАО|ОАО)\s*«[^»]+»"
    r"|(?:ИП|(?i:индивидуальн\w+ предпринимател\w+))\s+[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+){0,2}"
)
_IP_LONG = re.compile(r"^индивидуальн\w+ предпринимател\w+", re.IGNORECASE)
_INN = re.compile(r"ИНН\s+(\d{12}|\d{10})\b")
_NUMBER = re.compile(r"№\s*([^\s,]+)")
_PARENT = re.compile(
    r"к\s+договору\s+(?:[а-яё ]+?\s+)?№\s*(?P<number>\S+)\s+от\s+(?P<date>\d{2}\.\d{2}\.\d{4}"
    r"|«?\d{1,2}»?\s+[а-яё]+\s+\d{4})",
    re.IGNORECASE,
)
_EFFECTIVE = re.compile(
    r"вступает\s+в\s+силу\s+с\s+(?P<date>.+?)(?:\s+и\s|\.\s*$|$)", re.IGNORECASE
)
_CLAUSE_REF = re.compile(r"(?:пункт(?:ом|а|е)?|п\.)\s*(\d+\.\d+)", re.IGNORECASE)
_MONTHS = {
    m: i
    for i, m in enumerate(
        (
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ),
        start=1,
    )
}
_DATE_DOTS = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")
_DATE_WORDS = re.compile(r"«?(\d{1,2})»?\s+([а-яё]+)\s+(\d{4})", re.IGNORECASE)

# Пункты доп. соглашения, которые ничего не меняют: их не подмешиваем к ответу.
_BOILERPLATE = re.compile(
    r"остальн\w+ услови|во всём остальном|во всем остальном|неотъемлем\w+ част\w+ договора\.?$",
    re.IGNORECASE,
)
_CONTRACT_KINDS = (
    ("поставки", "supply"),
    ("оказание услуг", "services"),
    ("услуг", "services"),
    ("аренды", "lease"),
    ("подряда", "works"),
)
KIND_NAMES = {
    "supply": "договор поставки",
    "services": "договор оказания услуг",
    "lease": "договор аренды",
    "works": "договор подряда",
}


def parse_date(text: str) -> date | None:
    """Первая дата в строке: «31.01.2023», «31 января 2023», «“22” февраля 2024 г.»."""
    candidates: list[tuple[int, date]] = []
    for m in _DATE_DOTS.finditer(text):
        try:
            candidates.append((m.start(), date(int(m[3]), int(m[2]), int(m[1]))))
        except ValueError:
            continue
    for m in _DATE_WORDS.finditer(text):
        month = _MONTHS.get(m[2].lower())
        if month is None:
            continue
        try:
            candidates.append((m.start(), date(int(m[3]), month, int(m[1]))))
        except ValueError:
            continue
    return min(candidates)[1] if candidates else None


@dataclass(slots=True)
class DocMeta:
    """Сведения о документе, прочитанные из его текста."""

    doc_id: str
    doc_kind: str  # contract | amendment | regulation
    title: str
    path: str
    number: str | None = None
    doc_date: date | None = None
    contract_kind: str | None = None
    counterparty: str | None = None
    counterparty_inn: str | None = None
    # Для доп. соглашения: договор, к которому оно относится, и дата вступления в силу.
    parent_number: str | None = None
    parent_date: date | None = None
    parent_doc_id: str | None = None
    effective_date: date | None = None

    @property
    def contract_number(self) -> str | None:
        """Номер договора, к которому относится документ (у соглашения — номер договора)."""
        return self.parent_number if self.doc_kind == "amendment" else self.number

    @property
    def contract_date(self) -> date | None:
        return self.parent_date if self.doc_kind == "amendment" else self.doc_date

    def caption(self) -> str:
        """Короткое название для заголовка куска и ссылки в ответе."""
        if self.doc_kind == "amendment":
            text = f"Доп. соглашение № {self.number or '?'}"
            if self.doc_date:
                text += f" от {self.doc_date:%d.%m.%Y}"
            text += f" к договору № {self.parent_number or '?'}"
            if self.parent_date:
                text += f" от {self.parent_date:%d.%m.%Y}"
            if self.effective_date:
                text += f" (действует с {self.effective_date:%d.%m.%Y})"
        elif self.doc_kind == "contract":
            kind = KIND_NAMES.get(self.contract_kind or "", "договор")
            text = f"{kind[:1].upper()}{kind[1:]} № {self.number or '?'}"
            if self.doc_date:
                text += f" от {self.doc_date:%d.%m.%Y}"
        else:
            text = self.title[:1] + self.title[1:].lower()
            if self.doc_date:
                text += f" от {self.doc_date:%d.%m.%Y}"
        if self.counterparty:
            text += f", {self.counterparty}"
        return text


@dataclass(slots=True)
class Chunk:
    """Кусок индекса: пункт документа с метаданными."""

    chunk_id: str  # <doc_id>#<метка>, как id фрагментов P18
    doc_id: str
    label: str  # номер пункта («2.2»), «preamble» или «req-N»
    section: str | None
    text: str
    start: int
    end: int
    # Для доп. соглашения: какие пункты договора меняет этот пункт; служебный ли пункт.
    amends: tuple[str, ...] = ()
    boilerplate: bool = False
    meta: DocMeta | None = field(default=None, repr=False)

    @property
    def clause(self) -> str | None:
        return self.label if self.label[:1].isdigit() else None

    def heading(self) -> str:
        """Заголовок для модели и для индекса: документ, контрагент, раздел, пункт."""
        assert self.meta is not None
        parts = [self.meta.caption()]
        if self.section:
            parts.append(f"раздел «{self.section}»")
        if self.clause:
            parts.append(f"п. {self.clause}")
        elif self.label.startswith("req"):
            parts.append("реквизиты стороны")
        return ", ".join(parts)

    def index_text(self) -> str:
        """Текст для эмбеддингов и BM25: заголовок + сам пункт."""
        return f"{self.heading()}\n{self.text}"

    def metadata(self) -> dict[str, Any]:
        """Метаданные для Chroma: только скаляры, даты — числом ГГГГММДД для фильтров."""
        assert self.meta is not None
        m = self.meta

        def ymd(d: date | None) -> int:
            return int(d.strftime("%Y%m%d")) if d else 0

        return {
            "doc_id": self.doc_id,
            "doc_kind": m.doc_kind,
            "label": self.label,
            "section": self.section or "",
            "counterparty": m.counterparty or "",
            "counterparty_inn": m.counterparty_inn or "",
            "contract_number": m.contract_number or "",
            "contract_id": m.parent_doc_id if m.doc_kind == "amendment" else self.doc_id,
            "doc_date": ymd(m.doc_date),
            "contract_date": ymd(m.contract_date),
            "start": self.start,
            "end": self.end,
        }

    def to_json(self) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "meta"}
        data["amends"] = list(self.amends)
        return data


@dataclass(slots=True)
class _Paragraph:
    text: str
    start: int
    end: int


def _paragraphs(text: str) -> list[_Paragraph]:
    """Абзацы — куски между пустыми строками, со смещениями в исходном тексте."""
    result: list[_Paragraph] = []
    for m in re.finditer(r"[^\n](?:.|\n(?!\s*\n))*", text):
        body = m.group(0).rstrip()
        if body.strip():
            result.append(_Paragraph(body, m.start(), m.start() + len(body)))
    return result


def _split_long(p: _Paragraph, max_chars: int) -> list[_Paragraph]:
    """Пункт длиннее предела режется по предложениям (в корпусе P18 таких нет)."""
    if len(p.text) <= max_chars:
        return [p]
    # Границы предложений: начало каждого предложения в тексте пункта.
    starts = [0] + [m.end() for m in re.finditer(r"(?<=[.;!?])\s+", p.text)]
    sentences = [(a, b) for a, b in zip(starts, [*starts[1:], len(p.text)], strict=True) if a < b]
    pieces: list[_Paragraph] = []
    begin, last = sentences[0][0], sentences[0][0]
    for a, b in sentences:
        if b - begin > max_chars and last > begin:
            pieces.append(_piece(p, begin, last))
            begin = a
        last = b
    pieces.append(_piece(p, begin, len(p.text)))
    return pieces


def _piece(p: _Paragraph, a: int, b: int) -> _Paragraph:
    body = p.text[a:b].rstrip()
    return _Paragraph(body, p.start + a, p.start + a + len(body))


def _doc_kind(title: str, path: Path) -> str:
    upper = title.upper()
    if upper.startswith("ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ"):
        return "amendment"
    if upper.startswith("ДОГОВОР"):
        return "contract"
    return "regulation"


def _contract_kind(title: str) -> str | None:
    lower = title.lower()
    for marker, kind in _CONTRACT_KINDS:
        if marker in lower:
            return kind
    return None


def _counterparty(text: str, our_names: Iterable[str]) -> str | None:
    ours = {n.replace(" ", "") for n in our_names}
    for m in _ORG.finditer(text):
        name = _IP_LONG.sub("ИП", re.sub(r"\s+", " ", m.group(0)).strip())
        if name.replace(" ", "") not in ours:
            return name
    return None


def chunk_document(
    text: str,
    doc_id: str,
    path: str,
    *,
    our_names: Iterable[str] = (),
    our_inns: Iterable[str] = (),
    max_chars: int = 1200,
) -> tuple[DocMeta, list[Chunk]]:
    """Разобрать markdown документа на метаданные и куски-пункты."""
    paragraphs = _paragraphs(text)
    if not paragraphs or not paragraphs[0].text.startswith("# "):
        raise ValueError(f"{doc_id}: документ должен начинаться с заголовка «# …»")
    title = paragraphs[0].text[2:].strip()
    kind = _doc_kind(title, Path(path))
    meta = DocMeta(doc_id=doc_id, doc_kind=kind, title=title, path=path)
    number = _NUMBER.search(title)
    meta.number = number.group(1) if number else None
    meta.contract_kind = _contract_kind(title) if kind == "contract" else None
    our_names = tuple(our_names)
    our_inns = set(our_inns)

    chunks: list[Chunk] = []
    preamble: list[_Paragraph] = []
    section: str | None = None
    in_requisites = False
    current: _Paragraph | None = None
    current_label: str | None = None

    def flush() -> None:
        nonlocal current, current_label
        if current is not None and current_label is not None:
            for i, piece in enumerate(_split_long(current, max_chars)):
                label = current_label if i == 0 else f"{current_label}~{i + 1}"
                chunks.append(
                    Chunk(
                        f"{doc_id}#{label}",
                        doc_id,
                        label,
                        section,
                        piece.text,
                        piece.start,
                        piece.end,
                    )
                )
        current, current_label = None, None

    req_count = 0
    started = False  # закончилась ли вводная часть
    for p in paragraphs[1:]:
        heading = _SECTION.match(p.text)
        if heading:
            flush()
            started = True
            section = heading.group(2).strip()
            in_requisites = "реквизит" in section.lower()
            continue
        clause = _CLAUSE.match(p.text)
        requisite = _REQUISITES.match(p.text)
        if in_requisites and requisite:
            flush()
            req_count += 1
            current, current_label = p, f"req-{req_count}"
            continue
        if clause:
            flush()
            started = True
            current, current_label = p, clause.group(1)
            continue
        if not started:
            preamble.append(p)
            continue
        if current is not None:
            # Продолжение пункта (абзац без номера) — к тому же куску.
            current = _Paragraph(text[current.start : p.end], current.start, p.end)
        else:
            current, current_label = p, f"p{p.start}"
    flush()

    # Вводная часть — от заголовка до первого раздела или пункта.
    head_end = preamble[-1].end if preamble else paragraphs[0].end
    head_text = text[0:head_end]
    chunks.insert(0, Chunk(f"{doc_id}#preamble", doc_id, "preamble", None, head_text, 0, head_end))

    meta.doc_date = parse_date(" ".join(p.text for p in preamble[:3])) if preamble else None
    if kind == "amendment":
        parent = _PARENT.search(head_text)
        if parent:
            meta.parent_number = parent.group("number")
            meta.parent_date = parse_date(parent.group("date"))
        # Дата соглашения — строка «г. …, дата» (после ссылки на договор).
        for p in preamble:
            if p.text.startswith("г.") or _DATE_DOTS.fullmatch(p.text.strip()):
                meta.doc_date = parse_date(p.text) or meta.doc_date
                break
    if kind == "regulation":
        meta.doc_date = parse_date(head_text)
    meta.counterparty = _counterparty(head_text, our_names)

    for chunk in chunks:
        chunk.meta = meta
        if chunk.label.startswith("req"):
            inn = _INN.search(chunk.text)
            if inn and inn.group(1) not in our_inns and meta.counterparty_inn is None:
                meta.counterparty_inn = inn.group(1)
        if kind == "amendment" and chunk.clause:
            effective = _EFFECTIVE.search(chunk.text)
            if effective:
                meta.effective_date = parse_date(effective.group("date"))
                chunk.boilerplate = True  # отдельно в контекст не нужен: дата — в заголовке
            elif _BOILERPLATE.search(chunk.text):
                chunk.boilerplate = True
            chunk.amends = tuple(dict.fromkeys(_CLAUSE_REF.findall(chunk.text)))
    if kind == "amendment" and meta.effective_date is None:
        meta.effective_date = meta.doc_date
    return meta, chunks


def chunk_corpus(
    corpus_dir: Path,
    *,
    our_names: Iterable[str] = (),
    our_inns: Iterable[str] = (),
    max_chars: int = 1200,
) -> tuple[dict[str, DocMeta], list[Chunk]]:
    """Нарезать все `*.md` папки корпуса; связать доп. соглашения с договорами."""
    metas: dict[str, DocMeta] = {}
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(corpus_dir).as_posix()
        meta, doc_chunks = chunk_document(
            text, path.stem, rel, our_names=our_names, our_inns=our_inns, max_chars=max_chars
        )
        metas[meta.doc_id] = meta
        chunks.extend(doc_chunks)
    by_key = {(m.number, m.doc_date): m for m in metas.values() if m.doc_kind == "contract"}
    for meta in metas.values():
        if meta.doc_kind != "amendment":
            continue
        parent = by_key.get((meta.parent_number, meta.parent_date))
        if parent is not None:
            meta.parent_doc_id = parent.doc_id
            meta.contract_kind = parent.contract_kind
            meta.counterparty = meta.counterparty or parent.counterparty
            meta.counterparty_inn = meta.counterparty_inn or parent.counterparty_inn
    return metas, chunks
