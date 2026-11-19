"""Разбор вопроса: контрагенты, номера договоров, период — для фильтров по метаданным.

Контрагент ищется по известным названиям (из индекса или реестра) по леммам, поэтому
«по Вектору», «у «Маяка»», «с Регион-Продукт» находят ООО «Вектор», ООО «Маяк»,
ООО «Регион-Продукт». Из пересекающихся совпадений остаётся самое длинное: в вопросе
про «Маяк-Сервис» не находится заодно ООО «Маяк».
"""

import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

from docai.rag.chunking import parse_date
from docai.rag.lemmas import lemma

_PART = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_FORM = re.compile(r"^(ООО|АО|ПАО|ЗАО|НАО|ОАО|ИП)\s*")
_FORM_BEFORE = re.compile(r"\b(ООО|АО|ПАО|ЗАО|НАО|ОАО|ИП)\s*«?\s*$")
_ORG_IN_TEXT = re.compile(r"\b(?:ООО|АО|ПАО|ЗАО|НАО|ОАО)\s*«([^»]+)»")
_DATE = r"\d{2}\.\d{2}\.\d{4}"
_CONTRACT_REF = re.compile(
    r"договор\w*(?:\s+[а-яё-]+){0,3}?\s*№\s*(?P<number>[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё/\-.]*)"
    rf"(?:\s+от\s+(?P<date>{_DATE}))?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:за|в|во)\s+(\d{4})\s*(?:год|г\.)", re.IGNORECASE)
_THIS_YEAR = re.compile(r"\b(?:за|в)\s+(?:этом|текущем|этот|текущий)\s+году?\b", re.IGNORECASE)
_RANGE = re.compile(rf"\bс\s+({_DATE})\s+по\s+({_DATE})", re.IGNORECASE)
_QUARTER = re.compile(
    r"\b(I{1,3}|IV|[1-4]|перв\w*|втор\w*|трет\w*|четв[её]рт\w*)\s*(?:-?й\s+)?квартал\w*\s+(\d{4})",
    re.IGNORECASE,
)
_MONTHS = (
    "январ",
    "феврал",
    "март",
    "апрел",
    "ма",
    "июн",
    "июл",
    "август",
    "сентябр",
    "октябр",
    "ноябр",
    "декабр",
)
_MONTH = re.compile(
    r"\b(?:в|за)\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йея]\w*|июн\w*|июл\w*|август\w*"
    r"|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+(\d{4})",
    re.IGNORECASE,
)
_QUARTER_NUMBERS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "перв": 1, "втор": 2, "трет": 3, "четв": 4}


@dataclass(frozen=True)
class ContractRef:
    number: str
    date: dt.date | None = None


@dataclass
class QueryInfo:
    counterparties: list[str] = field(default_factory=list)
    unknown_orgs: list[str] = field(default_factory=list)
    contracts: list[ContractRef] = field(default_factory=list)
    period: tuple[date, date] | None = None
    contract_kinds: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, object]:
        return {
            "counterparties": self.counterparties,
            "unknown_orgs": self.unknown_orgs,
            "contracts": [
                {"number": c.number, "date": c.date.isoformat() if c.date else None}
                for c in self.contracts
            ],
            "period": [d.isoformat() for d in self.period] if self.period else None,
            "contract_kinds": sorted(self.contract_kinds),
        }


def split_name(name: str) -> tuple[str | None, str]:
    """«ООО «Маяк»» → («ООО», «Маяк»); «ИП Потапова Любовь Леонидовна» → («ИП», ФИО)."""
    form = _FORM.match(name)
    rest = name[form.end() :] if form else name
    quoted = re.search(r"«([^»]+)»", rest)
    return (form.group(1) if form else None), (quoted.group(1) if quoted else rest).strip()


@dataclass(frozen=True)
class _Token:
    lemma: str
    start: int
    capital: bool


def _tokens(text: str) -> list[_Token]:
    return [
        _Token(lemma(m.group(0).lower().replace("ё", "е")), m.start(), m.group(0)[0].isupper())
        for m in _PART.finditer(text)
    ]


class NameMatcher:
    """Поиск известных названий организаций в тексте вопроса."""

    def __init__(self, names: Iterable[str]) -> None:
        self.keys: list[tuple[tuple[str, ...], str, str | None]] = []
        for name in sorted(set(names)):
            form, core = split_name(name)
            parts = tuple(t.lemma for t in _tokens(core))
            if form == "ИП" and parts:
                parts = parts[:1]  # у ИП достаточно фамилии
            if parts:
                self.keys.append((parts, name, form))

    def find(self, text: str) -> list[str]:
        tokens = _tokens(text)
        lemmas_seq = [t.lemma for t in tokens]
        spans: list[tuple[int, int, str, str | None]] = []
        for parts, name, form in self.keys:
            n = len(parts)
            for i in range(len(tokens) - n + 1):
                if tuple(lemmas_seq[i : i + n]) == parts and tokens[i].capital:
                    spans.append((i, i + n, name, form))
        # Оставить самые длинные: совпадение внутри более длинного отбрасывается.
        kept: list[tuple[int, int, str, str | None]] = []
        for span in spans:
            inside = any(
                o[0] <= span[0] and span[1] <= o[1] and (o[1] - o[0]) > (span[1] - span[0])
                for o in spans
            )
            if not inside:
                kept.append(span)
        # Если перед названием указана форма (ООО/АО), лишние формы с тем же названием убрать.
        result: list[str] = []
        for start, _end, name, form in kept:
            explicit = _FORM_BEFORE.search(text[: tokens[start].start])
            if explicit and form and explicit.group(1) != form:
                continue
            if name not in result:
                result.append(name)
        return result


def _quarter(token: str) -> int:
    t = token.lower()
    if t in _QUARTER_NUMBERS:
        return _QUARTER_NUMBERS[t]
    if t.isdigit():
        return int(t)
    return _QUARTER_NUMBERS.get(t[:4], 0)


def _month_end(year: int, month: int) -> date:
    first_next = date(year + (month == 12), month % 12 + 1, 1)
    return first_next - timedelta(days=1)


def find_period(text: str, today: date) -> tuple[date, date] | None:
    """Период из вопроса: диапазон дат, квартал, месяц, год, «в этом году»."""
    m = _RANGE.search(text)
    if m:
        start, end = parse_date(m.group(1)), parse_date(m.group(2))
        if start and end:
            return start, end
    m = _QUARTER.search(text)
    if m:
        q, year = _quarter(m.group(1)), int(m.group(2))
        if 1 <= q <= 4:
            return date(year, 3 * q - 2, 1), _month_end(year, 3 * q)
    m = _MONTH.search(text)
    if m:
        word = m.group(1).lower()
        month = next(
            (
                i
                for i, stem in enumerate(_MONTHS, start=1)
                if word.startswith(stem) and (stem != "ма" or word[2:3] in "йяе")
            ),
            None,
        )
        if month:
            year = int(m.group(2))
            return date(year, month, 1), _month_end(year, month)
    m = _YEAR.search(text)
    if m:
        year = int(m.group(1))
        return date(year, 1, 1), date(year, 12, 31)
    if _THIS_YEAR.search(text):
        return date(today.year, 1, 1), today
    return None


def analyze(text: str, matcher: NameMatcher, today: date) -> QueryInfo:
    info = QueryInfo()
    info.counterparties = matcher.find(text)
    for m in _CONTRACT_REF.finditer(text):
        number = m.group("number").rstrip(".-/")
        when = parse_date(m.group("date")) if m.group("date") else None
        ref = ContractRef(number, when)
        if ref not in info.contracts:
            info.contracts.append(ref)
    known_cores = {split_name(n)[1].lower() for n in info.counterparties}
    for m in _ORG_IN_TEXT.finditer(text):
        if m.group(1).lower() not in known_cores and not info.counterparties:
            info.unknown_orgs.append(m.group(0))
    info.period = find_period(text, today)
    info.contract_kinds = contract_kinds(text)
    return info


_KIND_WORDS = (
    (re.compile(r"поставк|поставщик", re.IGNORECASE), "supply"),
    (re.compile(r"оказани\w*\s+услуг|услуг\w*\s*\)", re.IGNORECASE), "services"),
    (re.compile(r"аренд", re.IGNORECASE), "lease"),
    (re.compile(r"подряд", re.IGNORECASE), "works"),
)


def contract_kinds(text: str) -> set[str]:
    """Вид договора, названный в вопросе: «договор аренды», «(договор оказания услуг)»."""
    kinds: set[str] = set()
    for m in re.finditer(r"договор\w*[^.?!;]{0,40}", text, re.IGNORECASE):
        fragment = m.group(0)
        for pattern, kind in _KIND_WORDS:
            if pattern.search(fragment):
                kinds.add(kind)
    return kinds
