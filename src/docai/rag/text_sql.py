"""Текст-в-SQL по реестру первичных документов (таблица `registry` в SQLite).

Модель пишет запрос, но выполняется он только после проверки:
1. текст — один оператор, начинается с SELECT или WITH, без ключевых слов изменения
   данных (INSERT, UPDATE, DROP, ATTACH, PRAGMA …);
2. база открыта только на чтение (`mode=ro`), а авторизатор SQLite разрешает лишь чтение
   таблиц и функции — даже пропущенный проверкой текстом запрос ничего не изменит;
3. число шагов виртуальной машины ограничено, строк в ответе — не больше `max_rows`.

Контрагент, договор и период находятся в вопросе без модели (`docai.rag.query`) и идут
в промпт подсказкой: маленькой модели проще подставить готовый ИНН, чем угадать написание.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from docai.rag.query import QueryInfo

SQL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"sql": {"type": "string"}},
    "required": ["sql"],
}
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

SQL_PROMPT = """Напиши один запрос SQLite к таблице registry — реестру входящих первичных документов АО «Тайга-Строй». Мы покупатель, документы нам присылают продавцы (контрагенты).

Таблица registry:
- doc_type TEXT — вид документа: 'upd' (УПД), 'bill' (счёт на оплату), 'torg12' (накладная ТОРГ-12), 'vat_invoice' (счёт-фактура), 'act' (акт)
- doc_type_name TEXT — вид документа по-русски
- number TEXT — номер документа
- date TEXT — дата документа, 'ГГГГ-ММ-ДД'
- seller_name TEXT, seller_inn TEXT — продавец (контрагент)
- contract_number TEXT, contract_date TEXT — номер и дата ('ГГГГ-ММ-ДД') договора
- amount_without_vat REAL, vat_amount REAL, amount_with_vat REAL — суммы в рублях: без НДС, НДС, с НДС

Правила:
- Только SELECT.
- Подсказки найдены в вопросе программой и верны: перенеси в запрос каждое условие из подсказок и вычисли то, что в них указано.
- Не добавляй условий, которых нет в вопросе и подсказках. GROUP BY — только если он есть в подсказке.

Примеры:
Вопрос: Сколько актов получено от ООО «Пример» за март 2025 года?
Подсказки:
- контрагент ООО «Пример»: seller_inn = '1234567890'
- виды документов: doc_type = 'act'
- период: date BETWEEN '2025-03-01' AND '2025-03-31'
- вычислить: COUNT(*)
SQL: SELECT COUNT(*) FROM registry WHERE seller_inn = '1234567890' AND doc_type = 'act' AND date BETWEEN '2025-03-01' AND '2025-03-31'

Вопрос: Сколько НДС выставил нам АО «Образец» в счетах-фактурах за 2024 год?
Подсказки:
- контрагент АО «Образец»: seller_inn = '0987654321'
- виды документов: doc_type = 'vat_invoice'
- период: date BETWEEN '2024-01-01' AND '2024-12-31'
- вычислить: ROUND(SUM(vat_amount), 2)
SQL: SELECT ROUND(SUM(vat_amount), 2) FROM registry WHERE seller_inn = '0987654321' AND doc_type = 'vat_invoice' AND date BETWEEN '2024-01-01' AND '2024-12-31'

Вопрос: Какой самый дорогой счёт прислал ООО «Пример»?
Подсказки:
- контрагент ООО «Пример»: seller_inn = '1234567890'
- виды документов: doc_type = 'bill'
- вычислить: doc_type_name, number, date, amount_with_vat … ORDER BY amount_with_vat DESC LIMIT 1
SQL: SELECT doc_type_name, number, date, amount_with_vat FROM registry WHERE seller_inn = '1234567890' AND doc_type = 'bill' ORDER BY amount_with_vat DESC LIMIT 1

Вопрос: Какой продавец прислал больше всего документов в 2024 году?
Подсказки:
- период: date BETWEEN '2024-01-01' AND '2024-12-31'
- вычислить: seller_name, COUNT(*) AS n … GROUP BY seller_inn ORDER BY n DESC LIMIT 1
SQL: SELECT seller_name, COUNT(*) AS n FROM registry WHERE date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY seller_inn ORDER BY n DESC LIMIT 1

Вопрос: {question}
{hints}
Ответь JSON: {{"sql": "запрос"}}"""

ANSWER_PROMPT = """Вопрос бухгалтера: {question}

Запрос к реестру первичных документов:
{sql}

Результат ({count} стр.):
{table}

Сформулируй ответ на вопрос по результату в 1–2 предложениях на русском. Числа бери только из результата, суммы пиши в рублях с копейками. Ответь JSON: {{"answer": "ответ"}}"""

_FORBIDDEN = re.compile(
    r"\b(?:insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|"
    r"reindex|analyze|begin|commit|rollback|savepoint|release|load_extension)\b",
    re.IGNORECASE,
)
_STRING = re.compile(r"'(?:[^']|'')*'")
_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)

# Коды действий авторизатора SQLite (sqlite3.h), которые разрешены: чтение и функции.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


class SqlRejectedError(ValueError):
    """Запрос не прошёл проверку: не SELECT, несколько операторов, запрещённые слова."""


def validate_sql(sql: str) -> str:
    """Очистить и проверить запрос; вернуть его без `;` в конце."""
    text = _FENCE.sub("", sql.strip()).strip()
    text = re.sub(r"^sql\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip().rstrip(";").strip()
    if not text:
        raise SqlRejectedError("пустой запрос")
    bare = _STRING.sub("''", text)
    if "--" in bare or "/*" in bare:
        raise SqlRejectedError("комментарии в запросе не допускаются")
    if ";" in bare:
        raise SqlRejectedError("допускается только один оператор")
    if not re.match(r"(?:select|with)\b", bare, re.IGNORECASE):
        raise SqlRejectedError("допускается только SELECT")
    word = _FORBIDDEN.search(bare)
    if word:
        raise SqlRejectedError(f"запрещённое слово: {word.group(0)}")
    return text


@dataclass
class SqlResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False


def run_select(
    db_path: Path, sql: str, *, max_rows: int = 50, max_vm_steps: int = 5_000_000
) -> SqlResult:
    """Выполнить проверенный SELECT на базе, открытой только на чтение."""
    checked = validate_sql(sql)
    uri = db_path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    steps = 0

    def authorizer(action: int, *_: Any) -> int:
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def progress() -> int:
        nonlocal steps
        steps += 1000
        return 1 if steps > max_vm_steps else 0

    try:
        con.set_authorizer(authorizer)
        con.set_progress_handler(progress, 1000)
        cursor = con.execute(checked)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        raise SqlRejectedError(f"SQLite: {exc}") from exc
    finally:
        con.close()
    return SqlResult(checked, columns, [list(r) for r in rows[:max_rows]], len(rows) > max_rows)


@dataclass
class SqlHints:
    lines: list[str] = field(default_factory=list)
    # Что обязано быть в запросе: значения-литералы и фрагменты (в нижнем регистре, без пробелов).
    literals: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    has_period: bool = False
    compute: list[str] = field(default_factory=list)
    what: str | None = None

    def text(self) -> str:
        if not self.lines:
            return "Подсказки: нет"
        return "Подсказки:\n" + "\n".join(f"- {x}" for x in self.lines)


# Виды документов по словам вопроса (порядок важен: «счёт-фактура» раньше «счёта»).
_DOC_TYPES: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"отгрузочн", re.I), ("upd", "torg12", "act")),
    (re.compile(r"сч[её]т\w*[- ]фактур\w*", re.I), ("vat_invoice",)),
    (re.compile(r"\bупд\b", re.I), ("upd",)),
    (re.compile(r"торг-?12|накладн\w*", re.I), ("torg12",)),
    (re.compile(r"\bакт(?:ов|ы|а|ам)?\b", re.I), ("act",)),
    (re.compile(r"\bсч[её]т(?:ов|а|ам)?\b(?![- ]фактур)", re.I), ("bill",)),
]


def doc_types(question: str) -> list[str]:
    """Виды документов, названные в вопросе; пусто — все виды."""
    text = question
    found: list[str] = []
    for pattern, kinds in _DOC_TYPES:
        if pattern.search(text):
            found.extend(k for k in kinds if k not in found)
            text = pattern.sub(" ", text)
            if len(kinds) > 1:
                # «(УПД, ТОРГ-12, акты)» после «отгрузочных» — расшифровка, а не новые виды.
                text = re.sub(r"\([^)]*\)", " ", text)
    return found


def aggregate_hint(question: str) -> str | None:
    """Что вычислить: по словам вопроса, без модели."""
    q = question.lower().replace("ё", "е")
    if re.search(r"больше всего|чаще всего|самый активн", q):
        return "seller_name, COUNT(*) AS n … GROUP BY seller_inn ORDER BY n DESC LIMIT 1"
    if re.search(r"сколько\s+(?:разных\s+)?(?:контрагент|продав|поставщик)", q):
        return "COUNT(DISTINCT seller_inn)"
    if re.search(r"самый крупн|самый дорог|наибольш|максимальн", q):
        return (
            "doc_type_name, number, date, amount_with_vat … ORDER BY amount_with_vat DESC LIMIT 1"
        )
    if re.search(r"последн", q):
        return "doc_type_name, number, date, amount_with_vat … ORDER BY date DESC LIMIT 1"
    if re.search(r"сколько\s+ндс|сумм\w*\s+ндс|ндс\s+предъяв", q):
        return "ROUND(SUM(vat_amount), 2)"
    if re.search(r"без\s+ндс", q) and re.search(r"сумм", q):
        return "ROUND(SUM(amount_without_vat), 2)"
    if re.search(r"сумм", q):
        return "ROUND(SUM(amount_with_vat), 2)"
    if re.search(r"сколько", q):
        return "COUNT(*)"
    return None


def registry_sellers(db_path: Path) -> dict[str, str]:
    """Продавцы реестра: название → ИНН."""
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT DISTINCT seller_name, seller_inn FROM registry").fetchall()
    finally:
        con.close()
    return {name: inn for name, inn in rows if name}


def build_hints(info: QueryInfo, sellers: dict[str, str], question: str = "") -> SqlHints:
    hints = SqlHints()
    for name in info.counterparties:
        inn = sellers.get(name)
        if inn:
            hints.lines.append(f"контрагент {name}: seller_inn = '{inn}'")
            hints.literals.append(f"'{inn}'")
    for ref in info.contracts:
        line = f"договор № {ref.number}: contract_number = '{ref.number}'"
        hints.literals.append(f"'{ref.number}'")
        if ref.date:
            line += f" AND contract_date = '{ref.date.isoformat()}'"
            hints.literals.append(f"'{ref.date.isoformat()}'")
        hints.lines.append(line)
    kinds = doc_types(question)
    hints.doc_types = kinds
    if len(kinds) == 1:
        hints.lines.append(f"виды документов: doc_type = '{kinds[0]}'")
    elif kinds:
        listed = ", ".join(f"'{k}'" for k in kinds)
        hints.lines.append(f"виды документов: doc_type IN ({listed})")
    if info.period:
        start, end = info.period
        hints.lines.append(f"период: date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'")
        hints.literals += [f"'{start.isoformat()}'", f"'{end.isoformat()}'"]
        hints.has_period = True
    what = aggregate_hint(question)
    if what:
        hints.lines.append(f"вычислить: {what}")
        hints.what = what
        hints.compute = [
            _squash(part) for part in re.findall(r"[A-Z]+\([^)]*\)+|ORDER BY \w+ \w+", what)
        ]
    return hints


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


_DOC_FILTER = re.compile(r"\bdoc_type\s*(?:=|in\b)", re.IGNORECASE)
_DATE_FILTER = re.compile(r"\bdate\s*(?:between|>=|<=|>|<|=|like)", re.IGNORECASE)


def hint_problems(sql: str, hints: SqlHints) -> list[str]:
    """Чего из подсказок нет в запросе и что в нём лишнее (пусто — запрос согласован)."""
    squashed = _squash(sql)
    problems = [f"нет условия со значением {lit}" for lit in hints.literals if lit not in sql]
    codes = set(re.findall(r"'(upd|bill|torg12|vat_invoice|act)'", sql))
    if hints.doc_types:
        if codes != set(hints.doc_types):
            listed = ", ".join(f"'{k}'" for k in hints.doc_types)
            problems.append(f"виды документов должны быть ровно: {listed}")
    elif _DOC_FILTER.search(sql):
        problems.append("лишнее условие на doc_type: в вопросе вид документа не указан")
    if not hints.has_period and _DATE_FILTER.search(sql):
        problems.append("лишнее условие на date: в вопросе период не указан")
    problems += [f"нужно вычислить {part}" for part in hints.compute if part not in squashed]
    if hints.what is not None:
        lowered = " ".join(sql.lower().split())
        for clause in ("group by", "order by", "limit"):
            if clause in lowered and clause not in hints.what.lower():
                problems.append(f"лишний {clause.upper()}: в подсказке его нет")
    return problems


def sql_prompt(question: str, hints: SqlHints, today: date) -> str:
    return SQL_PROMPT.format(today=today.isoformat(), hints=hints.text(), question=question.strip())


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return (
            f"{Decimal(str(value)):.2f}"
            if value != int(value) or abs(value) >= 1000
            else (str(int(value)))
        )
    return "" if value is None else str(value)


def format_table(result: SqlResult) -> str:
    lines = [" | ".join(result.columns)]
    lines += [" | ".join(_cell(v) for v in row) for row in result.rows]
    return "\n".join(lines)


def answer_prompt(question: str, result: SqlResult) -> str:
    return ANSWER_PROMPT.format(
        question=question.strip(),
        sql=result.sql,
        count=len(result.rows),
        table=format_table(result),
    )


_NUM = re.compile(r"\d[\d\s ]*(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    found = set()
    for m in _NUM.finditer(text):
        raw = re.sub(r"[\s ]", "", m.group(0)).replace(",", ".")
        try:
            value = Decimal(raw)
        except ArithmeticError:
            continue
        found.add(format(value.normalize(), "f"))
    return found


def numbers_supported(answer: str, question: str, result: SqlResult) -> bool:
    """Все числа ответа есть в результате запроса или в самом вопросе."""
    allowed = _numbers(question) | _numbers(format_table(result))
    for row in result.rows:
        for value in row:
            if isinstance(value, int | float):
                allowed.add(format(Decimal(str(value)).normalize(), "f"))
                allowed.add(format(Decimal(str(round(value, 2))).quantize(Decimal("0.01")), "f"))
            elif isinstance(value, str):
                allowed |= _numbers(value)
                # Дата ГГГГ-ММ-ДД в ответе пишется как ДД.ММ.ГГГГ.
                m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
                if m:
                    allowed |= _numbers(f"{m[3]}.{m[2]}.{m[1]}") | {m[1], m[2], m[3]}
    allowed |= {format(Decimal(x).normalize(), "f") for x in list(allowed)}
    return _numbers(answer) <= allowed


def fallback_answer(result: SqlResult) -> str:
    """Ответ без модели: значение или строки результата."""
    if not result.rows:
        return "По реестру таких документов нет."
    if len(result.rows) == 1 and len(result.columns) == 1:
        return f"По реестру: {_cell(result.rows[0][0])}."
    parts = [
        ", ".join(f"{c}: {_cell(v)}" for c, v in zip(result.columns, row, strict=True))
        for row in result.rows[:10]
    ]
    return "По реестру: " + "; ".join(parts) + "."
