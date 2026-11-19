"""Вопросы к корпусу с эталонными ответами и id фрагментов-источников.

Категории (по методике EnterpriseRAG-Bench, данные свои, D3): простой факт,
перефразированный вопрос, несколько условий одного договора, доп. соглашение отменяет
условие, сравнение двух договоров, агрегат по реестру, ответа нет в корпусе.

Эталон считается из условий договоров (`Contract`) и таблицы реестра, а не из текста,
поэтому формулировка в договоре на эталон не влияет. Каждый вопрос несёт:
`answer` — эталонный ответ; `keys` — ключевые факты, которые должны быть в ответе
(см. `facts.normalize_answer`); `sources` — фрагменты с цитатами; `superseded` —
фрагменты со старой редакцией (для доп. соглашений); `sql` — запрос к реестру.
"""

import random
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from docai.rag import facts as F
from docai.rag.amendments import EFFECTIVE
from docai.rag.corpus import ContractBundle, Corpus
from docai.rag.document import Fragment
from docai.rag.registry import COLUMNS, RegistryRow
from docai.rag.regulations import REG_AMENDMENT, REG_FACTS, REG_SUPERSEDED
from docai.rag.terms import KIND_GEN, Kind
from docai.rag.words import date_short, money_answer, money_digits, plural
from docai.synth.directory import Entity

CATEGORIES = {
    "simple_fact": "простой факт",
    "paraphrase": "перефразированный вопрос",
    "multi_condition": "несколько условий из одного договора",
    "amendment_override": "доп. соглашение отменяет условие",
    "comparison": "сравнение двух договоров",
    "registry_aggregate": "агрегат по реестру",
    "no_answer": "ответа нет в корпусе",
}
# (вопросов по договорам, по регламентам) в каждой категории; всего 200.
QUOTAS = {
    "simple_fact": (32, 8),
    "paraphrase": (24, 6),
    "multi_condition": (26, 4),
    "amendment_override": (28, 2),
    "comparison": (25, 0),
    "registry_aggregate": (25, 0),
    "no_answer": (17, 3),
}
NO_ANSWER = "В корпусе нет ответа на этот вопрос."


@dataclass(slots=True)
class Question:
    category: str
    question: str
    answer: str
    keys: list[str]
    route: str  # rag | table | none
    sources: list[dict[str, str]] = field(default_factory=list)
    superseded: list[dict[str, str]] = field(default_factory=list)
    wrong_answers: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    registry_rows: list[str] = field(default_factory=list)
    sql: str | None = None
    sql_detail: str | None = None  # запрос, подтверждающий детализацию в ответе (разбивку)
    expected_value: str | None = None
    template: str = ""
    qid: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.qid,
            "category": self.category,
            "category_name": CATEGORIES[self.category],
            "question": self.question,
            "answer": self.answer,
            "keys": self.keys,
            "route": self.route,
            "sources": self.sources,
            "superseded": self.superseded,
            "wrong_answers": self.wrong_answers,
            "doc_ids": self.doc_ids,
            "registry_rows": self.registry_rows,
            "sql": self.sql,
            "sql_detail": self.sql_detail,
            "expected_value": self.expected_value,
            "template": self.template,
            "author": "generator",
        }


def _src(fragment: Fragment) -> dict[str, str]:
    return {"fragment_id": fragment.fragment_id, "doc_id": fragment.doc_id, "quote": fragment.text}


# --- Как назвать контрагента и договор --------------------------------------------------


def short_name(entity: Entity) -> str:
    """«Вектор-Снаб» для ООО «Вектор-Снаб», «ИП Петров» для ИП Петров Пётр Петрович."""
    if entity.is_individual:
        return "ИП " + entity.name.removeprefix("ИП ").split()[0]
    return entity.name.split("«", 1)[1].rstrip("»")


class Names:
    """Ссылки на договоры: однозначные в пределах корпуса."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.by_cp: dict[str, list[ContractBundle]] = defaultdict(list)
        for b in corpus.bundles:
            self.by_cp[b.base.them.inn].append(b)
        # Короткое имя считается однозначным, если оно у одного контрагента.
        owners: dict[str, set[str]] = defaultdict(set)
        for b in corpus.bundles:
            owners[short_name(b.base.them)].add(b.base.them.inn)
        self.unique_short = {name for name, inns in owners.items() if len(inns) == 1}

    def cp(self, entity: Entity, rng: random.Random, colloquial: bool = False) -> str:
        short = short_name(entity)
        if colloquial and short in self.unique_short:
            return rng.choice((f"«{short}»", f"«{short}»", short)) if not entity.is_individual \
                else short  # fmt: skip
        return entity.name

    def unique_kind(self, b: ContractBundle) -> bool:
        return sum(x.base.kind is b.base.kind for x in self.by_cp[b.base.them.inn]) == 1

    def only_one(self, b: ContractBundle) -> bool:
        return len(self.by_cp[b.base.them.inn]) == 1

    def formal(
        self, b: ContractBundle, case: str, rng: random.Random, with_cp: bool | None = None
    ) -> str:
        """«договору поставки № П-001/25 от 12.03.2025 с ООО «Вектор»» в нужном падеже."""
        noun = {"dat": "договору", "prep": "договоре", "gen": "договора", "nom": "договор"}[case]
        c = b.base
        text = f"{noun} {KIND_GEN[c.kind]} № {c.number} от {date_short(c.date)}"
        if with_cp is None:
            with_cp = rng.random() < 0.6
        if with_cp:
            text += f" с {c.them.name}"
        return text

    def loose(self, b: ContractBundle, case: str, rng: random.Random) -> str:
        """Ссылка через контрагента («договору аренды с «Вектор»»), если она однозначна."""
        noun = {"dat": "договору", "prep": "договоре", "gen": "договора", "nom": "договор"}[case]
        c = b.base
        name = self.cp(c.them, rng, colloquial=True)
        if self.only_one(b) and rng.random() < 0.5:
            return f"{noun} с {name}"
        if self.unique_kind(b):
            return f"{noun} {KIND_GEN[c.kind]} с {name}"
        return f"{noun} {KIND_GEN[c.kind]} № {c.number} с {name}"

    def hint(self, b: ContractBundle) -> str:
        """Уточнение для разговорного вопроса, если у контрагента несколько договоров."""
        if self.only_one(b):
            return ""
        if self.unique_kind(b):
            return f" (договор {KIND_GEN[b.base.kind]})"
        return f" (договор № {b.base.number})"


def _cite(b: ContractBundle, fragment: Fragment) -> str:
    c = b.base
    where = f"договор {KIND_GEN[c.kind]} № {c.number} от {date_short(c.date)}"
    if fragment.doc_id != b.contract_id:
        number = fragment.doc_id.rsplit("DS", 1)[1]
        return f"доп. соглашение № {number} к договору № {c.number}, п. {fragment.label}"
    if fragment.label.startswith("req"):
        return f"{where}, реквизиты сторон"
    return f"{where}, п. {fragment.label}"


# --- Шаблоны вопросов по фактам ----------------------------------------------------------

Template = Callable[[ContractBundle, "Ctx"], str]


@dataclass(slots=True)
class Ctx:
    names: Names
    rng: random.Random
    used_pairs: set[tuple[str, str]] = field(default_factory=set)  # (договор, факт) уже спрошены


def _simple_templates(fact: str, kind: Kind) -> list[Callable[[str, str, str], str]]:
    """Вопросы языком договора. Аргументы: ссылка в дательном, предложном, родительном."""
    t: dict[str, list[Callable[[str, str, str], str]]] = {
        F.PAYMENT: [
            lambda d, p, g: f"Какой срок оплаты установлен по {d}?",
            lambda d, p, g: f"В какой срок производится оплата по {d}?",
            lambda d, p, g: f"Каков порядок оплаты по {d}?",
        ],
        F.PRICE: {
            Kind.SUPPLY: [lambda d, p, g: f"Какова предельная сумма поставок по {d}?"],
            Kind.SERVICES: [lambda d, p, g: f"Какова ежемесячная стоимость услуг по {d}?"],
            Kind.LEASE: [lambda d, p, g: f"Какой размер арендной платы установлен по {d}?"],
            Kind.WORKS: [lambda d, p, g: f"Какова стоимость работ по {d}?"],
        }[kind],
        F.PENALTY_US: [
            lambda d, p, g: f"Какая неустойка предусмотрена за просрочку оплаты по {d}?",
            lambda d, p, g: f"Каков размер пени за нарушение срока оплаты в {p}?",
        ],
        F.PENALTY_THEM: [
            {
                Kind.SUPPLY: lambda d, p, g: (
                    f"Какая неустойка установлена за просрочку поставки по {d}?"
                ),
                Kind.SERVICES: lambda d, p, g: (
                    f"Какая неустойка установлена за нарушение сроков оказания услуг по {d}?"
                ),
                Kind.WORKS: lambda d, p, g: (
                    f"Какая неустойка установлена за нарушение срока окончания работ по {d}?"
                ),
                Kind.LEASE: lambda d, p, g: "",
            }[kind]
        ],
        F.ACCEPTANCE: [
            {
                Kind.SUPPLY: lambda d, p, g: (
                    f"В какой срок можно предъявить претензию по качеству товара по {d}?"
                ),
                Kind.SERVICES: lambda d, p, g: (
                    f"В какой срок заказчик должен подписать акт оказанных услуг по {d}?"
                ),
                Kind.WORKS: lambda d, p, g: (
                    f"В какой срок заказчик подписывает акт КС-2 и справку КС-3 по {d}?"
                ),
                Kind.LEASE: lambda d, p, g: f"В какой срок помещение передаётся арендатору по {d}?",
            }[kind]
        ],
        F.VALIDITY: [
            lambda d, p, g: f"Каков срок действия {g}?",
            lambda d, p, g: f"До какого момента действует {g.replace('договора', 'договор', 1)}?",
        ],
        F.PROLONGATION: [
            lambda d, p, g: f"Предусмотрена ли автоматическая пролонгация {g}?",
            lambda d, p, g: f"Продлевается ли автоматически {g.replace('договора', 'договор', 1)}?",
        ],
        F.TERMINATION: [
            lambda d, p, g: (
                f"За сколько дней нужно уведомить другую сторону об одностороннем отказе от {g}?"
            ),
        ],
        F.CLAIMS: [lambda d, p, g: f"Каков срок ответа на претензию по {d}?"],
        F.COURT: [lambda d, p, g: f"В каком суде рассматриваются споры по {d}?"],
        F.FORCE_MAJEURE: [
            lambda d, p, g: f"В какой срок нужно уведомить о форс-мажорных обстоятельствах по {d}?",
        ],
        F.DELIVERY: [lambda d, p, g: f"Каковы срок и условия поставки по {d}?"],
        F.PREMISES: [lambda d, p, g: f"Какова площадь и адрес арендуемого помещения по {d}?"],
        F.DEPOSIT: [lambda d, p, g: f"Какой обеспечительный платёж предусмотрен в {p}?"],
        F.INDEXATION: [lambda d, p, g: f"Предусмотрена ли индексация арендной платы в {p}?"],
        F.UTILITIES: [lambda d, p, g: f"Включены ли коммунальные услуги в арендную плату по {d}?"],
        F.WORKS_PERIOD: [lambda d, p, g: f"Каковы сроки выполнения работ по {d}?"],
        F.WARRANTY: [lambda d, p, g: f"Какой гарантийный срок установлен в {p}?"],
        F.RETENTION: [lambda d, p, g: f"Какое гарантийное удержание предусмотрено в {p}?"],
        F.REQUISITES: [
            lambda d, p, g: f"На какой расчётный счёт перечислять оплату по {d}?",
            lambda d, p, g: f"Какие банковские реквизиты контрагента указаны в {p}?",
        ],
    }
    return t.get(fact, [])


def _paraphrase_templates(fact: str, b: ContractBundle) -> list[Callable[[str], str]]:
    """Разговорные вопросы без слов из договора. Аргумент — имя контрагента."""
    c = b.base
    kind = c.kind
    p = b.effective.payment
    event = {
        "delivery": "того как они привезут товар",
        "upd": "подписания накладной",
        "invoice": "того как придёт их счёт",
        "act": "подписания акта",
        "ks": "приёмки работ",
    }.get(p.trigger, "")
    t: dict[str, list[Callable[[str], str]]] = {
        F.PAYMENT: (
            [lambda n: f"Сколько времени у нас есть, чтобы рассчитаться с {n} после {event}?"]
            if p.scheme == "deferral" and p.prepay_percent == 0
            else [lambda n: f"Нужно ли платить {n} вперёд и сколько?"]
            if p.scheme == "deferral"
            else [lambda n: f"До какого числа надо перечислять {n} деньги за аренду?"]
            if kind is Kind.LEASE
            else [lambda n: f"Когда мы должны каждый месяц платить {n} за их работу?"]
        ),
        F.PRICE: {
            Kind.SUPPLY: [lambda n: f"На какую максимальную сумму можно закупить товар у {n}?"],
            Kind.SERVICES: [lambda n: f"Во сколько нам обходится в месяц обслуживание у {n}?"],
            Kind.LEASE: [lambda n: f"Сколько мы платим {n} за офис в месяц?"],
            Kind.WORKS: [lambda n: f"Во сколько нам обойдутся работы, которые делает {n}?"],
        }[kind],
        F.PENALTY_US: [
            lambda n: f"Что нам грозит, если мы задержим деньги {n}?",
            lambda n: f"Сколько {n} может начислить нам за опоздание с платежом?",
        ],
        F.PENALTY_THEM: [
            {
                Kind.SUPPLY: lambda n: (
                    f"Что мы можем взыскать с {n}, если они опоздают с поставкой?"
                ),
                Kind.SERVICES: lambda n: (
                    f"Что мы можем взыскать с {n}, если они затянут с услугами?"
                ),
                Kind.WORKS: lambda n: (
                    f"Что мы можем взыскать с {n}, если работы не закончат вовремя?"
                ),
                Kind.LEASE: lambda n: "",
            }[kind]
        ],
        F.ACCEPTANCE: [
            {
                Kind.SUPPLY: lambda n: f"Сколько у нас дней, чтобы пожаловаться {n} на брак?",
                Kind.SERVICES: lambda n: (
                    f"Если мы не подпишем акт от {n} и ничего не возразим, что будет и через сколько?"
                ),
                Kind.WORKS: lambda n: (
                    f"Сколько дней у нас на проверку и подписание актов от {n} за работы?"
                ),
                Kind.LEASE: lambda n: (
                    f"Через сколько после подписания договора {n} должен отдать нам помещение?"
                ),
            }[kind]
        ],
        F.VALIDITY: [lambda n: f"На какой срок мы заключили договор с {n}?"],
        F.PROLONGATION: [lambda n: f"Продлится ли договор с {n} сам, если мы ничего не сделаем?"],
        F.TERMINATION: [
            lambda n: (
                f"Можем ли мы выйти из договора с {n} досрочно и за сколько надо предупредить?"
            )
        ],
        F.CLAIMS: [lambda n: f"Сколько {n} может тянуть с ответом на нашу претензию?"],
        F.COURT: [lambda n: f"Куда идти судиться с {n}, если не договоримся?"],
        F.FORCE_MAJEURE: [
            lambda n: (
                f"Если у {n} случится пожар или наводнение, за сколько дней они должны нам сообщить?"
            )
        ],
        F.DELIVERY: [
            lambda n: f"Через сколько дней после заказа {n} отгрузит товар и кто его везёт?"
        ],
        F.PREMISES: [lambda n: f"Какой метраж у офиса, который мы снимаем у {n}, и где он?"],
        F.DEPOSIT: [
            lambda n: f"Сколько мы должны держать у {n} в качестве гарантийного депозита за офис?"
        ],
        F.INDEXATION: [lambda n: f"На сколько {n} может поднимать аренду каждый год?"],
        F.UTILITIES: [lambda n: f"Платим ли мы {n} за свет и отопление отдельно от аренды?"],
        F.WORKS_PERIOD: [lambda n: f"К какому числу {n} должен всё доделать?"],
        F.WARRANTY: [lambda n: f"Сколько действует гарантия {n} на сделанные работы?"],
        F.RETENTION: [
            lambda n: f"Какую часть денег мы придерживаем у себя до конца гарантии по работам {n}?"
        ],
        F.REQUISITES: [lambda n: f"Куда переводить деньги {n} — какой у них счёт?"],
    }
    return [f for f in t.get(fact, []) if f("x")]


# Факты, о которых задаём вопросы (предмет договора — слишком общий).
QUESTION_FACTS = (
    F.PAYMENT, F.PRICE, F.PENALTY_US, F.PENALTY_THEM, F.ACCEPTANCE, F.VALIDITY, F.PROLONGATION,
    F.TERMINATION, F.CLAIMS, F.COURT, F.FORCE_MAJEURE, F.DELIVERY, F.PREMISES, F.DEPOSIT,
    F.INDEXATION, F.UTILITIES, F.WORKS_PERIOD, F.WARRANTY, F.RETENTION, F.REQUISITES,
)  # fmt: skip


def _fact_question(
    category: str, b: ContractBundle, fact: str, ctx: Ctx, *, paraphrase: bool, suffix: str = ""
) -> Question | None:
    rng = ctx.rng
    if paraphrase:
        options = _paraphrase_templates(fact, b)
        if not options:
            return None
        name = ctx.names.cp(b.base.them, rng, colloquial=True)
        text = rng.choice(options)(name)
        text = text[:-1] + suffix + ctx.names.hint(b) + "?"
    else:
        options = _simple_templates(fact, b.base.kind)
        options = [o for o in options if o("x", "x", "x")]
        if not options:
            return None
        with_cp = None
        d = ctx.names.formal(b, "dat", rng, with_cp)
        p = d.replace("договору", "договоре", 1)
        g = d.replace("договору", "договора", 1)
        text = rng.choice(options)(d, p, g)
        if suffix and text.endswith("?"):
            text = text[:-1] + suffix + "?"
    history = b.history(fact)
    fragment, value, _ = history[-1]
    answer = f"{_capital(value.answer)} ({_cite(b, fragment)})."
    question = Question(
        category,
        text,
        answer,
        list(value.keys),
        "rag",
        sources=[_src(fragment)],
        doc_ids=sorted({fragment.doc_id}),
        template=f"{'para' if paraphrase else 'simple'}:{fact}",
    )
    if category == "amendment_override":
        older = history[:-1]
        question.superseded = [_src(fr) for fr, _, _ in older]
        question.wrong_answers = [v.answer for _, v, _ in older if v.answer != value.answer]
        where = history[-1][2]
        amendment = next(a for a in b.amendments if a.doc.doc_id == fragment.doc_id)
        effective = amendment.effective
        effective_fragment = amendment.doc.fact_fragment(EFFECTIVE)
        assert effective_fragment is not None
        question.sources.append(_src(effective_fragment))
        question.answer = (
            f"{_capital(value.answer)} — по {where.replace('доп. соглашение', 'доп. соглашению')} "
            f"(действует с {date_short(effective)}, {_cite(b, fragment)}). В первоначальной "
            f"редакции договора было: {history[0][1].answer}."
        )
        question.doc_ids = sorted({fr.doc_id for fr, _, _ in history})
    return question


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


# --- Категории по договорам --------------------------------------------------------------


def _round_robin_facts(
    corpus: Corpus, ctx: Ctx, n: int, category: str, *, paraphrase: bool
) -> list[Question]:
    """n вопросов о неизменённых фактах; факты чередуются, договоры не повторяются."""
    rng = ctx.rng
    used: set[str] = set()
    result: list[Question] = []
    facts = list(QUESTION_FACTS)
    rng.shuffle(facts)
    attempts = 0
    while len(result) < n and attempts < n * 50:
        attempts += 1
        fact = facts[len(result) % len(facts)] if attempts % 3 else rng.choice(facts)
        pool = [
            b for b in corpus.bundles
            if b.contract_id not in used and fact in b.facts() and not b.touched(fact)
            and (b.contract_id, fact) not in ctx.used_pairs
        ]  # fmt: skip
        if not pool:
            continue
        b = rng.choice(pool)
        q = _fact_question(category, b, fact, ctx, paraphrase=paraphrase)
        if q is None:
            continue
        used.add(b.contract_id)
        ctx.used_pairs.add((b.contract_id, fact))
        result.append(q)
    return result


_MULTI = [
    ((F.VALIDITY, F.PROLONGATION), None,
     lambda d, p, g: f"Каков срок действия {g} и продлевается ли он автоматически?"),
    ((F.PAYMENT, F.PENALTY_US), None,
     lambda d, p, g: f"В какой срок нужно платить по {d} и какая неустойка предусмотрена за просрочку оплаты?"),
    ((F.DELIVERY, F.ACCEPTANCE), Kind.SUPPLY,
     lambda d, p, g: f"Каковы срок поставки и срок предъявления претензий по качеству по {d}?"),
    ((F.PRICE, F.PAYMENT), Kind.LEASE,
     lambda d, p, g: f"Какой размер арендной платы и до какого числа она вносится по {d}?"),
    ((F.PRICE, F.DEPOSIT, F.INDEXATION), Kind.LEASE,
     lambda d, p, g: f"Назовите размер арендной платы, обеспечительного платежа и условия индексации по {d}."),
    ((F.PRICE, F.UTILITIES), Kind.LEASE,
     lambda d, p, g: f"Сколько стоит аренда по {d} и входят ли в неё коммунальные услуги?"),
    ((F.WORKS_PERIOD, F.WARRANTY), Kind.WORKS,
     lambda d, p, g: f"Каковы сроки выполнения работ и гарантийный срок по {d}?"),
    ((F.WARRANTY, F.RETENTION), Kind.WORKS,
     lambda d, p, g: f"Какой гарантийный срок и какое гарантийное удержание установлены в {p}?"),
    ((F.PRICE, F.PAYMENT), Kind.WORKS,
     lambda d, p, g: f"Какова стоимость работ по {d} и как она оплачивается?"),
    ((F.CLAIMS, F.COURT), None,
     lambda d, p, g: f"Каков срок ответа на претензию и в каком суде разрешаются споры по {d}?"),
    ((F.PENALTY_US, F.PENALTY_THEM), None,
     lambda d, p, g: f"Какие неустойки установлены для каждой из сторон в {p}?"),
    ((F.VALIDITY, F.TERMINATION), None,
     lambda d, p, g: f"Каков срок действия {g} и за сколько дней можно уведомить об одностороннем отказе?"),
    ((F.ACCEPTANCE, F.PAYMENT), Kind.SERVICES,
     lambda d, p, g: f"В какой срок подписывается акт и в какой срок производится оплата по {d}?"),
    ((F.PRICE, F.PAYMENT), Kind.SERVICES,
     lambda d, p, g: f"Сколько стоят услуги по {d} и когда их нужно оплачивать?"),
    ((F.PAYMENT, F.DELIVERY, F.PENALTY_THEM), Kind.SUPPLY,
     lambda d, p, g: f"Назовите по {d} условия оплаты, срок поставки и неустойку за просрочку поставки."),
]  # fmt: skip


def _multi_questions(corpus: Corpus, ctx: Ctx, n: int) -> list[Question]:
    rng = ctx.rng
    used: set[str] = set()
    result: list[Question] = []
    attempts = 0
    while len(result) < n and attempts < n * 60:
        attempts += 1
        facts, kind, template = _MULTI[(len(result) + attempts) % len(_MULTI)]
        pool = [
            b for b in corpus.bundles
            if b.contract_id not in used
            and (kind is None or b.base.kind is kind)
            and all(f in b.facts() and not b.touched(f) for f in facts)
            and not any((b.contract_id, f) in ctx.used_pairs for f in facts)
        ]  # fmt: skip
        if not pool:
            continue
        b = rng.choice(pool)
        d = ctx.names.formal(b, "dat", rng)
        text = template(
            d, d.replace("договору", "договоре", 1), d.replace("договору", "договора", 1)
        )
        parts, keys, sources = [], [], []
        for fact in facts:
            fragment = b.source(fact)
            assert fragment is not None
            value = b.facts()[fact]
            parts.append(f"{F.LABELS[fact]} — {value.answer} ({_cite(b, fragment)})")
            keys += list(value.keys)
            sources.append(_src(fragment))
        used.add(b.contract_id)
        ctx.used_pairs.update((b.contract_id, f) for f in facts)
        result.append(
            Question(
                "multi_condition",
                text,
                _capital("; ".join(parts)) + ".",
                keys,
                "rag",
                sources=sources,
                doc_ids=[b.contract_id],
                template="multi:" + "+".join(facts),
            )
        )
    return result


def _amendment_questions(corpus: Corpus, ctx: Ctx, n: int) -> list[Question]:
    rng = ctx.rng
    candidates = [
        (b, fact) for b in corpus.bundles for fact in QUESTION_FACTS
        if fact in b.facts() and b.changed(fact)
    ]  # fmt: skip
    rng.shuffle(candidates)
    by_fact: dict[str, list[tuple[ContractBundle, str]]] = defaultdict(list)
    for b, fact in candidates:
        by_fact[fact].append((b, fact))
    order = sorted(by_fact, key=lambda f: (-len(by_fact[f]), f))
    used: set[str] = set()
    result: list[Question] = []
    suffixes = ("", " с учётом дополнительных соглашений", " в действующей редакции", " сейчас")
    while len(result) < n and any(by_fact.values()):
        for fact in order:
            if len(result) >= n or not by_fact[fact]:
                continue
            b, _ = by_fact[fact].pop()
            if b.contract_id in used:
                continue
            paraphrase = rng.random() < 0.4
            suffix = rng.choice(suffixes)
            q = _fact_question("amendment_override", b, fact, ctx, paraphrase=paraphrase,
                               suffix=suffix)  # fmt: skip
            if q is None:
                q = _fact_question("amendment_override", b, fact, ctx, paraphrase=not paraphrase,
                                   suffix=suffix)  # fmt: skip
            if q is None:
                continue
            used.add(b.contract_id)
            result.append(q)
    return result


_GREATER = {
    F.PAYMENT: ("срок оплаты длиннее", "срок оплаты"),
    F.PENALTY_US: ("неустойка за просрочку оплаты выше", "неустойка за просрочку оплаты"),
    F.PRICE: ("цена выше", "цена"),
    F.ACCEPTANCE: ("срок приёмки больше", "срок приёмки"),
    F.VALIDITY: ("срок действия дольше", "срок действия"),
    F.PROLONGATION: ("об отказе от пролонгации нужно предупреждать раньше", "срок уведомления об отказе от пролонгации"),
    F.TERMINATION: ("срок уведомления об одностороннем отказе больше", "срок уведомления об одностороннем отказе"),
    F.CLAIMS: ("срок ответа на претензию больше", "срок ответа на претензию"),
    F.FORCE_MAJEURE: ("срок уведомления о форс-мажоре больше", "срок уведомления о форс-мажоре"),
    F.DELIVERY: ("срок поставки больше", "срок поставки"),
    F.PREMISES: ("арендуемое помещение больше", "площадь помещения"),
    F.DEPOSIT: ("обеспечительный платёж больше", "обеспечительный платёж"),
    F.INDEXATION: ("ежегодная индексация аренды выше", "индексация аренды"),
    F.WARRANTY: ("гарантийный срок дольше", "гарантийный срок"),
    F.RETENTION: ("гарантийное удержание больше", "гарантийное удержание"),
}  # fmt: skip
_PRICE_WORD = {
    "money:lease": ("арендная плата выше", "арендная плата"),
    "money:services": ("ежемесячная стоимость услуг выше", "ежемесячная стоимость услуг"),
    "money:works": ("стоимость работ выше", "стоимость работ"),
    "money:supply": ("предельная сумма поставок больше", "предельная сумма поставок"),
}


def _comparison_questions(corpus: Corpus, ctx: Ctx, n: int) -> list[Question]:
    rng = ctx.rng
    names = ctx.names
    pools: dict[tuple[str, str], list[ContractBundle]] = defaultdict(list)
    for b in corpus.bundles:
        for fact, value in b.facts().items():
            if fact in _GREATER and value.value is not None and value.unit:
                pools[(fact, value.unit)].append(b)
    keys_order = sorted(k for k, v in pools.items() if len(v) >= 2)
    result: list[Question] = []
    used: set[str] = set()
    n_equal = 3
    attempts = 0
    while len(result) < n and attempts < n * 80:
        attempts += 1
        fact, unit = keys_order[attempts % len(keys_order)]
        pool = [b for b in pools[(fact, unit)] if b.contract_id not in used]
        if len(pool) < 2:
            continue
        a = rng.choice(pool)
        same_kind = [x for x in pool if x is not a and x.base.kind is a.base.kind]
        if not same_kind:
            continue
        same_cp = [x for x in same_kind if x.base.them.inn == a.base.them.inn]
        b = rng.choice(same_cp) if same_cp and rng.random() < 0.6 else rng.choice(same_kind)
        va, vb = a.facts()[fact], b.facts()[fact]
        assert va.value is not None and vb.value is not None
        want_equal = len(result) >= n - n_equal
        if (va.value == vb.value) != want_equal:
            continue
        greater, label = _PRICE_WORD[unit] if fact == F.PRICE else _GREATER[fact]
        ra = names.formal(a, "nom", rng, with_cp=True)
        rb = names.formal(b, "nom", rng, with_cp=True)
        fa, fb = a.source(fact), b.source(fact)
        assert fa is not None and fb is not None
        both = (
            f"По договору № {a.base.number} с {a.base.them.name} — {va.answer} ({_cite(a, fa)}); "
            f"по договору № {b.base.number} с {b.base.them.name} — {vb.answer} ({_cite(b, fb)})."
        )
        if want_equal:
            text = f"Одинаковые ли условия ({label}) в двух договорах: {ra} и {rb}?"
            answer = f"Да, одинаковые. {both}"
            template = f"compare-equal:{fact}"
        else:
            if rng.random() < 0.5:
                text = f"В каком из договоров {greater}: {ra} или {rb}?"
            else:
                text = f"Сравните два договора — {ra} и {rb}: где {greater}?"
            winner = a if va.value > vb.value else b
            answer = f"Больше по договору № {winner.base.number} с {winner.base.them.name}. {both}"
            template = f"compare:{fact}"
        keys = list(va.keys) + list(vb.keys)
        if not want_equal:
            keys.append(winner.base.number)
        used.update((a.contract_id, b.contract_id))
        result.append(
            Question(
                "comparison",
                text,
                answer,
                keys,
                "rag",
                sources=[_src(fa), _src(fb)],
                doc_ids=sorted({fa.doc_id, fb.doc_id}),
                template=template,
            )
        )
    return result


# --- Реестр ------------------------------------------------------------------------------

_TYPE_PLURAL_GEN = {
    "upd": "УПД",
    "bill": "счетов на оплату",
    "act": "актов",
    "torg12": "накладных ТОРГ-12",
    "vat_invoice": "счетов-фактур",
}
_MONTHS_PREP = (
    "январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре",
    "октябре", "ноябре", "декабре",
)  # fmt: skip
_QUARTERS = ("I", "II", "III", "IV")
SHIPPING_TYPES = ("upd", "torg12", "act")


def _period(
    rng: random.Random,
    rows: list[RegistryRow],
    kinds: tuple[str, ...] = ("year", "quarter", "month"),
) -> tuple[str, str, str, str]:
    """(текст «в марте 2025 года», SQL-условие, начало ISO, конец ISO) по датам строк."""
    d = date.fromisoformat(rng.choice(rows).date)
    kind = rng.choice(kinds)
    if kind == "year":
        start, end = date(d.year, 1, 1), date(d.year, 12, 31)
        text = f"за {d.year} год"
    elif kind == "quarter":
        q = (d.month - 1) // 3
        start = date(d.year, q * 3 + 1, 1)
        end = date(d.year + (q == 3), (q * 3 + 3) % 12 + 1, 1)
        end = date.fromordinal(end.toordinal() - 1)
        text = f"за {_QUARTERS[q]} квартал {d.year} года"
    else:
        start = date(d.year, d.month, 1)
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        end = date.fromordinal(nxt.toordinal() - 1)
        text = f"в {_MONTHS_PREP[d.month - 1]} {d.year} года"
    return (
        text,
        f"date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'",
        start.isoformat(),
        end.isoformat(),
    )


def _in(rows: list[RegistryRow], start: str, end: str) -> list[RegistryRow]:
    return [r for r in rows if start <= r.date <= end]


def _docs(n: int) -> str:
    return f"{n} {plural(n, ('документ', 'документа', 'документов'))}"


def _money_key(amount: Decimal) -> str:
    return money_digits(amount).replace(" ", "")


def _registry_questions(corpus: Corpus, ctx: Ctx, n: int) -> list[Question]:
    rng = ctx.rng
    rows = corpus.registry
    by_cp: dict[str, list[RegistryRow]] = defaultdict(list)
    for r in rows:
        by_cp[r.seller_inn].append(r)
    big = sorted((inn for inn, rs in by_cp.items() if len(rs) >= 4), key=lambda i: -len(by_cp[i]))
    by_contract: dict[str, list[RegistryRow]] = defaultdict(list)
    for r in rows:
        by_contract[r.contract_id].append(r)
    result: list[Question] = []
    seen: set[str] = set()

    def add(text: str, answer: str, keys: list[str], used_rows: list[RegistryRow], sql: str,
            value: str, template: str, sql_detail: str | None = None) -> None:  # fmt: skip
        if text in seen:
            return
        seen.add(text)
        result.append(
            Question(
                "registry_aggregate",
                text,
                answer,
                keys,
                "table",
                registry_rows=sorted(r.doc_id for r in used_rows),
                sql=sql,
                sql_detail=sql_detail,
                expected_value=value,
                template=template,
            )
        )

    makers = ["count_type", "sum_shipping", "count_contract", "sum_bills", "top_cp", "max_doc",
              "last_doc", "distinct_cp", "vat_sum"]  # fmt: skip
    attempts = 0
    while len(result) < n and attempts < n * 40:
        attempts += 1
        maker = makers[len(result) % len(makers)] if attempts % 4 else rng.choice(makers)
        if sum(q.template == maker for q in result) >= 4:  # не больше 4 вопросов одного вида
            continue
        inn = rng.choice(big)
        cp_rows = by_cp[inn]
        cp = cp_rows[0].seller_name
        where_cp = f"seller_inn = '{inn}'"
        if maker == "count_type":
            doc_type = rng.choice(sorted({r.doc_type for r in cp_rows}))
            period, cond, start, end = _period(
                rng, [r for r in cp_rows if r.doc_type == doc_type], ("year", "quarter")
            )
            found = [r for r in _in(cp_rows, start, end) if r.doc_type == doc_type]
            if len(found) < 2:
                continue
            listing = ", ".join(
                f"№ {r.number} от {date_short(date.fromisoformat(r.date))}" for r in found
            )
            add(
                f"Сколько {_TYPE_PLURAL_GEN[doc_type]} получено от {cp} {period}?",
                f"{len(found)}: {listing} (реестр первичных документов).",
                [str(len(found))], found,
                f"SELECT COUNT(*) FROM registry WHERE {where_cp} AND doc_type = '{doc_type}' AND {cond}",
                str(len(found)), "count_type",
            )  # fmt: skip
        elif maker == "sum_shipping":
            ship = [r for r in cp_rows if r.doc_type in SHIPPING_TYPES]
            if not ship:
                continue
            period, cond, start, end = _period(rng, ship, ("year", "quarter"))
            found = _in(ship, start, end)
            if len(found) < 2:
                continue
            total = sum((r.amount_with_vat for r in found), Decimal(0))
            add(
                f"На какую общую сумму с НДС получено отгрузочных документов (УПД, ТОРГ-12, акты) "
                f"от {cp} {period}?",
                f"{money_answer(total)}, документов: {len(found)} (реестр первичных документов).",
                [_money_key(total)], found,
                f"SELECT ROUND(SUM(CAST(amount_with_vat AS REAL)), 2) FROM registry WHERE {where_cp} "
                f"AND doc_type IN ('upd', 'torg12', 'act') AND {cond}",
                f"{total:.2f}", "sum_shipping",
            )  # fmt: skip
        elif maker == "count_contract":
            contract_id = rng.choice(sorted({r.contract_id for r in cp_rows}))
            found = by_contract[contract_id]
            r0 = found[0]
            kinds = Counter(r.doc_type_name for r in found)
            detail = ", ".join(f"{name} — {k}" for name, k in sorted(kinds.items()))
            add(
                f"Сколько документов зарегистрировано в реестре по договору № {r0.contract_number} "
                f"от {date_short(date.fromisoformat(r0.contract_date))} с {cp}?",
                f"{len(found)} ({detail}).",
                [str(len(found))], found,
                f"SELECT COUNT(*) FROM registry WHERE contract_id = '{contract_id}'",
                str(len(found)), "count_contract",
                f"SELECT doc_type_name, COUNT(*) FROM registry WHERE contract_id = '{contract_id}' "
                f"GROUP BY doc_type_name ORDER BY doc_type_name",
            )  # fmt: skip
        elif maker == "sum_bills":
            bills = [r for r in cp_rows if r.doc_type == "bill"]
            if not bills:
                continue
            year = date.fromisoformat(rng.choice(bills).date).year
            found = [r for r in bills if r.date.startswith(f"{year}-")]
            if len(found) < 2:
                continue
            total = sum((r.amount_with_vat for r in found), Decimal(0))
            add(
                f"На какую общую сумму {cp} выставил нам счетов на оплату в {year} году?",
                f"{money_answer(total)}, счетов: {len(found)} (реестр первичных документов).",
                [_money_key(total)], found,
                f"SELECT ROUND(SUM(CAST(amount_with_vat AS REAL)), 2) FROM registry WHERE {where_cp} "
                f"AND doc_type = 'bill' AND date LIKE '{year}-%'",
                f"{total:.2f}", "sum_bills",
            )  # fmt: skip
        elif maker == "top_cp":
            period, cond, start, end = _period(rng, rows)
            found = _in(rows, start, end)
            counts = Counter(r.seller_inn for r in found)
            if not counts:
                continue
            (top, k1), *rest = counts.most_common()
            if rest and rest[0][1] == k1:
                continue
            name = next(r.seller_name for r in found if r.seller_inn == top)
            add(
                f"От какого контрагента получено больше всего документов {period}?",
                f"{name} (ИНН {top}) — {_docs(k1)}; всего за период {_docs(len(found))} "
                f"(реестр первичных документов).",
                [short_name_from(name), str(k1)], [r for r in found if r.seller_inn == top],
                f"SELECT seller_name, COUNT(*) AS n FROM registry WHERE {cond} GROUP BY seller_inn "
                f"ORDER BY n DESC LIMIT 1",
                name, "top_cp",
            )  # fmt: skip
        elif maker == "max_doc":
            ordered = sorted(cp_rows, key=lambda r: r.amount_with_vat, reverse=True)
            if len(ordered) > 1 and ordered[0].amount_with_vat == ordered[1].amount_with_vat:
                continue
            r = ordered[0]
            add(
                f"Какой самый крупный по сумме документ получен от {cp}?",
                f"{r.doc_type_name} № {r.number} от {date_short(date.fromisoformat(r.date))} на "
                f"{money_answer(r.amount_with_vat)} с НДС (реестр первичных документов).",
                [r.number, _money_key(r.amount_with_vat)], [r],
                f"SELECT number, date, amount_with_vat FROM registry WHERE {where_cp} "
                f"ORDER BY CAST(amount_with_vat AS REAL) DESC LIMIT 1",
                r.number, "max_doc",
            )  # fmt: skip
        elif maker == "last_doc":
            ordered = sorted(cp_rows, key=lambda r: r.date, reverse=True)
            if len(ordered) > 1 and ordered[0].date == ordered[1].date:
                continue
            r = ordered[0]
            add(
                f"Когда получен последний документ от {cp} и что это за документ?",
                f"{date_short(date.fromisoformat(r.date))}: {r.doc_type_name} № {r.number} на "
                f"{money_answer(r.amount_with_vat)} (реестр первичных документов).",
                [date_short(date.fromisoformat(r.date)), r.number], [r],
                f"SELECT date, doc_type, number FROM registry WHERE {where_cp} ORDER BY date DESC LIMIT 1",
                r.date, "last_doc",
            )  # fmt: skip
        elif maker == "distinct_cp":
            period, cond, start, end = _period(rng, rows)
            if "год" in period and "квартал" not in period:
                continue
            found = _in(rows, start, end)
            k = len({r.seller_inn for r in found})
            add(
                f"Сколько разных контрагентов прислали нам документы {period}?",
                f"{k} {plural(k, ('контрагент', 'контрагента', 'контрагентов'))}, {_docs(len(found))} "
                f"(реестр первичных документов).",
                [str(k)], found,
                f"SELECT COUNT(DISTINCT seller_inn) FROM registry WHERE {cond}",
                str(k), "distinct_cp",
            )  # fmt: skip
        else:
            vat_rows = [r for r in cp_rows if r.doc_type in ("upd", "vat_invoice")]
            if not vat_rows:
                continue
            year = date.fromisoformat(rng.choice(vat_rows).date).year
            found = [r for r in vat_rows if r.date.startswith(f"{year}-")]
            if len(found) < 2:
                continue
            total = sum((r.vat_amount for r in found), Decimal(0))
            add(
                f"Сколько НДС предъявил нам {cp} по счетам-фактурам и УПД за {year} год?",
                f"{money_answer(total)}, документов: {len(found)} (реестр первичных документов).",
                [_money_key(total)], found,
                f"SELECT ROUND(SUM(CAST(vat_amount AS REAL)), 2) FROM registry WHERE {where_cp} "
                f"AND doc_type IN ('upd', 'vat_invoice') AND date LIKE '{year}-%'",
                f"{total:.2f}", "vat_sum",
            )  # fmt: skip
    return result


def short_name_from(full_name: str) -> str:
    if full_name.startswith("ИП "):
        return full_name.split()[1]
    return full_name.split("«", 1)[1].rstrip("»")


def run_sql(rows: list[RegistryRow], sql: str) -> list[tuple[Any, ...]]:
    """Выполнить эталонный SQL на реестре в памяти (SQLite из стандартной библиотеки)."""
    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE registry ({', '.join(f'{c} TEXT' for c in COLUMNS)})")
    con.executemany(
        f"INSERT INTO registry VALUES ({', '.join('?' for _ in COLUMNS)})",
        [r.as_csv() for r in rows],
    )
    return con.execute(sql).fetchall()


# --- Ответа нет --------------------------------------------------------------------------

_FAKE_FIRST = ("Кристалл", "Пеликан", "Сапфир", "Ангара", "Оникс", "Полюс", "Ирбис", "Кварц")
_FAKE_SECOND = ("Снаб", "Трейд", "Сервис", "Логистик", "Строй", "Ресурс")


def _no_answer_questions(corpus: Corpus, ctx: Ctx, n: int) -> list[Question]:
    rng = ctx.rng
    names = ctx.names
    result: list[Question] = []
    all_names = {b.base.them.name for b in corpus.bundles}
    all_numbers = {b.base.number for b in corpus.bundles}

    def add(text: str, answer: str, template: str, doc_ids: list[str] | None = None) -> None:
        result.append(
            Question(
                "no_answer", text, answer, [], "none", doc_ids=doc_ids or [], template=template
            )
        )

    # A. Контрагента нет в корпусе.
    fakes: list[str] = []
    while len(fakes) < 4:
        name = f"ООО «{rng.choice(_FAKE_FIRST)}-{rng.choice(_FAKE_SECOND)}»"
        if name not in all_names and name not in fakes:
            fakes.append(name)
    asks = [
        "Какой срок оплаты по договору поставки с {n}?",
        "До какого числа действует договор с {n}?",
        "Какая неустойка за просрочку оплаты предусмотрена в договоре с {n}?",
        "Сколько мы платим {n} за аренду офиса?",
    ]
    for name, ask in zip(fakes, asks, strict=True):
        add(
            ask.format(n=name),
            f"{NO_ANSWER} Договоров с {name} в корпусе нет.",
            "unknown_counterparty",
        )

    # B. Контрагент есть, договора с таким номером нет.
    pool = [b for b in corpus.bundles]
    rng.shuffle(pool)
    for b in pool[:4]:
        while True:
            number = f"{rng.randint(800, 999)}-{rng.choice(('П', 'У', 'А'))}/{rng.choice((24, 25))}"
            if number not in all_numbers:
                break
        existing = ", ".join(f"№ {x.base.number}" for x in names.by_cp[b.base.them.inn])
        add(
            f"Каков срок действия договора № {number} с {b.base.them.name}?",
            f"{NO_ANSWER} Договора № {number} с {b.base.them.name} в корпусе нет; с этим "
            f"контрагентом есть договоры: {existing}.",
            "unknown_number",
            [x.contract_id for x in names.by_cp[b.base.them.inn]],
        )

    # C. Условие могло бы быть, но в этом договоре его нет.
    missing = [
        (F.DEPOSIT, Kind.LEASE, "Какой обеспечительный платёж предусмотрен в {p}?",
         "обеспечительный платёж не предусмотрен"),
        (F.RETENTION, Kind.WORKS, "Какое гарантийное удержание предусмотрено в {p}?",
         "гарантийное удержание не предусмотрено"),
        (F.INDEXATION, Kind.LEASE, "На сколько ежегодно индексируется арендная плата по {d}?",
         "индексация арендной платы не предусмотрена"),
        (F.FORCE_MAJEURE, None, "В какой срок нужно уведомить о форс-мажоре по {d}?",
         "нет раздела об обстоятельствах непреодолимой силы и срока уведомления о них"),
        (F.TERMINATION, None, "За сколько дней нужно предупредить об одностороннем отказе от {g}?",
         "нет условия об одностороннем отказе с уведомлением"),
        (F.PENALTY_THEM, Kind.SUPPLY, "Какая неустойка установлена за просрочку поставки по {d}?",
         "неустойка за просрочку поставки не установлена"),
    ]  # fmt: skip
    used: set[str] = set()
    for fact, kind, ask, reason in missing:
        options = [
            b for b in corpus.bundles
            if fact not in b.facts() and (kind is None or b.base.kind is kind)
            and b.contract_id not in used
        ]  # fmt: skip
        if not options:
            continue
        b = rng.choice(options)
        used.add(b.contract_id)
        d = names.formal(b, "dat", rng, with_cp=True)
        text = ask.format(
            d=d, p=d.replace("договору", "договоре", 1), g=d.replace("договору", "договора", 1)
        )
        add(text, f"{NO_ANSWER} В договоре и доп. соглашениях к нему {reason}.", f"missing:{fact}",
            [x.doc_id for x in b.docs])  # fmt: skip

    # D. Таких сведений нет ни в одном документе.
    absent = [
        "Какой адрес электронной почты для направления претензий указан в {p}?",
        "Предусмотрено ли страхование товара при перевозке в {p}?",
        "Какой сотрудник контрагента (ФИО, телефон) назначен ответственным за исполнение обязательств в {p}?",
    ]
    kinds = [Kind.SUPPLY, Kind.SUPPLY, Kind.SERVICES]
    for ask, kind in zip(absent, kinds, strict=True):
        b = rng.choice(
            [x for x in corpus.bundles if x.base.kind is kind and x.contract_id not in used]
        )
        used.add(b.contract_id)
        p = names.formal(b, "prep", rng, with_cp=True)
        add(ask.format(p=p), f"{NO_ANSWER} В документах корпуса таких сведений нет.",
            "absent_concept", [x.doc_id for x in b.docs])  # fmt: skip
    return result[:n] if len(result) >= n else result


_REG_NO_ANSWER = [
    "Какой лимит расходов на корпоративную мобильную связь установлен в компании?",
    "Сколько дней ежегодного оплачиваемого отпуска положено сотрудникам по внутренним регламентам?",
    "В какие сроки выплачивается премия по итогам года?",
]


# --- Регламенты --------------------------------------------------------------------------

_REG_MULTI = [
    (("adv.per_diem", "adv.hotel"),
     "Какие суточные и какой лимит на проживание установлены для командировок?"),
    (("doc.transfer_deadline", "doc.scan"),
     "В какой срок первичные документы передаются в бухгалтерию и можно ли принять к учёту скан-копию?"),
    (("ctr.approval_term", "ctr.registration"),
     "Сколько времени отводится на согласование проекта договора и в какой срок подписанный договор передаётся в бухгалтерию?"),
    (("adv.term", "adv.report_term"),
     "На какой срок выдаются деньги под отчёт и когда нужно сдать авансовый отчёт?"),
    (("pay.request_deadline", "pay.no_contract_limit"),
     "За сколько дней подаётся заявка на оплату и можно ли платить по счёту без договора?"),
    (("acc.os_limit", "acc.depreciation"),
     "Какой лимит стоимости основных средств и какой способ амортизации установлены в учётной политике для бухгалтерского учёта?"),
]  # fmt: skip


def _strip_ref(answer: str) -> str:
    """Убрать из ответа ссылку на пункт в скобках в конце: «… (регламент …, п. 2.1).»."""
    return re.sub(r"\s*\([^()]*п\. [\d.]+\)\.?$", "", answer).rstrip(".")


def _reg_questions(corpus: Corpus, rng: random.Random) -> dict[str, list[Question]]:
    docs = {d.doc_id: d for d in corpus.regulations}
    amendment_doc, amended_doc = (docs[x] for x in REG_AMENDMENT)

    def source(key: str) -> Fragment:
        for doc in corpus.regulations:
            if doc.doc_id == amendment_doc.doc_id:
                continue
            frag = doc.fact_fragment(key)
            if frag is not None:
                return frag
        raise KeyError(key)

    def reg_q(category: str, key: str, text: str) -> Question:
        fact = REG_FACTS[key]
        frag = source(key)
        title = docs[frag.doc_id].meta["title"]
        q = Question(category, text, fact.answer, list(fact.keys), "rag", sources=[_src(frag)],
                     doc_ids=[frag.doc_id], template=f"reg:{key}")  # fmt: skip
        if not fact.answer.endswith(")."):
            q.answer = f"{fact.answer.rstrip('.')} ({title.lower()}, п. {frag.label})."
        return q

    amended = sorted(REG_SUPERSEDED)
    plain = sorted(k for k in REG_FACTS if k not in REG_SUPERSEDED)
    rng.shuffle(plain)
    out: dict[str, list[Question]] = defaultdict(list)
    n_simple, n_para = QUOTAS["simple_fact"][1], QUOTAS["paraphrase"][1]
    for key in plain[:n_simple]:
        out["simple_fact"].append(reg_q("simple_fact", key, REG_FACTS[key].simple))
    for key in plain[n_simple : n_simple + n_para]:
        out["paraphrase"].append(reg_q("paraphrase", key, REG_FACTS[key].paraphrase))
    multi = list(_REG_MULTI)
    rng.shuffle(multi)
    for keys, text in multi[: QUOTAS["multi_condition"][1]]:
        frags = [source(k) for k in keys]
        parts = [_strip_ref(REG_FACTS[k].answer) for k in keys]
        parts = parts[:1] + [p[:1].lower() + p[1:] for p in parts[1:]]
        title = docs[frags[0].doc_id].meta["title"]
        out["multi_condition"].append(
            Question(
                "multi_condition",
                text,
                "; ".join(parts) + f" ({title.lower()}, пп. {', '.join(f.label for f in frags)}).",
                [k for key in keys for k in REG_FACTS[key].keys],
                "rag",
                sources=[_src(f) for f in frags],
                doc_ids=[frags[0].doc_id],
                template="reg-multi:" + "+".join(keys),
            )
        )
    rng.shuffle(amended)
    for key in amended[: QUOTAS["amendment_override"][1]]:
        fact = REG_FACTS[key]
        new_frag = amendment_doc.fact_fragment(key)
        old_frag = amended_doc.fact_fragment(key)
        effective_frag = amendment_doc.fact_fragment("reg.effective")
        assert new_frag is not None and old_frag is not None and effective_frag is not None
        text = rng.choice((fact.simple, fact.paraphrase))
        out["amendment_override"].append(
            Question(
                "amendment_override",
                text,
                fact.answer,
                list(fact.keys),
                "rag",
                sources=[_src(new_frag), _src(effective_frag)],
                superseded=[_src(old_frag)],
                wrong_answers=[REG_SUPERSEDED[key]],
                doc_ids=[amendment_doc.doc_id, amended_doc.doc_id],
                template=f"reg-amend:{key}",
            )
        )
    for text in _REG_NO_ANSWER[: QUOTAS["no_answer"][1]]:
        out["no_answer"].append(
            Question(
                "no_answer",
                text,
                f"{NO_ANSWER} Во внутренних регламентах этого нет.",
                [],
                "none",
                template="reg-absent",
            )
        )
    return out


# --- Сборка ------------------------------------------------------------------------------


def generate_questions(corpus: Corpus, seed: int) -> list[Question]:
    """200 вопросов по квотам `QUOTAS`, id Q001…Q200 по порядку категорий."""
    names = Names(corpus)

    def ctx(tag: str) -> Ctx:
        return Ctx(names, random.Random(f"docai-rag:questions:{seed}:{tag}"))

    by_category: dict[str, list[Question]] = {
        "simple_fact": _round_robin_facts(
            corpus, ctx("simple"), QUOTAS["simple_fact"][0], "simple_fact", paraphrase=False
        ),
        "paraphrase": _round_robin_facts(
            corpus, ctx("para"), QUOTAS["paraphrase"][0], "paraphrase", paraphrase=True
        ),
        "multi_condition": _multi_questions(corpus, ctx("multi"), QUOTAS["multi_condition"][0]),
        "amendment_override": _amendment_questions(
            corpus, ctx("amend"), QUOTAS["amendment_override"][0]
        ),
        "comparison": _comparison_questions(corpus, ctx("compare"), QUOTAS["comparison"][0]),
        "registry_aggregate": _registry_questions(
            corpus, ctx("registry"), QUOTAS["registry_aggregate"][0]
        ),
        "no_answer": _no_answer_questions(corpus, ctx("none"), QUOTAS["no_answer"][0]),
    }
    regs = _reg_questions(corpus, random.Random(f"docai-rag:questions:{seed}:reg"))
    result: list[Question] = []
    for category in CATEGORIES:
        items = by_category[category] + regs.get(category, [])
        want = sum(QUOTAS[category])
        if len(items) != want:
            raise RuntimeError(
                f"категория {category}: получилось {len(items)} вопросов, нужно {want}"
            )
        result += items
    texts = Counter(q.question for q in result)
    duplicates = [t for t, k in texts.items() if k > 1]
    if duplicates:
        raise RuntimeError(f"повторяются вопросы: {duplicates[:3]}")
    for i, q in enumerate(result, start=1):
        q.qid = f"Q{i:03d}"
    return result
