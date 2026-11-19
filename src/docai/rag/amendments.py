"""Дополнительные соглашения: меняют условия договора, иногда отменяют их.

Соглашение генерируется как список изменений условий (новый `Contract`) и печатается
либо новой редакцией пункта («Пункт 2.3 Договора изложить в следующей редакции: …»),
либо короткой правкой («Срок оплаты … увеличить с 30 до 45 календарных дней»).
Действующие условия — исходные с применёнными по порядку соглашениями.
"""

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal

from docai.rag import facts as F
from docai.rag.contract_text import (
    CLAUSE_RENDERERS,
    KIND_DAT,
    Style,
    capitalize,
    party_intro,
    tidy_commas,
    with_style,
)
from docai.rag.document import Clause, CorpusDoc, Section
from docai.rag.terms import TODAY, Contract, Kind, Validity, new_bank_account
from docai.rag.words import days_gen, rate_text

# Пункт «Соглашение вступает в силу с …»: дата из него входит в эталон, значит и в источники.
EFFECTIVE = "amendment.effective"


@dataclass(slots=True)
class Amendment:
    """Доп. соглашение: документ и условия договора после него."""

    doc: CorpusDoc
    number: int
    signed: date
    effective: date
    changed: tuple[str, ...]  # ключи фактов, которые изменились
    before: Contract
    after: Contract


def _change_candidates(c: Contract) -> list[str]:
    options: list[str] = []
    p = c.payment
    if p.scheme == "deferral" and p.prepay_percent < 100:
        options += [F.PAYMENT, F.PAYMENT]
    if c.penalty_us is not None and c.penalty_us.kind == "percent":
        options.append(F.PENALTY_US)
    if c.validity.kind == "until_date":
        options += [F.VALIDITY, F.VALIDITY]
    if c.prolongation is not None and c.prolongation.auto:
        options.append(F.PROLONGATION)
    if c.price is not None and c.kind in (Kind.LEASE, Kind.SERVICES, Kind.WORKS):
        options.append(F.PRICE)
    options.append(F.REQUISITES)
    if c.kind is Kind.SUPPLY:
        options.append(F.DELIVERY)
    if c.kind is Kind.WORKS:
        options.append(F.WORKS_PERIOD)
    if c.termination_days is not None:
        options.append(F.TERMINATION)
    options.append(F.ACCEPTANCE)
    return options


def _changed_contract(c: Contract, fact: str, rng: random.Random, signed: date) -> Contract:
    if fact == F.PAYMENT:
        p = c.payment
        if p.prepay_percent and rng.random() < 0.5:
            new_prepay = rng.choice([x for x in (0, 20, 30, 50) if x != p.prepay_percent])
            return replace(c, payment=replace(p, prepay_percent=new_prepay,
                                              prepay_days=p.prepay_days or 5))  # fmt: skip
        assert p.days is not None
        new_days = rng.choice([x for x in (10, 14, 15, 20, 30, 45, 60, 90) if x != p.days])
        return replace(c, payment=replace(p, days=new_days))
    if fact == F.PENALTY_US:
        assert c.penalty_us is not None
        rates = [
            r for r in ("0.01", "0.03", "0.05", "0.1", "0.2") if Decimal(r) != c.penalty_us.rate
        ]
        return replace(c, penalty_us=replace(c.penalty_us, rate=Decimal(rng.choice(rates))))
    if fact == F.VALIDITY:
        assert c.validity.end is not None
        end = date(max(c.validity.end.year, signed.year) + rng.choice((1, 1, 2)), 12, 31)
        return replace(c, validity=Validity("until_date", end=end))
    if fact == F.PROLONGATION:
        assert c.prolongation is not None
        return replace(c, prolongation=replace(c.prolongation, auto=False, notice_days=None))
    if fact == F.PRICE:
        assert c.price is not None
        factor = Decimal(rng.choice(("1.05", "1.08", "1.1", "1.15", "1.2")))
        step = Decimal(100) if c.kind is not Kind.WORKS else Decimal(1000)
        return replace(c, price=(c.price * factor / step).quantize(Decimal(1)) * step)
    if fact == F.REQUISITES:
        bank, account = new_bank_account(rng, c.them, c.them_bank)
        return replace(c, them_bank=bank, them_account=account)
    if fact == F.DELIVERY:
        assert c.delivery_days is not None
        days = rng.choice([x for x in (2, 3, 5, 7, 10, 14, 20) if x != c.delivery_days])
        return replace(c, delivery_days=days)
    if fact == F.WORKS_PERIOD:
        assert c.works_end is not None
        return replace(c, works_end=c.works_end + timedelta(days=rng.choice((14, 30, 45, 60))))
    if fact == F.TERMINATION:
        assert c.termination_days is not None
        days = rng.choice([x for x in (10, 15, 30, 45, 60, 90) if x != c.termination_days])
        return replace(c, termination_days=days)
    if fact == F.ACCEPTANCE:
        days = rng.choice([x for x in (2, 3, 5, 7, 10, 14) if x != c.acceptance_days])
        return replace(c, acceptance_days=days)
    raise ValueError(fact)


def _restated(label: str, text: str, s: Style) -> str:
    return s.pick(
        f"Пункт {label} Договора изложить в следующей редакции: «{label}. {text}»",
        f"Стороны договорились изложить п. {label} Договора в новой редакции: «{text}»",
        f"Внести изменение в Договор, изложив пункт {label} следующим образом: «{text}»",
    )


def _short_change(fact: str, label: str, old: Contract, new: Contract, s: Style) -> str | None:
    """Короткая правка вместо новой редакции, если она однозначна; иначе None."""
    _, them = new.roles
    if fact == F.PAYMENT and old.payment.prepay_percent == 0 and new.payment.prepay_percent == 0:
        assert old.payment.days is not None and new.payment.days is not None
        verb = "увеличить" if new.payment.days > old.payment.days else "сократить"
        kind = new.payment.day_kind
        return s.pick(
            f"Срок оплаты, установленный пунктом {label} Договора, {verb} с "
            f"{days_gen(old.payment.days, kind)} до {days_gen(new.payment.days, kind)}.",
            f"Стороны пришли к соглашению изменить срок оплаты по п. {label} Договора: вместо "
            f"{days_gen(old.payment.days, kind)} оплата производится в течение "
            f"{days_gen(new.payment.days, kind)} с той же даты.",
        )
    if (
        fact == F.PENALTY_US
        and old.penalty_us
        and new.penalty_us
        and not new.penalty_us.cap_percent
    ):
        return s.pick(
            f"Размер неустойки за просрочку оплаты, указанный в п. {label} Договора, изменить "
            f"с {rate_text(old.penalty_us.rate, 0)} на {rate_text(new.penalty_us.rate, 0)} "
            f"за каждый день просрочки.",
            f"Пени за нарушение срока оплаты (п. {label} Договора) установить в размере "
            f"{rate_text(new.penalty_us.rate, 2)} за каждый день просрочки вместо "
            f"{rate_text(old.penalty_us.rate, 2)}.",
        )
    if fact == F.VALIDITY:
        end = s.date(new.validity.end)
        return s.pick(
            f"Продлить срок действия Договора до {end}.",
            f"Стороны пришли к соглашению продлить действие Договора по {end} включительно.",
        )
    if fact == F.PROLONGATION:
        return s.pick(
            f"Пункт {label} Договора (об автоматической пролонгации) исключить. По окончании "
            f"срока действия Договор прекращается.",
            f"Исключить из Договора пункт {label}; автоматическое продление Договора "
            f"не применяется, продление возможно только отдельным соглашением.",
        )
    if fact == F.REQUISITES:
        b = new.them_bank
        return s.pick(
            f"Изложить банковские реквизиты {them.gen} в следующей редакции: р/с "
            f"{new.them_account} в {b.name}, БИК {b.bik}, к/с {b.corr_account}. Оплата по "
            f"прежним реквизитам с даты вступления в силу Соглашения не производится.",
            f"В связи со сменой обслуживающего банка {them.nom} просит перечислять платежи "
            f"по Договору на новый счёт: р/с {new.them_account} в {b.name}, БИК {b.bik}, "
            f"к/с {b.corr_account}.",
        )
    if fact == F.PRICE and new.kind is Kind.LEASE:
        assert new.price is not None
        return f"Установить размер арендной платы {s.money(new.price)} в месяц."
    return None


def _effective_date(rng: random.Random, signed: date) -> date:
    if rng.random() < 0.5:
        return signed
    first_next = date(signed.year + signed.month // 12, signed.month % 12 + 1, 1)
    return first_next


def generate_amendments(
    c: Contract, contract_doc: CorpusDoc, s: Style, seed: int
) -> list[Amendment]:
    """0, 1 или 2 доп. соглашения к договору (у ~30% договоров есть хотя бы одно)."""
    rng = random.Random(f"docai-rag:amend:{seed}:{c.contract_id}")
    count = rng.choices((0, 1, 2), weights=(68, 25, 7))[0]
    result: list[Amendment] = []
    current = c
    last = c.date
    latest = TODAY - timedelta(days=20)
    for number in range(1, count + 1):
        signed = last + timedelta(days=rng.randint(30, 420))
        end = current.end_date
        if end is not None and not (current.prolongation and current.prolongation.auto):
            latest_allowed = min(latest, end - timedelta(days=5))
        else:
            latest_allowed = latest
        if signed > latest_allowed:
            break
        effective = _effective_date(rng, signed)
        if effective > TODAY:
            effective = signed
        candidates = _change_candidates(current)
        n_changes = 1 if rng.random() < 0.7 else 2
        changed: list[str] = []
        while candidates and len(changed) < n_changes:
            fact = rng.choice(candidates)
            candidates = [x for x in candidates if x != fact]
            if fact == F.PROLONGATION and F.VALIDITY in changed:
                continue
            if fact == F.VALIDITY and F.PROLONGATION in changed:
                continue
            changed.append(fact)
        before = current
        clauses: list[Clause] = []
        ds = with_style(s, random.Random(f"docai-rag:amend-text:{seed}:{c.contract_id}:{number}"))
        for fact in changed:
            after = _changed_contract(current, fact, rng, signed)
            label = contract_doc.fact_labels.get(fact, "")
            if fact == F.REQUISITES:
                label = "раздела «Адреса и реквизиты сторон»"
            text = (
                None if ds.rng.random() < 0.35 else _short_change(fact, label, current, after, ds)
            )
            if text is None:
                if fact == F.REQUISITES:
                    text = _short_change(fact, label, current, after, ds)
                else:
                    text = _restated(label, CLAUSE_RENDERERS[fact](after, ds), ds)
            assert text is not None
            clauses.append(Clause(text, (fact,)))
            current = after
        clauses.append(
            Clause(
                ds.pick(
                    f"Настоящее Соглашение вступает в силу с {ds.date(effective)} и является "
                    f"неотъемлемой частью Договора.",
                    f"Соглашение вступает в силу с {ds.date(effective)}.",
                ),
                (EFFECTIVE,),
            )
        )
        clauses.append(
            Clause(
                ds.pick(
                    "Остальные условия Договора, не затронутые настоящим Соглашением, остаются "
                    "неизменными, и Стороны подтверждают по ним свои обязательства.",
                    "Во всём остальном, что не предусмотрено настоящим Соглашением, Стороны "
                    "руководствуются условиями Договора.",
                )
            )
        )
        our, them = c.roles
        preamble = (
            f"{capitalize(party_intro(c.our, our.nom, ds))} и {party_intro(c.them, them.nom, ds)}, "
            f"совместно именуемые «Стороны», заключили настоящее дополнительное соглашение "
            f"(далее — «Соглашение») к Договору № {c.number} от {ds.date(c.date)} "
            f"(далее — «Договор») о нижеследующем:"
        )
        preamble = tidy_commas(preamble)
        doc = CorpusDoc(
            doc_id=f"{c.contract_id}-DS{number}",
            doc_kind="amendment",
            header=[
                f"ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № {number}",
                f"к {KIND_DAT[c.kind]} № {c.number} от {ds.date(c.date)}",
                f"г. {c.city}, {ds.date(signed)}",
            ],
            preamble=preamble,
            sections=[Section(None, clauses, numbered=False)],
            meta={
                "parent_id": c.contract_id,
                "number": number,
                "date": signed.isoformat(),
                "effective": effective.isoformat(),
                "changed": list(changed),
                "counterparty": c.them.name,
                "counterparty_inn": c.them.inn,
            },
        )
        doc.render()
        result.append(Amendment(doc, number, signed, effective, tuple(changed), before, current))
        last = signed
    return result
