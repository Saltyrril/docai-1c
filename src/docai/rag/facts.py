"""Факты договора: эталонные ответы, ключевые слова для проверки и значения для сравнения.

Ответ строится из условий (`Contract`), а не из напечатанного текста. Ключевые факты
(`keys`) — строки, которые должны встретиться в ответе системы после нормализации
(`normalize_answer`); альтернативы внутри одной строки разделены «|».
"""

import re
from dataclasses import dataclass
from decimal import Decimal

from docai.rag.terms import TRIGGER_ANSWER, Contract, Kind, Penalty
from docai.rag.words import (
    date_short,
    days_gen,
    days_nom,
    decimal_text,
    money_answer,
    money_digits,
    months_nom,
)

SUBJECT = "subject"
PRICE = "price"
PAYMENT = "payment"
PENALTY_US = "penalty.us"
PENALTY_THEM = "penalty.them"
ACCEPTANCE = "acceptance"
VALIDITY = "validity"
PROLONGATION = "prolongation"
TERMINATION = "termination"
CLAIMS = "claims"
COURT = "court"
FORCE_MAJEURE = "force_majeure"
DELIVERY = "supply.delivery"
PREMISES = "lease.premises"
DEPOSIT = "lease.deposit"
INDEXATION = "lease.indexation"
UTILITIES = "lease.utilities"
WORKS_PERIOD = "works.period"
WARRANTY = "works.warranty"
RETENTION = "works.retention"
REQUISITES = "requisites.them"

# Короткое имя суда для ключевых фактов (ответ может написать «АС Свердловской обл.»).
_COURT_KEYS = {
    "Москва": "москв",
    "Санкт-Петербург": "санкт-петербург",
    "Казань": "татарстан",
    "Екатеринбург": "свердловск",
    "Новосибирск": "новосибирск",
    "Нижний Новгород": "нижегородск",
    "Самара": "самарск",
    "Краснодар": "краснодарск",
    "Пермь": "пермск",
    "Воронеж": "воронежск",
}

LABELS = {
    SUBJECT: "предмет договора",
    PRICE: "цена договора",
    PAYMENT: "срок и порядок оплаты",
    PENALTY_US: "неустойка за просрочку оплаты",
    PENALTY_THEM: "неустойка за просрочку со стороны контрагента",
    ACCEPTANCE: "порядок и срок приёмки",
    VALIDITY: "срок действия договора",
    PROLONGATION: "условие о пролонгации",
    TERMINATION: "срок уведомления об одностороннем отказе",
    CLAIMS: "срок ответа на претензию",
    COURT: "суд для разрешения споров",
    FORCE_MAJEURE: "срок уведомления о форс-мажоре",
    DELIVERY: "срок и условия поставки",
    PREMISES: "площадь и адрес помещения",
    DEPOSIT: "обеспечительный платёж",
    INDEXATION: "индексация арендной платы",
    UTILITIES: "оплата коммунальных услуг",
    WORKS_PERIOD: "сроки выполнения работ",
    WARRANTY: "гарантийный срок",
    RETENTION: "гарантийное удержание",
    REQUISITES: "банковские реквизиты контрагента",
}


@dataclass(frozen=True, slots=True)
class Fact:
    answer: str
    keys: tuple[str, ...]
    value: Decimal | None = None  # число для сравнения двух договоров
    unit: str | None = None  # сравнивать можно только значения с одинаковой единицей


def normalize_answer(text: str) -> str:
    """Нормализация для проверки ключевых фактов: регистр, «ё», пробелы в числах, «%».

    «85 000,00 руб.» → «85000,00 руб.»; «0.1 %» → «0,1%»; «30 %» → «30%».
    """
    text = text.lower().replace("ё", "е").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    text = re.sub(r"(?<=\d) +%", "%", text)
    return re.sub(r"\s+", " ", text)


_NUMBER = re.compile(r"\d+(?:,\d+)?")


def _alt_found(alt: str, norm: str) -> bool:
    """Число ищется целиком («3» не находится в «КС-3» и «30»), остальное — подстрокой."""
    alt = normalize_answer(alt)
    if _NUMBER.fullmatch(alt):
        # «405300» находится в «405300,00», но «3» — не в «3,5», а «0,1» — не в «0,15».
        tail = r"(?!\d)" if "," in alt else r"(?!\d)(?!,(?!00(?!\d))\d)"
        return re.search(rf"(?<![\w\-/.,]){re.escape(alt)}{tail}", norm) is not None
    return alt in norm


def keys_found(answer: str, keys: tuple[str, ...] | list[str]) -> bool:
    """Все ли ключевые факты есть в ответе (у каждого — хотя бы одна альтернатива)."""
    norm = normalize_answer(answer)
    return all(any(_alt_found(alt, norm) for alt in key.split("|")) for key in keys)


def _pct_key(p: int) -> str:
    return f"{p}%|{p} процент"


def _money_key(amount: Decimal) -> str:
    return money_digits(amount, kopecks=False).replace(" ", "")


def _penalty_fact(penalty: Penalty, base: str) -> Fact:
    if penalty.kind == "key_rate":
        return Fact(
            f"1/300 ключевой ставки Банка России от {base} за каждый день просрочки",
            ("1/300", "ключев"),
        )
    rate = decimal_text(penalty.rate)
    answer = f"{rate}% от {base} за каждый день просрочки"
    keys: tuple[str, ...] = (f"{rate}%|{rate} процент",)
    if penalty.cap_percent:
        answer += f", но не более {penalty.cap_percent}%"
        keys += (_pct_key(penalty.cap_percent),)
    return Fact(answer, keys, penalty.rate, "rate")


def _payment_fact(c: Contract) -> Fact:
    p = c.payment
    if p.scheme == "monthly":
        when = "текущего месяца" if p.due_month == "current" else "месяца, следующего за расчётным"
        what = "арендная плата вносится" if c.kind is Kind.LEASE else "оплата производится"
        return Fact(f"{what} ежемесячно, не позднее {p.due_day}-го числа {when}", (str(p.due_day),))
    if p.prepay_percent == 100:
        return Fact(
            f"100% предоплата в течение {days_gen(p.prepay_days, 'banking')} "
            f"с даты получения счёта",
            ("100%|100 процент|полная предоплата|стопроцент", str(p.prepay_days)),
        )
    assert p.days is not None
    tail = f"в течение {days_gen(p.days, p.day_kind)} с {TRIGGER_ANSWER[p.trigger]}"
    stem = {"calendar": "календарн", "working": "рабоч", "banking": "банковск"}[p.day_kind]
    if p.prepay_percent:
        advance = "аванс" if c.kind is Kind.WORKS else "предоплата"
        answer = (
            f"{advance} {p.prepay_percent}% в течение {days_gen(p.prepay_days, 'banking')} "
            f"с даты получения счёта, оставшиеся {100 - p.prepay_percent}% — {tail}"
        )
        return Fact(answer, (_pct_key(p.prepay_percent), str(p.days), stem))
    return Fact(tail, (str(p.days), stem), Decimal(p.days), f"days:{p.day_kind}")


def contract_facts(c: Contract) -> dict[str, Fact]:
    """Все факты договора, которые в нём есть (отсутствующие условия — не ключи словаря)."""
    facts: dict[str, Fact] = {SUBJECT: Fact(c.subject, (c.subject.split()[0][:6],))}
    if c.price is not None:
        money = money_answer(c.price)
        text = {
            Kind.SUPPLY: f"общая стоимость товара по договору не более {money}, включая НДС",
            Kind.SERVICES: f"{money} в месяц, включая НДС",
            Kind.LEASE: f"{money} в месяц",
            Kind.WORKS: f"{money}, включая НДС (твёрдая цена)",
        }[c.kind]
        facts[PRICE] = Fact(text, (_money_key(c.price),), c.price, f"money:{c.kind}")
    facts[PAYMENT] = _payment_fact(c)

    if c.penalty_us is not None:
        facts[PENALTY_US] = _penalty_fact(c.penalty_us, "неоплаченной в срок суммы")
    else:
        facts[PENALTY_US] = Fact(
            "конкретный размер неустойки договором не установлен: стороны отвечают "
            "в соответствии с законодательством РФ",
            ("законодательств",),
        )
    if c.penalty_them is not None:
        base = {
            Kind.SUPPLY: "стоимости не поставленного в срок товара",
            Kind.SERVICES: "стоимости услуг, оказанных с нарушением срока",
            Kind.WORKS: "стоимости работ",
        }[c.kind]
        facts[PENALTY_THEM] = _penalty_fact(c.penalty_them, base)

    n = days_gen(c.acceptance_days, "working")
    facts[ACCEPTANCE] = Fact(
        {
            Kind.SUPPLY: f"количество проверяется при приёмке, претензии по качеству — "
            f"в течение {n} с даты приёмки товара",
            Kind.SERVICES: f"в течение {n} с даты получения акта заказчик подписывает его или "
            f"направляет мотивированный отказ, иначе услуги считаются принятыми",
            Kind.WORKS: f"в течение {n} с даты получения акта КС-2 и справки КС-3 заказчик "
            f"подписывает их или направляет мотивированный отказ",
            Kind.LEASE: f"помещение передаётся по акту приёма-передачи в течение {n} "
            f"с даты подписания договора",
        }[c.kind],
        (str(c.acceptance_days),),
        Decimal(c.acceptance_days),
        f"acceptance:{c.kind}",
    )

    v = c.validity
    if v.kind == "until_date":
        assert v.end is not None
        facts[VALIDITY] = Fact(f"до {date_short(v.end)} включительно", (date_short(v.end),))
    elif v.kind == "months":
        assert v.months is not None
        facts[VALIDITY] = Fact(
            f"{months_nom(v.months)} с даты подписания ({date_short(c.date)})",
            (str(v.months),),
            Decimal(v.months),
            "months",
        )
    else:
        facts[VALIDITY] = Fact("до полного исполнения сторонами своих обязательств", ("исполнени",))

    pr = c.prolongation
    if pr is not None:
        if pr.auto:
            assert pr.notice_days is not None
            period = "тот же срок" if pr.period == "same" else "следующий календарный год"
            facts[PROLONGATION] = Fact(
                f"да, продлевается автоматически на {period}, если ни одна сторона не заявит "
                f"о прекращении не позднее чем за {days_nom(pr.notice_days)} до окончания срока",
                (str(pr.notice_days),),
                Decimal(pr.notice_days),
                "days:notice",
            )
        else:
            facts[PROLONGATION] = Fact(
                "нет, автоматическая пролонгация не предусмотрена: продление — только "
                "дополнительным соглашением",
                ("не предусмотрен|не продлева|не пролонгир|отменен|исключ|не применя",),
            )

    if c.termination_days is not None:
        facts[TERMINATION] = Fact(
            f"односторонний отказ возможен с письменным уведомлением другой стороны не менее "
            f"чем за {days_nom(c.termination_days)}",
            (str(c.termination_days),),
            Decimal(c.termination_days),
            "days:termination",
        )
    facts[CLAIMS] = Fact(
        f"{days_nom(c.claim_days)} с даты получения претензии; претензионный порядок обязателен",
        (str(c.claim_days),),
        Decimal(c.claim_days),
        "days:claims",
    )
    facts[COURT] = Fact(c.court, (_COURT_KEYS[c.city],))
    if c.force_majeure_days is not None:
        facts[FORCE_MAJEURE] = Fact(
            f"уведомить другую сторону в течение {days_gen(c.force_majeure_days)} "
            f"с момента наступления обстоятельств непреодолимой силы",
            (str(c.force_majeure_days),),
            Decimal(c.force_majeure_days),
            "days:fm",
        )

    if c.kind is Kind.SUPPLY:
        assert c.delivery_days is not None
        how = (
            "доставка силами и за счёт поставщика на склад покупателя"
            if c.delivery_by_supplier
            else "самовывоз со склада поставщика"
        )
        keys = (str(c.delivery_days),) + (() if c.delivery_by_supplier else ("самовывоз",))
        facts[DELIVERY] = Fact(
            f"в течение {days_gen(c.delivery_days, c.delivery_day_kind)} с даты получения "
            f"заявки, {how}",
            keys,
            Decimal(c.delivery_days),
            f"days:{c.delivery_day_kind}",
        )
    if c.kind is Kind.LEASE:
        assert c.area is not None
        facts[PREMISES] = Fact(
            f"{decimal_text(c.area)} кв. м по адресу: {c.premises_address}",
            (decimal_text(c.area),),
            c.area,
            "area",
        )
        if c.deposit is not None:
            facts[DEPOSIT] = Fact(
                money_answer(c.deposit), (_money_key(c.deposit),), c.deposit, "money:deposit"
            )
        if c.indexation is not None:
            facts[INDEXATION] = Fact(
                f"ежегодная индексация на {c.indexation}%",
                (_pct_key(c.indexation),),
                Decimal(c.indexation),
                "percent:indexation",
            )
        facts[UTILITIES] = (
            Fact("коммунальные услуги включены в арендную плату", ("включен|входят|входит",))
            if c.utilities_included
            else Fact(
                "коммунальные услуги в арендную плату не входят и оплачиваются отдельно "
                "по показаниям приборов учёта",
                ("отдельно|не входят|не включ",),
            )
        )
    if c.kind is Kind.WORKS:
        assert c.works_start and c.works_end and c.warranty_months
        facts[WORKS_PERIOD] = Fact(
            f"с {date_short(c.works_start)} по {date_short(c.works_end)}",
            (date_short(c.works_end),),
        )
        facts[WARRANTY] = Fact(
            f"{months_nom(c.warranty_months)} с даты подписания акта о приёмке работ",
            (str(c.warranty_months),),
            Decimal(c.warranty_months),
            "months:warranty",
        )
        if c.retention_percent is not None:
            facts[RETENTION] = Fact(
                f"{c.retention_percent}% от стоимости выполненных работ, удержанная сумма "
                f"выплачивается после окончания гарантийного срока",
                (_pct_key(c.retention_percent),),
                Decimal(c.retention_percent),
                "percent:retention",
            )
    facts[REQUISITES] = Fact(
        f"р/с {c.them_account} в {c.them_bank.name}, БИК {c.them_bank.bik}, "
        f"к/с {c.them_bank.corr_account}",
        (c.them_account, c.them_bank.bik),
    )
    return facts
