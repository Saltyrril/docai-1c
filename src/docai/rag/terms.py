"""Условия договоров как данные: из них строится и текст договора, и эталонные ответы.

Договор генерируется сначала как набор условий (`Contract`), а уже потом печатается
текстом в одной из формулировок. Поэтому эталон не зависит от того, как именно
условие сформулировано: ответ берётся из условий, цитата — из напечатанного пункта.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum

from faker import Faker

from docai.synth.directory import Bank, Entity, get_directory
from docai.synth.generator import make_contract_number
from docai.synth.requisites import make_settlement_account

TODAY = date(2026, 11, 19)  # «сейчас» для вопросов: все доп. соглашения вступили в силу раньше
CONTRACT_DATE_FROM = date(2023, 1, 9)
CONTRACT_DATE_TO = date(2026, 8, 31)


class Kind(StrEnum):
    SUPPLY = "supply"
    SERVICES = "services"
    LEASE = "lease"
    WORKS = "works"


KIND_TITLE = {
    Kind.SUPPLY: "Договор поставки",
    Kind.SERVICES: "Договор возмездного оказания услуг",
    Kind.LEASE: "Договор аренды нежилого помещения",
    Kind.WORKS: "Договор подряда",
}
# «по договору поставки», «по договору оказания услуг»
KIND_GEN = {
    Kind.SUPPLY: "поставки",
    Kind.SERVICES: "оказания услуг",
    Kind.LEASE: "аренды",
    Kind.WORKS: "подряда",
}


@dataclass(frozen=True, slots=True)
class Role:
    """Название стороны в договоре с падежами (Поставщик, Поставщика, Поставщику...)."""

    nom: str

    def _form(self, soft: str, hard: str) -> str:
        return self.nom[:-1] + soft if self.nom.endswith("ь") else self.nom + hard

    @property
    def gen(self) -> str:
        return self._form("я", "а")

    @property
    def dat(self) -> str:
        return self._form("ю", "у")

    @property
    def acc(self) -> str:
        return self.gen  # одушевлённые: винительный = родительный

    @property
    def ins(self) -> str:
        return self._form("ем", "ом")


# (наша сторона, сторона контрагента)
ROLES = {
    Kind.SUPPLY: (Role("Покупатель"), Role("Поставщик")),
    Kind.SERVICES: (Role("Заказчик"), Role("Исполнитель")),
    Kind.LEASE: (Role("Арендатор"), Role("Арендодатель")),
    Kind.WORKS: (Role("Заказчик"), Role("Подрядчик")),
}

# Город договора и арбитражный суд субъекта, где он находится.
CITIES = (
    ("Москва", "Арбитражный суд города Москвы"),
    ("Санкт-Петербург", "Арбитражный суд города Санкт-Петербурга и Ленинградской области"),
    ("Казань", "Арбитражный суд Республики Татарстан"),
    ("Екатеринбург", "Арбитражный суд Свердловской области"),
    ("Новосибирск", "Арбитражный суд Новосибирской области"),
    ("Нижний Новгород", "Арбитражный суд Нижегородской области"),
    ("Самара", "Арбитражный суд Самарской области"),
    ("Краснодар", "Арбитражный суд Краснодарского края"),
    ("Пермь", "Арбитражный суд Пермского края"),
    ("Воронеж", "Арбитражный суд Воронежской области"),
)

# Категории каталога P3 → что поставляется по договору.
SUPPLY_GOODS = {
    "канцтовары": "канцелярские товары и офисные принадлежности",
    "стройматериалы": "строительные и отделочные материалы",
    "продукты": "продукты питания и напитки для офиса",
}
# (категория каталога P3 или «прочее», описание услуг)
SERVICES = (
    ("IT-услуги", "абонентское обслуживание компьютерной техники и локальной сети"),
    ("IT-услуги", "сопровождение и обновление программ 1С"),
    ("IT-услуги", "администрирование серверов и резервное копирование данных"),
    ("IT-услуги", "техническая поддержка пользователей и настройка рабочих мест"),
    ("транспорт", "перевозка грузов автомобильным транспортом"),
    ("транспорт", "транспортно-экспедиционное обслуживание"),
    ("прочее", "ежедневная уборка офисных помещений"),
    ("прочее", "охрана офиса и пропускной режим"),
    ("прочее", "техническое обслуживание кондиционеров и вентиляции"),
    ("прочее", "курьерская доставка документов по городу"),
)
WORKS = (
    "ремонт офисного помещения",
    "монтаж системы видеонаблюдения",
    "устройство кровли складского здания",
    "монтаж систем вентиляции и кондиционирования",
    "замена окон и остекление",
    "электромонтажные работы",
    "устройство наливных полов на складе",
    "монтаж стеллажного оборудования",
)

TRIGGER_ANSWER = {
    "delivery": "даты поставки товара",
    "upd": "даты подписания УПД (товарной накладной)",
    "invoice": "даты получения счёта",
    "act": "даты подписания акта",
    "ks": "даты подписания акта КС-2 и справки КС-3",
}


@dataclass(frozen=True, slots=True)
class Payment:
    """Порядок оплаты.

    scheme="deferral": предоплата `prepay_percent`% (0 — без неё) в течение `prepay_days`
    банковских дней с даты счёта, остаток — в течение `days` дней вида `day_kind` с события
    `trigger`. При предоплате 100% остатка нет. scheme="monthly": ежемесячно до `due_day`
    числа текущего (`due_month="current"`) или следующего месяца.
    """

    scheme: str
    prepay_percent: int = 0
    prepay_days: int = 0
    days: int | None = None
    day_kind: str = "calendar"
    trigger: str = ""
    due_day: int | None = None
    due_month: str = "current"


@dataclass(frozen=True, slots=True)
class Penalty:
    """Неустойка: `rate`% в день (kind="percent") или 1/300 ключевой ставки (kind="key_rate")."""

    kind: str
    rate: Decimal = Decimal(0)
    cap_percent: int | None = None


@dataclass(frozen=True, slots=True)
class Validity:
    """Срок действия: до даты, на N месяцев с подписания или до исполнения обязательств."""

    kind: str  # until_date | months | until_fulfilment
    end: date | None = None
    months: int | None = None


@dataclass(frozen=True, slots=True)
class Prolongation:
    auto: bool
    notice_days: int | None = None
    period: str = "same"  # same — на тот же срок; year — на следующий календарный год


@dataclass(frozen=True, slots=True)
class Contract:
    """Условия одного договора нашей организации с контрагентом."""

    contract_id: str
    kind: Kind
    number: str
    date: date
    city: str
    court: str
    our: Entity
    them: Entity
    them_bank: Bank
    them_account: str
    subject: str
    category: str  # категория каталога P3 (для привязки документов реестра) или «прочее»
    price: Decimal | None
    payment: Payment
    penalty_us: Penalty | None  # за просрочку оплаты (платим мы)
    penalty_them: Penalty | None  # за просрочку поставки, услуг, работ (платит контрагент)
    acceptance_days: int
    validity: Validity
    prolongation: Prolongation | None  # None — срок «до исполнения», продлевать нечего
    termination_days: int | None
    claim_days: int
    force_majeure_days: int | None
    # поставка
    delivery_days: int | None = None
    delivery_day_kind: str = "working"
    delivery_by_supplier: bool = True
    # аренда
    premises_address: str | None = None
    area: Decimal | None = None
    deposit: Decimal | None = None
    indexation: int | None = None
    utilities_included: bool | None = None
    # подряд
    works_address: str | None = None
    works_start: date | None = None
    works_end: date | None = None
    warranty_months: int | None = None
    retention_percent: int | None = None
    weight: float = 1.0  # насколько часто по договору приходят документы (реестр)

    @property
    def roles(self) -> tuple[Role, Role]:
        return ROLES[self.kind]

    @property
    def end_date(self) -> date | None:
        """Дата окончания срока (без учёта пролонгации), если её можно вычислить."""
        if self.validity.kind == "until_date":
            return self.validity.end
        if self.validity.kind == "months" and self.validity.months:
            return add_months(self.date, self.validity.months) - timedelta(days=1)
        return None

    def is_active(self, on: date) -> bool:
        """Действует ли договор на дату (с автопролонгацией — бессрочно после подписания)."""
        if on < self.date:
            return False
        end = self.end_date
        if end is None:
            return True
        if self.prolongation is not None and self.prolongation.auto:
            return True
        return on <= end


def add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    days_in_month = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(d.day, days_in_month))


# --- Генерация ---------------------------------------------------------------------------

_KIND_QUOTA = {Kind.SUPPLY: 110, Kind.SERVICES: 90, Kind.LEASE: 45, Kind.WORKS: 55}


def kind_plan(n: int) -> list[Kind]:
    """Виды договоров в пропорции 110 : 90 : 45 : 55 (для n=300 — ровно эти числа)."""
    total = sum(_KIND_QUOTA.values())
    kinds: list[Kind] = []
    for kind, quota in _KIND_QUOTA.items():
        kinds += [kind] * round(n * quota / total)
    while len(kinds) < n:
        kinds.append(Kind.SUPPLY)
    return kinds[:n]


def pick_our_company() -> Entity:
    """«Наша» организация: фиксированная организация из справочника P3 (не ИП)."""
    return get_directory().organizations[0]


def _contract_date(rng: random.Random) -> date:
    span = (CONTRACT_DATE_TO - CONTRACT_DATE_FROM).days
    return CONTRACT_DATE_FROM + timedelta(days=rng.randint(0, span))


def _price_rub(rng: random.Random, low: int, high: int, step: int) -> Decimal:
    return Decimal(rng.randrange(low, high, step))


def _penalty(rng: random.Random, *, allow_key_rate: bool) -> Penalty:
    if allow_key_rate and rng.random() < 0.25:
        return Penalty("key_rate")
    rate = rng.choice((Decimal("0.01"), Decimal("0.05"), Decimal("0.1"), Decimal("0.1")))
    if rng.random() < 0.2:
        rate = rng.choice((Decimal("0.03"), Decimal("0.2"), Decimal("0.3"), Decimal("0.5")))
    cap = rng.choice((None, 5, 10, 10, 20)) if rng.random() < 0.7 else None
    return Penalty("percent", rate, cap)


def _payment(rng: random.Random, kind: Kind) -> Payment:
    day_kind = rng.choices(("calendar", "working", "banking"), weights=(60, 30, 10))[0]
    if kind is Kind.LEASE:
        return Payment(
            "monthly",
            due_day=rng.choice((5, 10, 15, 20, 25)),
            due_month=rng.choice(("current", "current", "next")),
        )
    if kind is Kind.SERVICES and rng.random() < 0.35:
        return Payment("monthly", due_day=rng.choice((5, 10, 15, 20)), due_month="next")
    days = rng.choice((5, 7, 10, 14, 15, 20, 30, 30, 45, 60, 90))
    if kind is Kind.SUPPLY:
        prepay = rng.choices((0, 30, 50, 100), weights=(55, 20, 15, 10))[0]
        trigger = rng.choice(("delivery", "upd", "upd", "invoice"))
    elif kind is Kind.SERVICES:
        prepay = rng.choices((0, 50, 100), weights=(80, 10, 10))[0]
        trigger = rng.choice(("act", "act", "invoice"))
    else:  # подряд: аванс почти всегда
        prepay = rng.choices((0, 20, 30, 50), weights=(15, 25, 40, 20))[0]
        trigger = "ks"
    if prepay == 100:
        return Payment("deferral", 100, rng.choice((3, 5, 10)), None, "banking", "invoice")
    return Payment(
        "deferral",
        prepay,
        rng.choice((3, 5, 10)) if prepay else 0,
        days,
        day_kind,
        trigger,
    )


def _validity(rng: random.Random, kind: Kind, signed: date) -> Validity:
    if kind is Kind.WORKS:
        return Validity("until_fulfilment")
    choice = rng.choices(("until_date", "months", "until_fulfilment"), weights=(55, 35, 10))[0]
    if kind is Kind.LEASE and choice == "until_fulfilment":
        choice = "months"
    if choice == "until_date":
        year = signed.year + rng.choice((0, 0, 1, 1, 2))
        end = date(year, 12, 31)
        if end <= signed + timedelta(days=60):
            end = date(year + 1, 12, 31)
        return Validity("until_date", end=end)
    if choice == "months":
        months = rng.choice((11, 11, 12, 24, 36)) if kind is Kind.LEASE else rng.choice((12, 24))
        return Validity("months", months=months)
    return Validity("until_fulfilment")


def _prolongation(rng: random.Random, validity: Validity) -> Prolongation | None:
    if validity.kind == "until_fulfilment":
        return None
    if rng.random() < 0.6:
        return Prolongation(
            True,
            rng.choice((10, 14, 15, 30, 30, 60, 90)),
            "year" if validity.kind == "until_date" and rng.random() < 0.5 else "same",
        )
    return Prolongation(False)


def _address(rng: random.Random, fake: Faker, city: str) -> str:
    street = fake.street_name()
    building = f"д. {rng.randint(1, 140)}"
    if rng.random() < 0.3:
        building += f", стр. {rng.randint(1, 6)}"
    room = f", помещение {rng.randint(1, 40)}" if rng.random() < 0.6 else ""
    return f"г. {city}, {street}, {building}{room}"


def _contract_number(rng: random.Random, kind: Kind, signed: date, used: set[str]) -> str:
    prefix = {Kind.SUPPLY: "П", Kind.SERVICES: "У", Kind.LEASE: "А", Kind.WORKS: "ПД"}[kind]
    while True:
        style = rng.randrange(4)
        if style == 0:
            number = make_contract_number(rng)
        elif style == 1:
            number = f"{prefix}-{rng.randint(1, 999):03d}/{signed.year % 100}"
        elif style == 2:
            number = f"{rng.randint(1, 299)}-{prefix}"
        else:
            number = f"{signed.year}/{rng.randint(1, 450)}"
        if number not in used:
            used.add(number)
            return number


def generate_contracts(n: int, seed: int) -> list[Contract]:
    """n договоров нашей организации с контрагентами из справочников P3.

    Контрагентов меньше, чем договоров: у части контрагентов 2–4 договора, чтобы
    вопросы «по договору с X» требовали уточнения, а сравнение было внутри одного
    контрагента. Каждый договор строится своим генератором случайных чисел из
    (seed, номер), и его условия не зависят от остальных договоров.
    """
    directory = get_directory()
    our = pick_our_company()
    pool = [e for e in directory.entities if e.inn != our.inn]
    plan_rng = random.Random(f"docai-rag:plan:{seed}")
    n_counterparties = max(1, round(n * 0.62))
    counterparties = plan_rng.sample(pool, min(n_counterparties, len(pool)))
    owners = list(counterparties)
    while len(owners) < n:
        # Повторные договоры чаще у первых контрагентов списка: появляются «крупные» партнёры.
        owners.append(counterparties[int(len(counterparties) * plan_rng.random() ** 2)])
    plan_rng.shuffle(owners)
    kinds = kind_plan(n)
    plan_rng.shuffle(kinds)

    fake = Faker("ru_RU")
    used_numbers: set[str] = set()
    contracts: list[Contract] = []
    for index in range(1, n + 1):
        rng = random.Random(f"docai-rag:contract:{seed}:{index}")
        fake.seed_instance(f"docai-rag:address:{seed}:{index}")
        kind = kinds[index - 1]
        them = owners[index - 1]
        signed = _contract_date(rng)
        city, court = rng.choice(CITIES)
        validity = _validity(rng, kind, signed)
        payment = _payment(rng, kind)
        common = {
            "contract_id": f"K-{index:04d}",
            "kind": kind,
            "number": _contract_number(rng, kind, signed, used_numbers),
            "date": signed,
            "city": city,
            "court": court,
            "our": our,
            "them": them,
            "them_bank": them.bank,
            "them_account": them.account,
            "payment": payment,
            "acceptance_days": rng.choice((2, 3, 5, 5, 7, 10, 14)),
            "validity": validity,
            "prolongation": _prolongation(rng, validity),
            "claim_days": rng.choice((7, 10, 10, 15, 20, 30)),
            "force_majeure_days": rng.choice((3, 5, 7, 10, 14)) if rng.random() < 0.6 else None,
            "weight": rng.paretovariate(1.3),
        }
        no_penalty = rng.random() < 0.08
        penalty_us = None if no_penalty else _penalty(rng, allow_key_rate=True)
        if kind is Kind.SUPPLY:
            category = rng.choice(tuple(SUPPLY_GOODS))
            price = _price_rub(rng, 300_000, 5_000_000, 50_000) if rng.random() < 0.4 else None
            contract = Contract(
                **common,
                subject=SUPPLY_GOODS[category],
                category=category,
                price=price,
                penalty_us=penalty_us,
                penalty_them=None if no_penalty else _penalty(rng, allow_key_rate=False),
                termination_days=rng.choice((30, 60)) if rng.random() < 0.4 else None,
                delivery_days=rng.choice((2, 3, 5, 7, 10, 14)),
                delivery_day_kind=rng.choice(("working", "working", "calendar")),
                delivery_by_supplier=rng.random() < 0.75,
            )
        elif kind is Kind.SERVICES:
            category, subject = rng.choice(SERVICES)
            monthly = payment.scheme == "monthly" or rng.random() < 0.3
            price = _price_rub(rng, 15_000, 400_000, 2_500) if monthly else None
            contract = Contract(
                **common,
                subject=subject,
                category=category,
                price=price,
                penalty_us=penalty_us,
                penalty_them=None if no_penalty else _penalty(rng, allow_key_rate=False),
                termination_days=rng.choice((10, 30, 30, 60)) if rng.random() < 0.7 else None,
            )
        elif kind is Kind.LEASE:
            area = Decimal(rng.randint(180, 9_000)) / 10
            rate = Decimal(rng.randrange(800, 3_200, 50))
            rent = (area * rate).quantize(Decimal(1))
            rent = (rent / 100).quantize(Decimal(1)) * 100
            contract = Contract(
                **common,
                subject="аренда нежилого помещения под офис",
                category="прочее",
                price=rent,
                penalty_us=penalty_us,
                penalty_them=None,
                termination_days=rng.choice((30, 60, 90, 90)) if rng.random() < 0.8 else None,
                premises_address=_address(rng, fake, city),
                area=area,
                deposit=rent * rng.choice((1, 1, 2)) if rng.random() < 0.65 else None,
                indexation=rng.choice((3, 4, 5, 7, 10)) if rng.random() < 0.5 else None,
                utilities_included=rng.random() < 0.4,
            )
        else:
            start = signed + timedelta(days=rng.randint(5, 30))
            contract = Contract(
                **common,
                subject=rng.choice(WORKS),
                category="прочее",
                price=_price_rub(rng, 150_000, 9_000_000, 10_000),
                penalty_us=penalty_us,
                penalty_them=None if no_penalty else _penalty(rng, allow_key_rate=False),
                termination_days=rng.choice((15, 30)) if rng.random() < 0.2 else None,
                works_address=_address(rng, fake, city),
                works_start=start,
                works_end=start + timedelta(days=rng.choice((20, 30, 45, 60, 90, 120))),
                warranty_months=rng.choice((12, 12, 24, 36, 60)),
                retention_percent=rng.choice((3, 5, 5, 10)) if rng.random() < 0.45 else None,
            )
        contracts.append(contract)
    return contracts


def new_bank_account(rng: random.Random, entity: Entity, current: Bank) -> tuple[Bank, str]:
    """Новый банк и расчётный счёт контрагента (для доп. соглашения о смене реквизитов)."""
    banks = [b for b in get_directory().banks if b.bik != current.bik]
    bank = rng.choice(banks)
    return bank, make_settlement_account(rng, bank.bik, individual=entity.is_individual)
