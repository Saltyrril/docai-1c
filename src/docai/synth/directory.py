"""Справочники: вымышленные организации, ИП и банки.

Справочники строятся один раз из фиксированного `DIRECTORY_SEED`, поэтому не зависят
от seed документов: одни и те же 300 организаций и 50 ИП в любом запуске (при той же
версии Faker, она закреплена в uv.lock). Названия придуманы; совпадение с реальными
организациями случайно, ИНН сгенерированы по алгоритму, а не взяты из реестров.
"""

import random
from dataclasses import dataclass
from functools import cache

from faker import Faker

from docai.synth.requisites import (
    make_bik,
    make_corr_account,
    make_inn10,
    make_inn12,
    make_kpp,
    make_settlement_account,
    tax_office_code,
)

DIRECTORY_SEED = 20261114
N_ORGANIZATIONS = 300
N_INDIVIDUALS = 50
N_BANKS = 24


@dataclass(frozen=True, slots=True)
class Bank:
    name: str
    bik: str
    corr_account: str


@dataclass(frozen=True, slots=True)
class Entity:
    """Организация или ИП со своими реквизитами и счётом в банке."""

    name: str
    inn: str
    kpp: str | None
    address: str
    bank: Bank
    account: str

    @property
    def is_individual(self) -> bool:
        return len(self.inn) == 12


@dataclass(frozen=True, slots=True)
class Directory:
    banks: tuple[Bank, ...]
    organizations: tuple[Entity, ...]
    individuals: tuple[Entity, ...]

    @property
    def entities(self) -> tuple[Entity, ...]:
        return self.organizations + self.individuals


_NAME_FIRST = (
    "Вектор", "Альфа", "Орион", "Прогресс", "Регион", "Меридиан", "Север", "Горизонт",
    "Статус", "Триумф", "Эталон", "Магистраль", "Союз", "Аврора", "Балтика", "Ладога",
    "Сибирь", "Урал", "Волга", "Дон", "Кедр", "Лотос", "Маяк", "Нева", "Омега",
    "Парус", "Рубин", "Сфера", "Тайга", "Фаворит", "Центр", "Юнион", "Ярмарка",
    "Бриз", "Гранит", "Диалог", "Заря", "Импульс", "Квадрат", "Лидер", "Мастер",
    "Новатор", "Основа", "Потенциал", "Ракурс", "Старт", "Терра", "Успех", "Формат",
)  # fmt: skip
_NAME_SECOND = (
    "Снаб", "Трейд", "Строй", "Сервис", "Групп", "Логистик", "Торг", "Инвест",
    "Технологии", "Системы", "Монтаж", "Ресурс", "Проект", "Альянс", "Опт",
    "Комплект", "Продукт", "Транс", "Софт", "Партнёр",
)  # fmt: skip
_BANK_FIRST = (
    "Северный", "Волжский", "Уральский", "Балтийский", "Сибирский", "Донской", "Таёжный",
    "Столичный", "Купеческий", "Речной", "Южный", "Радужный", "Новый", "Центральный",
    "Заречный", "Торговый",
)  # fmt: skip
_BANK_SECOND = ("Кредит", "Капитал", "Торговый Банк", "Деловой Банк", "Финанс", "Эксперт Банк")


def _build_banks(rng: random.Random) -> tuple[Bank, ...]:
    biks: set[str] = set()
    names: set[str] = set()
    banks: list[Bank] = []
    while len(banks) < N_BANKS:
        bik = make_bik(rng)
        form = rng.choice(("АО", "ПАО", "ООО"))
        name = f"{form} «{rng.choice(_BANK_FIRST)} {rng.choice(_BANK_SECOND)}»"
        if bik in biks or name in names:
            continue
        biks.add(bik)
        names.add(name)
        banks.append(Bank(name, bik, make_corr_account(bik)))
    return tuple(banks)


def _address(rng: random.Random, fake: Faker) -> str:
    """Индекс, город и улица из Faker ru_RU; номера домов и помещений — свои, правдоподобные."""
    street = fake.street_name()
    house = str(rng.randint(1, 120))
    if rng.random() < 0.3:
        house += f"/{rng.randint(1, 5)}"
    tail = f", офис {rng.randint(1, 450)}" if rng.random() < 0.5 else ""
    return f"{fake.postcode()}, г. {fake.city_name()}, {street}, д. {house}{tail}"


def _build_organizations(rng: random.Random, fake: Faker, banks: tuple[Bank, ...]) -> list[Entity]:
    seen_names: set[str] = set()
    seen_inns: set[str] = set()
    result: list[Entity] = []
    while len(result) < N_ORGANIZATIONS:
        form = rng.choices(("ООО", "АО"), weights=(85, 15))[0]
        if rng.random() < 0.25:
            name = f"{form} «{rng.choice(_NAME_FIRST)}»"
        else:
            name = f"{form} «{rng.choice(_NAME_FIRST)}-{rng.choice(_NAME_SECOND)}»"
        office = tax_office_code(rng)
        inn = make_inn10(rng, office)
        if name in seen_names or inn in seen_inns:
            continue
        seen_names.add(name)
        seen_inns.add(inn)
        bank = rng.choice(banks)
        result.append(
            Entity(
                name=name,
                inn=inn,
                kpp=make_kpp(rng, office),
                address=_address(rng, fake),
                bank=bank,
                account=make_settlement_account(rng, bank.bik, individual=False),
            )
        )
    return result


def _build_individuals(rng: random.Random, fake: Faker, banks: tuple[Bank, ...]) -> list[Entity]:
    seen_names: set[str] = set()
    seen_inns: set[str] = set()
    result: list[Entity] = []
    while len(result) < N_INDIVIDUALS:
        if rng.random() < 0.7:
            fio = f"{fake.last_name_male()} {fake.first_name_male()} {fake.middle_name_male()}"
        else:
            last, first = fake.last_name_female(), fake.first_name_female()
            fio = f"{last} {first} {fake.middle_name_female()}"
        name = f"ИП {fio}"
        inn = make_inn12(rng)
        if name in seen_names or inn in seen_inns:
            continue
        seen_names.add(name)
        seen_inns.add(inn)
        bank = rng.choice(banks)
        result.append(
            Entity(
                name=name,
                inn=inn,
                kpp=None,
                address=_address(rng, fake),
                bank=bank,
                account=make_settlement_account(rng, bank.bik, individual=True),
            )
        )
    return result


@cache
def get_directory(seed: int = DIRECTORY_SEED) -> Directory:
    """Справочники (строятся один раз на процесс)."""
    rng = random.Random(seed)
    fake = Faker("ru_RU")
    fake.seed_instance(seed)
    banks = _build_banks(rng)
    organizations = _build_organizations(rng, fake, banks)
    individuals = _build_individuals(rng, fake, banks)
    return Directory(banks, tuple(organizations), tuple(individuals))
