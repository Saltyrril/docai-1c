"""Генерация валидных реквизитов: ИНН, КПП, БИК, корреспондентский и расчётный счета.

Контрольные цифры считаются по алгоритмам из `docai.validation`, поэтому всё, что
выдаёт этот модуль, проходит проверки P2. Номера случайные (не взяты у реальных
организаций), но формально корректны.
"""

import random

from docai.validation.requisites import account_key, inn_check_digits

# Коды субъектов РФ (первые две цифры ИНН/КПП) — реальные, чтобы реквизиты выглядели правдоподобно.
REGION_CODES = (
    "01", "02", "03", "05", "16", "18", "22", "23", "24", "25", "26", "27", "29", "31",
    "34", "36", "38", "39", "42", "43", "45", "46", "47", "50", "52", "53", "54", "55",
    "56", "58", "59", "61", "63", "64", "66", "67", "69", "70", "72", "74", "76", "77",
    "78", "86",
)  # fmt: skip

# Причина постановки на учёт в КПП: 01 — по месту нахождения организации (самая частая).
_KPP_REASONS = ("01", "01", "01", "01", "01", "43", "44", "45", "50")


def tax_office_code(rng: random.Random) -> str:
    """Код налогового органа: регион (2 цифры) + номер инспекции (2 цифры)."""
    return rng.choice(REGION_CODES) + f"{rng.randint(1, 49):02d}"


def make_inn10(rng: random.Random, office: str | None = None) -> str:
    """ИНН юридического лица (10 цифр)."""
    office = office or tax_office_code(rng)
    body = office + f"{rng.randint(0, 99999):05d}"
    return body + inn_check_digits(body)


def make_inn12(rng: random.Random, office: str | None = None) -> str:
    """ИНН индивидуального предпринимателя (12 цифр)."""
    office = office or tax_office_code(rng)
    body = office + f"{rng.randint(0, 999999):06d}"
    return body + inn_check_digits(body)


def make_kpp(rng: random.Random, office: str) -> str:
    """КПП: код налогового органа, причина постановки, порядковый номер."""
    return office + rng.choice(_KPP_REASONS) + f"{rng.randint(1, 999):03d}"


def make_bik(rng: random.Random) -> str:
    """БИК кредитной организации: 04 + регион + номер РКЦ + условный номер банка (050–999)."""
    region = f"{rng.randint(1, 99):02d}"
    rkc = f"{rng.randint(1, 99):02d}"
    return "04" + region + rkc + f"{rng.randint(50, 999):03d}"


def make_corr_account(bik: str) -> str:
    """Корреспондентский счёт банка: 30101810, ключ, нули, последние 3 цифры БИК."""
    draft = "30101810" + "0" + "00000000" + bik[-3:]
    key = account_key(draft, bik, in_credit_institution=False)
    return draft[:8] + key + draft[9:]


def make_settlement_account(rng: random.Random, bik: str, *, individual: bool) -> str:
    """Расчётный счёт в рублях: 40702 (организация) или 40802 (ИП) + ключ по БИК."""
    balance = "40802" if individual else "40702"
    draft = balance + "810" + "0" + f"{rng.randint(0, 99999999999):011d}"
    key = account_key(draft, bik, in_credit_institution=True)
    return draft[:8] + key + draft[9:]
