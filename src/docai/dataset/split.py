"""Разбиение train/val/test по продавцам.

Все документы одного продавца (по ИНН) попадают в одну часть. Иначе модель на тесте
видела бы реквизиты, названия и адреса продавца из train, и метрика измеряла бы
запоминание, а не умение читать новые документы.
"""

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

SPLITS = ("train", "val", "test")


def _order_key(seed: int, inn: str) -> str:
    return hashlib.sha256(f"docai-split:{seed}:{inn}".encode()).hexdigest()


def split_by_seller(
    sellers: Sequence[str], shares: Mapping[str, float], seed: int
) -> dict[str, str]:
    """Вернуть {ИНН продавца → часть}. `sellers` — ИНН по одному на документ.

    Продавцы перебираются в порядке SHA-256(seed, ИНН) (результат не зависит от порядка
    документов) и по одному отдаются той части, у которой сейчас наибольшее отставание
    от цели в документах. Так доли близки к заданным, а крупные продавцы не смещают их.
    """
    counts: dict[str, int] = defaultdict(int)
    for inn in sellers:
        counts[inn] += 1
    total = sum(counts.values())
    targets = {name: shares[name] * total for name in SPLITS if shares.get(name, 0) > 0}
    filled = dict.fromkeys(targets, 0)
    assignment: dict[str, str] = {}
    for inn in sorted(counts, key=lambda value: _order_key(seed, value)):
        part = max(targets, key=lambda name: (targets[name] - filled[name]) / targets[name])
        assignment[inn] = part
        filled[part] += counts[inn]
    return assignment
