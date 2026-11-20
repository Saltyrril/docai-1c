"""Эталонный ответ модели для SFT: черновик `Draft` по тому, что видно на странице.

Правила (D77):
- форма — ровно черновик, который конвейер просит у модели (`docai.extract.draft.Draft`):
  те же ключи в том же порядке, отсутствующее — null;
- значение берётся из эталонного JSON, но только если поле напечатано на этой странице
  (или в этом куске текста) — по разметке `layout.json` P4/P5. Чего не видно — null:
  промпты требуют «ничего не придумывай», а обучать модель выдумывать нельзя.
  Так, код ОКЕИ в счёте и акте не печатается — в эталоне ответа его нет;
- даты — как напечатаны («05.02.2026», D47), числа — JSON-числа с точкой, разрядность
  как в эталоне («12200.00»), JSON — компактный, кириллица без экранирования.
"""

import json
import re
from decimal import Decimal
from typing import Any

PARTY_KEYS = ("name", "inn", "kpp", "address")
BANK_KEYS = ("bank_name", "bik", "corr_account", "account")
BASIS_KEYS = ("name", "number", "date")
LINE_KEYS = (
    "name",
    "unit_name",
    "unit_code",
    "quantity",
    "price",
    "amount_without_vat",
    "vat_rate",
    "vat_amount",
    "amount_with_vat",
)
TOTALS_KEYS = ("amount_without_vat", "vat_amount", "amount_with_vat")
NUMERIC = {
    "quantity",
    "price",
    "amount_without_vat",
    "vat_amount",
    "amount_with_vat",
}
DATE_FIELDS = {"date", "basis.date"}
_LINE = re.compile(r"^lines\[(\d+)\]\.(\w+)$")


def visible_fields(layout_fields: list[dict[str, Any]], page: int | None = None) -> dict[str, str]:
    """Поля разметки → {путь поля: напечатанный текст}. `page` — только эта страница
    (у разметки PDF номер страницы в каждом поле; у разметки картинки — одна страница)."""
    visible: dict[str, str] = {}
    for item in layout_fields:
        if page is not None and item.get("page", page) != page:
            continue
        visible.setdefault(item["field"], str(item.get("text", "")))
    return visible


def _number(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))


def _value(ref: dict[str, Any] | None, key: str, path: str, visible: dict[str, str]) -> Any:
    if path not in visible or ref is None:
        return None
    if path in DATE_FIELDS:
        return visible[path] or None  # как напечатано
    value = ref.get(key)
    if value is None:
        return None
    if key in NUMERIC:
        return _number(value)
    return value


def _block(
    ref: dict[str, Any] | None, prefix: str, keys: tuple[str, ...], visible: dict[str, str]
) -> dict[str, Any]:
    return {key: _value(ref, key, f"{prefix}.{key}", visible) for key in keys}


def _optional_block(
    ref: dict[str, Any] | None, prefix: str, keys: tuple[str, ...], visible: dict[str, str]
) -> dict[str, Any] | None:
    block = _block(ref, prefix, keys, visible)
    return block if any(v is not None for v in block.values()) else None


def draft_target(ref: dict[str, Any], visible: dict[str, str]) -> dict[str, Any]:
    """Черновик-эталон для одной страницы или одного куска текста."""
    line_ids = sorted({int(m.group(1)) for path in visible if (m := _LINE.match(path))})
    ref_lines = ref.get("lines") or []
    lines = []
    for index in line_ids:
        if index >= len(ref_lines):
            continue
        lines.append(_block(ref_lines[index], f"lines[{index}]", LINE_KEYS, visible))
    upd_status = ref.get("upd_status") if "upd_status" in visible else None
    return {
        "doc_type": ref["doc_type"],
        "upd_status": int(upd_status) if upd_status is not None else None,
        "number": _value(ref, "number", "number", visible),
        "date": _value(ref, "date", "date", visible),
        "seller": _block(ref.get("seller"), "seller", PARTY_KEYS, visible),
        "buyer": _block(ref.get("buyer"), "buyer", PARTY_KEYS, visible),
        "seller_bank": _optional_block(ref.get("seller_bank"), "seller_bank", BANK_KEYS, visible),
        "basis": _optional_block(ref.get("basis"), "basis", BASIS_KEYS, visible),
        "lines": lines,
        "totals": _block(ref.get("totals"), "totals", TOTALS_KEYS, visible),
    }


def to_json(value: Any) -> str:
    """Компактный JSON; Decimal пишется как есть («12200.00»), без экспоненты."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        items = (f"{json.dumps(str(k), ensure_ascii=False)}:{to_json(v)}" for k, v in value.items())
        return "{" + ",".join(items) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(to_json(v) for v in value) + "]"
    raise TypeError(f"не умею записать в JSON: {type(value).__name__}")
