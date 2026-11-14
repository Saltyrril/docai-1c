"""FIELD_SPEC: каждое поле есть в схеме ровно у тех типов документов, где объявлено."""

import types
import typing
from decimal import Decimal

import pytest
from pydantic import BaseModel

from docai.schemas import (
    DOCUMENT_CLASSES,
    FIELD_SPEC,
    FIELDS_BY_NAME,
    DocType,
    FieldSpec,
    LineItem,
    NormKind,
    ValueType,
    fields_for,
    get_field_value,
    parse_document,
)
from factories import doc_data


def _unwrap(annotation: object) -> object:
    """list[X] → X, X | None → X."""
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        assert len(args) == 1
        return _unwrap(args[0])
    if origin is list:
        return _unwrap(typing.get_args(annotation)[0])
    return annotation


def _resolves(model: type[BaseModel], path: tuple[str, ...]) -> bool:
    current: object = model
    for part in path:
        if not (isinstance(current, type) and issubclass(current, BaseModel)):
            return False
        info = current.model_fields.get(part)
        if info is None:
            return False
        current = _unwrap(info.annotation)
    return True


@pytest.mark.parametrize("spec", FIELD_SPEC, ids=lambda s: s.name)
def test_field_exists_exactly_where_declared(spec: FieldSpec) -> None:
    for doc_type, cls in DOCUMENT_CLASSES.items():
        root = LineItem if spec.is_line_field else cls
        exists = _resolves(root, spec.path)
        assert exists == spec.applies_to(doc_type), (spec.name, doc_type)


def test_names_unique_and_index_complete() -> None:
    assert len(FIELDS_BY_NAME) == len(FIELD_SPEC)


def test_required_subset_of_doc_types() -> None:
    with pytest.raises(ValueError, match="required_in"):
        FieldSpec(
            "x", ValueType.STR, NormKind.TEXT, frozenset({DocType.UPD}), frozenset({DocType.BILL})
        )


def test_key_fields_are_declared() -> None:
    for name in ("seller.inn", "seller_bank.bik", "lines[].quantity", "totals.amount_with_vat"):
        assert name in FIELDS_BY_NAME
    assert FIELDS_BY_NAME["seller_bank.bik"].is_required(DocType.BILL)
    assert not FIELDS_BY_NAME["seller_bank.bik"].is_required(DocType.ACT)
    assert not FIELDS_BY_NAME["seller.kpp"].is_required(DocType.UPD)  # у ИП КПП нет


def test_fields_for_doc_type() -> None:
    names = {spec.name for spec in fields_for(DocType.VAT_INVOICE)}
    assert "seller_bank.bik" not in names
    assert "basis.number" not in names
    assert "upd_status" not in names
    assert "upd_status" in {spec.name for spec in fields_for(DocType.UPD)}


def test_get_field_value() -> None:
    doc = parse_document(doc_data("bill"))
    assert get_field_value(doc, FIELDS_BY_NAME["seller.inn"]) == "7707083893"
    assert get_field_value(doc, FIELDS_BY_NAME["lines[].quantity"]) == [Decimal(2), Decimal(3)]
    assert get_field_value(doc, FIELDS_BY_NAME["upd_status"]) is None  # у счёта нет
    act = parse_document(doc_data("act", seller_bank=None))
    assert get_field_value(act, FIELDS_BY_NAME["seller_bank.bik"]) is None
