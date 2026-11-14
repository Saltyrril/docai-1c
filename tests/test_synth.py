"""Тесты генератора синтетических данных (P3)."""

import json
import re
import time
from collections import Counter
from datetime import date

import pytest
from typer.testing import CliRunner

from docai.cli import app
from docai.schemas.common import VatRate
from docai.schemas.documents import parse_document
from docai.synth.catalog import CATEGORY_QUOTA, get_catalog
from docai.synth.directory import N_INDIVIDUALS, N_ORGANIZATIONS, get_directory
from docai.synth.generator import (
    DATE_FROM,
    DATE_TO,
    generate_document,
    generate_documents,
    main_vat_rate,
)
from docai.synth.requisites import make_inn10, make_inn12, make_settlement_account
from docai.validation import (
    has_errors,
    validate_account,
    validate_bik,
    validate_corr_account,
    validate_document,
    validate_inn,
    validate_kpp,
)

SEED = 42


@pytest.fixture(scope="module")
def batch():
    return generate_documents(1000, SEED)


# --- Справочники -----------------------------------------------------------------------


def test_directory_sizes_and_uniqueness():
    d = get_directory()
    assert len(d.organizations) == N_ORGANIZATIONS == 300
    assert len(d.individuals) == N_INDIVIDUALS == 50
    assert len({e.inn for e in d.entities}) == 350
    assert len({e.name for e in d.entities}) == 350
    assert all(e.kpp is None and e.name.startswith("ИП ") for e in d.individuals)


def test_directory_requisites_are_valid():
    for e in get_directory().entities:
        assert not validate_inn(e.inn)
        assert (e.kpp is None) == e.is_individual
        if e.kpp:
            assert not validate_kpp(e.kpp)
        assert not validate_bik(e.bank.bik)
        assert not validate_account(e.account, e.bank.bik)
        assert not validate_corr_account(e.bank.corr_account, e.bank.bik)
        assert len(e.inn) == (12 if e.is_individual else 10)


def test_directory_is_stable():
    first = get_directory()
    get_directory.cache_clear()
    assert get_directory() == first


def test_inn_generators():
    import random

    rng = random.Random(1)
    assert all(not validate_inn(make_inn10(rng)) for _ in range(500))
    assert all(not validate_inn(make_inn12(rng)) for _ in range(500))
    bik = get_directory().banks[0].bik
    assert not validate_account(make_settlement_account(rng, bik, individual=True), bik)


# --- Каталог ---------------------------------------------------------------------------


def test_catalog_has_500_items_in_five_categories():
    catalog = get_catalog()
    assert len(catalog) == 500
    assert Counter(i.category for i in catalog) == CATEGORY_QUOTA
    assert len({i.name for i in catalog}) == 500


def test_catalog_units_and_prices():
    for item in get_catalog():
        assert re.fullmatch(r"\d{3}", item.unit_code)
        assert item.unit_name
        assert item.price > 0
        assert item.qty_max >= 1


# --- Документы -------------------------------------------------------------------------


def test_determinism_same_seed():
    a = [d.document.model_dump_json() for d in generate_documents(60, 7)]
    b = [d.document.model_dump_json() for d in generate_documents(60, 7)]
    assert a == b


def test_different_seed_differs():
    a = [d.document.model_dump_json() for d in generate_documents(20, 1)]
    b = [d.document.model_dump_json() for d in generate_documents(20, 2)]
    assert a != b


def test_document_does_not_depend_on_n():
    short = generate_documents(5, SEED)
    long = generate_documents(30, SEED)
    assert [d.document for d in short] == [d.document for d in long[:5]]
    assert generate_document(SEED, 3).document == short[2].document


def test_clean_documents_pass_validation(batch):
    clean = [d for d in batch if not d.has_errors]
    assert len(clean) > 900
    for d in clean:
        assert validate_document(d.document) == [], d.file_name


def test_corrupted_documents_are_caught(batch):
    broken = [d for d in batch if d.has_errors]
    assert 20 <= len(broken) <= 90  # около 5% от 1000
    for d in broken:
        assert has_errors(validate_document(d.document)), d.file_name
        assert d.expected_codes
    kinds = {k for d in broken for k in d.error_kinds}
    assert kinds == {"line_amount", "line_vat", "total", "invalid_inn"}


def test_error_rate_extremes():
    assert not any(d.has_errors for d in generate_documents(100, SEED, error_rate=0))
    assert all(d.has_errors for d in generate_documents(50, SEED, error_rate=1))


def test_all_doc_types_and_line_counts(batch):
    assert {d.document.doc_type.value for d in batch} == {
        "upd", "bill", "act", "torg12", "vat_invoice",
    }  # fmt: skip
    counts = [len(d.document.lines) for d in batch]
    assert min(counts) == 1
    assert max(counts) == 25
    assert all(1 <= c <= 25 for c in counts)


def test_dates_and_vat_rates_follow_date(batch):
    seen = Counter()
    for d in batch:
        doc = d.document
        assert DATE_FROM <= doc.date <= DATE_TO
        for line in doc.lines:
            seen[line.vat_rate] += 1
            if line.vat_rate is VatRate.R22:
                assert doc.date >= date(2026, 1, 1)
            if line.vat_rate is VatRate.R20:
                assert doc.date < date(2026, 1, 1)
    assert main_vat_rate(date(2025, 12, 31)) is VatRate.R20
    assert main_vat_rate(date(2026, 1, 1)) is VatRate.R22
    total = sum(seen.values())
    assert (seen[VatRate.R22] + seen[VatRate.R20]) / total > 0.6
    for rare in (VatRate.R10, VatRate.NO_VAT, VatRate.R5, VatRate.R7):
        assert seen[rare] > 0


def test_number_formats(batch):
    numbers = [d.document.number for d in batch]
    assert any(re.fullmatch(r"\d{1,4}", n) for n in numbers)
    assert any(re.fullmatch(r"[А-Я]-\d{4}", n) for n in numbers)
    assert any(re.fullmatch(r"[А-Я]{2}-\d{6}", n) for n in numbers)


def test_basis_is_not_after_document(batch):
    with_basis = 0
    for d in batch:
        basis = getattr(d.document, "basis", None)
        if basis is not None:
            with_basis += 1
            assert basis.date <= d.document.date
            assert basis.number
    assert with_basis > 500


def test_bill_has_bank_and_seller_differs_from_buyer(batch):
    for d in batch:
        doc = d.document
        assert doc.seller.inn != doc.buyer.inn or d.error_kinds
        if doc.doc_type.value == "bill":
            assert doc.seller_bank.account


def test_json_roundtrip(batch):
    for d in batch[:100]:
        data = json.loads(d.document.model_dump_json())
        assert parse_document(data) == d.document


def test_generation_is_fast():
    started = time.perf_counter()
    generate_documents(1000, 5)
    assert time.perf_counter() - started < 60


# --- CLI -------------------------------------------------------------------------------


def test_cli_generate(tmp_path):
    out = tmp_path / "json"
    result = CliRunner().invoke(
        app, ["synth", "generate", "--n", "25", "--seed", "42", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    files = sorted(out.glob("*.json"))
    assert len(files) == 25
    manifest = [json.loads(x) for x in (out / "_manifest.jsonl").read_text("utf-8").splitlines()]
    assert len(manifest) == 25
    for record in manifest:
        doc = parse_document((out / record["file"]).read_text("utf-8"))
        assert doc.doc_type.value == record["doc_type"]
        assert "error_kinds" not in json.loads((out / record["file"]).read_text("utf-8"))


def test_cli_same_seed_same_files(tmp_path):
    runner = CliRunner()
    for name in ("a", "b"):
        runner.invoke(
            app, ["synth", "generate", "--n", "15", "--seed", "3", "--out", str(tmp_path / name)]
        )
    for f in (tmp_path / "a").iterdir():
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()
