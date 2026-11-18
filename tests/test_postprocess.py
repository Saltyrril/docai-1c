"""P10: нормализация после LLM, автоисправления, уверенность, статус, измерение порч."""

import math
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docai.cli import app
from docai.eval import CORRUPTIONS, corrupt, evaluate_clean, evaluate_corruptions, same_document
from docai.extract import Mode, extract
from docai.extract import pipeline as pipeline_module
from docai.postprocess import (
    DocStatus,
    PostprocessConfig,
    SourceText,
    autofix_document,
    normalize_document,
    postprocess,
    score_fields,
)
from docai.postprocess.confidence import involved_fields, logit, sigmoid
from docai.postprocess.evidence import verbatim
from docai.schemas.common import VatRate
from docai.schemas.documents import parse_document
from docai.schemas.fields import NormKind
from docai.synth.generator import generate_documents
from docai.validation import IssueCode, validate_document
from factories import doc_data, line
from test_extract import TEXT_PDF, FakeOllama, answer, draft_data


def make(doc_type: str = "upd", **overrides):
    return parse_document(doc_data(doc_type, **overrides))


def set_line(doc, index: int, **changes):
    result = doc.model_copy(deep=True)
    result.lines[index] = result.lines[index].model_copy(update=changes)
    return result


def set_totals(doc, **changes):
    result = doc.model_copy(deep=True)
    result.totals = result.totals.model_copy(update=changes)
    return result


# ---------- нормализация ----------


def test_normalize_clean_document_changes_nothing() -> None:
    doc = make()
    result, log = normalize_document(doc)
    assert log == []
    assert result == doc


def test_normalize_logs_every_change() -> None:
    doc = make("bill")
    doc.seller.inn = "ИНН 7707 083 893"
    doc.seller.kpp = "7736 01001"
    doc.number = "№ 12"
    doc.seller_bank.bik = "04 4525 225"
    doc.lines[0].unit_code = "6"
    doc.lines[0].name = "Бумага   А4 белая"
    result, log = normalize_document(doc)
    assert result.seller.inn == "7707083893"
    assert result.seller.kpp == "773601001"
    assert result.number == "12"
    assert result.seller_bank.bik == "044525225"
    assert result.lines[0].unit_code == "006"
    assert result.lines[0].name == "Бумага А4 белая"
    assert {c.field for c in log} == {
        "seller.inn",
        "seller.kpp",
        "number",
        "seller_bank.bik",
        "lines[0].unit_code",
        "lines[0].name",
    }
    assert all(c.kind == "normalize" and c.before != c.after for c in log)
    assert doc.number == "№ 12"  # исходный документ не изменён


def test_normalize_splits_inn_and_kpp_written_together() -> None:
    doc = make()
    doc.buyer.inn = "7710140679/771301001"
    doc.buyer.kpp = None
    result, log = normalize_document(doc)
    assert (result.buyer.inn, result.buyer.kpp) == ("7710140679", "771301001")
    assert [c.rule for c in log] == ["split_inn_kpp", "split_inn_kpp"]
    keep = make()
    keep.buyer.inn = "7710140679/111111111"  # КПП уже есть — не затираем
    assert normalize_document(keep)[0].buyer.kpp == "771301001"


def test_normalize_empty_optional_becomes_none() -> None:
    doc = make()
    doc.seller.kpp = "  "
    result, _ = normalize_document(doc)
    assert result.seller.kpp is None


# ---------- автоисправления ----------


@pytest.mark.parametrize(
    ("attr", "wrong", "right"),
    [
        ("amount_without_vat", "2010.00", "2000.00"),
        ("vat_amount", "441.00", "440.00"),
        ("amount_with_vat", "2540.00", "2440.00"),
    ],
)
def test_autofix_single_wrong_line_amount(attr: str, wrong: str, right: str) -> None:
    good = make()
    bad = set_line(good, 0, **{attr: Decimal(wrong)})
    assert validate_document(bad)
    fixed, log = autofix_document(bad)
    assert same_document(fixed, good)
    assert [c.field for c in log] == [f"lines[0].{attr}"]
    assert log[0].kind == "autofix"
    assert log[0].before == wrong
    assert Decimal(log[0].after) == Decimal(right)
    assert "Строка 1" in log[0].reason
    assert validate_document(fixed) == []
    assert bad.lines[0].model_dump()[attr] == Decimal(wrong)  # исходный не изменён


def test_autofix_does_not_touch_quantity_or_price() -> None:
    """Неверное количество: непонятно, ошиблись в количестве, цене или сумме."""
    good = make()
    for changes in ({"quantity": Decimal(3)}, {"price": Decimal("1100.00")}):
        bad = set_line(good, 0, **changes)
        fixed, log = autofix_document(bad)
        assert log == []
        assert same_document(fixed, bad)


def test_autofix_does_not_touch_wrong_vat_rate() -> None:
    bad = set_line(make(), 0, vat_rate=VatRate.R20)
    _, log = autofix_document(bad)
    assert log == []


def test_autofix_total_equal_to_lines_sum() -> None:
    good = make()
    bad = set_totals(good, amount_with_vat=Decimal("2661.99"))
    fixed, log = autofix_document(bad)
    assert same_document(fixed, good)
    assert [(c.field, c.rule) for c in log] == [
        ("totals.amount_with_vat", "total_equals_lines_sum")
    ]


def test_autofix_skips_missing_line() -> None:
    """Все итоги расходятся со строками: пропущена строка, а не ошибка в одном итоге."""
    bad = make()
    bad.lines.pop()
    fixed, log = autofix_document(bad)
    assert log == []
    assert same_document(fixed, bad)


def test_autofix_trusts_total_that_confirms_amount() -> None:
    """Сумма строки без НДС ≠ количество × цена, но итог по документу сходится с ней."""
    doc = make(
        "bill",
        lines=[
            line("2", "100.00", "250.00", None, None, None),
            line("1", "50.00", "50.00", None, None, None),
        ],
        totals={"amount_without_vat": "300.00", "vat_amount": None, "amount_with_vat": "300.00"},
    )
    assert any(i.code is IssueCode.LINE_AMOUNT_MISMATCH for i in validate_document(doc))
    _, log = autofix_document(doc)
    assert log == []


def test_autofix_uses_total_to_choose_among_few_numbers() -> None:
    """В строке только количество, цена и сумма; итог подтверждает исправление суммы."""
    lines = [
        line("2", "100.00", "250.00", None, None, None),
        line("1", "50.00", "50.00", None, None, None),
    ]
    doc = make(
        "bill",
        lines=lines,
        totals={"amount_without_vat": "250.00", "vat_amount": None, "amount_with_vat": "250.00"},
    )
    fixed, log = autofix_document(doc)
    assert [c.field for c in log] == ["lines[0].amount_without_vat"]
    assert fixed.lines[0].amount_without_vat == Decimal("200.00")


def test_autofix_skips_when_source_has_the_printed_value() -> None:
    """В тексте напечатано именно неверное значение: ошибка в самом документе."""
    bad = set_line(make(), 0, amount_without_vat=Decimal("2010.00"))
    printed = [SourceText.build("text_layer", "Бумага 2 1 000,00 2 010,00 22% 440,00 2 440,00")]
    assert autofix_document(bad, printed)[1] == []
    other = [SourceText.build("text_layer", "Бумага 2 1 000,00 2 000,00 22% 440,00 2 440,00")]
    assert len(autofix_document(bad, other)[1]) == 1


# ---------- дословное вхождение ----------


def test_verbatim_forms() -> None:
    src = [
        SourceText.build("text_layer", "Итого | 12 200,50 | ИНН 7707083893 | 14 ноября 2026 г. |")
    ]
    assert verbatim(NormKind.MONEY, Decimal("12200.50"), src) == (True, True)
    assert verbatim(NormKind.MONEY, Decimal("12200.51"), src) == (False, True)
    assert verbatim(NormKind.INN, "7707083893", src) == (True, True)
    assert verbatim(NormKind.INN, "7707083894", src) == (False, True)
    from datetime import date

    assert verbatim(NormKind.DATE, date(2026, 11, 14), src) == (True, True)
    assert verbatim(NormKind.DATE, date(2026, 11, 15), src) == (False, True)
    assert verbatim(NormKind.MONEY, Decimal("12200.50"), []) is None
    assert verbatim(NormKind.EXACT, "upd", src) is None


def test_verbatim_money_is_not_a_substring_of_a_longer_number() -> None:
    src = [SourceText.build("text_layer", "всего 112 200,50")]
    assert verbatim(NormKind.MONEY, Decimal("12200.50"), src) == (False, True)


def test_verbatim_short_value_is_weak() -> None:
    src = [SourceText.build("text_layer", "шт 2 1 000,00")]
    assert verbatim(NormKind.QUANTITY, Decimal(2), src) == (True, False)


def test_verbatim_text_ignores_case_quotes_and_table_bars() -> None:
    src = [SourceText.build("text_layer", "| Продавец: АО «ТБанк», Москва |")]
    assert verbatim(NormKind.TEXT, 'АО "Тбанк"', src) == (True, True)


# ---------- уверенность ----------


def field_conf(doc, issues=None, **kwargs):
    issues = validate_document(doc) if issues is None else issues
    return {f.field: f for f in score_fields(doc, issues, **kwargs)}


def test_confidence_formula_prior_plus_one_signal() -> None:
    """σ(logit(p0) + ln LR): адрес без проверок, один сигнал «нашёлся в тексте»."""
    cfg = PostprocessConfig()
    doc = make()
    src = [SourceText.build("text_layer", "Адрес: Москва")]
    plain = field_conf(doc, route="vision", config=cfg)["seller.address"]
    with_text = field_conf(doc, route="vision", sources=src, config=cfg)["seller.address"]
    prior = cfg.priors["vision"]["text"]
    assert plain.confidence == pytest.approx(prior, abs=1e-3)
    expected = sigmoid(logit(prior) + math.log(cfg.lr_found_strong))
    assert with_text.confidence == pytest.approx(expected, abs=1e-3)
    assert with_text.signals == {"verbatim": cfg.lr_found_strong}


def test_confidence_validation_error_lowers_and_pass_raises() -> None:
    good = make()
    bad = set_line(good, 0, amount_without_vat=Decimal("2010.00"))
    ok = field_conf(good, route="vision")
    broken = field_conf(bad, route="vision")
    name = "lines[0].amount_without_vat"
    assert ok[name].confidence > 0.9
    assert broken[name].confidence < 0.2
    assert broken[name].confidence < broken["lines[1].amount_without_vat"].confidence
    assert any("LINE_AMOUNT_MISMATCH" in r for r in broken[name].reasons)
    assert broken["lines[0].vat_amount"].confidence < ok["lines[0].vat_amount"].confidence


def test_confidence_inn_checksum() -> None:
    doc = make()
    doc.seller.inn = "7707083894"
    got = field_conf(doc, route="text")
    assert got["seller.inn"].confidence < 0.2
    assert got["buyer.inn"].confidence > 0.95


def test_confidence_text_missing_from_text_layer_is_strong_evidence() -> None:
    doc = make()
    src = [SourceText.build("text_layer", "Итого 2 561,99 ИНН 7707083893 7710140679")]
    got = field_conf(doc, route="text", sources=src)
    assert got["totals.amount_with_vat"].confidence > 0.97
    assert got["lines[0].name"].confidence < 0.5  # «Бумага» в тексте нет


def test_confidence_okei_absent_from_text_is_not_evidence() -> None:
    """Код ОКЕИ во многих формах не печатается: отсутствие в тексте не снижает уверенность."""
    doc = make()
    src = [SourceText.build("text_layer", "Бумага 2 шт 1 000,00")]
    plain = field_conf(doc, route="text")["lines[0].unit_code"].confidence
    with_text = field_conf(doc, route="text", sources=src)["lines[0].unit_code"]
    assert with_text.confidence == plain
    assert "verbatim" not in with_text.signals


def test_confidence_cross_path() -> None:
    doc = make()
    other = set_line(doc, 0, name="Другое")
    base = field_conf(doc, route="vision")["lines[0].name"].confidence
    got = field_conf(doc, route="vision", alternates=[other])
    assert got["lines[0].name"].confidence < base
    assert got["lines[1].name"].confidence > base  # вторая строка совпала


def test_confidence_missing_required_field_is_zero() -> None:
    doc = make()
    doc.seller.address = None
    got = field_conf(doc, route="text")
    assert got["seller.address"].confidence == 0
    assert got["seller.address"].value is None


def test_confidence_corrected_field_is_capped() -> None:
    doc = make()
    got = field_conf(doc, route="text", corrected=frozenset({"lines[0].amount_without_vat"}))
    assert got["lines[0].amount_without_vat"].confidence == PostprocessConfig().corrected_cap
    assert got["lines[0].amount_without_vat"].corrected


def test_confidence_validation_boost_is_capped() -> None:
    cfg = PostprocessConfig(validation_boost_cap=2.0)
    got = field_conf(make(), route="text", config=cfg)
    assert got["lines[0].amount_without_vat"].signals["validation"] == 2.0


def test_involved_fields_cover_the_equation() -> None:
    doc = set_line(make(), 0, amount_without_vat=Decimal("2010.00"))
    issue = next(i for i in validate_document(doc) if i.code is IssueCode.LINE_AMOUNT_MISMATCH)
    names = involved_fields(issue, len(doc.lines))
    assert {"lines[0].quantity", "lines[0].price"} <= set(names)


# ---------- статус ----------


def printed(doc) -> list[tuple[str, str]]:
    """Текст «как напечатано»: все значения документа."""
    parts = [doc.number, doc.seller.name, doc.seller.inn, doc.seller.kpp, doc.seller.address]
    parts += [doc.buyer.name, doc.buyer.inn, doc.buyer.kpp, doc.buyer.address]
    parts += [doc.date.strftime("%d.%m.%Y")]
    for ln in doc.lines:
        parts += [ln.name, ln.unit_name, ln.unit_code, str(ln.quantity), str(ln.price)]
        parts += [str(ln.amount_without_vat), str(ln.vat_amount), str(ln.amount_with_vat)]
    parts += [str(v) for v in (doc.totals.amount_without_vat, doc.totals.vat_amount)]
    parts += [str(doc.totals.amount_with_vat)]
    basis = getattr(doc, "basis", None)
    if basis is not None:
        parts += [basis.number, basis.name, basis.date.strftime("%d.%m.%Y")]
    return [("text_layer", " | ".join(str(p) for p in parts))]


def test_status_upload_for_clean_document_with_text() -> None:
    doc = make()
    result = postprocess(doc, route="text", texts=printed(doc))
    assert result.report.status is DocStatus.UPLOAD
    assert result.report.status_ru == "можно загружать"
    assert result.report.review_fields == []
    assert result.report.corrections == []
    assert result.report.mean_confidence and result.report.mean_confidence > 0.9


def test_status_review_lists_low_confidence_fields() -> None:
    doc = make()
    doc.seller.inn = "7707083894"  # ИНН с неверной цифрой: исправить нельзя
    result = postprocess(doc, route="text", texts=printed(make()))
    report = result.report
    assert report.status is DocStatus.REVIEW
    assert "seller.inn" in [f.field for f in report.review_fields]
    assert [f.confidence for f in report.review_fields] == sorted(
        f.confidence for f in report.review_fields
    )
    assert any(i.code is IssueCode.INN_CHECKSUM for i in report.issues_after)


def test_status_after_autofix_keeps_log_and_caps_confidence() -> None:
    good = make()
    bad = set_line(good, 0, amount_with_vat=Decimal("2540.00"))
    result = postprocess(bad, route="text", texts=printed(good))
    report = result.report
    assert same_document(result.document, good)
    assert [c.kind for c in report.corrections] == ["autofix"]
    assert report.issues_before and not report.issues_after
    fixed = next(f for f in report.fields if f.field == "lines[0].amount_with_vat")
    assert fixed.corrected
    assert fixed.confidence <= PostprocessConfig().corrected_cap


def test_status_review_when_correction_is_not_trusted() -> None:
    cfg = PostprocessConfig(corrected_cap=0.5)
    good = make()
    bad = set_line(good, 0, amount_with_vat=Decimal("2540.00"))
    report = postprocess(bad, route="text", texts=printed(good), config=cfg).report
    assert report.status is DocStatus.REVIEW
    assert "lines[0].amount_with_vat" in [f.field for f in report.review_fields]


def test_autofix_can_be_disabled() -> None:
    bad = set_line(make(), 0, amount_with_vat=Decimal("2540.00"))
    result = postprocess(bad, route="text", config=PostprocessConfig(autofix=False))
    assert result.report.corrections == []
    assert result.report.status is DocStatus.REVIEW


def test_status_reject_without_document_or_lines() -> None:
    assert postprocess(None, route="vision").report.status is DocStatus.REJECT
    empty = make()
    empty.lines = []
    assert postprocess(empty, route="text").report.status is DocStatus.REJECT


def test_status_reject_when_most_fields_unread() -> None:
    """OCR без подтверждений: цифры не сходятся, ИНН неверны, текста нет в источнике."""
    doc = make()
    doc.seller.inn = "7707083894"
    doc.buyer.inn = "7710140678"
    doc.lines[0].price = Decimal("1")
    doc.lines[1].price = Decimal("2")
    doc.totals.amount_with_vat = Decimal("5")
    result = postprocess(doc, route="ocr", texts=[("ocr", "совсем другой текст")])
    assert result.report.status is DocStatus.REJECT
    assert "не прочитана" in result.report.reasons[0]


def test_xml_route_is_never_autofixed() -> None:
    """В XML сумма взята у отправителя; исправлять её — значит выдумывать данные."""
    bad = set_line(make(), 0, amount_with_vat=Decimal("2540.00"))
    report = postprocess(bad, route="xml").report
    assert report.corrections == []
    assert report.status is DocStatus.REVIEW


# ---------- конвейер ----------


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docai.extract.vram.shutil.which", lambda name: None)


def test_pipeline_fixes_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    data = draft_data()
    data["lines"][0]["amount_with_vat"] = 2540.00  # модель ошиблась в цифре
    monkeypatch.setattr(pipeline_module, "chat", FakeOllama(answer(data)))
    result = extract([TEXT_PDF])
    report = result.postprocess
    assert report is not None
    assert report.status_ru in {"нужна проверка", "отклонить", "можно загружать"}
    sums = [
        c.field for c in report.corrections if c.kind == "autofix" and c.rule != "doc_type_by_title"
    ]
    assert sums == ["lines[0].amount_with_vat"]
    # Подменённый ответ — «upd», а PDF — счёт на оплату: тип исправлен по заголовку (D50).
    assert result.document["doc_type"] == "bill"
    assert Decimal(result.document["lines"][0]["amount_with_vat"]) == Decimal("2440.00")
    assert "postprocess" in result.meta.timings_s


def test_pipeline_cross_check_agreement_raises_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeOllama(answer(draft_data()))
    monkeypatch.setattr(pipeline_module, "chat", fake)
    single = extract([TEXT_PDF], Mode.TEXT).postprocess
    both = extract([TEXT_PDF], Mode.TEXT, cross_check=Mode.TEXT)
    assert both.meta.cross_check == "text"
    assert len(fake.bodies) == 3
    assert both.postprocess.mean_confidence > single.mean_confidence


def test_pipeline_cli_prints_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline_module, "chat", FakeOllama(answer(draft_data())))
    out = tmp_path / "r.json"
    result = CliRunner().invoke(app, ["extract", str(TEXT_PDF), "--out", str(out), "-q"])
    assert result.exit_code == 0, result.output
    assert "статус:" in result.output
    assert '"postprocess"' in out.read_text(encoding="utf-8")


# ---------- порчи и критерий готовности ----------


@pytest.fixture(scope="module")
def clean_docs():
    return [sd.document for sd in generate_documents(80, seed=2026, error_rate=0.0)]


def test_every_corruption_is_applicable_and_changes_the_document(clean_docs) -> None:
    for corruption in CORRUPTIONS:
        changed = 0
        for index, doc in enumerate(clean_docs):
            bad = corrupt(doc, corruption, f"t:{index}")
            changed += bad is not None and not same_document(bad, doc)
        assert changed >= 10, corruption.name


def test_corruptions_are_deterministic(clean_docs) -> None:
    for corruption in CORRUPTIONS[:6]:
        a = corrupt(clean_docs[3], corruption, "s")
        b = corrupt(clean_docs[3], corruption, "s")
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            assert same_document(a, b)


def test_validation_catches_at_least_95_percent_of_rule_errors(clean_docs) -> None:
    stats = [s for s in evaluate_corruptions(clean_docs) if s.group == "rule"]
    applied = sum(s.applied for s in stats)
    caught = sum(s.caught for s in stats)
    assert applied > 1000
    assert caught / applied >= 0.95, [(s.name, s.caught, s.applied) for s in stats]
    assert sum(s.wrong_fix for s in stats) == 0  # автоисправление не портит документ


def test_validation_has_no_false_alarms_on_clean_documents(clean_docs) -> None:
    stats = evaluate_clean(clean_docs)
    assert stats.documents == len(clean_docs)
    assert stats.false_alarm_rate <= 0.02
    assert stats.with_fixes == 0


def test_text_source_catches_errors_outside_validation_rules(clean_docs) -> None:
    """Опечатку в наименовании правила не видят, а текст-источник — видит."""
    texts = {id(d): printed(d) for d in clean_docs}
    wanted = [c for c in CORRUPTIONS if c.group == "outside"]
    stats = evaluate_corruptions(
        clean_docs, corruptions=wanted, texts_for=lambda d: texts[id(d)], route="text"
    )
    short_numbers = {"number_digit", "basis_number_digit"}  # «12» легко встречается в тексте
    for s in stats:
        assert s.caught == 0, s.name  # правила молчат...
        limit = 0.35 if s.name in short_numbers else 0.1
        assert s.silent / s.applied < limit, (s.name, s.silent, s.applied)  # ...но статус — нет
