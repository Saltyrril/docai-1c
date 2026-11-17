"""Методика оценки (P11): метрики на ручных мини-примерах с известным ответом."""

import itertools
import json
import math
import random
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from docai.cli import app
from docai.eval import power as power_mod
from docai.eval import predict as predict_mod
from docai.eval.assign import linear_assignment
from docai.eval.calibration import auroc, reliability_bins, reliability_svg
from docai.eval.compare import compare_documents, match_rows
from docai.eval.config import EvalConfig, load_eval_config
from docai.eval.metrics import comparison_counts, count_matrix, main_metrics
from docai.eval.predict import (
    MAX_CONSECUTIVE_FAILURES,
    Prediction,
    fingerprint,
    fingerprint_config,
    load_cached,
    run_predictions,
)
from docai.eval.run import compare_runs, evaluate
from docai.eval.sample import EvalItem, proportional_quotas, select_dataset_items
from docai.eval.stats import (
    bootstrap,
    mcnemar,
    paired_bootstrap,
    ratio,
    resample_weights,
    totals,
)
from docai.eval.text import anls_score, levenshtein, normalized_distance, similarity
from docai.extract import ExtractConfig, ExtractionResult, Mode, RawExtraction
from docai.extract.result import Meta
from docai.postprocess import FieldConfidence
from docai.schemas.documents import parse_document
from factories import doc_data, line

ROOT = Path(__file__).resolve().parents[1]


def doc(doc_type: str = "upd", **overrides):
    return parse_document(doc_data(doc_type, **overrides))


def compare(ref, pred, **kw):
    return compare_documents(ref, pred, **kw)


# ---- Левенштейн и ANLS -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("", "", 0),
     ("ромашка", "ромашка", 0), ("ромашка", "ромошка", 1), ("flaw", "lawn", 2)],
)  # fmt: skip
def test_levenshtein(a, b, expected):
    assert levenshtein(a, b) == expected
    assert levenshtein(b, a) == expected


def test_normalized_distance_and_similarity():
    assert normalized_distance("kitten", "sitting") == pytest.approx(3 / 7)
    assert normalized_distance("", "") == 0
    assert similarity("abcd", "abce") == pytest.approx(0.75)
    assert similarity(None, None) == 1


def test_anls_threshold():
    assert anls_score("abcd", "abcd") == 1
    assert anls_score("abce", "abcd") == pytest.approx(0.75)  # NL = 0,25 < 0,5
    assert anls_score("ab", "abcd") == 0  # NL = 0,5 — не меньше порога
    assert anls_score("abc", "abcd") == pytest.approx(0.75)
    assert anls_score(None, "abcd") == 0
    assert anls_score("ab", "abcd", threshold=0.6) == pytest.approx(0.5)


# ---- венгерский алгоритм ---------------------------------------------------------------


def brute_force(cost):
    n, m = len(cost), len(cost[0])
    best = math.inf
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            best = min(best, sum(cost[i][c] for i, c in enumerate(cols)))
    else:
        for rows in itertools.permutations(range(n), m):
            best = min(best, sum(cost[r][j] for j, r in enumerate(rows)))
    return best


def test_assignment_known_example():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    assert linear_assignment(cost) == [(0, 1), (1, 0), (2, 2)]


def test_assignment_empty_and_ragged():
    assert linear_assignment([]) == []
    assert linear_assignment([[]]) == []
    with pytest.raises(ValueError):
        linear_assignment([[1, 2], [3]])


@pytest.mark.parametrize("seed", range(40))
def test_assignment_matches_brute_force(seed):
    rng = random.Random(seed)
    n, m = rng.randint(1, 6), rng.randint(1, 6)
    cost = [[rng.choice([0, 0.25, 0.5, 1, rng.random()]) for _ in range(m)] for _ in range(n)]
    pairs = linear_assignment(cost)
    assert len(pairs) == min(n, m)
    assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == len(pairs)
    assert sum(cost[i][j] for i, j in pairs) == pytest.approx(brute_force(cost))


# ---- сравнение документа ---------------------------------------------------------------


def test_identical_document_is_perfect():
    ref = doc("upd")
    c = compare(ref, doc("upd"))
    assert c.ok and c.type_ok and c.extracted
    assert c.header_correct == len(c.header)
    assert c.rows_matched == c.rows_exact == 2
    cells = c.cells
    assert cells.ref == cells.pred == cells.correct
    assert c.f1 == 1


def test_normalization_before_compare():
    ref = doc("upd")
    pred = doc(
        "upd",
        number="№ 123",
        seller={**doc_data()["seller"], "name": "пао  сбербанк", "inn": "7707 083 893"},
    )
    c = compare(ref, pred)
    assert c.ok, c.required_errors


def test_header_counts_by_hand():
    """УПД: 16 полей шапки применимы; у фабрики пусты 3 (адреса не везде, банк не у УПД)."""
    ref = doc("upd")
    pred = doc("upd", seller={**doc_data()["seller"], "inn": "7707083894"})
    c = compare(ref, pred)
    wrong = [f.field for f in c.header if not f.correct]
    assert wrong == ["seller.inn"]
    assert c.required_errors == ["seller.inn"]
    assert not c.ok
    assert c.header_correct == len(c.header) - 1


def test_optional_field_error_keeps_doc_ok():
    ref = doc("upd")
    pred = doc("upd", seller={**doc_data()["seller"], "kpp": "773601002"})
    c = compare(ref, pred)
    assert [f.field for f in c.header if not f.correct] == ["seller.kpp"]
    assert c.ok


def test_hallucinated_header_value():
    data = doc_data("upd")
    data["seller"] = {**data["seller"], "kpp": None}
    ref = parse_document(data)
    c = compare(ref, doc("upd"))
    assert ("seller.kpp", "773601001") in c.empty_ref
    counts = comparison_counts(c, 10)
    assert counts["hdr:halluc"] >= 1


def test_reordered_rows_are_matched():
    """Строки в другом порядке: по номеру всё было бы неверно, по сопоставлению — верно."""
    ref = doc("upd")
    data = doc_data("upd")
    data["lines"] = list(reversed(data["lines"]))
    c = compare(ref, parse_document(data))
    assert sorted((p.ref_index, p.pred_index) for p in c.pairs) == [(0, 1), (1, 0)]
    assert c.rows_exact == 2
    assert c.ok


def test_missing_row():
    ref = doc("upd")
    data = doc_data("upd")
    data["lines"] = data["lines"][1:]
    c = compare(ref, parse_document(data))
    assert (c.n_ref_rows, c.n_pred_rows, c.rows_matched) == (2, 1, 1)
    assert c.pairs[0].ref_index == 1
    assert not c.ok
    assert "строк 1 вместо 2" in c.required_errors
    cells = c.cells
    assert cells.correct == cells.pred  # всё, что вернула модель, верно
    assert cells.ref == 2 * cells.pred


def test_extra_row_and_cell_counts_by_hand():
    """Лишняя строка: 9 непустых ячеек каждая (у УПД 9 полей строки)."""
    ref = doc("upd")
    data = doc_data("upd")
    data["lines"].append(line("1", "10.00", "10.00", "22%", "2.20", "12.20", "Скрепки"))
    c = compare(ref, parse_document(data))
    assert (c.n_ref_rows, c.n_pred_rows, c.rows_matched, c.rows_exact) == (2, 3, 2, 2)
    cells = c.cells
    assert (cells.ref, cells.pred, cells.correct) == (18, 27, 18)
    assert not c.ok


def test_wrong_cell_in_matched_row():
    ref = doc("upd")
    data = doc_data("upd")
    data["lines"][0]["quantity"] = "20"
    c = compare(ref, parse_document(data))
    assert c.rows_matched == 2 and c.rows_exact == 1
    assert c.cells.correct == 17
    assert c.line_fields["lines[].quantity"].correct == 1
    assert "lines[0].quantity" in c.required_errors


def test_row_below_threshold_is_not_matched():
    ref = doc("upd")
    data = doc_data("upd")
    data["lines"] = [
        line("7", "1.00", "7.00", "10%", "0.70", "7.70", "Совсем другое"),
        data["lines"][1],
    ]
    pairs = match_rows(ref, parse_document(data), threshold=0.5)
    assert [(i, j) for i, j, _ in pairs] == [(1, 1)]


def test_failed_extraction():
    c = compare(doc("bill"), None)
    assert not c.extracted and not c.ok
    assert c.header_correct == 0 and c.cells.correct == 0
    assert "документ не собран" in c.required_errors
    assert c.f1 == 0


def test_wrong_doc_type_fails_upd_fields():
    """УПД прочитан как счёт-фактура: статуса и основания у счёта-фактуры нет — ошибки."""
    ref = doc("upd")
    data = doc_data("vat_invoice")
    pred = parse_document(data)
    c = compare(ref, pred)
    assert not c.type_ok and not c.ok
    assert {"doc_type", "upd_status"} <= set(c.required_errors)
    wrong = {f.field for f in c.header if not f.correct}
    assert {"basis.name", "basis.number", "basis.date"} <= wrong


def test_anls_on_text_fields():
    ref = doc("upd")
    data = doc_data("upd")
    data["buyer"] = {**data["buyer"], "name": "АО «ТБанк»."}  # одна лишняя точка
    c = compare(ref, parse_document(data))
    buyer = next(f for f in c.header if f.field == "buyer.name")
    assert not buyer.correct
    assert buyer.anls == pytest.approx(1 - 1 / len("ао тбанк."))


def test_confidence_points_follow_row_matching():
    ref = doc("upd")
    data = doc_data("upd")
    data["lines"] = list(reversed(data["lines"]))
    data["lines"][0]["quantity"] = "30"  # предсказанная строка 0 — это строка 1 эталона
    pred = parse_document(data)
    conf = [
        FieldConfidence(field="number", value="123", confidence=0.9),
        FieldConfidence(field="lines[0].quantity", value="30", confidence=0.95),
        FieldConfidence(field="lines[1].quantity", value="2", confidence=0.6),
        FieldConfidence(field="lines[5].price", value="1", confidence=0.5),
    ]
    c = compare(ref, pred, confidences=conf)
    assert [(p.field, p.correct) for p in c.confidence] == [
        ("number", True),
        ("lines[0].quantity", False),
        ("lines[1].quantity", True),
        ("lines[5].price", False),
    ]


# ---- статистика -------------------------------------------------------------------------


def test_resample_weights_rows_sum_to_n():
    w = resample_weights(7, 50, seed=1)
    assert w.shape == (50, 7)
    assert (w.sum(axis=1) == 7).all()
    assert (resample_weights(7, 50, seed=1) == w).all()


def test_bootstrap_micro_average_and_constant_ci():
    # Два документа: 9 из 10 и 1 из 1 верных → micro = 10/11, а не среднее долей 0,95.
    matrix = np.array([[9, 10], [1, 1]], dtype=float)
    est = bootstrap(matrix, ["c", "n"], {"acc": ratio("c", "n")}, n_resamples=500, seed=3)
    assert est["acc"].value == pytest.approx(10 / 11)
    assert est["acc"].low <= est["acc"].value <= est["acc"].high
    constant = np.array([[1, 1]] * 20, dtype=float)
    est = bootstrap(constant, ["c", "n"], {"acc": ratio("c", "n")}, n_resamples=200)
    assert (est["acc"].low, est["acc"].high) == (1.0, 1.0)


def test_bootstrap_ci_covers_known_proportion():
    """Доля 0,5 на 200 документах: интервал ≈ ±1,96·√(0,25/200) ≈ ±0,069."""
    matrix = np.array([[i % 2, 1] for i in range(200)], dtype=float)
    est = bootstrap(matrix, ["ok", "n"], {"p": ratio("ok", "n")}, n_resamples=2000, seed=0)
    assert est["p"].value == 0.5
    assert est["p"].high - est["p"].low == pytest.approx(2 * 1.96 * math.sqrt(0.25 / 200), abs=0.02)


def test_missing_counter_is_zero():
    s = totals(np.array([[1.0]]), ["a"])
    assert s["нет"] == 0


def test_paired_bootstrap():
    a = np.array([[i % 2, 1] for i in range(100)], dtype=float)
    same = paired_bootstrap(a, a.copy(), ["ok", "n"], {"p": ratio("ok", "n")}, n_resamples=300)
    assert same["p"].diff == 0 and same["p"].low == 0 and same["p"].high == 0
    better = a.copy()
    better[:, 0] = 1
    est = paired_bootstrap(a, better, ["ok", "n"], {"p": ratio("ok", "n")}, n_resamples=500)
    assert est["p"].diff == pytest.approx(0.5)
    assert est["p"].low > 0 and est["p"].p_value < 0.01
    with pytest.raises(ValueError):
        paired_bootstrap(a, a[:10], ["ok", "n"], {"p": ratio("ok", "n")})


def test_mcnemar_exact_and_chi2():
    # b = 1, c = 9: p = 2 · (C(10,0) + C(10,1)) / 2¹⁰ = 22/1024.
    a = [True] * 1 + [False] * 9 + [True] * 5
    b = [False] * 1 + [True] * 9 + [True] * 5
    res = mcnemar(a, b)
    assert (res.b, res.c, res.method) == (1, 9, "точный биномиальный")
    assert res.p_value == pytest.approx(22 / 1024)
    # b = 10, c = 30: χ² = (|10 − 30| − 1)² / 40 = 9,025; p ≈ 0,002663.
    a = [True] * 10 + [False] * 30
    b = [False] * 10 + [True] * 30
    res = mcnemar(a, b)
    assert res.statistic == pytest.approx(9.025)
    assert res.p_value == pytest.approx(0.002663, abs=1e-5)
    assert mcnemar([True, False], [True, False]).p_value == 1.0


# ---- калибровка ------------------------------------------------------------------------


def _cal_counts(points):
    """Счётчики калибровки одного «документа» из списка (уверенность, верно)."""
    from docai.eval.compare import ConfidencePoint

    c = compare(doc("bill"), doc("bill"))
    c.confidence = [ConfidencePoint("x", conf, ok) for conf, ok in points]
    return comparison_counts(c, 10)


def test_ece_and_brier_by_hand():
    # Корзина 9: уверенность 0,95 ×2, верно 1 → |1 − 1,9| = 0,9; корзина 1: |0 − 0,3| = 0,3.
    # ECE = (0,9 + 0,3) / 4 = 0,3; MCE = max(0,45; 0,15) = 0,45.
    points = [(0.95, True), (0.95, False), (0.15, False), (0.15, False)]
    matrix, keys = count_matrix([_cal_counts(points)])
    fns = {m.name: m.fn for m in main_metrics(10)}
    s = totals(matrix, keys)
    assert float(fns["ece"](s)) == pytest.approx(0.3)
    assert float(fns["mce"](s)) == pytest.approx(0.45)
    brier = (0.05**2 + 0.95**2 + 0.15**2 + 0.15**2) / 4
    assert float(fns["brier"](s)) == pytest.approx(brier)
    bins = reliability_bins(s, 10)
    assert bins[9]["fields"] == 2 and bins[9]["share_correct"] == 0.5
    assert bins[1]["mean_confidence"] == pytest.approx(0.15)


def test_perfectly_calibrated_has_zero_ece():
    points = [(1.0, True)] * 5 + [(0.0, False)] * 5
    matrix, keys = count_matrix([_cal_counts(points)])
    ece = {m.name: m.fn for m in main_metrics(10)}["ece"]
    assert float(ece(totals(matrix, keys))) == pytest.approx(0)


def test_auroc_by_hand():
    assert auroc([0.9, 0.8, 0.3, 0.2], [True, True, False, False]) == 1
    assert auroc([0.9, 0.8, 0.3, 0.2], [False, False, True, True]) == 0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    # Пары (верное, неверное): (0,9; 0,8) — 1, (0,9; 0,4) — 1, (0,5; 0,8) — 0, (0,5; 0,4) — 1 → 3/4.
    assert auroc([0.9, 0.5, 0.8, 0.4], [True, True, False, False]) == pytest.approx(0.75)
    assert math.isnan(auroc([0.9], [True]))


def test_reliability_svg_is_valid_xml():
    import xml.etree.ElementTree as ET

    points = [(0.95, True), (0.55, False), (0.15, False)]
    matrix, keys = count_matrix([_cal_counts(points)])
    svg = reliability_svg(reliability_bins(totals(matrix, keys), 10), "проверка")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "prefers-color-scheme: dark" in svg


# ---- выборка ---------------------------------------------------------------------------


def test_proportional_quotas():
    counts = {"upd": 95, "torg12": 65, "bill": 63, "vat_invoice": 42, "act": 39}
    q = proportional_quotas(counts, 100)
    assert sum(q.values()) == 100
    assert q == {"upd": 31, "torg12": 21, "bill": 21, "vat_invoice": 14, "act": 13}
    assert proportional_quotas({"b": 1, "a": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    with pytest.raises(ValueError):
        proportional_quotas({"a": 1}, 2)


def make_dataset(root: Path, per_type: int = 4) -> None:
    """Маленький датасет в формате манифеста P6 (картинки и PDF — пустые файлы)."""
    rows = []
    for t, doc_type in enumerate(["upd", "bill", "act"]):
        for k in range(per_type):
            doc_id = f"{t * 10 + k:05d}_{doc_type}"
            (root / "json").mkdir(parents=True, exist_ok=True)
            data = doc_data(doc_type, number=str(k + 1))
            (root / "json" / f"{doc_id}.json").write_text(
                parse_document(data).model_dump_json(), encoding="utf-8"
            )
            for level in ("clean", "scan", "photo"):
                image = f"images/{doc_id}_p1_{level}_200dpi.png"
                rows.append(
                    {
                        "id": f"{doc_id}_{level}_200dpi",
                        "doc_id": doc_id,
                        "doc_type": doc_type,
                        "split": "test" if k < per_type - 1 else "train",
                        "level": level,
                        "dpi": 200,
                        "pages": 1,
                        "json": f"json/{doc_id}.json",
                        "pdf": f"pdf/{doc_id}.pdf",
                        "images": [image],
                        "has_errors": False,
                    }
                )
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_config(tmp_path: Path, **data) -> EvalConfig:
    make_dataset(tmp_path / "ds")
    return EvalConfig.model_validate(
        {
            "name": "t",
            "data": {"dataset": tmp_path / "ds", "n_docs": 6, **data},
            "extract": {"config": None},
            "bootstrap": {"n_resamples": 200},
            "report": {"out_dir": tmp_path / "reports", "date": "2026-11-16", "worst": 2},
            "predictions_dir": tmp_path / "cache",
            # Тесты не должны зависеть от того, заряжается ли ноутбук.
            "power_guard": False,
        }
    )


def test_select_dataset_items(tmp_path):
    cfg = eval_config(tmp_path)
    items = select_dataset_items(cfg)
    assert len(items) == 6
    assert sorted(it.doc_type for it in items) == ["act", "act", "bill", "bill", "upd", "upd"]
    assert [it.level for it in items] == ["clean", "scan", "photo"] * 2
    assert all(it.inputs[0].suffix == (".pdf" if it.level == "clean" else ".png") for it in items)
    assert [it.item_id for it in select_dataset_items(cfg)] == [it.item_id for it in items]
    cfg_all = eval_config(tmp_path, level_assignment="all", exclude_doc_ids=["00000_upd"])
    items_all = select_dataset_items(cfg_all)
    assert len(items_all) == 18
    assert "00000_upd" not in {it.doc_id for it in items_all}


def test_config_rejects_text_mode_with_images(tmp_path):
    with pytest.raises(ValueError):
        EvalConfig.model_validate({"name": "t", "extract": {"mode": "text"}})
    with pytest.raises(ValueError):
        EvalConfig.model_validate({"name": "с пробелом"})


def test_repo_eval_config_loads():
    cfg = load_eval_config(ROOT / "configs/eval/p11_baseline.yaml")
    assert cfg.name == "p11_baseline"
    assert cfg.extract.build().model == "qwen3.5:2b-q8_0"
    assert "00016_upd" in cfg.data.exclude_doc_ids


# ---- кэш ответов и сквозной прогон -----------------------------------------------------


def fake_raw(item_truth: Path, *, break_quantity: bool = False) -> RawExtraction:
    data = json.loads(item_truth.read_text(encoding="utf-8"))
    if break_quantity:
        data["lines"][0]["quantity"] = "999"
    meta = Meta(
        docai_version="test",
        started_at="2026-11-16T00:00:00+00:00",
        source=["x"],
        mode="auto",
        route="text",
        pages=1,
        timings_s={"total": 2.0},
    )
    return RawExtraction(
        result=ExtractionResult(document=data, valid=True, meta=meta),
        texts=[("text_layer", "")],
    )


class FakeExtract:
    def __init__(self, items: list[EvalItem], broken: set[str] = frozenset(), fail: bool = False):
        self.truth = {tuple(it.inputs): it.truth for it in items}
        self.ids = {tuple(it.inputs): it.item_id for it in items}
        self.broken = broken
        self.fail = fail
        self.calls = 0

    def __call__(self, paths, mode, config, cross_check):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Ollama недоступна")
        key = tuple(paths)
        return fake_raw(self.truth[key], break_quantity=self.ids[key] in self.broken)


def test_predictions_are_cached(tmp_path):
    cfg = eval_config(tmp_path)
    items = select_dataset_items(cfg)
    fake = FakeExtract(items)
    config = ExtractConfig()
    run_predictions(items, config, Mode.AUTO, cfg.cache_dir, extract_fn=fake)
    assert fake.calls == 6
    run_predictions(items, config, Mode.AUTO, cfg.cache_dir, extract_fn=fake)
    assert fake.calls == 6  # всё из кэша
    # Постобработка в отпечаток не входит, модель — входит.
    changed_pp = config.model_copy(
        update={"postprocess": config.postprocess.model_copy(update={"autofix": False})}
    )
    assert fingerprint(items[0], changed_pp, Mode.AUTO, None) == fingerprint(
        items[0], config, Mode.AUTO, None
    )
    other_model = config.model_copy(update={"model": "qwen3.5:2b-q4_K_M"})
    run_predictions(items, other_model, Mode.AUTO, cfg.cache_dir, extract_fn=fake)
    assert fake.calls == 12


def test_exceptions_are_recorded_and_stop_the_run(tmp_path):
    cfg = eval_config(tmp_path)
    items = select_dataset_items(cfg)
    fake = FakeExtract(items, fail=True)
    with pytest.raises(RuntimeError, match="исключения подряд"):
        run_predictions(items, ExtractConfig(), Mode.AUTO, cfg.cache_dir, extract_fn=fake)
    assert fake.calls == MAX_CONSECUTIVE_FAILURES
    saved = Prediction.model_validate_json(
        (cfg.cache_dir / f"{items[0].item_id}.json").read_text(encoding="utf-8")
    )
    assert saved.exception and "Ollama" in saved.exception
    ok = FakeExtract(items)
    run_predictions(items, ExtractConfig(), Mode.AUTO, cfg.cache_dir, extract_fn=ok)
    assert ok.calls == 6  # неудачи повторены


def test_evaluate_end_to_end(tmp_path):
    from docai.eval.report import write_report

    cfg = eval_config(tmp_path)
    items = select_dataset_items(cfg)
    broken = {items[0].item_id, items[1].item_id}
    result, _, records = evaluate(cfg, extract_fn=FakeExtract(items, broken))
    o = result["overall"]
    assert o["doc_ok"]["value"] == pytest.approx(4 / 6)
    assert o["extracted"]["value"] == 1
    assert o["header_accuracy"]["value"] == 1
    assert o["rows_f1"]["value"] == 1
    assert o["sec_per_page"]["value"] == pytest.approx(2.0, abs=0.1)
    assert o["doc_ok"]["low"] <= o["doc_ok"]["value"] <= o["doc_ok"]["high"]
    assert set(result["groups"]) == {"doc_type", "level", "route"}
    assert result["groups"]["level"]["values"]["clean"]["documents"] == 2
    assert result["fields"]["lines[].quantity"]["accuracy"]["value"] < 1
    assert result["raw_vs_post"]["metrics"]["doc_ok"]["diff"] == pytest.approx(0)

    out = write_report(result, records, cfg)
    assert out == tmp_path / "reports" / "2026-11-16_t"
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Худшие примеры" in report and "lines[].quantity" in report
    assert "999" in report
    loaded = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert loaded["overall"]["doc_ok"]["value"] == pytest.approx(4 / 6)
    assert (out / "reliability.svg").is_file()
    assert yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))["name"] == "t"

    # Сравнение прогона с самим собой: разность 0, Макнемар без расхождений.
    same = compare_runs(loaded, loaded, cfg)
    assert same["common_documents"] == 6
    assert same["metrics"]["doc_ok"]["diff"] == 0
    assert same["mcnemar_doc_ok"]["p_value"] == 1


def test_cli_eval_and_compare(tmp_path, monkeypatch):
    cfg = eval_config(tmp_path)
    items = select_dataset_items(cfg)
    monkeypatch.setattr(predict_mod, "_default_extract", FakeExtract(items))
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(app, ["eval", "--config", str(path)])
    assert res.exit_code == 0, res.output
    metrics = tmp_path / "reports" / "2026-11-16_t" / "metrics.json"
    assert metrics.is_file()
    res = runner.invoke(app, ["eval-compare", str(metrics), str(metrics)])
    assert res.exit_code == 0, res.output
    assert "Макнемар" in res.output


def test_basis_date_field_type():
    """Дата сравнивается как дата: «15.01.2026» и date(2026, 1, 15) равны."""
    ref = doc("bill")
    data = doc_data("bill")
    data["basis"] = {**data["basis"], "date": date(2026, 1, 15)}
    assert compare(ref, parse_document(data)).ok


# ---- новые поля конфига и питание (D47) ------------------------------------------------


def test_fingerprint_ignores_legacy_default_date_format(tmp_path):
    """Поле date_format со значением iso не меняет отпечаток: кэш P11 остаётся годным."""
    cfg = eval_config(tmp_path)
    item = select_dataset_items(cfg)[0]
    iso = ExtractConfig(date_format="iso")
    printed = ExtractConfig(date_format="printed")
    # При iso содержимое отпечатка такое же, как до появления поля date_format.
    assert "date_format" not in fingerprint_config(iso)
    assert fingerprint_config(printed)["date_format"] == "printed"
    assert "postprocess" not in fingerprint_config(iso)
    assert fingerprint(item, printed, Mode.AUTO, None) != fingerprint(item, iso, Mode.AUTO, None)


def test_model_failure_is_cached(tmp_path):
    """Зациклившаяся генерация — ответ модели: кэшируется. Сбой Ollama — повторяется."""
    cfg = eval_config(tmp_path)
    items = select_dataset_items(cfg)[:1]
    path = cfg.cache_dir / f"{items[0].item_id}.json"

    def failing(message):
        def fn(paths, mode, config, cross_check):
            raise RuntimeError(message)

        return fn

    aborted = 'Ollama вернула 500: {"error":"prediction aborted, token repeat limit reached"}'
    run_predictions(items, ExtractConfig(), Mode.AUTO, cfg.cache_dir, extract_fn=failing(aborted))
    fp = fingerprint(items[0], ExtractConfig(), Mode.AUTO, None)
    assert load_cached(path, fp) is not None
    run_predictions(
        items, ExtractConfig(), Mode.AUTO, cfg.cache_dir, extract_fn=failing("connection refused")
    )
    # Кэш с зацикливанием уже есть — документ не пересчитывался, запись прежняя.
    assert "prediction aborted" in load_cached(path, fp).exception
    path.unlink()
    run_predictions(
        items, ExtractConfig(), Mode.AUTO, cfg.cache_dir, extract_fn=failing("connection refused")
    )
    assert load_cached(path, fp) is None


def test_eval_config_date_format_override():
    cfg = load_eval_config(ROOT / "configs/eval/p11_date_printed.yaml")
    built = cfg.extract.build()
    assert (built.date_format, built.text_prompt, built.vision_prompt) == (
        "printed",
        "text_v2",
        "vision_v2",
    )
    base = load_eval_config(ROOT / "configs/eval/p11_baseline.yaml")
    assert base.extract.build().date_format == "iso"
    # Эксперимент сравнивается с базой на тех же документах.
    assert [i.item_id for i in select_dataset_items(cfg)] == [
        i.item_id for i in select_dataset_items(base)
    ]


def test_power_watch_detects_battery_and_sleep(monkeypatch):
    state = {"ac": True}
    monkeypatch.setattr(power_mod, "on_ac_power", lambda: state["ac"])
    with power_mod.PowerWatch(poll_s=0.01) as watch:
        pass
    assert watch.clean
    with power_mod.PowerWatch(poll_s=0.01) as watch:
        state["ac"] = False
    assert watch.battery and not watch.clean
    state["ac"] = True
    # Сон: настенные часы прыгнули на час между двумя замерами.
    clock = iter([1000.0, 1000.5, 4600.0])
    with power_mod.PowerWatch(poll_s=10, clock=lambda: next(clock, 4601.0)) as watch:
        pass
    assert watch.clean  # 1000,0 → 1000,5: разрыв полсекунды
    watch._sample()
    assert watch.slept and not watch.clean
    assert watch.reason == "сон"


def test_power_guard_waits_and_recomputes(tmp_path, monkeypatch):
    """Батарея во время документа: ответ не сохраняется, документ считается заново от сети."""
    cfg = eval_config(tmp_path)
    items = select_dataset_items(cfg)[:2]
    state = {"ac": True, "unplug_once": True}
    monkeypatch.setattr(power_mod, "on_ac_power", lambda: state["ac"])
    monkeypatch.setattr(power_mod, "WAIT_POLL_S", 0)
    monkeypatch.setattr(power_mod, "_sleep", lambda s: state.update(ac=True))
    fake = FakeExtract(items)

    def extract_with_unplug(paths, mode, config, cross_check):
        if state["unplug_once"]:
            state["unplug_once"] = False
            state["ac"] = False  # зарядку выдернули посреди документа
        return fake(paths, mode, config, cross_check)

    notes: list[str] = []
    preds = run_predictions(
        items,
        ExtractConfig(),
        Mode.AUTO,
        cfg.cache_dir,
        extract_fn=extract_with_unplug,
        power_guard=True,
        notice=notes.append,
    )
    assert fake.calls == 3  # первый документ посчитан дважды
    assert all(p.raw is not None and p.on_ac for p in preds.values())
    assert any("пересчитаю" in n for n in notes)
    assert any("жду" in n for n in notes)


def test_p_value_format():
    from docai.eval.report import fmt_p

    assert fmt_p(0.000244) == "< 0,001"
    assert fmt_p(0.0293) == "0,029"
    assert fmt_p(0.0293, eq=True) == "= 0,029"
    assert fmt_p(float("nan")) == "—"
