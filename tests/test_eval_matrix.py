"""Матрица экспериментов P12: порядок документов, партии с кэшем, VRAM-проверка, сводка."""

import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml
from typer.testing import CliRunner

from docai.cli import app
from docai.eval.matrix import (
    MatrixConfig,
    active_cells,
    build_cells,
    cell_items,
    check_vram,
    load_matrix_config,
    plan_docs,
    progress_table,
    real_items,
    run_matrix,
    skip_reason,
)
from docai.eval.matrix_charts import level_bars_svg, scatter_svg
from docai.eval.matrix_report import aggregate, write_matrix_report
from test_eval import fake_raw, make_dataset

ROOT = Path(__file__).resolve().parents[1]


def matrix_config(tmp_path: Path, **over) -> MatrixConfig:
    make_dataset(tmp_path / "ds", per_type=7)  # test: 6 документов каждого типа = 18
    data = {
        "name": "m",
        "batch_docs": 6,
        "data": {"dataset": tmp_path / "ds", "n_docs": 18, "include_real": False},
        "extract_config": None,
        "paths": {
            "text": {"mode": "text", "levels": ["clean"]},
            "vision": {"mode": "vision", "levels": ["clean", "scan", "photo"]},
        },
        "models": [
            {"id": "a", "tag": "tag-a", "paths": ["text", "vision"]},
            {"id": "b", "tag": "tag-b", "paths": ["vision"], "require_full_vram": True},
        ],
        "bootstrap": {"n_resamples": 100},
        "report": {"out_dir": tmp_path / "reports", "date": "2026-11-18"},
        "predictions_dir": tmp_path / "cache",
        "power_guard": False,
    }
    data.update(over)
    return MatrixConfig.model_validate(data)


class Fake:
    """Подменяет извлечение: возвращает эталон; помнит порядок вызовов."""

    def __init__(self, cfg: MatrixConfig):
        self.truth = {}
        for cell in build_cells(cfg):
            for _, item in cell_items(cell, plan_docs(cfg), []):
                self.truth[tuple(item.inputs)] = item.truth
        self.calls: list[tuple[str, str]] = []

    def __call__(self, paths, mode, config, cross_check):
        self.calls.append((config.model, Path(paths[0]).name))
        return fake_raw(self.truth[tuple(paths)])


def test_plan_docs_is_stratified_in_every_prefix(tmp_path):
    cfg = matrix_config(tmp_path)
    plan = plan_docs(cfg)
    assert [p.order for p in plan] == list(range(18))
    assert len({p.doc_id for p in plan}) == 18
    assert plan == plan_docs(cfg)  # воспроизводимо
    # в каждом начале из 6 документов — по два каждого типа, по два на уровень
    for start in range(0, 18, 6):
        chunk = plan[start : start + 6]
        assert set(Counter(p.doc_type for p in chunk).values()) == {2}
        assert set(Counter(p.level for p in chunk).values()) == {2}
    for start in range(0, 18, 3):  # блоки по три: каждый уровень по разу
        assert {p.level for p in plan[start : start + 3]} == {"clean", "scan", "photo"}


def test_cell_items_follow_path_levels(tmp_path):
    cfg = matrix_config(tmp_path)
    cells = {c.id: c for c in build_cells(cfg)}
    plan = plan_docs(cfg)
    text = cell_items(cells["a__text"], plan, [])
    assert {it.level for _, it in text} == {"clean"}
    assert all(it.inputs[0].suffix == ".pdf" for _, it in text)
    vision = cell_items(cells["a__vision"], plan, [])
    assert Counter(it.level for _, it in vision) == {"clean": 6, "scan": 6, "photo": 6}
    assert all(it.inputs[0].suffix == ".png" for _, it in vision)  # и clean — картинкой
    # у text и vision на clean одни и те же элементы, если документ раздан на clean
    clean_vision = {it.item_id for _, it in vision if it.level == "clean"}
    assert clean_vision <= {it.item_id for _, it in text}
    assert real_items(cfg) == []


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="только clean"):
        matrix_config(
            tmp_path,
            paths={
                "text": {"mode": "text", "levels": ["clean", "scan"]},
                "vision": {"mode": "vision", "levels": ["clean"]},
            },
        )
    with pytest.raises(ValueError, match="делиться"):
        matrix_config(tmp_path, batch_docs=4)
    with pytest.raises(ValueError, match="неизвестный путь"):
        matrix_config(tmp_path, models=[{"id": "a", "tag": "t", "paths": ["nope"]}])


def test_vram_gate(tmp_path):
    cfg = matrix_config(tmp_path)
    model_b = cfg.models[1]
    assert "не проверена" in skip_reason(cfg, model_b)
    assert [c.id for c in active_cells(cfg)] == ["a__text", "a__vision"]

    gib = 2**30
    asked = []
    result = check_vram(
        cfg,
        chat_fn=lambda econfig, prompt: asked.append((econfig.model, econfig.options.num_ctx)),
        placement_fn=lambda url, tag: {"size": 4 * gib, "size_vram": 3 * gib},
        peak_fn=lambda: 3900,
    )
    assert asked == [("tag-b", 8192)]
    assert result["b"]["fits"] is False and result["b"]["vram_share"] == 0.75
    assert "75%" in skip_reason(cfg, model_b) and "8192" in skip_reason(cfg, model_b)
    assert [c.id for c in active_cells(cfg)] == ["a__text", "a__vision"]

    check_vram(
        cfg,
        chat_fn=lambda *a: None,
        placement_fn=lambda url, tag: {"size": 3 * gib, "size_vram": 3 * gib},
        peak_fn=lambda: 3500,
    )
    assert skip_reason(cfg, model_b) is None
    assert "b__vision" in [c.id for c in active_cells(cfg)]
    with pytest.raises(ValueError, match="нет ячеек"):
        active_cells(cfg, ["zzz"])


def test_static_skip_reason(tmp_path):
    cfg = matrix_config(tmp_path)
    cfg.models[0].skip_reason = "нет модели"
    assert skip_reason(cfg, cfg.models[0]) == "нет модели"
    rows = progress_table(cfg)
    assert rows[0]["skipped"] == "нет модели"


def test_run_in_batches_and_resume(tmp_path):
    cfg = matrix_config(tmp_path)
    fake = Fake(cfg)
    kw = {"extract_fn": fake, "models_fn": lambda url: set()}
    fresh = run_matrix(cfg, docs=6, **kw)
    assert fresh == {"a__text": 6, "a__vision": 6}
    assert len(fake.calls) == 12
    # партия идёт по ячейкам подряд: сначала 6 документов text, затем 6 vision
    assert {m for m, _ in fake.calls} == {"tag-a"}
    assert all(name.endswith(".pdf") for _, name in fake.calls[:6])
    assert all(name.endswith(".png") for _, name in fake.calls[6:])

    again = run_matrix(cfg, docs=6, **kw)  # продолжение: ничего нового
    assert again == {"a__text": 0, "a__vision": 0}
    assert len(fake.calls) == 12
    rest = run_matrix(cfg, **kw)  # остальные 12 документов
    assert rest == {"a__text": 12, "a__vision": 12}
    only = run_matrix(cfg, only=["a__vision"], **kw)
    assert only == {"a__vision": 0}

    table = [r for r in progress_table(cfg) if "skipped" not in r]
    assert {r["cell"]: r["done"] for r in table} == {"a__text": 18, "a__vision": 18}
    assert all(r["measured"] for r in table)


def test_run_reports_missing_models(tmp_path):
    cfg = matrix_config(tmp_path)
    with pytest.raises(RuntimeError, match="tag-a"):
        run_matrix(cfg, models_fn=lambda url: {"other"})


def test_aggregate_uses_common_prefix_and_writes_report(tmp_path):
    cfg = matrix_config(tmp_path)
    fake = Fake(cfg)
    kw = {"extract_fn": fake, "models_fn": lambda url: set()}
    run_matrix(cfg, docs=12, **kw)
    run_matrix(cfg, docs=18, only=["a__text"], **kw)  # text ушла вперёд, vision — нет
    data = aggregate(cfg)
    assert data["documents"] == 12 and data["ready_in_all_cells"] == 12
    assert {r["cell"] for r in data["cells"]} == {"a__text", "a__vision"}
    for row in data["cells"]:
        assert row["overall"]["doc_ok"]["value"] == 1
        assert row["overall"]["cells_f1"]["value"] == 1
    levels = next(r for r in data["cells"] if r["cell"] == "a__vision")["levels"]
    assert {v["documents"] for v in levels.values()} == {4}
    assert next(r for r in data["cells"] if r["cell"] == "a__text")["levels"].keys() == {"clean"}
    assert data["skipped"][0]["model"] == "b"
    assert data["real"] == []
    assert aggregate(cfg, docs=6)["documents"] == 6

    out = write_matrix_report(cfg, data)
    assert out == tmp_path / "reports" / "2026-11-18_m"
    report = (out / "report.md").read_text(encoding="utf-8")
    for needle in ("Сводная таблица", "По уровням входа", "Пропущено", "не проверена", "Реальные"):
        assert needle in report
    assert "размеченных реальных" in report.lower()
    for svg in ("quality_vs_speed.svg", "quality_by_level.svg"):
        ElementTree.fromstring((out / svg).read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_quality"] in {"a__text", "a__vision"}
    assert yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))["name"] == "m"


def test_aggregate_without_data_fails(tmp_path):
    with pytest.raises(ValueError, match="нет документов"):
        aggregate(matrix_config(tmp_path))


def est(v, lo=None, hi=None):
    return {"value": v, "low": lo, "high": hi}


def test_charts_are_valid_xml():
    both = {"cells_f1": est(0.7, 0.6, 0.8), "doc_ok": est(0.1, 0.05, 0.2)}
    points = [
        {"model": "m1", "path": p, "x": x, "y": both}
        for p, x in (("text", 20.0), ("vision", 60.0), ("ocr", 300.0))
    ] + [{"model": "m2", "path": "text", "x": None, "y": {}}]
    metrics = (("cells_f1", "Ячейки"), ("doc_ok", "Без правок"))
    ElementTree.fromstring(scatter_svg(points, metrics, "t & <u>"))
    rows = [
        {"label": "a / text", "levels": {"clean": {"cells_f1": est(0.9, 0.8, 0.95)}}},
        {"label": "b / vision", "levels": {lv: {"cells_f1": est(0.5)} for lv in ("clean", "scan")}},
    ]
    ElementTree.fromstring(level_bars_svg(rows, metrics, "уровни"))


def test_repo_matrix_config_and_cli_plan():
    path = ROOT / "configs/eval/p12_matrix.yaml"
    cfg = load_matrix_config(path)
    assert [m.tag for m in cfg.models] == [
        "qwen3.5:0.8b",
        "qwen3.5:2b-q4_K_M",
        "qwen3.5:2b-q8_0",
        "qwen3.5:4b-q4_K_M",
    ]
    four = cfg.models[3]
    assert four.paths == ["text"] and four.require_full_vram
    assert cfg.data.n_docs == 300 and cfg.extract_config == Path("configs/extract_d48.yaml")
    res = CliRunner().invoke(app, ["eval-matrix", "plan", "-c", str(path)])
    assert res.exit_code == 0, res.output
