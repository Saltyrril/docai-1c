"""Тесты сборки датасета (P6)."""

import json
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from docai.cli import app
from docai.dataset.build import BuildError, build_dataset
from docai.dataset.card import compute_stats, render_card
from docai.dataset.config import DatasetConfig, load_config
from docai.dataset.split import split_by_seller

SHARES = {"train": 0.8, "val": 0.1, "test": 0.1}


def _sellers() -> list[str]:
    # 100 продавцов, у каждого от 1 до 12 документов.
    return [f"inn{i:03d}" for i in range(100) for _ in range(1 + i % 12)]


def test_split_one_seller_one_part() -> None:
    sellers = _sellers()
    assignment = split_by_seller(sellers, SHARES, seed=1)
    assert set(assignment) == set(sellers)
    assert set(assignment.values()) == {"train", "val", "test"}


def test_split_shares_close_to_target() -> None:
    sellers = _sellers()
    assignment = split_by_seller(sellers, SHARES, seed=1)
    for part, share in SHARES.items():
        got = sum(1 for s in sellers if assignment[s] == part) / len(sellers)
        assert abs(got - share) < 0.05, part


def test_split_deterministic_and_seed_dependent() -> None:
    sellers = _sellers()
    first = split_by_seller(sellers, SHARES, seed=1)
    assert first == split_by_seller(list(reversed(sellers)), SHARES, seed=1)
    assert first != split_by_seller(sellers, SHARES, seed=2)


def test_split_zero_share_part_stays_empty() -> None:
    shares = {"train": 0.9, "val": 0.0, "test": 0.1}
    assignment = split_by_seller(_sellers(), shares, seed=1)
    assert "val" not in set(assignment.values())


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(split={"train": 0.5, "val": 0.1, "test": 0.1})
    with pytest.raises(ValidationError):
        DatasetConfig(dpi=[123])
    with pytest.raises(ValidationError):
        DatasetConfig(unknown_field=1)
    path = tmp_path / "c.yaml"
    path.write_text("seed: 7\nn_documents: 5\nlevels: [clean]\n", encoding="utf-8")
    cfg = load_config(path)
    assert (cfg.seed, cfg.n_documents) == (7, 5)


def test_shipped_config_is_v1() -> None:
    cfg = load_config(Path("configs/dataset_v1.yaml"))
    assert cfg.n_documents == 3000
    assert [level.value for level in cfg.levels] == ["clean", "scan", "photo"]


def _small(tmp_path: Path, name: str) -> DatasetConfig:
    return DatasetConfig(
        seed=11,
        n_documents=24,
        out=tmp_path / name,
        levels=["clean", "scan"],
        dpi=[150],
        workers=2,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    build_dataset(_small(root, "a"))
    return root


def test_manifest_rows_and_files(built: Path) -> None:
    out = built / "a"
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 24 * 2
    required = {"id", "doc_type", "json", "pdf", "images", "level", "split", "seed"}
    for row in rows:
        assert required <= set(row)
        assert row["seed"] == 11
        assert row["pages"] == len(row["images"]) == len(row["image_layouts"])
        for rel in [row["json"], row["pdf"], row["pdf_layout"], *row["images"]]:
            assert (out / rel).is_file(), rel
    assert len({row["id"] for row in rows}) == len(rows)


def test_no_seller_in_two_splits(built: Path) -> None:
    rows = [
        json.loads(line)
        for line in (built / "a" / "manifest.jsonl").read_text("utf-8").splitlines()
    ]
    parts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        parts[row["seller_inn"]].add(row["split"])
    assert all(len(v) == 1 for v in parts.values())
    # Все уровни одного документа лежат в одной части.
    doc_parts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc_parts[row["doc_id"]].add(row["split"])
    assert all(len(v) == 1 for v in doc_parts.values())


def test_rebuild_same_seed_identical_manifest(built: Path) -> None:
    build_dataset(_small(built, "b"))
    for name in ("manifest.jsonl", "build_info.json"):
        assert (built / "a" / name).read_bytes() == (built / "b" / name).read_bytes(), name
    for path in sorted((built / "a" / "json").glob("0*.json")):
        assert path.read_bytes() == (built / "b" / "json" / path.name).read_bytes()


def test_nonempty_out_requires_force(built: Path) -> None:
    with pytest.raises(BuildError):
        build_dataset(_small(built, "a"))


def test_stats_and_card(built: Path) -> None:
    stats = compute_stats(built / "a")
    assert stats["documents"] == 24
    assert sum(stats["split_docs"].values()) == 24
    card = render_card(stats)
    assert "## Разбиение train / val / test" in card
    assert "Ставка" in card


def test_cli_build_and_card(tmp_path: Path) -> None:
    runner = CliRunner()
    config = tmp_path / "c.yaml"
    config.write_text("levels: [clean]\ndpi: [150]\nworkers: 2\n", encoding="utf-8")
    out = tmp_path / "ds"
    result = runner.invoke(
        app, ["dataset", "build", "--config", str(config), "--out", str(out), "--n", "6"]
    )
    assert result.exit_code == 0, result.output
    again = runner.invoke(
        app, ["dataset", "build", "--config", str(config), "--out", str(out), "--n", "6"]
    )
    assert again.exit_code == 1
    card = tmp_path / "DATASET.md"
    done = runner.invoke(app, ["dataset", "card", "--dataset", str(out), "--out", str(card)])
    assert done.exit_code == 0, done.output
    assert card.read_text("utf-8").startswith("# Карточка датасета")
