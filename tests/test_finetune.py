"""Шаг P21: SFT-датасет для дообучения, правки /feedback, ноутбук и Modelfile."""

import datetime as dt
import json
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from docai.extract.draft import (
    Draft,
    DraftBank,
    DraftBasis,
    DraftLine,
    DraftParty,
    DraftTotals,
    parse_draft,
)
from docai.extract.pipeline import _page_info
from docai.extract.prompts import render_prompt
from docai.finetune.build import SftBuildError, build_sft_dataset
from docai.finetune.check import pick_items
from docai.finetune.config import FeedbackOptions, Mix, SftConfig, load_config
from docai.finetune.feedback import collect_feedback, corrected_values, nest, printed_view
from docai.finetune.targets import (
    BANK_KEYS,
    BASIS_KEYS,
    LINE_KEYS,
    PARTY_KEYS,
    TOTALS_KEYS,
    draft_target,
    to_json,
    visible_fields,
)
from docai.service.store import Store
from docai.synth.generator import generate_document

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- эталон ответа


def _ref(index: int = 1, doc_type: str | None = None) -> dict[str, Any]:
    """Эталонный JSON документа генератора P3 (как в json/ датасета)."""
    i = index
    while True:
        doc = generate_document(42, i, error_rate=0.0).document
        if doc_type is None or doc.doc_type.value == doc_type:
            return json.loads(doc.model_dump_json(exclude_none=False))
        i += 1


def _paths(node: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out += _paths(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out += _paths(value, f"{prefix}[{i}]")
    elif node is not None and prefix != "doc_type":
        out.append(prefix)
    return out


def _printed(ref: dict[str, Any], path: str) -> str:
    node: Any = ref
    for part in path.replace("]", "").replace("[", ".").split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    if path in ("date", "basis.date"):
        return dt.date.fromisoformat(node).strftime("%d.%m.%Y")
    return str(node)


def _layout(ref: dict[str, Any], skip: tuple[str, ...] = (), pages: int = 1) -> list[dict]:
    """Разметка, где напечатано всё, кроме `skip`. При pages=2 строки с первой и итоги —
    на второй странице, шапка — на первой."""
    fields = []
    for path in _paths(ref):
        if any(path.endswith(s) for s in skip):
            continue
        page = 1
        if pages == 2 and (
            path.startswith("totals")
            or (path.startswith("lines[") and not path.startswith("lines[0]"))
        ):
            page = 2
        fields.append(
            {"field": path, "page": page, "text": _printed(ref, path), "bbox": [0, 0, 1, 1]}
        )
    return fields


def test_target_keys_follow_draft_schema() -> None:
    """Порядок ключей эталона = порядок полей черновика: грамматика Ollama идёт по схеме."""
    assert tuple(DraftParty.model_fields) == PARTY_KEYS
    assert tuple(DraftBank.model_fields) == BANK_KEYS
    assert tuple(DraftBasis.model_fields) == BASIS_KEYS
    assert tuple(DraftLine.model_fields) == LINE_KEYS
    assert tuple(DraftTotals.model_fields) == TOTALS_KEYS
    ref = _ref(doc_type="bill")
    target = draft_target(ref, visible_fields(_layout(ref)))
    assert tuple(target) == tuple(Draft.model_fields)


def test_target_takes_only_printed_fields() -> None:
    ref = _ref(doc_type="bill")
    assert ref["lines"][0]["unit_code"]  # в эталоне код ОКЕИ есть…
    layout = _layout(ref, skip=("unit_code", "seller.kpp"))  # …но в счёте он не печатается
    target = draft_target(ref, visible_fields(layout))
    assert all(line["unit_code"] is None for line in target["lines"])
    assert target["seller"]["kpp"] is None
    assert target["seller"]["inn"] == ref["seller"]["inn"]
    assert len(target["lines"]) == len(ref["lines"])


def test_target_dates_as_printed_numbers_as_in_reference() -> None:
    ref = _ref(doc_type="bill")
    text = to_json(draft_target(ref, visible_fields(_layout(ref))))
    assert f'"date":"{_printed(ref, "date")}"' in text
    price = ref["lines"][0]["price"]
    assert f'"price":{price},' in text  # число без кавычек, разрядность как в эталоне
    assert "\\u04" not in text  # кириллица не экранирована
    draft, notes = parse_draft(json.loads(text, parse_float=Decimal), "printed")
    assert notes == []
    assert draft.date == dt.date.fromisoformat(ref["date"])
    assert draft.lines[0].price == Decimal(price)


def test_target_page_split() -> None:
    """Двухстраничный документ: на второй странице — строки 2…n и итоги, шапки нет."""
    ref = _ref(doc_type="upd")
    while len(ref["lines"]) < 2:
        ref = _ref(index=int(ref["number"][-2:] or 1) + 7, doc_type="upd")
    layout = _layout(ref, pages=2)
    first = draft_target(ref, visible_fields(layout, page=1))
    second = draft_target(ref, visible_fields(layout, page=2))
    assert len(first["lines"]) == 1 and first["totals"]["amount_with_vat"] is None
    assert len(second["lines"]) == len(ref["lines"]) - 1
    assert second["number"] is None and second["seller"]["inn"] is None
    assert second["doc_type"] == "upd"  # тип обязателен в черновике на каждой странице
    assert second["totals"]["amount_with_vat"] == Decimal(ref["totals"]["amount_with_vat"])
    whole = draft_target(ref, visible_fields(layout))
    assert len(whole["lines"]) == len(ref["lines"])


def test_to_json_decimal_without_exponent() -> None:
    assert to_json({"a": Decimal("100"), "b": Decimal("12200.00"), "c": None}) == (
        '{"a":100,"b":12200.00,"c":null}'
    )
    with pytest.raises(TypeError):
        to_json({"a": object()})


# ---------------------------------------------------------------- маленький датасет P6


def _image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _fake_dataset(root: Path) -> Path:
    """Мини-копия датасета P6: train 8 документов (один двухстраничный), val 6, test 2."""
    rows = []
    plan = [("train", 8), ("val", 6), ("test", 2)]
    index = 1
    for split, n in plan:
        for k in range(n):
            ref = _ref(index)
            index = index + 1
            doc_id = f"{index:05d}_{ref['doc_type']}"
            pages = 2 if (split == "train" and k == 0 and len(ref["lines"]) > 1) else 1
            (root / "json").mkdir(parents=True, exist_ok=True)
            (root / "pdf").mkdir(parents=True, exist_ok=True)
            (root / "json" / f"{doc_id}.json").write_text(
                json.dumps(ref, ensure_ascii=False), encoding="utf-8"
            )
            layout = _layout(ref, pages=pages)
            (root / "pdf" / f"{doc_id}.layout.json").write_text(
                json.dumps({"fields": layout}, ensure_ascii=False), encoding="utf-8"
            )
            (root / "pdf" / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 fake " + doc_id.encode())
            for level in ("clean", "scan", "photo"):
                images, layouts = [], []
                for page in range(1, pages + 1):
                    ext = "png" if level == "clean" else "jpg"
                    name = f"images/{doc_id}_p{page}_{level}_200dpi.{ext}"
                    _image(root / name, (1653, 2339) if level != "photo" else (2339, 1653))
                    lay_name = name.rsplit(".", 1)[0] + ".layout.json"
                    page_fields = [
                        {k: v for k, v in f.items() if k != "page"}
                        for f in layout
                        if f["page"] == page
                    ]
                    (root / lay_name).write_text(
                        json.dumps({"fields": page_fields}, ensure_ascii=False), encoding="utf-8"
                    )
                    images.append(name)
                    layouts.append(lay_name)
                rows.append(
                    {
                        "id": f"{doc_id}_{level}_200dpi",
                        "doc_id": doc_id,
                        "doc_type": ref["doc_type"],
                        "split": split,
                        "level": level,
                        "dpi": 200,
                        "pages": pages,
                        "json": f"json/{doc_id}.json",
                        "pdf": f"pdf/{doc_id}.pdf",
                        "pdf_layout": f"pdf/{doc_id}.layout.json",
                        "images": images,
                        "image_layouts": layouts,
                        "has_errors": False,
                        "error_kinds": [],
                    }
                )
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return root


def _fake_pdf_text(path: Path, tables: str) -> list[str]:
    assert tables == "grid"
    return [f"Текст документа {path.stem}\nстрока таблицы | 1 | 2 |"]


def _config(tmp_path: Path, dataset: Path, **kw: Any) -> SftConfig:
    defaults: dict[str, Any] = {
        "dataset": dataset,
        "out": tmp_path / "out" / "sft_test",
        "n_train": 6,
        "n_val": 3,
        "feedback": FeedbackOptions(enabled=False),
        "train_config": ROOT / "configs" / "finetune" / "train_v1.yaml",
    }
    defaults.update(kw)
    return SftConfig(**defaults)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _fake_dataset(tmp_path_factory.mktemp("p6"))


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_dataset_format_and_mix(tmp_path: Path, dataset: Path) -> None:
    config = _config(tmp_path, dataset)
    result = build_sft_dataset(config, pdf_text=_fake_pdf_text, workers=2)
    out = result.out
    train = _records(out / "train.jsonl")
    val = _records(out / "val.jsonl")
    assert len(train) >= 6 and len(val) >= 3
    test_docs = {
        r["doc_id"]
        for r in map(
            json.loads, (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if r["split"] == "test"
    }
    assert not test_docs & {r["meta"]["doc_id"] for r in train + val}
    assert {r["meta"]["split"] for r in train} == {"train"}
    kinds = {r["meta"]["kind"] for r in train}
    assert kinds == {"text", "clean", "scan", "photo"}
    for record in train + val:
        user, assistant = record["messages"]
        assert user["role"] == "user" and assistant["role"] == "assistant"
        content = user["content"]
        assert content[-1]["type"] == "text"
        if record["meta"]["kind"] == "text":
            assert len(content) == 1
            assert content[0]["text"].startswith("Ты извлекаешь реквизиты")
        else:
            # картинка перед текстом — так же картинки ставит рендерер Ollama для qwen3.5
            assert content[0]["type"] == "image"
            image = Image.open(out / content[0]["image"])
            assert max(image.size) == 1280  # как `max_image_side` конвейера
            assert list(image.size) == record["meta"]["image_size"]
            index, total = map(int, record["meta"]["part"].split()[1].split("/"))
            assert content[1]["text"] == render_prompt(
                "vision_v2", page_info=_page_info(index, total)
            )
        target = assistant["content"][0]["text"]
        draft, notes = parse_draft(json.loads(target, parse_float=Decimal), "printed")
        assert notes == [] and draft.doc_type.value == record["meta"]["doc_type"]
    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["counts"]["train"]["examples"] == len(train)
    assert yaml.safe_load((out / "train_config.yaml").read_text(encoding="utf-8"))["export"]["name"]


def test_build_two_page_document_gives_two_examples(tmp_path: Path, dataset: Path) -> None:
    config = _config(tmp_path, dataset, n_train=40, mix=Mix(text=0, clean=1, scan=0, photo=0))
    build_sft_dataset(config, pdf_text=_fake_pdf_text, workers=2)
    train = _records(config.out / "train.jsonl")
    two = [r for r in train if r["meta"]["part"].endswith("/2")]
    assert [r["meta"]["part"] for r in two] == ["page 1/2", "page 2/2"]
    assert "страница 2 из 2" in two[1]["messages"][0]["content"][1]["text"]
    second = json.loads(two[1]["messages"][1]["content"][0]["text"])
    assert second["number"] is None and second["lines"]


def test_build_is_deterministic_and_archived(tmp_path: Path, dataset: Path) -> None:
    a = _config(tmp_path / "a", dataset)
    b = _config(tmp_path / "b", dataset)
    build_sft_dataset(a, pdf_text=_fake_pdf_text, workers=2)
    result = build_sft_dataset(b, pdf_text=_fake_pdf_text, workers=2)
    for name in ("train.jsonl", "val.jsonl"):
        assert (a.out / name).read_bytes() == (b.out / name).read_bytes()
    assert result.archive is not None
    with zipfile.ZipFile(result.archive) as zf:
        names = set(zf.namelist())
    assert {"sft_test/train.jsonl", "sft_test/val.jsonl", "sft_test/train_config.yaml"} <= names
    assert any(n.startswith("sft_test/images/train/") for n in names)
    with pytest.raises(SftBuildError, match="не пуста"):
        build_sft_dataset(a, pdf_text=_fake_pdf_text)
    build_sft_dataset(a, pdf_text=_fake_pdf_text, force=True, workers=2)


def test_text_prompt_matches_pipeline(tmp_path: Path, dataset: Path) -> None:
    config = _config(tmp_path, dataset, mix=Mix(text=1, clean=0, scan=0, photo=0))
    build_sft_dataset(config, pdf_text=_fake_pdf_text, workers=2)
    record = _records(config.out / "train.jsonl")[0]
    text = _fake_pdf_text(dataset / "pdf" / f"{record['meta']['doc_id']}.pdf", "grid")[0]
    expected = render_prompt(
        "text_v3", source="текстовый слой PDF", source_title="Текст документа", document_text=text
    )
    assert record["messages"][0]["content"][0]["text"] == expected


def test_text_layer_too_short_is_skipped(tmp_path: Path, dataset: Path) -> None:
    config = _config(tmp_path, dataset, mix=Mix(text=1, clean=0, scan=0, photo=0))
    build_sft_dataset(config, pdf_text=lambda path, tables: ["мало"], workers=2)
    assert _records(config.out / "train.jsonl") == []


def test_config_files_are_valid() -> None:
    config = load_config(ROOT / "configs" / "finetune" / "sft_v1.yaml")
    assert config.vision_prompt == "vision_v2" and config.text_prompt == "text_v3"
    assert config.feedback.include_unknown is False
    extract = yaml.safe_load((ROOT / "configs" / "extract.yaml").read_text(encoding="utf-8"))
    # обучение видит то же, что конвейер
    assert config.vision_prompt == extract["vision_prompt"]
    assert config.text_prompt == extract["text_prompt"]
    assert config.max_image_side == extract["max_image_side"]
    assert config.max_text_chars == extract["max_text_chars"]
    assert config.text_tables == extract["text_tables"]
    train = yaml.safe_load(config.train_config.read_text(encoding="utf-8"))
    assert train["model"]["load_in_4bit"] is False
    assert train["export"]["name"] == "docai-qwen35-2b-ft"
    with pytest.raises(ValueError, match="давать 1"):
        Mix(text=0.5, clean=0.5, scan=0.5, photo=0)


# ---------------------------------------------------------------- правки /feedback


def _service_db(data_dir: Path, jobs: list[tuple[str, Path, str, dict[str, Any]]]) -> None:
    """База сервиса P14 с заданиями и правками (схема — из `docai.service.store`)."""
    store = Store(data_dir / "service.db")
    store.close()
    import sqlite3

    db = sqlite3.connect(data_dir / "service.db")
    for job_id, file, kind, result_doc in jobs:
        upload = data_dir / "uploads" / job_id / f"source{file.suffix}"
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(file.read_bytes())
        result = {"document": result_doc, "valid": True, "meta": {}}
        db.execute(
            "INSERT INTO jobs (id, status, filename, stored_name, source_kind, mode, size_bytes,"
            " created_at, result_json) VALUES (?, 'done', ?, ?, ?, 'auto', ?, '2026-11-20', ?)",
            (
                job_id,
                file.name,
                upload.name,
                kind,
                upload.stat().st_size,
                json.dumps(result, ensure_ascii=False),
            ),
        )
        corrections = [
            {"field": "number", "before": result_doc["number"], "after": "77-А", "changed": True}
        ]
        db.execute(
            "INSERT INTO feedback (job_id, created_at, changed, total, corrections_json)"
            " VALUES (?, '2026-11-20', 1, 1, ?)",
            (job_id, json.dumps(corrections, ensure_ascii=False)),
        )
    db.commit()
    db.close()


def test_feedback_classification_and_opt_in(tmp_path: Path, dataset: Path) -> None:
    data_dir = tmp_path / "service"
    data_dir.mkdir()
    synth_image = next((dataset / "images").glob("*_scan_200dpi.jpg"))
    unknown_image = tmp_path / "photo_from_phone.png"
    _image(unknown_image, (1000, 1400))
    xml = tmp_path / "edo.xml"
    xml.write_text("<Файл/>", encoding="utf-8")
    doc = json.loads(json.dumps(_ref(3)))
    _service_db(
        data_dir,
        [
            ("job_synth", synth_image, "image", doc),
            ("job_unknown", unknown_image, "image", doc),
            ("job_xml", xml, "xml", doc),
        ],
    )
    report = collect_feedback(data_dir, [dataset])
    status = {job.job_id: job.status for job in report.jobs}
    assert status == {"job_synth": "synthetic", "job_unknown": "unknown", "job_xml": "xml"}
    unknown = next(job for job in report.jobs if job.job_id == "job_unknown")
    assert unknown.values["number"] == "77-А"  # исправление человека заменило ответ модели

    # По умолчанию правки к файлам не из синтетики в датасет не идут.
    config = _config(
        tmp_path, dataset, feedback=FeedbackOptions(data_dir=data_dir, synthetic_dirs=[])
    )
    result = build_sft_dataset(config, pdf_text=_fake_pdf_text, workers=2)
    assert result.feedback["by_status"] == {"synthetic": 1, "unknown": 1, "xml": 1}
    assert not any(r["meta"]["source"] == "feedback" for r in _records(config.out / "train.jsonl"))

    # Только по решению владельца.
    config = _config(
        tmp_path / "opt_in",
        dataset,
        feedback=FeedbackOptions(data_dir=data_dir, synthetic_dirs=[], include_unknown=True),
    )
    build_sft_dataset(config, pdf_text=_fake_pdf_text, workers=2)
    fb = [r for r in _records(config.out / "train.jsonl") if r["meta"]["source"] == "feedback"]
    assert len(fb) == 1
    target = json.loads(fb[0]["messages"][1]["content"][0]["text"])
    assert target["number"] == "77-А"
    assert target["date"] == dt.date.fromisoformat(doc["date"]).strftime("%d.%m.%Y")
    assert max(Image.open(config.out / fb[0]["messages"][0]["content"][0]["image"]).size) == 1280


def test_feedback_helpers() -> None:
    assert nest({"seller.inn": "1", "lines[1].price": "2", "number": None}) == {
        "seller": {"inn": "1"},
        "lines": [{}, {"price": "2"}],
        "number": None,
    }
    ref, visible = printed_view(
        {"date": "2026-11-14", "doc_type": "bill", "number": None, "seller.inn": "7707083893"}
    )
    assert visible == {"date": "14.11.2026", "seller.inn": "7707083893"}
    assert ref["doc_type"] == "bill"
    values = corrected_values(
        None, [{"field": "number", "after": " 5 "}, {"field": "date", "after": ""}]
    )
    assert values == {"number": "5", "date": None}


def test_feedback_without_service_db(tmp_path: Path) -> None:
    assert collect_feedback(tmp_path / "nothing", []).jobs == []


# ---------------------------------------------------------------- проверка модели, ноутбук


def test_check_picks_val_documents(dataset: Path) -> None:
    items = pick_items(dataset)
    assert [kind for kind, _ in items] == ["text", "text", "clean", "scan", "photo"]
    manifest = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    val_docs = {r["doc_id"] for r in manifest if r["split"] == "val"}
    for kind, path in items:
        assert path.stem.split("_p1_")[0] in val_docs
        assert path.suffix == (".pdf" if kind == "text" else path.suffix)


def test_notebook_is_valid_and_uses_train_config() -> None:
    path = ROOT / "notebooks" / "finetune_qwen35_2b.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for needle in (
        "train_config.yaml",
        "load_in_4bit",
        "use_gradient_checkpointing",
        "UnslothVisionDataCollator",
        'resize="max"',  # без уменьшения картинок: как в Ollama (1280 px → 1120 токенов)
        "train_on_responses_only",
        "resume_from_checkpoint",
        "convert_hf_to_gguf.py",
        "--mmproj",
    ):
        assert needle in code, needle
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), "ноутбук хранится без вывода"


def test_modelfile_matches_export_names() -> None:
    modelfile = (ROOT / "notebooks" / "Modelfile.docai-qwen35-2b-ft").read_text(encoding="utf-8")
    name = yaml.safe_load(
        (ROOT / "configs" / "finetune" / "train_v1.yaml").read_text(encoding="utf-8")
    )["export"]["name"]
    froms = [
        line.split(maxsplit=1)[1] for line in modelfile.splitlines() if line.startswith("FROM ")
    ]
    assert froms == [f"./{name}.Q8_0.gguf", f"./{name}.mmproj-f16.gguf"]
    assert "RENDERER qwen3.5" in modelfile and "PARSER qwen3.5" in modelfile
    assert "PARAMETER presence_penalty 0" in modelfile


def test_extract_ft_config_differs_only_by_model() -> None:
    base = yaml.safe_load((ROOT / "configs" / "extract.yaml").read_text(encoding="utf-8"))
    ft = yaml.safe_load((ROOT / "configs" / "extract_ft.yaml").read_text(encoding="utf-8"))
    assert ft.pop("model") == "docai-qwen35-2b-ft"
    base.pop("model")
    assert ft == base
