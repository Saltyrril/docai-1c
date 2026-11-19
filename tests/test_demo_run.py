"""Тесты вспомогательного скрипта демонстрации scripts/demo_run.py (P17)."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_run.py"


@pytest.fixture(scope="module")
def demo_run():
    spec = importlib.util.spec_from_file_location("demo_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_run"] = module
    spec.loader.exec_module(module)
    return module


def test_summary_counts_done_and_errors(demo_run, capsys) -> None:
    rows = [
        {
            "file": "a.xml",
            "status": "done",
            "route": "xml",
            "doc_type": "upd",
            "document_status": "upload",
            "fields_need_review": 0,
            "fields_total": 42,
            "total_s": 1.0,
        },
        {"file": "b.jpg", "status": "failed", "error": "Ollama недоступна", "total_s": 2.0},
        {
            "file": "c.pdf",
            "status": "done",
            "route": "text",
            "doc_type": "bill",
            "document_status": "review",
            "fields_need_review": 4,
            "fields_total": 39,
            "total_s": 9.0,
        },
    ]
    demo_run.print_summary(rows)
    out = capsys.readouterr().out
    assert "Обработано: 2 из 3" in out
    assert "ОШИБКА: Ollama недоступна" in out
    assert "review — 1, upload — 1" in out


def test_upload_to_dead_port_reports_error_not_exception(demo_run, tmp_path) -> None:
    doc = tmp_path / "x.pdf"
    doc.write_bytes(b"%PDF-1.4")
    row = demo_run.run_one("http://127.0.0.1:9", doc, timeout_s=5)
    assert row["status"] == "error"
    assert row["error"]
