"""Заглушка: проверяет, что пакет и все подмодули импортируются, а CLI запускается."""

import importlib

import pytest
from typer.testing import CliRunner

import docai
from docai.cli import app

SUBPACKAGES = [
    "schemas",
    "validation",
    "synth",
    "render",
    "edo",
    "labeling",
    "extract",
    "rag",
    "service",
    "eval",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"docai.{name}")
    assert module.__doc__, f"у docai.{name} должно быть описание"


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert docai.__version__ in result.output
