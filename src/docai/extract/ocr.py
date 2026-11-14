"""OCR-путь для сравнения: Tesseract на CPU (вызов программы, без Python-обёрток)."""

import shutil
import subprocess
from functools import cache
from pathlib import Path

from PIL import Image

from docai.extract.images import to_png_bytes

# Стандартный путь установщика UB Mannheim (сборка Tesseract для Windows).
WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

INSTALL_HINT = (
    "Tesseract не найден. Установка (PowerShell):\n"
    "  winget install --id UB-Mannheim.TesseractOCR -e\n"
    "Затем перезапустите терминал или укажите путь в конфиге: tesseract_cmd."
)

LANG_HINT = (
    "В Tesseract нет языка «{lang}». Скачайте rus.traineddata (PowerShell от администратора):\n"
    '  Invoke-WebRequest "https://github.com/tesseract-ocr/tessdata_fast/raw/main/rus.traineddata"'
    ' -OutFile "{tessdata}\\rus.traineddata"'
)


class OcrError(RuntimeError):
    """Tesseract не установлен, нет языка или распознавание упало."""


def find_tesseract(configured: str | None = None) -> Path:
    """Путь к tesseract.exe: из конфига, из PATH или стандартный путь установщика."""
    if configured:
        path = Path(configured)
        if path.is_file():
            return path
        raise OcrError(f"tesseract_cmd указывает на несуществующий файл: {configured}")
    found = shutil.which("tesseract")
    if found:
        return Path(found)
    if WINDOWS_DEFAULT.is_file():
        return WINDOWS_DEFAULT
    raise OcrError(INSTALL_HINT)


@cache
def _languages(cmd: Path) -> frozenset[str]:
    out = subprocess.run(
        [str(cmd), "--list-langs"], capture_output=True, check=True, timeout=60
    ).stdout.decode("utf-8", errors="replace")
    # Первая строка — «List of available languages in "...":».
    return frozenset(line.strip() for line in out.splitlines()[1:] if line.strip())


def tesseract_version(cmd: Path) -> str:
    out = subprocess.run([str(cmd), "--version"], capture_output=True, check=True, timeout=60)
    text = (out.stdout or out.stderr).decode("utf-8", errors="replace")
    return text.splitlines()[0].strip() if text else "?"


def check_languages(cmd: Path, lang: str) -> None:
    available = _languages(cmd)
    for code in lang.split("+"):
        if code not in available:
            raise OcrError(LANG_HINT.format(lang=code, tessdata=cmd.parent / "tessdata"))


def ocr_image(cmd: Path, image: Image.Image, lang: str, psm: int) -> str:
    """Распознать одну страницу. Картинка передаётся через stdin, текст читается из stdout."""
    result = subprocess.run(
        [str(cmd), "stdin", "stdout", "-l", lang, "--psm", str(psm)],
        input=to_png_bytes(image),
        capture_output=True,
        timeout=600,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise OcrError(f"Tesseract завершился с кодом {result.returncode}: {error}")
    return result.stdout.decode("utf-8", errors="replace").strip()
