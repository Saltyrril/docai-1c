"""P0: модель читает русский текст с картинки и возвращает JSON. Замеряет время и пиковую VRAM.

Запуск (нужны Ollama, nvidia-smi и Pillow):
    uv run --with pillow python reports/2026-11-14_p0_env_check/vision_test.py \
        qwen3.5:2b-q8_0 qwen3.5:2b-q4_K_M
"""

import base64
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PNG = HERE / "test_invoice.png"
TEXT = "ИНН 7707083893, сумма 12 200,00 руб."
PROMPT = (
    "Прочитай текст на изображении и верни только JSON вида "
    '{"inn": "...", "sum": "..."} без пояснений.'
)
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"


def make_png() -> None:
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 40)
    img = Image.new("RGB", (900, 120), "white")
    ImageDraw.Draw(img).text((20, 35), TEXT, fill="black", font=font)
    img.save(PNG)


def vram_used() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0])


def run(model: str) -> tuple[str, float, float, int]:
    peak = [0]
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], vram_used())
            time.sleep(0.1)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    image_b64 = base64.b64encode(PNG.read_bytes()).decode()
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": 0,
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
    }
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp_raw:
        resp = json.load(resp_raw)
    dt = time.time() - t0
    stop.set()
    thread.join()
    return resp["message"]["content"], dt, resp.get("load_duration", 0) / 1e9, peak[0]


if __name__ == "__main__":
    make_png()
    print(f"baseline VRAM: {vram_used()} MiB")
    for m in sys.argv[1:]:
        for i in (1, 2):  # оба запуска «холодные»: keep_alive=0 выгружает модель
            ans, dt, load, peak = run(m)
            record = {
                "model": m,
                "run": i,
                "answer": ans,
                "total_s": round(dt, 2),
                "load_s": round(load, 2),
                "peak_vram_MiB": peak,
                "peak_vram_GB": round(peak / 1024, 2),
            }
            print(json.dumps(record, ensure_ascii=False))
