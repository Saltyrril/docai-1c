"""Минимальный клиент HTTP API Ollama (`/api/chat`) на стандартной библиотеке."""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from docai.extract.config import ExtractConfig


class OllamaError(RuntimeError):
    """Ollama недоступна или вернула ошибку."""


@dataclass
class ChatResult:
    """Ответ модели и счётчики Ollama (длительности — в секундах)."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    done_reason: str | None
    total_s: float
    load_s: float
    prompt_eval_s: float
    eval_s: float
    thinking: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def parse_json(self) -> dict[str, Any]:
        """Ответ как JSON; числа — Decimal, чтобы не терять копейки."""
        return json.loads(self.content, parse_float=Decimal)


def build_chat_request(
    config: ExtractConfig,
    prompt: str,
    schema: dict[str, Any],
    images_b64: list[str] | None = None,
) -> dict[str, Any]:
    """Тело запроса `/api/chat`: схема в `format`, рассуждения выключены, сэмплинг фиксирован."""
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if images_b64:
        message["images"] = images_b64
    return {
        "model": config.model,
        "messages": [message],
        "stream": False,
        "format": schema,
        "think": config.think,
        "keep_alive": config.keep_alive,
        "options": config.options.model_dump(),
    }


def _seconds(ns: int | None) -> float:
    return round((ns or 0) / 1e9, 3)


def chat(config: ExtractConfig, body: dict[str, Any]) -> ChatResult:
    """Отправить запрос и дождаться полного ответа (без стриминга)."""
    url = config.ollama_url.rstrip("/") + "/api/chat"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama вернула {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise OllamaError(
            f"Ollama недоступна по адресу {config.ollama_url} ({exc}). Запустите: ollama serve"
        ) from exc
    if "error" in data:
        raise OllamaError(f"Ollama: {data['error']}")
    message = data.get("message") or {}
    return ChatResult(
        content=message.get("content", ""),
        thinking=message.get("thinking") or None,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        done_reason=data.get("done_reason"),
        total_s=_seconds(data.get("total_duration")),
        load_s=_seconds(data.get("load_duration")),
        prompt_eval_s=_seconds(data.get("prompt_eval_duration")),
        eval_s=_seconds(data.get("eval_duration")),
        raw={k: v for k, v in data.items() if k != "message"},
    )
