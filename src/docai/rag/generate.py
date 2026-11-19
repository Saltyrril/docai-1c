"""Ответ модели по найденным фрагментам: каждая фраза со ссылкой [n], ответ — JSON.

Модель (Qwen3.5-2B в Ollama, на GPU) получает пронумерованные фрагменты с заголовками
(документ, контрагент, раздел, пункт; у доп. соглашения — дата вступления в силу) и
отвечает только по ним. Ссылки [n] сопоставляются с id фрагментов здесь, а не моделью:
номер в скобках модель ставит надёжнее, чем переписывает id.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from docai.extract.config import ExtractConfig
from docai.extract.ollama import chat
from docai.rag.config import LlmConfig
from docai.rag.facts import normalize_answer
from docai.rag.lemmas import lemmas
from docai.rag.search import Hit

NOT_FOUND = "В документах не нашёл."

# Сначала ответ, потом флаг: если флаг идёт первым, 2B-модель чаще решает «не нашёл»
# раньше, чем прочитает фрагменты (D71).
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "found": {"type": "boolean"},
    },
    "required": ["answer", "found"],
}

PROMPT = """Ты помогаешь бухгалтеру АО «Тайга-Строй». Ответь на вопрос только по фрагментам документов ниже.

Вопрос: {question}

Правила:
1. Бери факты только из фрагментов. Ничего не додумывай и не используй свои знания.
2. В конце каждого предложения ставь номер фрагмента, из которого взят факт: [1] или [2][3].
3. Если доп. соглашение изменяет пункт договора, действует редакция соглашения с самой поздней датой вступления в силу: отвечай по ней и ссылайся на соглашение.
4. Сроки, суммы, проценты и названия переписывай точно, как во фрагменте.
5. Вопрос может называть условие своими словами, не как в договоре («депозит» — обеспечительный платёж, «задержать деньги» — просрочка оплаты, «сколько ждать ответа на претензию» — срок рассмотрения претензии): сопоставляй по смыслу.
6. Если пункт прямо говорит, что условия нет или что оно действует иначе (например, «продление — только дополнительным соглашением»), это тоже ответ: found = true, ответь «нет» и сошлись на этот пункт.
7. Если вопрос сравнивает два договора, приведи значение по каждому договору со ссылкой на его пункт и скажи, где больше (или что одинаково).
8. Если во фрагментах нет ответа на вопрос или речь в них о другом договоре или контрагенте, верни found = false и answer = "{not_found}".
9. Отвечай кратко: 1–3 предложения на русском языке.

Фрагменты:
{fragments}

Вопрос: {question}

Ответь JSON: {{"answer": "ответ со ссылками [n]", "found": true или false}}"""

_REF = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])")
# Сокращение перед точкой («г. Воронеж», «ул. Ленина», «алл. Привольная») — не конец фразы.
_ABBREVIATION = re.compile(r"(?:^|[\s(«])[а-яёa-z]{1,4}\.$")


def split_sentences(text: str) -> list[str]:
    pieces = [x for x in _SENTENCE.split(text) if x.strip()]
    result: list[str] = []
    for piece in pieces:
        if result and _ABBREVIATION.search(result[-1]):
            result[-1] = f"{result[-1]} {piece}"
        else:
            result.append(piece)
    return result


def format_fragments(hits: Sequence[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        blocks.append(f"[{n}] {hit.chunk.heading()}\n{hit.chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    return PROMPT.format(
        not_found=NOT_FOUND, fragments=format_fragments(hits), question=question.strip()
    )


def ollama_config(llm: LlmConfig) -> ExtractConfig:
    """Клиент Ollama из P9 принимает ExtractConfig: берём из него адрес и таймаут."""
    return ExtractConfig(
        model=llm.model,
        ollama_url=llm.ollama_url,
        think=llm.think,
        keep_alive=llm.keep_alive,
        timeout_s=llm.timeout_s,
        options=llm.options,
    )


def ask_llm(llm: LlmConfig, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Запрос к Ollama со схемой JSON; ответ — словарь и счётчики Ollama."""
    body = {
        "model": llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema,
        "think": llm.think,
        "keep_alive": llm.keep_alive,
        "options": llm.options.model_dump(),
    }
    result = chat(ollama_config(llm), body)
    try:
        data = json.loads(result.content)
    except json.JSONDecodeError:
        data = {}
    return (data if isinstance(data, dict) else {}), result


@dataclass
class Citation:
    n: int
    fragment_id: str
    doc_id: str
    label: str
    document: str
    quote: str
    start: int
    end: int
    path: str
    # model — ссылку [n] поставила модель; matched — программа по совпадению чисел и слов.
    source: str = "model"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "fragment_id": self.fragment_id,
            "doc_id": self.doc_id,
            "label": self.label,
            "document": self.document,
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "path": self.path,
            "source": self.source,
        }


@dataclass
class ParsedAnswer:
    found: bool
    answer: str
    citations: list[Citation]
    uncited_sentences: int
    raw: dict[str, Any]
    matched_sentences: int = 0


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _anchors(text: str) -> set[str]:
    """Числа и даты фразы: «405 300,00» → «405300,00»; номера пунктов «2.4» тоже числа."""
    norm = normalize_answer(text)
    return {m.group(0).replace(".", ",") for m in _NUMBER.finditer(norm)}


def _content(text: str) -> set[str]:
    return {w for w in lemmas(text) if len(w) > 3 and not w[0].isdigit()}


def attribute(sentence: str, hits: Sequence[Hit]) -> list[int]:
    """Номера фрагментов [n], подтверждающих фразу без ссылки.

    Числа фразы (сроки, суммы, проценты, даты) ищутся во фрагментах: жадно берутся фрагменты,
    покрывающие больше всего ещё не покрытых чисел (в сравнении двух договоров — оба). Если
    чисел нет или ни одно не нашлось — фрагмент с наибольшей долей общих слов (не меньше 0,4).
    """
    numbers = _anchors(sentence)
    words = _content(sentence)
    frag_numbers = [_anchors(h.chunk.text) for h in hits]
    frag_words = [_content(h.chunk.index_text()) for h in hits]
    chosen: list[int] = []
    left = set(numbers)
    while left:
        best = max(range(len(hits)), key=lambda i: (len(frag_numbers[i] & left), -i))
        gain = frag_numbers[best] & left
        if not gain:
            break
        chosen.append(best + 1)
        left -= gain
    if chosen:
        return sorted(chosen)
    if not words:
        return []
    ratios = [len(words & fw) / len(words) for fw in frag_words]
    best = max(range(len(hits)), key=lambda i: (ratios[i], -i))
    return [best + 1] if ratios[best] >= 0.4 else []


def parse_answer(data: dict[str, Any], hits: Sequence[Hit]) -> ParsedAnswer:
    """Проверить ответ: ссылки только на выданные фрагменты; фразы без ссылок — сверить
    с фрагментами (`attribute`); если ссылок нет совсем — «не нашёл»."""
    text = str(data.get("answer") or "").strip()
    found = bool(data.get("found")) and bool(text)
    if not found or text.startswith(NOT_FOUND[:-1]):
        return ParsedAnswer(False, NOT_FOUND, [], 0, data)

    def keep(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= len(hits) else ""

    text = _REF.sub(keep, text).strip()
    sentences = split_sentences(text)
    uncited = 0
    matched: set[int] = set()
    rebuilt: list[str] = []
    for sentence in sentences:
        if _REF.search(sentence):
            rebuilt.append(sentence)
            continue
        uncited += 1
        refs = attribute(sentence, hits)
        matched.update(refs)
        if refs:
            body = sentence.rstrip()
            tail = ""
            if body and body[-1] in ".!?":
                body, tail = body[:-1], body[-1]
            sentence = body + "".join(f" [{n}]" for n in refs) + tail
        rebuilt.append(sentence)
    text = " ".join(rebuilt)
    numbers: list[int] = []
    for m in _REF.finditer(text):
        n = int(m.group(1))
        if n not in numbers:
            numbers.append(n)
    if not numbers:
        return ParsedAnswer(False, NOT_FOUND, [], uncited, data)
    model_refs = {int(m.group(1)) for x in sentences for m in _REF.finditer(x)}
    citations = []
    for n in numbers:
        chunk = hits[n - 1].chunk
        assert chunk.meta is not None
        citations.append(
            Citation(
                n=n,
                fragment_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                label=chunk.label,
                document=chunk.heading(),
                quote=chunk.text.strip(),
                start=chunk.start,
                end=chunk.end,
                path=chunk.meta.path,
                source="model" if n in model_refs else "matched",
            )
        )
    return ParsedAnswer(True, text, citations, uncited, data, matched_sentences=len(matched))
