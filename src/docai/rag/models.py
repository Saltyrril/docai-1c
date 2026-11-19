"""Модели на CPU: эмбеддинги Qwen3-Embedding и реранкеры (Qwen3-Reranker, bge-reranker-v2-m3).

GPU занята одной LLM в Ollama, поэтому всё здесь считается на процессоре (`device="cpu"`).
torch и transformers импортируются лениво: CLI и тесты с заглушками их не грузят.

Веса скачиваются с Hugging Face один раз (при первой индексации); дальше модели читаются
из локального кэша без обращения к сети.
"""

import copy
import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol

from docai.rag.config import EmbeddingConfig, RerankerConfig

log = logging.getLogger("docai.rag")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def _torch(threads: int) -> Any:
    import torch

    torch.set_num_threads(threads)
    return torch


def _pretrained(loader: Any, name: str, **kwargs: Any) -> Any:
    """Сначала из локального кэша (без сети), при отсутствии — скачать."""
    try:
        return loader(name, local_files_only=True, **kwargs)
    except OSError:
        log.info("модели %s нет в кэше, скачиваю с Hugging Face", name)
        return loader(name, **kwargs)


class Qwen3Embedder:
    """Qwen3-Embedding через sentence-transformers; векторы нормированы (косинус)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        torch = _torch(config.threads)
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        self.name = config.model
        self.config = config
        # Веса Qwen3-Embedding хранятся в bf16; на Ryzen 5600H (нет AVX-512 BF16) float32
        # считает быстрее, поэтому тип задаётся явно.
        self.model = _pretrained(
            SentenceTransformer,
            config.model,
            device="cpu",
            model_kwargs={"dtype": torch.float32},
        )
        self.model.max_seq_length = config.max_seq_length
        log.info("эмбеддинги %s загружены за %.1f с", self.name, time.perf_counter() - started)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text],
            prompt=self.config.query_instruction,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()


class Qwen3Reranker:
    """Qwen3-Reranker: вероятность ответа «yes» на «отвечает ли пункт на вопрос».

    Считается только последняя позиция и только две строки выходного слоя («yes»/«no»):
    полный выход по словарю 150 тыс. токенов для каждой позиции на CPU — лишняя четверть
    времени.
    """

    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, config: RerankerConfig, threads: int = 6) -> None:
        self.torch = _torch(threads)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.name = config.model
        self.config = config
        self.tokenizer = _pretrained(
            AutoTokenizer.from_pretrained, config.model, padding_side="left"
        )
        model = _pretrained(
            AutoModelForCausalLM.from_pretrained, config.model, dtype=self.torch.float32
        )
        self.model = model.eval()
        yes = self.tokenizer.convert_tokens_to_ids("yes")
        no = self.tokenizer.convert_tokens_to_ids("no")
        weight = model.get_output_embeddings().weight
        self.head = weight[[no, yes]].detach().clone()
        self.prefix_ids = self.tokenizer.encode(self._PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self._SUFFIX, add_special_tokens=False)
        log.info("реранкер %s загружен за %.1f с", self.name, time.perf_counter() - started)

    def _pairs(self, query: str, passages: Sequence[str]) -> list[str]:
        return [
            f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {p}"
            for p in passages
        ]

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if self.config.prefix_cache:
            return self._score_cached(query, passages)
        return self._score_plain(query, passages)

    def _score_cached(self, query: str, passages: Sequence[str]) -> list[float]:
        """Общая часть (системный промпт, инструкция, вопрос) считается один раз.

        Её KV-кэш копируется на пачку пунктов, и модель досчитывает только сам пункт и
        хвост промпта: токенов на пару вдвое меньше. Пачка выравнивается паддингом справа,
        оценка берётся с последнего настоящего токена каждой строки (причинное внимание:
        паддинг после него на результат не влияет).
        """
        torch = self.torch
        head = f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>:"
        head_ids = self.prefix_ids + self.tokenizer.encode(head, add_special_tokens=False)
        budget = max(16, self.config.max_length - len(head_ids) - len(self.suffix_ids))
        pad = self.tokenizer.pad_token_id or 0
        scores: list[float] = []
        with torch.inference_mode():
            prefix = self.model.model(input_ids=torch.tensor([head_ids]), use_cache=True)
            cache = prefix.past_key_values
            for i in range(0, len(passages), self.config.batch_size):
                batch = passages[i : i + self.config.batch_size]
                rows = [
                    self.tokenizer.encode(" " + p, add_special_tokens=False)[:budget]
                    + self.suffix_ids
                    for p in batch
                ]
                width = max(len(r) for r in rows)
                ids = torch.full((len(rows), width), pad, dtype=torch.long)
                mask = torch.zeros((len(rows), width), dtype=torch.long)
                for j, r in enumerate(rows):
                    ids[j, : len(r)] = torch.tensor(r)
                    mask[j, : len(r)] = 1
                full_mask = torch.cat(
                    [torch.ones((len(rows), len(head_ids)), dtype=torch.long), mask], dim=1
                )
                positions = (len(head_ids) + torch.arange(width)).unsqueeze(0).expand(len(rows), -1)
                batch_cache = copy.deepcopy(cache)
                batch_cache.batch_repeat_interleave(len(rows))
                hidden = self.model.model(
                    input_ids=ids,
                    attention_mask=full_mask,
                    position_ids=positions,
                    past_key_values=batch_cache,
                    use_cache=True,
                ).last_hidden_state
                last = mask.sum(dim=1) - 1
                logits = hidden[torch.arange(len(rows)), last] @ self.head.T
                scores.extend(torch.log_softmax(logits, dim=1)[:, 1].exp().tolist())
        return scores

    def _score_plain(self, query: str, passages: Sequence[str]) -> list[float]:
        """Как в карточке модели: каждая пара целиком, паддинг слева."""
        torch = self.torch
        budget = self.config.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        scores: list[float] = []
        pairs = self._pairs(query, passages)
        for i in range(0, len(pairs), self.config.batch_size):
            batch = pairs[i : i + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=budget,
            )
            encoded["input_ids"] = [
                self.prefix_ids + ids + self.suffix_ids for ids in encoded["input_ids"]
            ]
            inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model.model(**inputs).last_hidden_state[:, -1, :]
                logits = hidden @ self.head.T  # [batch, 2]: «no», «yes»
                probs = torch.log_softmax(logits, dim=1)[:, 1].exp()
            scores.extend(probs.tolist())
        return scores


class BgeReranker:
    """bge-reranker-v2-m3 (кросс-энкодер XLM-RoBERTa): сигмоида от логита пары."""

    def __init__(self, config: RerankerConfig, threads: int = 6) -> None:
        self.torch = _torch(threads)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        started = time.perf_counter()
        self.name = config.model
        self.config = config
        self.tokenizer = _pretrained(AutoTokenizer.from_pretrained, config.model)
        model = _pretrained(
            AutoModelForSequenceClassification.from_pretrained,
            config.model,
            dtype=self.torch.float32,
        )
        self.model = model.eval()
        log.info("реранкер %s загружен за %.1f с", self.name, time.perf_counter() - started)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        torch = self.torch
        # Пары похожей длины — в одну пачку: короткий пункт не дополняется паддингом
        # до длинной вводной части (на CPU паддинг считается как настоящие токены).
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))
        scores = [0.0] * len(passages)
        for i in range(0, len(order), self.config.batch_size):
            idx = order[i : i + self.config.batch_size]
            inputs = self.tokenizer(
                [query] * len(idx),
                [passages[j] for j in idx],
                padding=True,
                truncation="only_second",
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**inputs).logits.view(-1).float()
            for j, value in zip(idx, torch.sigmoid(logits).tolist(), strict=True):
                scores[j] = value
        return scores


def make_reranker(config: RerankerConfig, threads: int = 6) -> Reranker | None:
    if config.kind == "none":
        return None
    if config.kind == "qwen3":
        return Qwen3Reranker(config, threads)
    return BgeReranker(config, threads)
