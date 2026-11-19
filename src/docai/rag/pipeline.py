"""RAG-конвейер целиком: вопрос → маршрут → (поиск → реранкер → ответ) или (SQL → ответ).

    pipeline = RagPipeline(load_rag_config())
    result = pipeline.ask("Какой срок оплаты по договору с ООО «Вектор»?")

Модели эмбеддингов и реранкера живут в памяти процесса (CPU) и грузятся при первом
вопросе или в `warm_up()`; LLM — в Ollama на GPU. Вопросы обрабатываются по одному
(замок): модели на CPU не потокобезопасны, а GPU одна.
"""

import logging
import threading
import time
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

from docai.extract.ollama import ChatResult
from docai.rag.config import RagConfig
from docai.rag.generate import ANSWER_SCHEMA as RAG_SCHEMA
from docai.rag.generate import NOT_FOUND, ask_llm, build_prompt, parse_answer
from docai.rag.index import IndexPaths, load_manifest
from docai.rag.models import Embedder, Qwen3Embedder, Reranker, make_reranker
from docai.rag.query import NameMatcher, QueryInfo, analyze
from docai.rag.router import route_question
from docai.rag.search import RagIndex, Retriever
from docai.rag.text_sql import (
    ANSWER_SCHEMA as SQL_ANSWER_SCHEMA,
)
from docai.rag.text_sql import (
    SQL_SCHEMA,
    SqlRejectedError,
    SqlResult,
    answer_prompt,
    build_hints,
    fallback_answer,
    hint_problems,
    numbers_supported,
    registry_sellers,
    run_select,
    sql_prompt,
)

log = logging.getLogger("docai.rag")

# Вызов LLM: промпт и JSON-схема → разобранный JSON и счётчики Ollama (в тестах — None).
Llm = Callable[[str, dict[str, Any]], tuple[dict[str, Any], ChatResult | None]]


class CitationOut(BaseModel):
    n: int = Field(description="Номер ссылки [n] в тексте ответа")
    fragment_id: str = Field(description="Id фрагмента: <документ>#<пункт>, например K-0013#2.2")
    doc_id: str
    label: str = Field(description="Пункт документа: «2.2», «preamble», «req-1»")
    document: str = Field(description="Документ, контрагент, раздел и пункт")
    quote: str = Field(description="Полный текст пункта")
    start: int = Field(description="Смещение начала пункта в файле документа")
    end: int
    path: str = Field(description="Файл документа относительно папки корпуса")
    source: Literal["model", "matched"] = Field(
        "model",
        description="model — ссылку поставила модель; matched — программа по совпадению "
        "чисел и слов фразы с пунктом (модель забыла ссылку)",
    )


class AskResult(BaseModel):
    question: str
    route: Literal["rag", "sql"]
    found: bool
    answer: str = Field(description="Ответ; в RAG каждая фраза со ссылкой [n]")
    citations: list[CitationOut] = Field(default_factory=list)
    reason: str | None = Field(None, description="Почему ответа нет (если found = false)")
    sql: str | None = Field(None, description="Выполненный запрос к реестру (маршрут sql)")
    sql_columns: list[str] | None = None
    sql_rows: list[list[str | None]] | None = Field(
        None, description="Строки результата; числа — строками с точкой, как во всём API"
    )
    filters: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    model: str
    reranker: str | None = None
    candidates: list[dict[str, Any]] | None = Field(
        None, description="Кандидаты поиска с рангами (только при debug)"
    )


def sql_cell(value: Any) -> str | None:
    """Ячейка результата SQL для API: число — строкой с точкой без экспоненты."""
    if value is None:
        return None
    if isinstance(value, float):
        return format(Decimal(repr(value)), "f")
    return str(value)


class RagPipeline:
    def __init__(
        self,
        config: RagConfig,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        llm: Llm | None = None,
        load_reranker: bool = True,
    ) -> None:
        self.config = config
        # Индекса нет вовсе — ошибка сразу; векторы не досчитаны — SQL по реестру уже работает,
        # а вопрос по документам получит IndexNotReadyError.
        load_manifest(config.index_dir)
        self._index: RagIndex | None = None
        self.today = date.fromisoformat(config.today)
        self._embedder = embedder
        self._reranker = reranker
        self._load_reranker = load_reranker and reranker is None and config.reranker.kind != "none"
        self._llm = llm or (lambda prompt, schema: ask_llm(config.llm, prompt, schema))
        self._lock = threading.Lock()
        registry = IndexPaths(config.index_dir).registry
        self.registry_db = registry if registry.is_file() else None
        self.sellers = registry_sellers(registry) if self.registry_db else {}
        self.seller_matcher = NameMatcher(self.sellers)

    # --- модели ---------------------------------------------------------------------------

    @property
    def index(self) -> RagIndex:
        if self._index is None:
            self._index = RagIndex(self.config.index_dir)
        return self._index

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Qwen3Embedder(self.config.embedding)
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        if self._load_reranker:
            self._reranker = make_reranker(self.config.reranker, self.config.embedding.threads)
            self._load_reranker = False
        return self._reranker

    def warm_up(self) -> None:
        """Загрузить модели CPU заранее (иначе первый вопрос дольше на 10–20 с)."""
        with self._lock:
            self.embedder.embed_query("прогрев")
            reranker = self.reranker
            if reranker is not None:
                reranker.score("прогрев", ["прогрев"])

    # --- ответ ----------------------------------------------------------------------------

    def ask(self, question: str, *, debug: bool = False) -> AskResult:
        question = question.strip()
        if not question:
            raise ValueError("пустой вопрос")
        with self._lock:
            started = time.perf_counter()
            route = route_question(question)
            if route == "sql" and self.registry_db is not None:
                result = self._ask_sql(question)
            else:
                result = self._ask_rag(question, debug=debug)
            result.timings["total_s"] = round(time.perf_counter() - started, 3)
            return result

    def _base(self, question: str, route: Literal["rag", "sql"]) -> AskResult:
        return AskResult(
            question=question,
            route=route,
            found=False,
            answer=NOT_FOUND,
            model=self.config.llm.model,
            reranker=None
            if route == "sql" or self.config.reranker.kind == "none"
            else self.config.reranker.model,
        )

    def _llm_timings(self, chat: ChatResult | None, prefix: str, timings: dict[str, float]) -> None:
        if chat is None:
            return
        timings[f"{prefix}_load_s"] = chat.load_s
        timings[f"{prefix}_prompt_tokens"] = chat.prompt_tokens
        timings[f"{prefix}_completion_tokens"] = chat.completion_tokens

    def _ask_rag(self, question: str, *, debug: bool) -> AskResult:
        result = self._base(question, "rag")
        timings = result.timings
        started = time.perf_counter()
        info = analyze(question, self.index.matcher, self.today)
        result.filters = {"query": info.as_dict()}
        timings["analyze_s"] = round(time.perf_counter() - started, 3)

        reason = self._missing(info)
        if reason is not None:
            result.reason = reason
            return result

        retriever = Retriever(self.index, self.embedder, self.reranker, self.config.retrieval)
        started = time.perf_counter()
        retrieval = retriever.retrieve(question, info)
        timings.update(retrieval.timings)
        timings["retrieve_s"] = round(time.perf_counter() - started, 3)
        result.filters.update(retrieval.filters)
        if debug:
            result.candidates = [h.as_dict() for h in retrieval.candidates[:30]]
        if not retrieval.context:
            result.reason = "поиск ничего не нашёл"
            return result

        started = time.perf_counter()
        data, chat = self._llm(build_prompt(question, retrieval.context), RAG_SCHEMA)
        timings["generate_s"] = round(time.perf_counter() - started, 3)
        self._llm_timings(chat, "llm", timings)
        parsed = parse_answer(data, retrieval.context)
        result.found = parsed.found
        result.answer = parsed.answer
        result.citations = [CitationOut(**c.as_dict()) for c in parsed.citations]
        if not parsed.found:
            result.reason = "в найденных фрагментах ответа нет"
        if debug:
            result.filters["context"] = [h.as_dict() for h in retrieval.context]
            result.filters["uncited_sentences"] = parsed.uncited_sentences
        return result

    def _missing(self, info: QueryInfo) -> str | None:
        """Названный в вопросе контрагент или договор отсутствует в корпусе — ответа нет."""
        if info.unknown_orgs and not info.counterparties:
            return "в документах нет контрагента " + ", ".join(info.unknown_orgs)
        if info.contracts and not self.index.contract_docs(info):
            refs = ", ".join(f"№ {c.number}" for c in info.contracts)
            who = f" с {', '.join(info.counterparties)}" if info.counterparties else ""
            return f"в документах нет договора {refs}{who}"
        return None

    def _ask_sql(self, question: str) -> AskResult:
        assert self.registry_db is not None
        result = self._base(question, "sql")
        timings = result.timings
        info = analyze(question, self.seller_matcher, self.today)
        result.filters = {"query": info.as_dict()}
        if info.unknown_orgs and not info.counterparties:
            result.reason = "в реестре нет контрагента " + ", ".join(info.unknown_orgs)
            result.answer = f"В реестре нет документов от {', '.join(info.unknown_orgs)}."
            return result
        hints = build_hints(info, self.sellers, question)
        prompt = sql_prompt(question, hints, self.today)

        started = time.perf_counter()
        # Две попытки: запрос, который не выполнился или расходится с подсказками, уходит
        # модели обратно с перечнем замечаний. Из выполнившихся берётся тот, где их меньше.
        sql_result: SqlResult | None = None
        best: tuple[int, SqlResult] | None = None
        error: str | None = None
        last_sql = ""
        for attempt in range(2):
            text = prompt
            if attempt:
                text += (
                    f"\n\nПредыдущий запрос:\n{last_sql}\nЗамечания: {error}. "
                    "Напиши исправленный запрос."
                )
            data, chat = self._llm(text, SQL_SCHEMA)
            self._llm_timings(chat, f"sql{attempt + 1}", timings)
            last_sql = str(data.get("sql") or "")
            try:
                candidate = run_select(
                    self.registry_db,
                    last_sql,
                    max_rows=self.config.sql.max_rows,
                    max_vm_steps=self.config.sql.max_vm_steps,
                )
            except SqlRejectedError as exc:
                error = "; ".join([str(exc), *hint_problems(last_sql, hints)])
                log.info("SQL отклонён (%s): %s", error, last_sql)
                continue
            problems = hint_problems(candidate.sql, hints)
            if best is None or len(problems) < best[0]:
                best = (len(problems), candidate)
            if not problems:
                break
            error = "; ".join(problems)
            log.info("SQL расходится с подсказками (%s): %s", error, last_sql)
        sql_result = best[1] if best else None
        timings["sql_s"] = round(time.perf_counter() - started, 3)
        if sql_result is None:
            result.reason = f"не удалось построить запрос к реестру: {error}"
            result.answer = "Не удалось ответить по реестру: запрос не прошёл проверку."
            return result
        result.sql = sql_result.sql
        result.sql_columns = sql_result.columns
        result.sql_rows = [[sql_cell(v) for v in row] for row in sql_result.rows]

        started = time.perf_counter()
        data, chat = self._llm(answer_prompt(question, sql_result), SQL_ANSWER_SCHEMA)
        timings["generate_s"] = round(time.perf_counter() - started, 3)
        self._llm_timings(chat, "llm", timings)
        answer = str(data.get("answer") or "").strip()
        if not answer or not numbers_supported(answer, question, sql_result):
            answer = fallback_answer(sql_result)
        result.answer = answer
        # «0 документов» — тоже ответ; ответа нет, только если строк нет или в них пусто.
        result.found = any(v is not None for row in sql_result.rows for v in row)
        if not result.found:
            result.reason = "по реестру документов нет"
        return result
