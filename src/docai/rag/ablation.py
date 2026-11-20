"""Прогон оценки RAG и абляций (P20): поиск по вариантам и ответы по конфигурациям.

Два этапа, оба с кэшем в `cache_dir` (JSONL, дописывается после каждого вопроса; остановка
и повторный запуск продолжают с места — как в оценке извлечения, D46):

1. `retrieval` — для вопросов с эталонными источниками четыре варианта поиска: только BM25,
   только векторы, гибрид (RRF), гибрид + реранкер. Модель-ответчик не нужна, считаются
   только ранги найденных пунктов.
2. `answers` — полный ответ конвейера (`RagPipeline.ask`) по каждой конфигурации абляции.
   Часть конфигураций не прогоняется заново там, где её путь совпадает с другой по
   построению (`reuse_from`): агрегаты по реестру идут через SQL и не зависят от поиска,
   а вопросы без таблиц при выключенном маршрутизаторе идут тем же путём RAG, что и при
   включённом (проверяется: `route_question` отправляет их в RAG).

Кэш привязан к отпечатку: настройки конвейера, манифест индекса и текст вопроса. Другой
отпечаток — запись игнорируется и считается заново.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from docai.rag.config import RagConfig, load_rag_config
from docai.rag.evaluate import load_jsonl
from docai.rag.generate import ask_llm
from docai.rag.index import load_manifest
from docai.rag.pipeline import RagPipeline
from docai.rag.query import analyze
from docai.rag.router import route_question
from docai.rag.search import Retriever

log = logging.getLogger("docai.rag.ablation")

Route = Literal["rag", "none", "table"]


class AblationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"
    reranker: bool = False
    router: bool = True
    # Какие маршруты вопросов (поле `route` эталона) прогоняются этой конфигурацией.
    run: list[Route]
    # Остальные маршруты берутся из этой конфигурации (путь не зависит от различия).
    reuse_from: str | None = None

    @property
    def retrieval_key(self) -> str:
        return self.mode + ("+rerank" if self.reranker else "")


class EvalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 2026
    n_resamples: int = 2000
    pipeline_config: Path
    questions: Path
    fragments: Path
    manual_questions: Path
    cache_dir: Path
    report_dir: Path
    ks: list[int] = Field(default_factory=lambda: [5, 10])
    # Результаты, по которым настраивался конвейер в P19: вопросы оттуда — «затронутые».
    tuned_on: list[Path] = Field(default_factory=list)
    # Ручной аудит автоматической проверки (что и как проверено глазами), попадает в отчёт.
    audit_notes: list[str] = Field(default_factory=list)
    ablations: list[AblationSpec]

    def spec(self, name: str) -> AblationSpec:
        return next(a for a in self.ablations if a.name == name)


def load_settings(path: Path) -> EvalSettings:
    return EvalSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def tuned_question_ids(paths: Iterable[Path]) -> set[str]:
    """Id вопросов, на которых настраивался конвейер (из результатов P19)."""
    ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        ids.update(data.get("questions") or [])
        ids.update(r["question"]["id"] for r in data.get("rows") or [] if "question" in r)
    return ids


# --- Кэш ------------------------------------------------------------------------------------


def _sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def question_hash(question: dict[str, Any]) -> str:
    return _sha([question["id"], question["question"]])


def config_for(base: RagConfig, spec: AblationSpec) -> RagConfig:
    """Конфиг конвейера для конфигурации абляции."""
    config = base.model_copy(deep=True)
    config.retrieval.mode = spec.mode
    config.use_router = spec.router
    if not spec.reranker:
        config.reranker.kind = "none"
    return config


def fingerprint(config: RagConfig) -> str:
    """Отпечаток: все настройки конвейера и манифест индекса (корпус, эмбеддинги)."""
    manifest = load_manifest(config.index_dir)
    return _sha({"config": config.model_dump(mode="json"), "index": manifest.get("fingerprint")})


class Cache:
    """JSONL: по записи на вопрос; побеждает последняя запись с подходящим отпечатком."""

    def __init__(self, path: Path, fp: str) -> None:
        self.path = path
        self.fp = fp
        self.rows: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("fp") == fp:
                    self.rows[row["id"]] = row

    def get(self, question: dict[str, Any]) -> dict[str, Any] | None:
        row = self.rows.get(question["id"])
        return row if row is not None and row["qhash"] == question_hash(question) else None

    def put(self, question: dict[str, Any], payload: dict[str, Any]) -> None:
        row = {"id": question["id"], "qhash": question_hash(question), "fp": self.fp, **payload}
        self.rows[question["id"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def retrieval_cache_path(settings: EvalSettings, qset: str) -> Path:
    return settings.cache_dir / qset / "retrieval.jsonl"


def answers_cache_path(settings: EvalSettings, qset: str, name: str) -> Path:
    return settings.cache_dir / qset / f"answers_{name}.jsonl"


# --- Рабочее место: общие модели ------------------------------------------------------------


class Workbench:
    """Один набор моделей CPU на все конфигурации (иначе реранкер грузился бы заново)."""

    def __init__(self, base: RagConfig) -> None:
        self.base = base
        self.root = RagPipeline(base)
        self.root.warm_up()

    def pipeline(self, spec: AblationSpec) -> RagPipeline:
        config = config_for(self.base, spec)
        pipeline = RagPipeline(
            config,
            embedder=self.root.embedder,
            reranker=self.root.reranker if spec.reranker else None,
            load_reranker=False,
        )
        pipeline.share_index(self.root.index)
        return pipeline

    def warm_up_llm(self) -> float:
        started = time.perf_counter()
        ask_llm(self.base.llm, 'Ответь JSON {"ok": true}', {"type": "object"})
        return time.perf_counter() - started


def _spans(hits: Sequence[Any]) -> list[list[Any]]:
    return [[h.chunk.chunk_id, h.chunk.doc_id, h.chunk.start, h.chunk.end] for h in hits]


def has_gold(question: dict[str, Any]) -> bool:
    """Вопрос участвует в оценке поиска: маршрут rag и есть эталонные источники."""
    return question["route"] == "rag" and bool(question.get("sources"))


# --- Этап 1: поиск --------------------------------------------------------------------------


def run_retrieval(
    settings: EvalSettings,
    bench: Workbench,
    questions: Sequence[dict[str, Any]],
    qset: str,
    say: Callable[[str], None] = print,
) -> None:
    base = bench.root
    variants: dict[str, AblationSpec] = {}
    for spec in settings.ablations:
        variants.setdefault(spec.retrieval_key, spec)
    fp = fingerprint(
        config_for(bench.base, AblationSpec(name="x", label="x", run=[], reranker=True))
    )
    cache = Cache(retrieval_cache_path(settings, qset), fp)
    # Запись вопроса пополняется недостающими вариантами: добавленная абляция не заставляет
    # пересчитывать остальные.
    todo = []
    for q in questions:
        if not has_gold(q):
            continue
        entry = cache.get(q)
        absent = [k for k in variants if entry is None or k not in entry["variants"]]
        if absent and not (entry is not None and entry.get("missing")):
            todo.append((q, entry, absent))
    say(f"Поиск: {len(todo)} вопросов из {sum(1 for q in questions if has_gold(q))} к расчёту")
    index = base.index
    for number, (q, entry, absent) in enumerate(todo, start=1):
        info = analyze(q["question"], index.matcher, base.today)
        missing = base.missing_reason(info)
        payload: dict[str, Any] = {
            "missing": missing,
            "variants": dict(entry["variants"]) if entry else {},
        }
        if missing is None:
            for key in absent:
                spec = variants[key]
                retriever = Retriever(
                    index,
                    base.embedder,
                    base.reranker if spec.reranker else None,
                    config_for(bench.base, spec).retrieval,
                )
                started = time.perf_counter()
                result = retriever.retrieve(q["question"], info)
                payload["variants"][key] = {
                    "ranked": _spans(result.candidates[:30]),
                    "context": _spans(result.context),
                    "seconds": round(time.perf_counter() - started, 3),
                }
        cache.put(q, payload)
        if number % 10 == 0 or number == len(todo):
            say(f"  поиск: {number}/{len(todo)}")


# --- Этап 2: ответы -------------------------------------------------------------------------


def answer_payload(result: dict[str, Any]) -> dict[str, Any]:
    keep = ("route", "found", "answer", "reason", "sql", "sql_rows", "timings")
    payload = {k: result.get(k) for k in keep}
    payload["citations"] = [
        {k: c[k] for k in ("n", "fragment_id", "doc_id", "start", "end", "quote", "source")}
        for c in result.get("citations") or []
    ]
    payload["query_filters"] = (result.get("filters") or {}).get("query")
    return payload


def run_answers(
    settings: EvalSettings,
    bench: Workbench,
    spec: AblationSpec,
    questions: Sequence[dict[str, Any]],
    qset: str,
    say: Callable[[str], None] = print,
    max_errors: int = 3,
) -> None:
    pipeline = bench.pipeline(spec)
    cache = Cache(answers_cache_path(settings, qset, spec.name), fingerprint(pipeline.config))
    todo = [q for q in questions if q["route"] in spec.run and cache.get(q) is None]
    total = sum(1 for q in questions if q["route"] in spec.run)
    say(f"Ответы [{spec.name}]: {len(todo)} вопросов из {total} к расчёту")
    errors = 0
    for number, q in enumerate(todo, start=1):
        try:
            result = pipeline.ask(q["question"]).model_dump()
        except Exception as exc:
            errors += 1
            log.warning("%s: %s", q["id"], exc)
            say(f"  {q['id']}: ошибка {exc!r} ({errors} подряд)")
            if errors >= max_errors:
                raise
            continue
        errors = 0
        cache.put(q, answer_payload(result))
        say(
            f"  [{spec.name}] {number}/{len(todo)} {q['id']} {q['category']} "
            f"{result['timings']['total_s']:.1f} с | {result['answer'][:90]!r}"
        )


def run_all(
    settings: EvalSettings,
    stages: Sequence[str],
    qset: str,
    say: Callable[[str], None] = print,
) -> None:
    questions_path = settings.questions if qset == "p18" else settings.manual_questions
    questions = load_jsonl(questions_path) if questions_path.is_file() else []
    if not questions:
        say(f"Набор «{qset}» пуст или файла нет: {questions_path}")
        return
    base = load_rag_config(settings.pipeline_config)
    bench = Workbench(base)
    if "retrieval" in stages:
        run_retrieval(settings, bench, questions, qset, say)
    if "answers" in stages:
        say(f"LLM загружена за {bench.warm_up_llm():.1f} с")
        for spec in settings.ablations:
            run_answers(settings, bench, spec, questions, qset, say)


def router_agreement(questions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Совпадает ли маршрут правил с эталонным: на этом держится `reuse_from`."""
    wrong = [
        q["id"]
        for q in questions
        if (route_question(q["question"]) == "sql") != (q["route"] == "table")
    ]
    return {"questions": len(questions), "disagree": wrong}
