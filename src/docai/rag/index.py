"""Индекс RAG: куски с леммами (BM25), векторы в Chroma, реестр в SQLite.

Папка индекса (по умолчанию `data/rag_index/`, в git не попадает):

    manifest.json   отпечаток (корпус, модель, нарезка) и сколько кусков уже с векторами
    chunks.jsonl    куски: id, документ, пункт, смещения, текст, метаданные, леммы
    docs.jsonl      метаданные документов
    chroma/         встроенная Chroma: векторы Qwen3-Embedding (косинус)
    registry.sqlite реестр первичных документов (таблица `registry`) для текста-в-SQL

Векторы на CPU считаются долго (десятки минут), поэтому они пишутся в Chroma пачками:
после остановки (Ctrl+C, сбой) повторный запуск находит уже записанные id и продолжает.
Если корпус, модель или нарезка поменялись, отпечаток не совпадёт — нужен `--rebuild`.
"""

import contextlib
import csv
import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from docai.rag.chunking import Chunk, DocMeta, chunk_corpus
from docai.rag.config import RagConfig
from docai.rag.lemmas import lemmas
from docai.rag.models import Embedder

INDEX_FORMAT = 1
CHUNKER_VERSION = "structure-v1"
COLLECTION_PREFIX = "chunks_"

# Колонки реестра и их типы в SQLite (суммы — числа, чтобы SUM работал без CAST).
REGISTRY_COLUMNS: dict[str, str] = {
    "doc_id": "TEXT",
    "doc_type": "TEXT",
    "doc_type_name": "TEXT",
    "number": "TEXT",
    "date": "TEXT",
    "seller_name": "TEXT",
    "seller_inn": "TEXT",
    "buyer_name": "TEXT",
    "buyer_inn": "TEXT",
    "contract_id": "TEXT",
    "contract_number": "TEXT",
    "contract_date": "TEXT",
    "amount_without_vat": "REAL",
    "vat_amount": "REAL",
    "amount_with_vat": "REAL",
    "vat_rates": "TEXT",
    "lines": "INTEGER",
    "p3_source": "TEXT",
}


class IndexMismatchError(RuntimeError):
    """Индекс собран по другому корпусу, модели или нарезке."""


class IndexNotReadyError(RuntimeError):
    """Индекса нет или векторы посчитаны не для всех кусков."""


@dataclass(frozen=True)
class IndexPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def chunks(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def docs(self) -> Path:
        return self.root / "docs.jsonl"

    @property
    def chroma(self) -> Path:
        return self.root / "chroma"

    @property
    def registry(self) -> Path:
        return self.root / "registry.sqlite"


def corpus_sha256(corpus_dir: Path) -> str:
    """Отпечаток корпуса: относительные пути и байты всех `*.md`."""
    digest = hashlib.sha256()
    for path in sorted(corpus_dir.rglob("*.md")):
        digest.update(path.relative_to(corpus_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint(config: RagConfig) -> dict[str, Any]:
    return {
        "format": INDEX_FORMAT,
        "chunker": CHUNKER_VERSION,
        "max_chunk_chars": config.max_chunk_chars,
        "corpus_sha256": corpus_sha256(config.corpus_dir),
        "embedding_model": config.embedding.model,
        "max_seq_length": config.embedding.max_seq_length,
    }


def chroma_client(path: Path) -> Any:
    """Встроенная Chroma без телеметрии: данные и события не уходят в сеть."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=False)
    )


def open_chroma(path: Path, name: str) -> Any:
    return chroma_client(path).get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )


def _date_json(d: date | None) -> str | None:
    return d.isoformat() if d else None


def meta_to_json(meta: DocMeta) -> dict[str, Any]:
    return {
        "doc_id": meta.doc_id,
        "doc_kind": meta.doc_kind,
        "title": meta.title,
        "path": meta.path,
        "number": meta.number,
        "doc_date": _date_json(meta.doc_date),
        "contract_kind": meta.contract_kind,
        "counterparty": meta.counterparty,
        "counterparty_inn": meta.counterparty_inn,
        "parent_number": meta.parent_number,
        "parent_date": _date_json(meta.parent_date),
        "parent_doc_id": meta.parent_doc_id,
        "effective_date": _date_json(meta.effective_date),
    }


def meta_from_json(data: dict[str, Any]) -> DocMeta:
    def d(key: str) -> date | None:
        value = data.get(key)
        return date.fromisoformat(value) if value else None

    return DocMeta(
        doc_id=data["doc_id"],
        doc_kind=data["doc_kind"],
        title=data["title"],
        path=data["path"],
        number=data.get("number"),
        doc_date=d("doc_date"),
        contract_kind=data.get("contract_kind"),
        counterparty=data.get("counterparty"),
        counterparty_inn=data.get("counterparty_inn"),
        parent_number=data.get("parent_number"),
        parent_date=d("parent_date"),
        parent_doc_id=data.get("parent_doc_id"),
        effective_date=d("effective_date"),
    )


def build_registry_db(csv_path: Path, db_path: Path) -> int:
    """Реестр из CSV (UTF-8 с BOM, «;») в SQLite. Возвращает число строк."""
    db_path.unlink(missing_ok=True)
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    con = sqlite3.connect(db_path)
    try:
        columns = ", ".join(f"{name} {kind}" for name, kind in REGISTRY_COLUMNS.items())
        con.execute(f"CREATE TABLE registry ({columns})")
        placeholders = ", ".join("?" for _ in REGISTRY_COLUMNS)

        def value(name: str, raw: str) -> Any:
            kind = REGISTRY_COLUMNS[name]
            if raw == "":
                return None
            if kind == "REAL":
                return float(raw)
            if kind == "INTEGER":
                return int(raw)
            return raw

        con.executemany(
            f"INSERT INTO registry VALUES ({placeholders})",
            [[value(n, row.get(n, "")) for n in REGISTRY_COLUMNS] for row in rows],
        )
        con.commit()
    finally:
        con.close()
    return len(rows)


def _batches(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass
class IndexReport:
    chunks: int
    documents: int
    registry_rows: int
    already_embedded: int
    embedded_now: int
    seconds: float
    complete: bool


ProgressCallback = Callable[[int, int], None]


def build_index(
    config: RagConfig,
    embedder_factory: Callable[[], Embedder],
    *,
    rebuild: bool = False,
    write_batch: int = 64,
    on_start: Callable[[int, int], None] | None = None,
    on_progress: ProgressCallback | None = None,
    stop_after: int | None = None,
) -> IndexReport:
    """Нарезать корпус, записать куски и реестр, досчитать недостающие векторы.

    `on_start(всего, уже есть)` и `on_progress(сделано_сейчас, пачка)` — для прогресс-бара.
    `stop_after` — остановиться после стольких новых векторов (проверка продолжения).
    """
    started = time.perf_counter()
    paths = IndexPaths(config.index_dir)
    current = fingerprint(config)
    saved: dict[str, Any] = {}
    if paths.manifest.is_file():
        saved = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != current and not rebuild:
            raise IndexMismatchError(
                "индекс собран по другому корпусу, модели или нарезке; "
                "пересоберите: uv run docai rag index --rebuild"
            )
    # Пересборка — в новую коллекцию: на Windows файлы Chroma могут быть открыты этим же
    # процессом (или сервисом), и папку не удалить; старые коллекции удаляются по возможности.
    fresh = rebuild or not saved.get("collection")
    collection_name = (
        f"{COLLECTION_PREFIX}{uuid.uuid4().hex[:10]}" if fresh else saved["collection"]
    )
    paths.root.mkdir(parents=True, exist_ok=True)

    metas, chunks = chunk_corpus(
        config.corpus_dir,
        our_names=config.our_company.names,
        our_inns=config.our_company.inns,
        max_chars=config.max_chunk_chars,
    )
    with paths.chunks.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            row = chunk.to_json()
            row["lemmas"] = lemmas(chunk.index_text())
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with paths.docs.open("w", encoding="utf-8", newline="\n") as fh:
        for meta in metas.values():
            fh.write(json.dumps(meta_to_json(meta), ensure_ascii=False) + "\n")
    registry_rows = (
        build_registry_db(config.registry_csv, paths.registry)
        if config.registry_csv.is_file()
        else 0
    )

    def write_manifest(embedded: int) -> None:
        manifest = {
            "fingerprint": current,
            "chunks": len(chunks),
            "documents": len(metas),
            "registry_rows": registry_rows,
            "embedded": embedded,
            "complete": embedded >= len(chunks),
            "collection": collection_name,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        paths.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    client = chroma_client(paths.chroma)
    if fresh:
        for old in client.list_collections():
            old_name = getattr(old, "name", old)
            if old_name != collection_name:
                # Файлы заняты — коллекция останется на диске, но не используется.
                with contextlib.suppress(Exception):
                    client.delete_collection(old_name)
    collection = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    existing = set(collection.get(include=[])["ids"])
    known = {c.chunk_id for c in chunks}
    stale = sorted(existing - known)
    if stale:  # кусков больше нет в корпусе (при том же отпечатке не бывает, но на всякий)
        collection.delete(ids=stale)
        existing -= set(stale)
    todo = [c for c in chunks if c.chunk_id not in existing]
    # Похожие по длине куски — в одну пачку: меньше паддинга на CPU.
    todo.sort(key=lambda c: len(c.index_text()))
    already = len(existing)
    write_manifest(already)
    if on_start:
        on_start(len(chunks), already)

    done = 0
    embedder: Embedder | None = None
    try:
        for batch in _batches(todo, write_batch):
            if stop_after is not None and done >= stop_after:
                break
            if embedder is None:
                embedder = embedder_factory()
            vectors = embedder.embed_documents([c.index_text() for c in batch])
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors,
                metadatas=[c.metadata() for c in batch],
            )
            done += len(batch)
            if on_progress:
                on_progress(done, len(batch))
    finally:
        write_manifest(already + done)
    return IndexReport(
        chunks=len(chunks),
        documents=len(metas),
        registry_rows=registry_rows,
        already_embedded=already,
        embedded_now=done,
        seconds=round(time.perf_counter() - started, 1),
        complete=already + done >= len(chunks),
    )


def load_manifest(root: Path) -> dict[str, Any]:
    paths = IndexPaths(root)
    if not paths.manifest.is_file():
        raise IndexNotReadyError(f"индекса нет в {root}; соберите: uv run docai rag index")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))
