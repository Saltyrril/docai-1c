"""Настройки RAG-конвейера (P19). Значения по умолчанию совпадают с `configs/rag/pipeline.yaml`."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from docai.extract.config import OllamaOptions


class OurCompany(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # «Наша» сторона договоров: контрагентом считается другая сторона (D61).
    names: list[str] = Field(default_factory=lambda: ["АО «Тайга-Строй»"])
    inns: list[str] = Field(default_factory=lambda: ["7704037472"])


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 16
    max_seq_length: int = 512
    # Инструкция для запроса (у Qwen3-Embedding запрос с инструкцией, документ — без).
    query_instruction: str = (
        "Instruct: Given a question in Russian about company contracts or internal "
        "regulations, retrieve the clause that answers it\nQuery: "
    )
    # Потоки PyTorch на CPU: по числу физических ядер (Ryzen 5 5600H — 6).
    threads: int = 6


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Выбор по сравнению на 50 вопросах: качество почти равное, bge в 2,5 раза быстрее (D67).
    kind: Literal["qwen3", "bge", "none"] = "bge"
    model: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 384
    batch_size: int = 8
    # Qwen3-Reranker: считать общий префикс (инструкция + вопрос) один раз (D67).
    prefix_cache: bool = True
    instruction: str = (
        "Given a question in Russian about company contracts or internal regulations, "
        "judge whether the clause answers it"
    )


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Какие поиски используются: оба с RRF (умолчание) или один из них — для абляций P20.
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"
    bm25_top_k: int = 30
    vector_top_k: int = 30
    rrf_k: int = 60
    # Сколько кандидатов после RRF идут в реранкер и сколько лучших — в контекст модели.
    rerank_candidates: int = 12
    # Первые N из BM25 и из векторов попадают к реранкеру в любом случае.
    per_list_guaranteed: int = 4
    context_top_k: int = 5
    # Пункты доп. соглашений, которые подмешиваются к найденным пунктам договора.
    max_amendment_chunks: int = 4
    use_filters: bool = True


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Как у извлечения (D48): при одной модели в VRAM другая квантизация — перезагрузка (D71).
    model: str = "qwen3.5:2b-q4_K_M"
    ollama_url: str = "http://127.0.0.1:11434"
    think: bool = False
    keep_alive: str = "5m"
    timeout_s: float = 120.0
    # num_ctx — как у извлечения (8192): при другом значении Ollama перезагружает модель.
    options: OllamaOptions = Field(
        default_factory=lambda: OllamaOptions(num_ctx=8192, num_predict=600)
    )


class SqlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rows: int = 50
    # Предел шагов виртуальной машины SQLite на один запрос (защита от тяжёлых запросов).
    max_vm_steps: int = 5_000_000


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_dir: Path = Path("data/rag/corpus")
    registry_csv: Path = Path("data/rag/registry.csv")
    index_dir: Path = Path("data/rag_index")
    # «Сегодня» для вопросов «за этот год», «последний документ» (дата набора P18, D64).
    today: str = "2026-11-19"
    # Маршрутизатор «агрегат по реестру → SQL». Выключен — всё идёт в RAG (абляция P20).
    use_router: bool = True
    max_chunk_chars: int = 1200
    our_company: OurCompany = Field(default_factory=OurCompany)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    sql: SqlConfig = Field(default_factory=SqlConfig)


DEFAULT_PIPELINE_CONFIG = Path("configs/rag/pipeline.yaml")


def load_rag_config(path: Path | None = None) -> RagConfig:
    """Конфиг из YAML; без пути — `configs/rag/pipeline.yaml`, если он есть, иначе умолчания."""
    if path is None:
        path = DEFAULT_PIPELINE_CONFIG if DEFAULT_PIPELINE_CONFIG.is_file() else None
    if path is None:
        return RagConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RagConfig.model_validate(data)
