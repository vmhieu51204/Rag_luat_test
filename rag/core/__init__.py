"""Core retrieval and legal-reference modules."""

from .embeddings import (
    run_pipeline,
    load_model,
    load_chroma,
    query,
    COLLECTION_NAME,
    MODEL_NAME,
    DEVICE,
    MAX_CHUNK_CHARS,
    BATCH_SIZE,
)
from .law_retriever import LawClauseRetriever
from .verdict_labels import extract_label_sets_from_verdict

__all__ = [
    "run_pipeline",
    "load_model",
    "load_chroma",
    "query",
    "COLLECTION_NAME",
    "MODEL_NAME",
    "DEVICE",
    "MAX_CHUNK_CHARS",
    "BATCH_SIZE",
    "LawClauseRetriever",
    "extract_label_sets_from_verdict",
]
