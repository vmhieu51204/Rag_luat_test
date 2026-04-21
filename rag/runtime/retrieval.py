"""Persistent retrieval runtime that keeps model and DB loaded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.core.embeddings import load_chroma, load_model, run_pipeline


@dataclass
class RetrievalRuntimeConfig:
    model_name: str
    device: str
    train_db_dir: str
    collection_name: str


class RetrievalRuntime:
    """Stateful runtime for retrieval operations.

    The runtime caches the embedding model and train collection handle so
    repeated queries do not reload heavyweight components.
    """

    def __init__(self, config: RetrievalRuntimeConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._train_collection: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = load_model(model_name=self.config.model_name, device=self.config.device)
        return self._model

    @property
    def train_collection(self) -> Any:
        if self._train_collection is None:
            self._train_collection = load_chroma(
                self.config.train_db_dir,
                collection_name=self.config.collection_name,
                create=False,
            )
        return self._train_collection

    def ensure_train_index(
        self,
        *,
        train_dir: str,
        content_fields: list[str] | None,
        max_chunk_chars: int,
        batch_size: int,
    ) -> None:
        run_pipeline(
            train_dir,
            self.config.train_db_dir,
            content_fields=content_fields,
            model_name=self.config.model_name,
            device=self.config.device,
            max_chunk_chars=max_chunk_chars,
            batch_size=batch_size,
            collection_name=self.config.collection_name,
        )

    def encode_query(self, query_text: str) -> list[list[float]]:
        return self.model.encode([query_text], normalize_embeddings=True).tolist()

    def query_train(
        self,
        *,
        query_text: str,
        top_k: int,
        exclude_doc_id: str | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        vec = self.encode_query(query_text)
        where = {"doc_id": {"$ne": exclude_doc_id}} if exclude_doc_id else None
        kwargs: dict[str, Any] = {
            "query_embeddings": vec,
            "n_results": top_k,
            "include": include or ["metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        return self.train_collection.query(**kwargs)

    def train_doc_count(self) -> int:
        return self.train_collection.count()
