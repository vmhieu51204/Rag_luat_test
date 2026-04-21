"""Centralized configuration defaults for the RAG package."""

from __future__ import annotations

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COLLECTION_NAME = "legal_chunks_vn"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_CHUNK_CHARS = 1500

ID_FIELD = "Ma_Ban_An"
VERDICT_FIELD = "PHAN_QUYET_CUA_TOA_SO_THAM"
LEGAL_BASIS_FIELD = "Can_Cu_Dieu_Luat"
LEGAL_SOURCE_FIELD = "Bo_Luat_Va_Van_Ban_Khac"

TRAIN_EMBED_CONTENT_FIELDS = ["Summary", "Tang_nang", "Giam_nhe"]
TEST_EMBED_CONTENT_FIELDS = ["Synthetic_summary"]
QUERY_CONTENT_FIELDS = ["Synthetic_summary"]

DEFAULT_RESULTS_OUT = "./output/eval_results.json"
DEFAULT_TRAIN_DB_DIR = "./output/chroma_db_train"
DEFAULT_TEST_DB_DIR = "./output/chroma_db_test"

DEFAULT_LAW_DOC_PATH = "law_doc.json"
DEFAULT_VERDICT_OUTPUT_DIR = "output/generated_verdict_from_eval"

DEFAULT_AISTUDIO_MODEL = "gemma-4-31b-it"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
