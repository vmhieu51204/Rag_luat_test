"""Retrieval utilities and caching for the Streamlit demo."""

from pathlib import Path
import streamlit as st
from rag.core.law_retriever import LawClauseRetriever
from rag.evaluation.eval_utils import load_articles_index
from rag.runtime.retrieval import RetrievalRuntime, RetrievalRuntimeConfig
from rag.config import DEFAULT_MODEL_NAME, DEFAULT_DEVICE, DEFAULT_COLLECTION_NAME

# Resolve paths relative to the repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR   = REPO_ROOT / "chunk" / "train"
TEST_DIR    = REPO_ROOT / "chunk" / "test"
LAW_JSON    = REPO_ROOT / "raw_law.json"
CASE_DB_DIR = REPO_ROOT / "output" / "reasoning_act_eval" / "case_db"

@st.cache_resource(show_spinner="Loading Law Articles Database...")
def get_law_retriever() -> LawClauseRetriever:
    """Loads and caches the LawClauseRetriever."""
    return LawClauseRetriever(LAW_JSON)

@st.cache_resource(show_spinner="Indexing Train Cases...")
def get_train_articles_index() -> dict:
    """Loads and caches the training case articles index."""
    train_articles_index, _ = load_articles_index(TRAIN_DIR)
    return train_articles_index

@st.cache_resource(show_spinner="Connecting to Case Vector Database...")
def get_case_runtime() -> RetrievalRuntime:
    """Loads and caches the RetrievalRuntime connection to Chroma DB."""
    return RetrievalRuntime(
        RetrievalRuntimeConfig(
            model_name=DEFAULT_MODEL_NAME,
            device="cpu",
            train_db_dir=str(CASE_DB_DIR),
            collection_name=DEFAULT_COLLECTION_NAME,
        )
    )
