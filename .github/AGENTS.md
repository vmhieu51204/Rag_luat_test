---
description: "Workspace agents for complete_repo legal caselaw analysis"
---

# Available Custom Agents

## Legal Case Retrieval

**ID**: `legal-case-retrieval`  
**File**: `.github/agents/legal-case-retrieval.agent.md`

Specialized agent for Vietnamese legal caselaw semantic search and retrieval using your RAG infrastructure.

### LLM & Embedding Models
- **Agent Reasoning**: VS Code Copilot (Claude Haiku 4.5)
- **Embeddings**: `BAAI/bge-m3` (multilingual, 384-dim vectors)
- **Vector Database**: ChromaDB at `./output/chroma_db_train/legal_chunks_vn`
- **Backend**: Pre-built `rag/runtime/retrieval.py::RetrievalRuntime`

### Use This Agent When:
- Finding similar past cases for a given test case
- Analyzing case law patterns across the corpus
- Performing semantic retrieval on legal summaries
- Needing JSON-structured case metadata and similarity scores
- Working with Synthetic_summary fields for retrieval

### Key Features:
- **BAAI/bge-m3** embeddings (no OpenAI calls, fully local)
- Top 5 ranked results by ChromaDB distance → similarity
- Extracts metadata: charges, defendants, verdicts, dates
- Confined to chunk/train folder for historical cases
- JSON output format with confidence scores (0-1)
- Integrates with existing RAG infrastructure

### Example Prompt:
```
Find 5 similar cases for 01-03-2024-Gia_Lai-2ta1457995t1cvn using the RAG pipeline
```

### Python Integration
```python
from rag.runtime.retrieval import RetrievalRuntime, RetrievalRuntimeConfig
from rag.config import DEFAULT_MODEL_NAME, DEFAULT_TRAIN_DB_DIR, DEFAULT_COLLECTION_NAME

config = RetrievalRuntimeConfig(
    model_name=DEFAULT_MODEL_NAME,
    device="cuda",
    train_db_dir=DEFAULT_TRAIN_DB_DIR,
    collection_name=DEFAULT_COLLECTION_NAME
)
runtime = RetrievalRuntime(config)
results = runtime.query_train(query_text="...", top_k=5)
```

---

## Configuration Notes

### RAG System Settings (rag/config.py)
- **Embedding Model**: `BAAI/bge-m3` 
- **Model Dimension**: 384
- **ChromaDB Directory**: `./output/chroma_db_train/`
- **Collection Name**: `legal_chunks_vn`
- **Query Fields**: `["Synthetic_summary"]`
- **Case ID Field**: `Ma_Ban_An`
- **Max Chunk Size**: 1500 characters

### Optional: Verdict Generation (not in standard agent)
For generating verdicts, the RAG system can also use:
- **Gemma-4-31b-it** via AiStudio/OpenRouter/OpenAI
- Configured in `rag/llm/` module
- Not enabled in basic retrieval agent by default

### Legal Clause Retrieval (Bonus)
Also available from `rag.core.law_retriever`:
- Retrieve law articles from `law_doc.json`
- Supports signatures: "174-4-a", "51-2", "51"
- Cached lookup, O(1) retrieval

See the agent's `.agent.md` file for detailed documentation and code examples.
