"""
embedding_store.py
==================
Standalone embedding pipeline: chunk JSON files, embed with SentenceTransformer,
store in ChromaDB, and query.

Provides both a Python API (importable functions) and a CLI.

Usage
-----
    # Embed all JSON files from a folder
    python embedding_store.py run \
        --input_dir ./json_folder \
        --db_dir    ./output/chroma_db \
        --content_fields Summary \
        --model_name BAAI/bge-m3 \
        --max_chunk_chars 1500

    # Query the DB after embedding
    python embedding_store.py query \
        --db_dir ./output/chroma_db \
        --text   "tinh tiet giam nhe trach nhiem hinh su" \
        --top_k  5
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from rag.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MODEL_NAME,
    ID_FIELD as CONFIG_ID_FIELD,
)

# ── Default configuration ──────────────────────────────────────────────────
CONTENT_FIELDS  = []       # text fields to chunk
ID_FIELD        = CONFIG_ID_FIELD
MAX_CHUNK_CHARS = DEFAULT_MAX_CHUNK_CHARS  # max chars per chunk; 0 = one chunk per field
MODEL_NAME      = DEFAULT_MODEL_NAME
COLLECTION_NAME = DEFAULT_COLLECTION_NAME
BATCH_SIZE      = DEFAULT_BATCH_SIZE        # lower to 8 if RAM is tight
DEVICE          = DEFAULT_DEVICE            # "cuda" if GPU available


# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------

def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks ≤ max_chars, breaking on sentence boundaries."""
    if not text or not text.strip():
        return []
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return [text.strip()]

    sentence_end = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end.split(text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars].strip())
                sentence = sentence[max_chars:]
            current = sentence.strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_document(
    data: dict,
    source_file: str,
    content_fields: list[str] | None = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    """Produce chunk records from a single JSON document."""
    fields = content_fields or CONTENT_FIELDS
    doc_id = data.get(ID_FIELD, Path(source_file).stem)
    chunks = []
    for field in fields:
        raw = (data.get(field) or "").strip()
        if not raw:
            continue
        sub_chunks = split_text(raw, max_chunk_chars)
        for idx, text in enumerate(sub_chunks):
            chunks.append({
                "doc_id":                 doc_id,
                "source_file":            source_file,
                "field":                  field,
                "chunk_index":            idx,
                "total_chunks_for_field": len(sub_chunks),
                "text":                   text,
            })
    return chunks


def chunk_folder(
    input_dir: Path,
    content_fields: list[str] | None = None,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> tuple[list[dict], list[dict]]:
    """
    Read all JSON files in input_dir.
    Returns (all_docs_metadata, all_chunks) where each doc_meta has keys:
        doc_id, source_file, n_chunks
    """
    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"  [WARN] No JSON files found in {input_dir}")
        return [], []

    all_docs, all_chunks = [], []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        chunks = chunk_document(
            data, f.name,
            content_fields=content_fields,
            max_chunk_chars=max_chunk_chars,
        )
        doc_id = data.get(ID_FIELD, f.stem)
        all_docs.append({"doc_id": doc_id, "source_file": f.name, "n_chunks": len(chunks)})
        all_chunks.extend(chunks)

    print(f"  Loaded {len(files)} files → {len(all_chunks)} chunks")
    return all_docs, all_chunks


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------

def load_model(model_name: str = MODEL_NAME, device: str = DEVICE):
    """Load a SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] Run: pip install sentence-transformers")
        sys.exit(1)
    print(f"  Loading {model_name} on {device} ...")
    model = SentenceTransformer(model_name, device=device)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def load_chroma(db_dir: str, collection_name: str = COLLECTION_NAME, *, create: bool = True):
    """
    Open (or create) a ChromaDB persistent collection.

    Parameters
    ----------
    db_dir : str
        Path to the ChromaDB directory.
    collection_name : str
        Name of the collection.
    create : bool
        If True, create the collection if it doesn't exist (get_or_create).
        If False, only get an existing collection (raises if missing).

    Returns
    -------
    collection : chromadb.Collection
    """
    try:
        import chromadb
    except ImportError:
        print("[ERROR] Run: pip install chromadb")
        sys.exit(1)
    client = chromadb.PersistentClient(path=db_dir)
    if create:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        collection = client.get_collection(name=collection_name)
    return collection


def embed_chunks(
    chunks: list[dict],
    db_dir: str,
    model_name: str = MODEL_NAME,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Embed chunks and upsert into ChromaDB."""
    if not chunks:
        print("  No chunks to embed.")
        return

    model = load_model(model_name=model_name, device=device)
    collection = load_chroma(db_dir, collection_name=collection_name, create=True)

    ids       = [f"{c['doc_id']}__{c['field']}__{c['chunk_index']:03d}" for c in chunks]
    texts     = [c["text"] for c in chunks]
    metadatas = [{
        "doc_id":                 c["doc_id"],
        "source_file":            c["source_file"],
        "field":                  c["field"],
        "chunk_index":            c["chunk_index"],
        "total_chunks_for_field": c["total_chunks_for_field"],
    } for c in chunks]

    # Skip already-embedded chunks (safe to re-run / resume)
    existing  = set(collection.get(ids=ids)["ids"])
    new_mask  = [i for i, cid in enumerate(ids) if cid not in existing]
    if not new_mask:
        print(f"  All {len(chunks)} chunks already in DB — nothing to do.")
        return
    print(f"  {len(existing)} already in DB, embedding {len(new_mask)} new chunks ...")

    ids_new   = [ids[i]       for i in new_mask]
    texts_new = [texts[i]     for i in new_mask]
    meta_new  = [metadatas[i] for i in new_mask]

    t0, all_vecs = time.time(), []
    for start in range(0, len(texts_new), batch_size):
        batch = texts_new[start : start + batch_size]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.extend(vecs.tolist())
        done = min(start + batch_size, len(texts_new))
        print(f"  {done}/{len(texts_new)} — {time.time()-t0:.1f}s", end="\r")

    print(f"\n  Embedding done in {time.time()-t0:.1f}s")

    CHROMA_BATCH = 500
    for start in range(0, len(ids_new), CHROMA_BATCH):
        sl = slice(start, start + CHROMA_BATCH)
        collection.upsert(
            ids=ids_new[sl],
            embeddings=all_vecs[sl],
            documents=texts_new[sl],
            metadatas=meta_new[sl],
        )

    print(f"  ✅ ChromaDB '{collection_name}' now has {collection.count()} documents.")


# ---------------------------------------------------------------------------
# 3. Query
# ---------------------------------------------------------------------------

def query(
    db_dir: str,
    text: str,
    top_k: int = 5,
    model_name: str = MODEL_NAME,
    device: str = DEVICE,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Query the ChromaDB collection and print results."""
    print(f'\nQuerying: "{text}"  (top {top_k})\n')
    model = load_model(model_name=model_name, device=device)
    collection = load_chroma(db_dir, collection_name=collection_name, create=False)

    vec     = model.encode([text], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
    ):
        score = 1 - dist
        print(f"  [{rank}] score={score:.4f}  doc={meta['doc_id']}  "
              f"field={meta['field']}  chunk={meta['chunk_index']}")
        print(f"       {doc[:200].replace(chr(10), ' ')}...")
        print()


# ---------------------------------------------------------------------------
# 4. Full pipeline  (chunk → embed)
# ---------------------------------------------------------------------------

def run_pipeline(
    input_dir: str,
    db_dir: str,
    content_fields: list[str] | None = None,
    model_name: str = MODEL_NAME,
    device: str = DEVICE,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    batch_size: int = BATCH_SIZE,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Chunk all JSON files in input_dir and embed into ChromaDB at db_dir."""
    input_path = Path(input_dir)
    fields = content_fields or CONTENT_FIELDS
    print(f"  Using content fields: {fields}")

    print("\n── Step 1: Chunking ─────────────────────────────────────")
    docs, chunks = chunk_folder(
        input_path,
        content_fields=fields,
        max_chunk_chars=max_chunk_chars,
    )
    if not docs:
        return

    print("\n── Step 2: Embedding all chunks ─────────────────────────")
    embed_chunks(
        chunks, db_dir,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        collection_name=collection_name,
    )

    print("\n── Done ─────────────────────────────────────────────────")
    print(f"  Embedded all chunks in : {db_dir}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk JSON files, embed with SentenceTransformer, store/query ChromaDB"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── run ────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Embed all JSON files from input_dir")
    p_run.add_argument("--input_dir",  required=True, help="Folder with raw JSON files")
    p_run.add_argument("--db_dir",     default="./output/chroma_db")
    p_run.add_argument(
        "--content_fields",
        default=None,
        help="Comma-separated list of JSON fields to chunk & embed "
             f"(default: {','.join(CONTENT_FIELDS)})",
    )
    p_run.add_argument(
        "--model_name", default=MODEL_NAME,
        help=f"SentenceTransformer model name (default: {MODEL_NAME})",
    )
    p_run.add_argument(
        "--device", default=DEVICE,
        help=f"Device for model inference (default: {DEVICE})",
    )
    p_run.add_argument(
        "--max_chunk_chars", type=int, default=MAX_CHUNK_CHARS,
        help=f"Max characters per chunk; 0 = no splitting (default: {MAX_CHUNK_CHARS})",
    )
    p_run.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Embedding batch size (default: {BATCH_SIZE})",
    )
    p_run.add_argument(
        "--collection_name", default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME})",
    )

    # ── query ──────────────────────────────────────────────────────────────
    p_q = sub.add_parser("query", help="Query the ChromaDB collection")
    p_q.add_argument("--db_dir", default="./output/chroma_db")
    p_q.add_argument("--text",   required=True)
    p_q.add_argument("--top_k",  type=int, default=5)
    p_q.add_argument("--model_name", default=MODEL_NAME)
    p_q.add_argument("--device", default=DEVICE)
    p_q.add_argument("--collection_name", default=COLLECTION_NAME)

    args = parser.parse_args()

    if args.cmd == "run":
        fields = (
            [f.strip() for f in args.content_fields.split(",")]
            if args.content_fields else None
        )
        run_pipeline(
            args.input_dir,
            args.db_dir,
            content_fields=fields,
            model_name=args.model_name,
            device=args.device,
            max_chunk_chars=args.max_chunk_chars,
            batch_size=args.batch_size,
            collection_name=args.collection_name,
        )
    elif args.cmd == "query":
        query(
            args.db_dir, args.text, args.top_k,
            model_name=args.model_name,
            device=args.device,
            collection_name=args.collection_name,
        )


if __name__ == "__main__":
    main()
