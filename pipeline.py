"""
pipeline.py
===========
Embedding pipeline: chunk all JSON files in a folder and embed into ChromaDB.

Usage
-----
    # Embed all JSON files from a folder
    python pipeline.py run \
        --input_dir ./json_folder \
        --db_dir    ./output/chroma_db

    # Query the DB after embedding
    python pipeline.py query \
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

CONTENT_FIELDS  = ["Summary"]   # text fields to chunk
ID_FIELD        = "Ma_Ban_An"
MAX_CHUNK_CHARS = 1500        # max chars per chunk; None = one chunk per field

MODEL_NAME      = "BAAI/bge-m3"
COLLECTION_NAME = "legal_chunks_vn"
BATCH_SIZE      = 32          # lower to 8 if RAM is tight
DEVICE          = "cuda"       # "cuda" if GPU available


# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------

def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks ≤ max_chars, breaking on sentence boundaries."""
    if not text or not text.strip():
        return []
    if max_chars is None or len(text) <= max_chars:
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

def chunk_document(data: dict, source_file: str) -> list[dict]:
    """Produce chunk records from a single JSON document."""
    doc_id = data.get(ID_FIELD, Path(source_file).stem)
    chunks = []
    for field in CONTENT_FIELDS:
        raw = (data.get(field) or "").strip()
        if not raw:
            continue
        sub_chunks = split_text(raw, MAX_CHUNK_CHARS)
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

def chunk_folder(input_dir: Path) -> tuple[list[dict], list[dict]]:
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
        chunks = chunk_document(data, f.name)
        doc_id = data.get(ID_FIELD, f.stem)
        all_docs.append({"doc_id": doc_id, "source_file": f.name, "n_chunks": len(chunks)})
        all_chunks.extend(chunks)

    print(f"  Loaded {len(files)} files → {len(all_chunks)} chunks")
    return all_docs, all_chunks

# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------

def load_model():

    from sentence_transformers import SentenceTransformer

    print(f"  Loading {MODEL_NAME} on {DEVICE} ...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model

def load_chroma(db_dir: str):
    import chromadb
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection

def embed_chunks(chunks: list[dict], db_dir: str) -> None:
    """Embed chunks and upsert into ChromaDB."""
    if not chunks:
        print("  No chunks to embed.")
        return

    model = load_model()
    _, collection = load_chroma(db_dir)

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
    for start in range(0, len(texts_new), BATCH_SIZE):
        batch = texts_new[start : start + BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.extend(vecs.tolist())
        done = min(start + BATCH_SIZE, len(texts_new))
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

    print(f"  ✅ ChromaDB '{COLLECTION_NAME}' now has {collection.count()} documents.")


# ---------------------------------------------------------------------------
# 3. Query
# ---------------------------------------------------------------------------

def query(db_dir: str, text: str, top_k: int = 5) -> None:
    print(f'\nQuerying: "{text}"  (top {top_k})\n')
    model = load_model()
    _, collection = load_chroma(db_dir)

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
# 4. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_dir: str, db_dir: str) -> None:
    input_path  = Path(input_dir)

    print("\n── Step 1: Chunking ─────────────────────────────────────")
    docs, chunks = chunk_folder(input_path)
    if not docs:
        return

    print("\n── Step 2: Embedding all chunks ─────────────────────────")
    embed_chunks(chunks, db_dir)

    print("\n── Done ─────────────────────────────────────────────────")
    print(f"  Embedded all chunks in : {db_dir}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk all JSON files in a folder and embed into ChromaDB"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Embed all JSON files from input_dir")
    p_run.add_argument("--input_dir",  required=True, help="Folder with raw JSON files")
    p_run.add_argument("--db_dir",     default="./output/chroma_db")

    p_q = sub.add_parser("query", help="Query the ChromaDB collection")
    p_q.add_argument("--db_dir", default="./output/chroma_db")
    p_q.add_argument("--text",   required=True)
    p_q.add_argument("--top_k",  type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "run":
        run_pipeline(args.input_dir, args.db_dir)
    elif args.cmd == "query":
        query(args.db_dir, args.text, args.top_k)


if __name__ == "__main__":
    main()