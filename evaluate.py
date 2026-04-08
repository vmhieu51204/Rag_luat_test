"""
evaluate.py
===========
Embeds train/test folders into separate ChromaDB directories, then
queries the train DB with test documents and measures how well the
retrieved train documents' articles (Cac_Dieu_Quyet_Dinh) overlap with
the test document ground truth.

Evaluation logic
----------------
    1. Run pipeline.py on train_dir -> train_db_dir.
    2. Run pipeline.py on test_dir  -> test_db_dir.
    3. For each test document that has Cac_Dieu_Quyet_Dinh:
             - build query from text fields
             - retrieve top-K chunks from train_db_dir
             - union retrieved train-doc article IDs
             - compute precision/recall/F1 against test doc ground truth
    4. Macro-average all metrics.

Usage
-----
    python evaluate.py \
    --train_dir   ./chunk/Chuong_XXII_chunked/train \
    --test_dir    ./chunk/Chuong_XXII_chunked/test \
    --train_db_dir ./output/chroma_db_train \
    --test_db_dir  ./output/chroma_db_test \
        --top_k       10 \
        --results_out ./output/eval_results.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Fields used to build the query text (same order as pipeline.py)
CONTENT_FIELDS  = ["Summary"]
ID_FIELD        = "Ma_Ban_An"
ARTICLES_FIELD  = "Cac_Dieu_Quyet_Dinh"
COLLECTION_NAME = "legal_chunks_vn"
MODEL_NAME      = "BAAI/bge-m3"
DEVICE          = "cuda"

# Helpers
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] Run: pip install sentence-transformers")
        sys.exit(1)
    print(f"  Loading {MODEL_NAME} on {DEVICE} ...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return model


def load_chroma(db_dir: str):
    try:
        import chromadb
    except ImportError:
        print("[ERROR] Run: pip install chromadb")
        sys.exit(1)
    client     = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def run_pipeline_embedding(input_dir: str, db_dir: str) -> None:
    """Run pipeline.py run to embed all JSON files from input_dir into db_dir."""
    pipeline_path = Path(__file__).with_name("pipeline.py")
    cmd = [
        sys.executable,
        str(pipeline_path),
        "run",
        "--input_dir",
        input_dir,
        "--db_dir",
        db_dir,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_article_signatures(articles) -> set[str]:
    """
    Given `Cac_Dieu_Quyet_Dinh` which may be a dict { defendant: [ [dieu, khoan, diem], ... ] },
    collapse into a set of unique string signatures for evaluation.
    """
    signatures = set()
    if isinstance(articles, dict):
        for val_list in articles.values():
            if isinstance(val_list, list):
                for item in val_list:
                    if isinstance(item, list):
                        signatures.add("-".join(str(i) for i in item))
                    else:
                        signatures.add(str(item))
            else:
                signatures.add(str(val_list))
    elif isinstance(articles, list):
        for a in articles:
            if isinstance(a, list):
                signatures.add("-".join(str(i) for i in a))
            else:
                signatures.add(str(a))
    elif articles:
        signatures.add(str(articles))
    return signatures

def load_articles_index(raw_dir: Path) -> dict[str, set[str]]:
    """
    Read every raw JSON in raw_dir and build:
        doc_id -> set of article numbers (Cac_Dieu_Quyet_Dinh)
    Only includes docs that actually have the field.
    """
    index = {}
    for f in raw_dir.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        doc_id   = data.get(ID_FIELD, f.stem)
        articles = data.get(ARTICLES_FIELD)
        if articles:
            index[doc_id] = extract_article_signatures(articles)
    return index

def load_test_docs(test_dir: Path) -> list[dict]:
    """
    Build one record per test document from raw JSON files.
    Returns list of:
        {doc_id, query_text, ground_truth_articles}
    Skips docs without Cac_Dieu_Quyet_Dinh.
    """
    test_docs = []
    skipped = []

    for f in sorted(test_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        doc_id = data.get(ID_FIELD, f.stem)
        articles = data.get(ARTICLES_FIELD)
        if not articles:
            skipped.append(doc_id)
            continue

        parts = []
        for field in CONTENT_FIELDS:
            val = (data.get(field) or "").strip()
            if val:
                parts.append(val)
        query_text = "\n\n".join(parts)
        if not query_text.strip():
            skipped.append(doc_id)
            continue

        test_docs.append({
            "doc_id":            doc_id,
            "query_text":        query_text,
            "ground_truth":      extract_article_signatures(articles),
        })

    if skipped:
        print(f"  [INFO] Skipped {len(skipped)} test doc(s) with missing data: "
              f"{skipped}")
    return test_docs

# Metrics
def precision_recall_f1(predicted: set[str], ground_truth: set[str]) -> tuple[float, float, float]:
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0
    tp        = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall    = tp / len(ground_truth)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

# Main evaluation
def evaluate(
    train_dir: str,
    test_dir: str,
    train_db_dir: str,
    test_db_dir: str,
    top_k: int,
    results_out: str,
    skip_embedding: bool,
) -> None:

    train_path = Path(train_dir)
    test_path = Path(test_dir)

    if not skip_embedding:
        print("\n── Step 1: Embed train folder ───────────────────────────")
        run_pipeline_embedding(train_dir, train_db_dir)

        print("\n── Step 2: Embed test folder ────────────────────────────")
        run_pipeline_embedding(test_dir, test_db_dir)
    else:
        print("\n── Step 1: Skipping embedding (--skip_embedding) ───────")

    print("\n── Step 3: Load model ───────────────────────────────────")
    model = load_model()

    print("\n── Step 4: Load train ChromaDB ──────────────────────────")
    collection = load_chroma(train_db_dir)
    print(f"  Train collection has {collection.count()} documents")

    print("\n── Step 5: Load test documents ──────────────────────────")
    test_docs = load_test_docs(test_path)
    print(f"  {len(test_docs)} test doc(s) with ground-truth articles")

    if not test_docs:
        print("\n  [WARN] No evaluable test documents found.")
        print(f"  Make sure test_dir files contain '{ARTICLES_FIELD}' and query text fields.")
        return

    # Build train doc_id -> articles lookup to resolve retrieved chunk owners.
    train_articles_index = load_articles_index(train_path)

    print(f"\n── Step 6: Query + evaluate (top_k={top_k}) ─────────────")
    per_doc_results = []

    for doc in test_docs:
        doc_id      = doc["doc_id"]
        query_text  = doc["query_text"]
        ground_truth = doc["ground_truth"]

        # Embed the query
        vec = model.encode([query_text], normalize_embeddings=True).tolist()

        # Retrieve top-K chunks; exclude chunks from the query doc itself
        results = collection.query(
            query_embeddings=vec,
            n_results=top_k,
            include=["metadatas", "distances"],
            where={"doc_id": {"$ne": doc_id}},   # exclude self
        )

        # Collect unique retrieved doc_ids in rank order
        retrieved_doc_ids: list[str] = []
        seen = set()
        for meta in results["metadatas"][0]:
            rid = meta["doc_id"]
            if rid not in seen:
                seen.add(rid)
                retrieved_doc_ids.append(rid)

        # Union of articles across all retrieved documents
        predicted_articles: set[str] = set()
        for rid in retrieved_doc_ids:
            predicted_articles |= train_articles_index.get(rid, set())

        p, r, f1 = precision_recall_f1(predicted_articles, ground_truth)

        result = {
            "doc_id":              doc_id,
            "ground_truth":        sorted(ground_truth),
            "retrieved_doc_ids":   retrieved_doc_ids,
            "predicted_articles":  sorted(predicted_articles),
            "matched_articles":    sorted(predicted_articles & ground_truth),
            "missed_articles":     sorted(ground_truth - predicted_articles),
            "extra_articles":      sorted(predicted_articles - ground_truth),
            "precision":           round(p,  4),
            "recall":              round(r,  4),
            "f1":                  round(f1, 4),
        }
        per_doc_results.append(result)

        print(f"\n  doc: {doc_id}")
        print(f"    ground truth : {sorted(ground_truth)}")
        print(f"    predicted    : {sorted(predicted_articles)}")
        print(f"    matched      : {sorted(predicted_articles & ground_truth)}")
        print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

    # Macro-average
    n = len(per_doc_results)
    macro_p  = sum(r["precision"] for r in per_doc_results) / n
    macro_r  = sum(r["recall"]    for r in per_doc_results) / n
    macro_f1 = sum(r["f1"]        for r in per_doc_results) / n

    print("\n── Results ──────────────────────────────────────────────")
    print(f"  Evaluated on : {n} test document(s)")
    print(f"  top_k        : {top_k}")
    print(f"  Macro P      : {macro_p:.4f}")
    print(f"  Macro R      : {macro_r:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")

    # Save results
    output = {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_db_dir": train_db_dir,
        "test_db_dir": test_db_dir,
        "top_k":     top_k,
        "n_docs":    n,
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
        "macro_f1":        round(macro_f1, 4),
        "per_doc":         per_doc_results,
    }
    out_path = Path(results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(f"\n  Full results saved to: {out_path}")



# CLI


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Embed train/test folders into separate DBs and evaluate retrieval "
            "quality using Cac_Dieu_Quyet_Dinh overlap"
        )
    )
    parser.add_argument(
        "--train_dir",
        required=True,
        help="Folder containing train JSON files",
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="Folder containing test JSON files",
    )
    parser.add_argument(
        "--train_db_dir",
        default="./output/chroma_db_train",
        help="ChromaDB output directory for embedded train set",
    )
    parser.add_argument(
        "--test_db_dir",
        default="./output/chroma_db_test",
        help="ChromaDB output directory for embedded test set",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--results_out",
        default="./output/eval_results.json",
        help="Where to write the JSON results file",
    )
    parser.add_argument(
        "--skip_embedding",
        action="store_true",
        help="Skip running pipeline.py embedding and evaluate using existing DBs",
    )
    args = parser.parse_args()
    evaluate(
        args.train_dir,
        args.test_dir,
        args.train_db_dir,
        args.test_db_dir,
        args.top_k,
        args.results_out,
        args.skip_embedding,
    )


if __name__ == "__main__":
    main()
#  /home/hieujayce/Downloads/complete_repo/.venv/bin/python evaluate.py --train_dir ./chunk/Chuong_XXII_chunked/train --test_dir ./chunk/Chuong_XXII_chunked/synth/split --train_db_dir ./output/chroma_db_train --test_db_dir ./output/chroma_db_test_split --top_k 5 --results_out ./output/eval_results_synth_split.json
# python evaluate.py --train_dir ./chunk/Chuong_XXII_chunked/train --test_dir ./chunk/Chuong_XXII_chunked/test --train_db_dir ./output/chroma_db_train --test_db_dir ./output/chroma_db_test --top_k 5 --results_out eval_results.json