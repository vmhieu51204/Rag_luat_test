"""Retrieval evaluation workflow for legal-basis overlap scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MODEL_NAME,
    DEFAULT_COLLECTION_NAME,
    ID_FIELD,
    LEGAL_BASIS_FIELD,
    QUERY_CONTENT_FIELDS,
    TEST_EMBED_CONTENT_FIELDS,
    TRAIN_EMBED_CONTENT_FIELDS,
    VERDICT_FIELD,
)
from rag.core.embeddings import run_pipeline
from rag.core.verdict_labels import extract_label_sets_from_verdict
from rag.runtime.retrieval import RetrievalRuntime, RetrievalRuntimeConfig


def load_articles_index(raw_dir: Path) -> tuple[dict[str, dict[str, set[str]]], list[dict]]:
    """Build train index: doc_id -> {'dieu_only', 'full_signature'}."""
    index: dict[str, dict[str, set[str]]] = {}
    skipped: list[dict] = []

    for f in sorted(raw_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        doc_id = data.get(ID_FIELD, f.stem)
        label_sets, stats, errors = extract_label_sets_from_verdict(data)
        if errors:
            skipped.append(
                {
                    "doc_id": doc_id,
                    "file": f.name,
                    "stage": "train_index",
                    "reasons": sorted(set(errors)),
                    "stats": stats,
                }
            )
            continue

        index[doc_id] = label_sets

    return index, skipped


def load_test_docs(test_dir: Path, query_fields: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Build test records with strict verdict and query-text checks."""
    fields = query_fields or QUERY_CONTENT_FIELDS
    test_docs: list[dict] = []
    skipped: list[dict] = []

    for f in sorted(test_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        doc_id = data.get(ID_FIELD, f.stem)
        label_sets, stats, errors = extract_label_sets_from_verdict(data)
        if errors:
            skipped.append(
                {
                    "doc_id": doc_id,
                    "file": f.name,
                    "stage": "test_ground_truth",
                    "reasons": sorted(set(errors)),
                    "stats": stats,
                }
            )
            continue

        parts = []
        missing_fields = []
        for field in fields:
            val = (data.get(field) or "").strip()
            if val:
                parts.append(val)
            else:
                missing_fields.append(field)

        query_text = "\n\n".join(parts).strip()
        if not query_text:
            skipped.append(
                {
                    "doc_id": doc_id,
                    "file": f.name,
                    "stage": "test_query",
                    "reasons": ["empty_query_text"],
                    "query_fields": fields,
                    "empty_fields": missing_fields,
                }
            )
            continue

        test_docs.append(
            {
                "doc_id": doc_id,
                "query_text": query_text,
                "ground_truth_dieu": label_sets["dieu_only"],
                "ground_truth_full": label_sets["full_signature"],
                "gt_stats": stats,
            }
        )

    return test_docs, skipped


def _print_skip_report(skipped: list[dict], title: str) -> None:
    if not skipped:
        return
    print(f"  [INFO] {title}: skipped {len(skipped)} file(s)")
    for item in skipped[:20]:
        reasons = ",".join(item.get("reasons", []))
        print(f"    - {item.get('doc_id')} ({item.get('stage')}): {reasons}")
    if len(skipped) > 20:
        print(f"    ... and {len(skipped) - 20} more")


def precision_recall_f1(predicted: set[str], ground_truth: set[str]) -> tuple[float, float, float]:
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def evaluate(
    train_dir: str,
    test_dir: str,
    train_db_dir: str,
    test_db_dir: str,
    top_k: int,
    results_out: str,
    skip_embedding: bool,
    train_embedding_fields: list[str] | None = None,
    test_embedding_fields: list[str] | None = None,
    query_content_fields: list[str] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    train_fields = train_embedding_fields or TRAIN_EMBED_CONTENT_FIELDS
    test_fields = test_embedding_fields or TEST_EMBED_CONTENT_FIELDS
    query_fields = query_content_fields or QUERY_CONTENT_FIELDS

    print(f"  Train embedding fields: {train_fields}")
    print(f"  Test  embedding fields: {test_fields}")
    print(f"  Query content fields  : {query_fields}")

    train_path = Path(train_dir)
    test_path = Path(test_dir)

    runtime = RetrievalRuntime(
        RetrievalRuntimeConfig(
            model_name=model_name,
            device=device,
            train_db_dir=train_db_dir,
            collection_name=collection_name,
        )
    )

    if not skip_embedding:
        print("\n── Step 1: Embed train folder ───────────────────────────")
        runtime.ensure_train_index(
            train_dir=train_dir,
            content_fields=train_fields,
            max_chunk_chars=max_chunk_chars,
            batch_size=batch_size,
        )

        print("\n── Step 2: Embed test folder ────────────────────────────")
        # Compatibility path: keep test embedding behavior unchanged.
        run_pipeline(
            test_dir,
            test_db_dir,
            content_fields=test_fields,
            model_name=model_name,
            device=device,
            max_chunk_chars=max_chunk_chars,
            batch_size=batch_size,
            collection_name=collection_name,
        )
    else:
        print("\n── Step 1: Skipping embedding (--skip_embedding) ───────")

    print("\n── Step 3: Load model ───────────────────────────────────")
    _ = runtime.model

    print("\n── Step 4: Load train ChromaDB ──────────────────────────")
    print(f"  Train collection has {runtime.train_doc_count()} documents")

    print("\n── Step 5: Load test documents ──────────────────────────")
    test_docs, test_skipped = load_test_docs(test_path, query_fields=query_fields)
    _print_skip_report(test_skipped, "Test loading")
    print(f"  {len(test_docs)} evaluable test doc(s)")

    if not test_docs:
        print("\n  [WARN] No evaluable test documents found.")
        print(f"  Check '{VERDICT_FIELD}', '{LEGAL_BASIS_FIELD}', and configured query fields.")
        return

    train_articles_index, train_skipped = load_articles_index(train_path)
    _print_skip_report(train_skipped, "Train label index")

    print(f"\n── Step 6: Query + evaluate (top_k={top_k}) ─────────────")
    per_doc_results = []

    for doc in test_docs:
        doc_id = doc["doc_id"]
        query_text = doc["query_text"]
        ground_truth_dieu = doc["ground_truth_dieu"]
        ground_truth_full = doc["ground_truth_full"]

        results = runtime.query_train(
            query_text=query_text,
            top_k=top_k,
            exclude_doc_id=doc_id,
            include=["metadatas", "distances"],
        )

        retrieved_doc_ids: list[str] = []
        seen = set()
        for meta in results["metadatas"][0]:
            rid = meta["doc_id"]
            if rid not in seen:
                seen.add(rid)
                retrieved_doc_ids.append(rid)

        predicted_articles: set[str] = set()
        predicted_full: set[str] = set()
        for rid in retrieved_doc_ids:
            labels = train_articles_index.get(rid)
            if not labels:
                continue
            predicted_articles |= labels["dieu_only"]
            predicted_full |= labels["full_signature"]

        p_dieu, r_dieu, f1_dieu = precision_recall_f1(predicted_articles, ground_truth_dieu)
        p_full, r_full, f1_full = precision_recall_f1(predicted_full, ground_truth_full)

        result = {
            "doc_id": doc_id,
            "ground_truth": sorted(ground_truth_full),
            "ground_truth_dieu": sorted(ground_truth_dieu),
            "retrieved_doc_ids": retrieved_doc_ids,
            "predicted_articles": sorted(predicted_full),
            "predicted_articles_dieu": sorted(predicted_articles),
            "matched_articles": sorted(predicted_full & ground_truth_full),
            "missed_articles": sorted(ground_truth_full - predicted_full),
            "extra_articles": sorted(predicted_full - ground_truth_full),
            "matched_articles_dieu": sorted(predicted_articles & ground_truth_dieu),
            "missed_articles_dieu": sorted(ground_truth_dieu - predicted_articles),
            "extra_articles_dieu": sorted(predicted_articles - ground_truth_dieu),
            "precision": round(p_full, 4),
            "recall": round(r_full, 4),
            "f1": round(f1_full, 4),
            "precision_dieu": round(p_dieu, 4),
            "recall_dieu": round(r_dieu, 4),
            "f1_dieu": round(f1_dieu, 4),
            "gt_stats": doc["gt_stats"],
        }
        per_doc_results.append(result)

        print(f"\n  doc: {doc_id}")
        print(f"    ground truth (full_signature): {sorted(ground_truth_full)}")
        print(f"    predicted              : {sorted(predicted_full)}")
        print(f"    matched                : {sorted(predicted_full & ground_truth_full)}")
        print(f"    P={p_full:.4f}  R={r_full:.4f}  F1={f1_full:.4f}")
        print(f"    DIEU P={p_dieu:.4f}  R={r_dieu:.4f}  F1={f1_dieu:.4f}")

    n = len(per_doc_results)
    macro_p = sum(r["precision"] for r in per_doc_results) / n
    macro_r = sum(r["recall"] for r in per_doc_results) / n
    macro_f1 = sum(r["f1"] for r in per_doc_results) / n
    macro_p_dieu = sum(r["precision_dieu"] for r in per_doc_results) / n
    macro_r_dieu = sum(r["recall_dieu"] for r in per_doc_results) / n
    macro_f1_dieu = sum(r["f1_dieu"] for r in per_doc_results) / n

    print("\n── Results ──────────────────────────────────────────────")
    print(f"  Evaluated on : {n} test document(s)")
    print(f"  top_k        : {top_k}")
    print(f"  Macro P      : {macro_p:.4f}")
    print(f"  Macro R      : {macro_r:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Macro P(dieu): {macro_p_dieu:.4f}")
    print(f"  Macro R(dieu): {macro_r_dieu:.4f}")
    print(f"  Macro F1(dieu): {macro_f1_dieu:.4f}")

    output = {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_db_dir": train_db_dir,
        "test_db_dir": test_db_dir,
        "top_k": top_k,
        "model_name": model_name,
        "collection_name": collection_name,
        "max_chunk_chars": max_chunk_chars,
        "train_embedding_fields": train_fields,
        "test_embedding_fields": test_fields,
        "query_content_fields": query_fields,
        "n_docs": n,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision_dieu": round(macro_p_dieu, 4),
        "macro_recall_dieu": round(macro_r_dieu, 4),
        "macro_f1_dieu": round(macro_f1_dieu, 4),
        "n_train_skipped": len(train_skipped),
        "n_test_skipped": len(test_skipped),
        "train_skipped": train_skipped,
        "test_skipped": test_skipped,
        "per_doc": per_doc_results,
    }
    out_path = Path(results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(f"\n  Full results saved to: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Embed train/test folders into separate DBs and evaluate retrieval "
            "quality using PHAN_QUYET_CUA_TOA_SO_THAM overlap"
        )
    )
    parser.add_argument("--train_dir", required=True, help="Folder containing train JSON files")
    parser.add_argument("--test_dir", required=True, help="Folder containing test JSON files")
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
        default=5,
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
        help="Skip embedding and evaluate using existing DBs",
    )
    parser.add_argument(
        "--train_embedding_fields",
        default=None,
        help=(
            "Comma-separated list of JSON fields to chunk & embed for train "
            f"(default: {','.join(TRAIN_EMBED_CONTENT_FIELDS)})"
        ),
    )
    parser.add_argument(
        "--test_embedding_fields",
        default=None,
        help=(
            "Comma-separated list of JSON fields to chunk & embed for test "
            f"(default: {','.join(TEST_EMBED_CONTENT_FIELDS)})"
        ),
    )
    parser.add_argument(
        "--query_content_fields",
        default=None,
        help=(
            "Comma-separated list of JSON fields used to build query text "
            f"(default: {','.join(QUERY_CONTENT_FIELDS)})"
        ),
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Device for model inference (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--max_chunk_chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=(
            "Max characters per chunk; 0 = no splitting "
            f"(default: {DEFAULT_MAX_CHUNK_CHARS})"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--collection_name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_fields = [f.strip() for f in args.train_embedding_fields.split(",")] if args.train_embedding_fields else None
    test_fields = [f.strip() for f in args.test_embedding_fields.split(",")] if args.test_embedding_fields else None
    query_fields = [f.strip() for f in args.query_content_fields.split(",")] if args.query_content_fields else None

    evaluate(
        args.train_dir,
        args.test_dir,
        args.train_db_dir,
        args.test_db_dir,
        args.top_k,
        args.results_out,
        args.skip_embedding,
        train_embedding_fields=train_fields,
        test_embedding_fields=test_fields,
        query_content_fields=query_fields,
        model_name=args.model_name,
        device=args.device,
        max_chunk_chars=args.max_chunk_chars,
        batch_size=args.batch_size,
        collection_name=args.collection_name,
    )


if __name__ == "__main__":
    main()
