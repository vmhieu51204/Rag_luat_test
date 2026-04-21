"""Evaluate embedding model performance with and without added law chunks.

This script keeps the original retrieval evaluator untouched and focuses on
retrieval-only metrics. It can benchmark one or more embedding models on:
- baseline index (case chunks only)
- law-augmented index (case chunks + raw_law chunks)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rag.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MODEL_NAME,
)
from rag.core.embeddings import run_pipeline
from rag.core.law_embeddings import embed_law_chunks
from rag.evaluation.retrieval_eval import (
    _print_skip_report,
    load_articles_index,
    load_test_docs,
    precision_recall_f1,
)
from rag.runtime.retrieval import RetrievalRuntime, RetrievalRuntimeConfig


def _slugify_model(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return slug or "model"


def _parse_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name")
    return models


def _collect_predictions_from_results(
    results: dict[str, Any],
    train_articles_index: dict[str, dict[str, set[str]]],
) -> tuple[list[str], set[str], set[str], int]:
    """Build predicted labels from retrieved case docs and law chunks."""
    retrieved_doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()
    predicted_dieu: set[str] = set()
    predicted_full: set[str] = set()
    law_hits = 0

    for meta in results.get("metadatas", [[]])[0]:
        if not isinstance(meta, dict):
            continue

        source_type = str(meta.get("source_type", "")).strip().lower()
        if source_type == "law":
            sig = str(meta.get("law_signature_full", "")).strip()
            if sig:
                law_hits += 1
                predicted_full.add(sig)
                dieu = str(meta.get("law_dieu", "")).strip()
                if dieu:
                    predicted_dieu.add(dieu)
                else:
                    predicted_dieu.add(sig.split("-", 1)[0])
            continue

        rid = str(meta.get("doc_id", "")).strip()
        if rid and rid not in seen_doc_ids:
            seen_doc_ids.add(rid)
            retrieved_doc_ids.append(rid)

    for rid in retrieved_doc_ids:
        labels = train_articles_index.get(rid)
        if not labels:
            continue
        predicted_dieu |= labels["dieu_only"]
        predicted_full |= labels["full_signature"]

    return retrieved_doc_ids, predicted_dieu, predicted_full, law_hits


def _query_case_only(
    runtime: RetrievalRuntime,
    *,
    query_text: str,
    exclude_doc_id: str,
    top_k_case: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Retrieve top-k non-law case docs with adaptive over-fetching."""
    if top_k_case <= 0:
        return [], []

    n_fetch = max(top_k_case * 8, 64)
    cap = max(top_k_case * 200, 800)

    while True:
        results = runtime.query_train(
            query_text=query_text,
            top_k=n_fetch,
            exclude_doc_id=exclude_doc_id,
            include=["metadatas", "distances"],
        )
        metas = results.get("metadatas", [[]])[0]
        case_doc_ids: list[str] = []
        case_metas: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()

        for meta in metas:
            if not isinstance(meta, dict):
                continue
            source_type = str(meta.get("source_type", "")).strip().lower()
            if source_type == "law":
                continue

            rid = str(meta.get("doc_id", "")).strip()
            if not rid or rid in seen_doc_ids:
                continue
            seen_doc_ids.add(rid)
            case_doc_ids.append(rid)
            case_metas.append(meta)
            if len(case_doc_ids) >= top_k_case:
                break

        if len(case_doc_ids) >= top_k_case or n_fetch >= cap:
            return case_doc_ids, case_metas

        n_fetch = min(n_fetch * 2, cap)


def _query_law_only(
    runtime: RetrievalRuntime,
    *,
    query_text: str,
    top_k_law: int,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Retrieve top-k law chunks only and return signature sets."""
    if top_k_law <= 0:
        return [], set(), set()

    vec = runtime.encode_query(query_text)
    results = runtime.train_collection.query(
        query_embeddings=vec,
        n_results=top_k_law,
        where={"source_type": "law"},
        include=["metadatas", "distances"],
    )
    metas = results.get("metadatas", [[]])[0]

    pred_full: set[str] = set()
    pred_dieu: set[str] = set()
    law_metas: list[dict[str, Any]] = []

    for meta in metas:
        if not isinstance(meta, dict):
            continue
        law_metas.append(meta)
        sig = str(meta.get("law_signature_full", "")).strip()
        if not sig:
            continue
        pred_full.add(sig)
        dieu = str(meta.get("law_dieu", "")).strip()
        if dieu:
            pred_dieu.add(dieu)
        else:
            pred_dieu.add(sig.split("-", 1)[0])

    return law_metas, pred_dieu, pred_full


def evaluate_single_configuration(
    *,
    model_name: str,
    device: str,
    train_dir: str,
    test_dir: str,
    case_db_dir: str,
    law_db_dir: str,
    test_db_dir: str,
    top_k_case: int,
    top_k_law: int,
    max_chunk_chars: int,
    batch_size: int,
    collection_name: str,
    train_embedding_fields: list[str],
    test_embedding_fields: list[str],
    query_content_fields: list[str],
    embed_law: bool,
    law_json: str,
    law_id: str,
) -> dict[str, Any]:
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    case_runtime = RetrievalRuntime(
        RetrievalRuntimeConfig(
            model_name=model_name,
            device=device,
            train_db_dir=case_db_dir,
            collection_name=collection_name,
        )
    )
    law_runtime = RetrievalRuntime(
        RetrievalRuntimeConfig(
            model_name=model_name,
            device=device,
            train_db_dir=law_db_dir,
            collection_name=collection_name,
        )
    )

    case_runtime.ensure_train_index(
        train_dir=train_dir,
        content_fields=train_embedding_fields,
        max_chunk_chars=max_chunk_chars,
        batch_size=batch_size,
    )

    run_pipeline(
        test_dir,
        test_db_dir,
        content_fields=test_embedding_fields,
        model_name=model_name,
        device=device,
        max_chunk_chars=max_chunk_chars,
        batch_size=batch_size,
        collection_name=collection_name,
    )

    law_chunks_added = 0
    if embed_law:
        law_chunks_added = embed_law_chunks(
            raw_law_path=law_json,
            db_dir=law_db_dir,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            collection_name=collection_name,
            law_id=law_id,
            max_chunk_chars=max_chunk_chars,
        )

    test_docs, test_skipped = load_test_docs(test_path, query_fields=query_content_fields)
    _print_skip_report(test_skipped, "Test loading")

    if not test_docs:
        return {
            "n_docs": 0,
            "top_k_case": top_k_case,
            "top_k_law": top_k_law,
            "macro_precision_case_only": 0.0,
            "macro_recall_case_only": 0.0,
            "macro_f1_case_only": 0.0,
            "macro_precision_law_only": 0.0,
            "macro_recall_law_only": 0.0,
            "macro_f1_law_only": 0.0,
            "macro_precision_combined": 0.0,
            "macro_recall_combined": 0.0,
            "macro_f1_combined": 0.0,
            "macro_precision_dieu_case_only": 0.0,
            "macro_recall_dieu_case_only": 0.0,
            "macro_f1_dieu_case_only": 0.0,
            "macro_precision_dieu_law_only": 0.0,
            "macro_recall_dieu_law_only": 0.0,
            "macro_f1_dieu_law_only": 0.0,
            "macro_precision_dieu_combined": 0.0,
            "macro_recall_dieu_combined": 0.0,
            "macro_f1_dieu_combined": 0.0,
            "n_test_skipped": len(test_skipped),
            "n_train_skipped": 0,
            "train_skipped": [],
            "test_skipped": test_skipped,
            "law_chunks_added": law_chunks_added,
            "per_doc": [],
        }

    train_articles_index, train_skipped = load_articles_index(train_path)
    _print_skip_report(train_skipped, "Train label index")

    per_doc: list[dict[str, Any]] = []
    query_law_db = True

    for doc in test_docs:
        doc_id = doc["doc_id"]
        retrieved_doc_ids, _case_metas = _query_case_only(
            case_runtime,
            query_text=doc["query_text"],
            exclude_doc_id=doc_id,
            top_k_case=top_k_case,
        )

        if query_law_db:
            try:
                law_metas, law_only_dieu, law_only_full = _query_law_only(
                    law_runtime,
                    query_text=doc["query_text"],
                    top_k_law=top_k_law,
                )
            except Exception:  # noqa: BLE001
                # Law DB may not exist in baseline mode; treat as empty retrieval.
                law_metas, law_only_dieu, law_only_full = [], set(), set()
                query_law_db = False
        else:
            law_metas, law_only_dieu, law_only_full = [], set(), set()

        case_only_dieu: set[str] = set()
        case_only_full: set[str] = set()
        for rid in retrieved_doc_ids:
            labels = train_articles_index.get(rid)
            if not labels:
                continue
            case_only_dieu |= labels["dieu_only"]
            case_only_full |= labels["full_signature"]

        combined_dieu = case_only_dieu | law_only_dieu
        combined_full = case_only_full | law_only_full

        ground_truth_dieu = doc["ground_truth_dieu"]
        ground_truth_full = doc["ground_truth_full"]

        p_case_dieu, r_case_dieu, f1_case_dieu = precision_recall_f1(case_only_dieu, ground_truth_dieu)
        p_case_full, r_case_full, f1_case_full = precision_recall_f1(case_only_full, ground_truth_full)

        p_law_dieu, r_law_dieu, f1_law_dieu = precision_recall_f1(law_only_dieu, ground_truth_dieu)
        p_law_full, r_law_full, f1_law_full = precision_recall_f1(law_only_full, ground_truth_full)

        p_comb_dieu, r_comb_dieu, f1_comb_dieu = precision_recall_f1(combined_dieu, ground_truth_dieu)
        p_comb_full, r_comb_full, f1_comb_full = precision_recall_f1(combined_full, ground_truth_full)

        per_doc.append(
            {
                "doc_id": doc_id,
                "ground_truth": sorted(ground_truth_full),
                "ground_truth_dieu": sorted(ground_truth_dieu),
                "retrieved_doc_ids": retrieved_doc_ids,
                "law_hits_in_top_k": len(law_metas),
                "predicted_articles_case_only": sorted(case_only_full),
                "predicted_articles_dieu_case_only": sorted(case_only_dieu),
                "predicted_articles_law_only": sorted(law_only_full),
                "predicted_articles_dieu_law_only": sorted(law_only_dieu),
                "predicted_articles_combined": sorted(combined_full),
                "predicted_articles_dieu_combined": sorted(combined_dieu),
                "matched_articles_case_only": sorted(case_only_full & ground_truth_full),
                "matched_articles_law_only": sorted(law_only_full & ground_truth_full),
                "matched_articles_combined": sorted(combined_full & ground_truth_full),
                "matched_articles_dieu_case_only": sorted(case_only_dieu & ground_truth_dieu),
                "matched_articles_dieu_law_only": sorted(law_only_dieu & ground_truth_dieu),
                "matched_articles_dieu_combined": sorted(combined_dieu & ground_truth_dieu),
                "precision_case_only": round(p_case_full, 4),
                "recall_case_only": round(r_case_full, 4),
                "f1_case_only": round(f1_case_full, 4),
                "precision_dieu_case_only": round(p_case_dieu, 4),
                "recall_dieu_case_only": round(r_case_dieu, 4),
                "f1_dieu_case_only": round(f1_case_dieu, 4),
                "precision_law_only": round(p_law_full, 4),
                "recall_law_only": round(r_law_full, 4),
                "f1_law_only": round(f1_law_full, 4),
                "precision_dieu_law_only": round(p_law_dieu, 4),
                "recall_dieu_law_only": round(r_law_dieu, 4),
                "f1_dieu_law_only": round(f1_law_dieu, 4),
                "precision_combined": round(p_comb_full, 4),
                "recall_combined": round(r_comb_full, 4),
                "f1_combined": round(f1_comb_full, 4),
                "precision_dieu_combined": round(p_comb_dieu, 4),
                "recall_dieu_combined": round(r_comb_dieu, 4),
                "f1_dieu_combined": round(f1_comb_dieu, 4),
            }
        )

    n_docs = len(per_doc)
    macro_precision_case_only = sum(r["precision_case_only"] for r in per_doc) / n_docs
    macro_recall_case_only = sum(r["recall_case_only"] for r in per_doc) / n_docs
    macro_f1_case_only = sum(r["f1_case_only"] for r in per_doc) / n_docs

    macro_precision_law_only = sum(r["precision_law_only"] for r in per_doc) / n_docs
    macro_recall_law_only = sum(r["recall_law_only"] for r in per_doc) / n_docs
    macro_f1_law_only = sum(r["f1_law_only"] for r in per_doc) / n_docs

    macro_precision_combined = sum(r["precision_combined"] for r in per_doc) / n_docs
    macro_recall_combined = sum(r["recall_combined"] for r in per_doc) / n_docs
    macro_f1_combined = sum(r["f1_combined"] for r in per_doc) / n_docs

    macro_precision_dieu_case_only = sum(r["precision_dieu_case_only"] for r in per_doc) / n_docs
    macro_recall_dieu_case_only = sum(r["recall_dieu_case_only"] for r in per_doc) / n_docs
    macro_f1_dieu_case_only = sum(r["f1_dieu_case_only"] for r in per_doc) / n_docs

    macro_precision_dieu_law_only = sum(r["precision_dieu_law_only"] for r in per_doc) / n_docs
    macro_recall_dieu_law_only = sum(r["recall_dieu_law_only"] for r in per_doc) / n_docs
    macro_f1_dieu_law_only = sum(r["f1_dieu_law_only"] for r in per_doc) / n_docs

    macro_precision_dieu_combined = sum(r["precision_dieu_combined"] for r in per_doc) / n_docs
    macro_recall_dieu_combined = sum(r["recall_dieu_combined"] for r in per_doc) / n_docs
    macro_f1_dieu_combined = sum(r["f1_dieu_combined"] for r in per_doc) / n_docs

    return {
        "n_docs": n_docs,
        "top_k_case": top_k_case,
        "top_k_law": top_k_law,
        "macro_precision_case_only": round(macro_precision_case_only, 4),
        "macro_recall_case_only": round(macro_recall_case_only, 4),
        "macro_f1_case_only": round(macro_f1_case_only, 4),
        "macro_precision_law_only": round(macro_precision_law_only, 4),
        "macro_recall_law_only": round(macro_recall_law_only, 4),
        "macro_f1_law_only": round(macro_f1_law_only, 4),
        "macro_precision_combined": round(macro_precision_combined, 4),
        "macro_recall_combined": round(macro_recall_combined, 4),
        "macro_f1_combined": round(macro_f1_combined, 4),
        "macro_precision_dieu_case_only": round(macro_precision_dieu_case_only, 4),
        "macro_recall_dieu_case_only": round(macro_recall_dieu_case_only, 4),
        "macro_f1_dieu_case_only": round(macro_f1_dieu_case_only, 4),
        "macro_precision_dieu_law_only": round(macro_precision_dieu_law_only, 4),
        "macro_recall_dieu_law_only": round(macro_recall_dieu_law_only, 4),
        "macro_f1_dieu_law_only": round(macro_f1_dieu_law_only, 4),
        "macro_precision_dieu_combined": round(macro_precision_dieu_combined, 4),
        "macro_recall_dieu_combined": round(macro_recall_dieu_combined, 4),
        "macro_f1_dieu_combined": round(macro_f1_dieu_combined, 4),
        "n_test_skipped": len(test_skipped),
        "n_train_skipped": len(train_skipped),
        "train_skipped": train_skipped,
        "test_skipped": test_skipped,
        "law_chunks_added": law_chunks_added,
        "per_doc": per_doc,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark embedding models on retrieval-only evaluation with and without "
            "raw_law chunk augmentation."
        )
    )
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--law_json", default="raw_law.json")
    parser.add_argument("--law_id", default="blhs")
    parser.add_argument("--models", default=DEFAULT_MODEL_NAME, help="Comma-separated model names")
    parser.add_argument("--device", default="cuda", help="Embedding device (default: cuda)")
    parser.add_argument("--top_k_case", type=int, default=5, help="Top-k retrieved past-case chunks")
    parser.add_argument("--top_k_law", type=int, default=10, help="Top-k retrieved law clause chunks")
    parser.add_argument("--max_chunk_chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--collection_name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--work_dir",
        default="output/law_model_eval",
        help="Root directory for model-specific case/law/test DBs",
    )
    parser.add_argument(
        "--results_out",
        default="output/law_model_eval/model_comparison.json",
        help="Path to write model comparison output JSON",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline (case-only) run and evaluate only law-augmented runs",
    )
    parser.add_argument("--train_embedding_fields", default="Summary")
    parser.add_argument("--test_embedding_fields", default="Synthetic_summary")
    parser.add_argument("--query_content_fields", default="Synthetic_summary")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    models = _parse_models(args.models)
    train_fields = [f.strip() for f in args.train_embedding_fields.split(",") if f.strip()]
    test_fields = [f.strip() for f in args.test_embedding_fields.split(",") if f.strip()]
    query_fields = [f.strip() for f in args.query_content_fields.split(",") if f.strip()]

    results: dict[str, Any] = {
        "device": args.device,
        "top_k_case": args.top_k_case,
        "top_k_law": args.top_k_law,
        "max_chunk_chars": args.max_chunk_chars,
        "batch_size": args.batch_size,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "law_json": args.law_json,
        "law_id": args.law_id,
        "models": {},
    }

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        model_slug = _slugify_model(model_name)
        model_root = work_dir / model_slug
        shared_case_db = str(model_root / "shared" / "case_db")
        shared_law_db = str(model_root / "shared" / "law_db")
        shared_test_db = str(model_root / "shared" / "test_db")

        print(f"\n=== Model: {model_name} ({args.device}) ===")

        model_result: dict[str, Any] = {}
        if not args.skip_baseline:
            print("  -> Running baseline (case-only)")
            model_result["baseline"] = evaluate_single_configuration(
                model_name=model_name,
                device=args.device,
                train_dir=args.train_dir,
                test_dir=args.test_dir,
                case_db_dir=shared_case_db,
                law_db_dir=shared_law_db,
                test_db_dir=shared_test_db,
                top_k_case=args.top_k_case,
                top_k_law=args.top_k_law,
                max_chunk_chars=args.max_chunk_chars,
                batch_size=args.batch_size,
                collection_name=args.collection_name,
                train_embedding_fields=train_fields,
                test_embedding_fields=test_fields,
                query_content_fields=query_fields,
                embed_law=False,
                law_json=args.law_json,
                law_id=args.law_id,
            )

        print("  -> Running law-augmented")
        model_result["with_law"] = evaluate_single_configuration(
            model_name=model_name,
            device=args.device,
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            case_db_dir=shared_case_db,
            law_db_dir=shared_law_db,
            test_db_dir=shared_test_db,
            top_k_case=args.top_k_case,
            top_k_law=args.top_k_law,
            max_chunk_chars=args.max_chunk_chars,
            batch_size=args.batch_size,
            collection_name=args.collection_name,
            train_embedding_fields=train_fields,
            test_embedding_fields=test_fields,
            query_content_fields=query_fields,
            embed_law=True,
            law_json=args.law_json,
            law_id=args.law_id,
        )

        if "baseline" in model_result:
            base_comb_precision = model_result["baseline"]["macro_precision_combined"]
            law_comb_precision = model_result["with_law"]["macro_precision_combined"]
            base_comb_recall = model_result["baseline"]["macro_recall_combined"]
            law_comb_recall = model_result["with_law"]["macro_recall_combined"]
            base_comb_f1 = model_result["baseline"]["macro_f1_combined"]
            law_comb_f1 = model_result["with_law"]["macro_f1_combined"]
            model_result["delta_with_law"] = {
                "macro_precision_combined": round(law_comb_precision - base_comb_precision, 4),
                "macro_recall_combined": round(law_comb_recall - base_comb_recall, 4),
                "macro_f1_combined": round(law_comb_f1 - base_comb_f1, 4),
            }
            print(
                "  Δ with law: "
                f"macro_precision_combined={model_result['delta_with_law']['macro_precision_combined']:+.4f}, "
                f"macro_recall_combined={model_result['delta_with_law']['macro_recall_combined']:+.4f}, "
                f"macro_f1_combined={model_result['delta_with_law']['macro_f1_combined']:+.4f}"
            )

        results["models"][model_name] = model_result

    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"\nSaved model comparison to: {out_path}")


if __name__ == "__main__":
    main()
