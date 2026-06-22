"""Evaluate ReAct-style judgment prediction over chunk/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MODEL_NAME,
)
from rag.core.embeddings import run_pipeline
from rag.core.law_retriever import LawClauseRetriever
from rag.evaluation.eval_utils import (
    _aggregate,
    _compact_defendant_item,
    _extract_doc_id,
    _extract_gt_defendants,
    _extract_phat_tu_months,
    _extract_pred_defendants,
    _macro_mean,
    _name_key,
    _parse_fields,
    _safe_float,
    _set_prf,
    _to_dieu_set,
    load_articles_index,
    save_eval_results,
)
from rag.generation.reasoning_act import (
    DEFAULT_REASON_ACT_TRAIN_FIELDS,
    DEFAULT_SENTENCING_CALIBRATION_FIELDS,
    MANDATORY_SUPPORTING_DIEU,
    SYNTHETIC_SUMMARY_FORMAT_NOTE,
    safe_run_reasoning_act,
)
from rag.llm.providers import LLMProvider, default_model_for_provider
from rag.runtime.retrieval import RetrievalRuntime, RetrievalRuntimeConfig

load_dotenv()


def _trace_to_dict(trace: Any) -> dict[str, Any] | None:
    if trace is None:
        return None
    if hasattr(trace, "model_dump"):
        return trace.model_dump()
    return trace


def _evaluate_single_doc(
    *,
    path: Path,
    data: dict[str, Any],
    case_runtime: RetrievalRuntime,
    train_dir: Path,
    train_articles_index: dict[str, dict[str, set[str]]],
    law_retriever: LawClauseRetriever,
    input_fields: list[str],
    query_fields: list[str],
    broad_top_k_case: int,
    top_k_case: int,
    max_additional_law_rounds: int,
    provider: LLMProvider,
    model_name: str,
    only_blhs: bool,
    use_provider_fallback: bool,
) -> dict[str, Any]:
    doc_id = _extract_doc_id(data, path.stem)
    gt_defendants = _extract_gt_defendants(data, only_blhs=only_blhs)

    react_result, error = safe_run_reasoning_act(
        data=data,
        doc_id=doc_id,
        law_retriever=law_retriever,
        case_runtime=case_runtime,
        train_dir=train_dir,
        train_articles_index=train_articles_index,
        provider=provider,
        model_name=model_name,
        use_provider_fallback=use_provider_fallback,
        input_fields=input_fields,
        query_fields=query_fields,
        broad_top_k_case=broad_top_k_case,
        top_k_case=top_k_case,
        max_additional_law_rounds=max_additional_law_rounds,
    )

    if react_result is None:
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "failed",
            "reason": (error or "generation_error").split(":", 1)[0],
            "defendants": [
                {
                    "Bi_Cao": item.get("Bi_Cao", ""),
                    "ground_truth": _compact_defendant_item(item),
                    "prediction": None,
                }
                for item in gt_defendants
            ],
            "trace": None,
            "error": error,
        }

    pred_output = react_result["prediction"]
    trace = react_result["trace"]
    pred_defendants = _extract_pred_defendants(pred_output, only_blhs=only_blhs)

    gt_by_name = {_name_key(item["Bi_Cao"]): item for item in gt_defendants if item.get("Bi_Cao")}
    pred_by_name = {_name_key(item["Bi_Cao"]): item for item in pred_defendants if item.get("Bi_Cao")}
    all_keys = sorted(set(gt_by_name) | set(pred_by_name))
    matched_keys = sorted(set(gt_by_name) & set(pred_by_name))
    gt_only = sorted(set(gt_by_name) - set(pred_by_name))
    pred_only = sorted(set(pred_by_name) - set(gt_by_name))

    clause_precision_values: list[float] = []
    clause_recall_values: list[float] = []
    clause_f1_values: list[float] = []
    phat_tu_sq_err_values: list[float] = []
    defendants: list[dict[str, Any]] = []

    for key in all_keys:
        gt_item = gt_by_name.get(key)
        pred_item = pred_by_name.get(key)
        gt_set = _to_dieu_set(set((gt_item or {}).get("Applied_Law_Clauses", [])))
        pred_set = _to_dieu_set(set((pred_item or {}).get("Applied_Law_Clauses", [])))
        prf = _set_prf(pred_set, gt_set)
        clause_precision_values.append(float(prf["precision"]))
        clause_recall_values.append(float(prf["recall"]))
        clause_f1_values.append(float(prf["f1"]))

        gt_months = _extract_phat_tu_months((gt_item or {}).get("Phat_Tu"))
        pred_months = _extract_phat_tu_months((pred_item or {}).get("Phat_Tu"))
        sq_err = float((pred_months - gt_months) ** 2)
        phat_tu_sq_err_values.append(sq_err)

        defendants.append(
            {
                "Bi_Cao": (gt_item or pred_item or {}).get("Bi_Cao", ""),
                "ground_truth": _compact_defendant_item(gt_item),
                "prediction": _compact_defendant_item(pred_item),
                "metrics": {
                    "law_clause_prf": prf,
                    "phat_tu_months": {
                        "ground_truth": gt_months,
                        "prediction": pred_months,
                        "squared_error": _safe_float(sq_err),
                    },
                },
            }
        )

    phat_tu_rmse_months = (
        _safe_float((sum(phat_tu_sq_err_values) / len(phat_tu_sq_err_values)) ** 0.5)
        if phat_tu_sq_err_values
        else 0.0
    )

    trace_dict = _trace_to_dict(trace) or {}
    return {
        "doc_id": doc_id,
        "source_file": path.name,
        "status": "processed",
        "reason": "ok",
        "llm_input_payload": react_result.get("llm_input_payload", {}),
        "prediction_raw": pred_output.model_dump(),
        "trace": trace_dict,
        "similar_case_context": {
            "similar_case_doc_ids": [item.get("doc_id") for item in trace_dict.get("similar_cases", [])],
        },
        "sentencing_calibration_context": {
            "mitigation_factors": trace_dict.get("mitigation_factors", []),
            "aggravation_factors": trace_dict.get("aggravation_factors", []),
            "calibration_case_doc_ids": [
                item.get("doc_id") for item in trace_dict.get("sentencing_calibration_cases", [])
            ],
            "calibration_cases": trace_dict.get("sentencing_calibration_cases", []),
        },
        "retrieved_law": {
            "offence_articles": trace_dict.get("retrieved_offence_articles", []),
            "additional_articles": trace_dict.get("retrieved_additional_articles", []),
            "supporting_articles": trace_dict.get("retrieved_supporting_articles", []),
            "mandatory_supporting_articles": list(MANDATORY_SUPPORTING_DIEU),
        },
        "additional_law_queries": trace_dict.get("additional_law_queries", []),
        "supporting_article_assessments": trace_dict.get("supporting_article_assessments", []),
        "rejected_candidates": trace_dict.get("rejected_candidates", []),
        "missing_facts": trace_dict.get("missing_facts", []),
        "defendant_alignment": {
            "matched_count": len(matched_keys),
            "gt_only_count": len(gt_only),
            "pred_only_count": len(pred_only),
            "gt_only_keys": gt_only,
            "pred_only_keys": pred_only,
        },
        "defendants": defendants,
        "doc_metrics": {
            "law_clause_precision_macro": _macro_mean(clause_precision_values),
            "law_clause_recall_macro": _macro_mean(clause_recall_values),
            "law_clause_f1_macro": _macro_mean(clause_f1_values),
            "phat_tu_rmse_months": phat_tu_rmse_months,
            "n_defendants_scored": len(all_keys),
        },
        "_usage": react_result.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReAct judgment prediction on test cases.")
    parser.add_argument("--test-dir", default="chunk/test")
    parser.add_argument("--train-dir", default="chunk/train")
    parser.add_argument("--law-json", default="raw_law.json")
    parser.add_argument("--case-db-dir", default="output/reasoning_act_eval/case_db")
    parser.add_argument("--results-out", default="output/reasoning_act_eval/results.json")
    parser.add_argument("--provider", choices=[p.value for p in LLMProvider], default="openrouter")
    parser.add_argument("--model", default=None)
    parser.add_argument("--embed-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS)
    parser.add_argument(
        "--input-fields",
        default="THONG_TIN_CHUNG.Thong_Tin_Bi_Cao,Synthetic_summary_2",
        help=(
            "Comma-separated fields passed to the ReAct LLM prompts. "
            "Synthetic_summary may be a list of separate first-person defendant stories."
        ),
    )
    parser.add_argument(
        "--query-fields",
        default="Synthetic_summary_2,THONG_TIN_CHUNG.Thong_Tin_Bi_Cao",
        help="Comma-separated fields used for similar-case retrieval.",
    )
    parser.add_argument(
        "--train-embedding-fields",
        default=",".join(DEFAULT_REASON_ACT_TRAIN_FIELDS + DEFAULT_SENTENCING_CALIBRATION_FIELDS),
    )
    parser.add_argument("--broad-top-k-case", type=int, default=64)
    parser.add_argument("--top-k-case", type=int, default=5)
    parser.add_argument(
        "--max-additional-law-rounds",
        type=int,
        default=1,
        help="Maximum extra legal-analysis rounds after the model requests additional BLHS signatures.",
    )
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--only-blhs", action="store_true", default=True)
    parser.add_argument("--include-non-blhs", action="store_false", dest="only_blhs")
    parser.add_argument("--disable-provider-fallback", action="store_true", default=False)
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        default=False,
        help="Use an existing case DB instead of embedding chunk/train.",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    law_json = Path(args.law_json)
    case_db_dir = Path(args.case_db_dir)
    results_out = Path(args.results_out)

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    if not law_json.exists():
        raise FileNotFoundError(f"Missing law JSON: {law_json}")
    if args.top_k_case < 1:
        raise ValueError("--top-k-case must be >= 1")
    if args.broad_top_k_case < args.top_k_case:
        raise ValueError("--broad-top-k-case must be >= --top-k-case")
    if args.max_additional_law_rounds < 0:
        raise ValueError("--max-additional-law-rounds must be >= 0")

    input_fields = _parse_fields(args.input_fields)
    query_fields = _parse_fields(args.query_fields)
    train_embedding_fields = _parse_fields(args.train_embedding_fields)
    provider = LLMProvider(args.provider)
    model_name = args.model or default_model_for_provider(provider)
    use_provider_fallback = not args.disable_provider_fallback

    if not args.skip_embedding:
        print("Preparing train case embeddings...")
        run_pipeline(
            str(train_dir),
            str(case_db_dir),
            content_fields=train_embedding_fields,
            model_name=args.embed_model,
            device=args.device,
            max_chunk_chars=args.max_chunk_chars,
            batch_size=args.batch_size,
            collection_name=args.collection_name,
        )

    train_articles_index, train_skipped = load_articles_index(train_dir)
    case_runtime = RetrievalRuntime(
        RetrievalRuntimeConfig(
            model_name=args.embed_model,
            device=args.device,
            train_db_dir=str(case_db_dir),
            collection_name=args.collection_name,
        )
    )
    law_retriever = LawClauseRetriever(law_json)

    files = sorted(test_dir.glob("*.json"))
    if args.first_n is not None:
        if args.first_n < 1:
            raise ValueError("--first-n must be >= 1")
        files = files[: args.first_n]

    print(f"Found {len(files)} test files")
    print(f"Provider={provider.value} | Model={model_name}")
    print(f"Provider fallback enabled={use_provider_fallback}")
    print(f"Input fields={input_fields}")
    print(f"Query fields={query_fields}")
    print(f"Train embedding fields={train_embedding_fields}")
    print(f"Synthetic_summary format={SYNTHETIC_SUMMARY_FORMAT_NOTE}")
    print(f"Mandatory supporting articles={list(MANDATORY_SUPPORTING_DIEU)}")
    print(f"Max additional law rounds={args.max_additional_law_rounds}")
    print(f"Train label index size={len(train_articles_index)} (skipped={len(train_skipped)})")

    per_doc: list[dict[str, Any]] = []
    completed_files: set[str] = set()
    if results_out.exists():
        try:
            with open(results_out, encoding="utf-8") as fh:
                existing_data = json.load(fh)
            per_doc = existing_data.get("per_doc", [])
            completed_files = {doc.get("source_file") for doc in per_doc if doc.get("source_file")}
            print(f"Resuming from {results_out}: {len(completed_files)} files already processed.")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load existing results from {results_out}: {exc}")

    config = {
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "law_json": str(law_json),
        "case_db_dir": str(case_db_dir),
        "provider": provider.value,
        "model": model_name,
        "provider_fallback": use_provider_fallback,
        "embedding_model": args.embed_model,
        "device": args.device,
        "collection_name": args.collection_name,
        "input_fields": input_fields,
        "query_fields": query_fields,
        "train_embedding_fields": train_embedding_fields,
        "synthetic_summary_format": SYNTHETIC_SUMMARY_FORMAT_NOTE,
        "broad_top_k_case": args.broad_top_k_case,
        "top_k_case": args.top_k_case,
        "max_additional_law_rounds": args.max_additional_law_rounds,
        "mandatory_supporting_articles": list(MANDATORY_SUPPORTING_DIEU),
        "only_blhs": args.only_blhs,
        "n_train_label_index": len(train_articles_index),
        "n_train_label_skipped": len(train_skipped),
    }

    for path in files:
        if path.name in completed_files:
            print(f"Already processed, skipping: {path.name}")
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        result = _evaluate_single_doc(
            path=path,
            data=data,
            case_runtime=case_runtime,
            train_dir=train_dir,
            train_articles_index=train_articles_index,
            law_retriever=law_retriever,
            input_fields=input_fields,
            query_fields=query_fields,
            broad_top_k_case=args.broad_top_k_case,
            top_k_case=args.top_k_case,
            max_additional_law_rounds=args.max_additional_law_rounds,
            provider=provider,
            model_name=model_name,
            only_blhs=args.only_blhs,
            use_provider_fallback=use_provider_fallback,
        )
        per_doc.append(result)
        print(f"{result['status']}: {path.name} ({result.get('reason', '')})")
        save_eval_results(results_out, config=config, summary=None, per_doc=per_doc)

    summary = _aggregate(per_doc)
    save_eval_results(results_out, config=config, summary=summary, per_doc=per_doc)

    print("DONE")
    print(f"Saved: {results_out}")


if __name__ == "__main__":
    main()
