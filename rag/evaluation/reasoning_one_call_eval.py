"""Evaluate one-call generic LLM judgment prediction over chunk/test.

This is the non-agentic baseline for ``reasoning_act_eval.py``. It uses the
same input fields and scoring shape, but performs exactly one structured LLM
call per document with no law lookup, no candidate loop, and no similar-case
retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from rag.evaluation.eval_utils import (
    _aggregate,
    _compact_defendant_item,
    _extract_doc_id,
    _extract_gt_defendants,
    _extract_input_payload,
    _extract_phat_tu_months,
    _extract_pred_defendants,
    _macro_mean,
    _name_key,
    _parse_fields,
    _safe_float,
    _set_prf,
    _to_dieu_set,
    save_eval_results,
)
from rag.generation.reasoning_act import MANDATORY_SUPPORTING_DIEU
from rag.generation.schemas import GenerationOutput, PredictedDefendant, PredictedLawClause, build_output_schema_instruction
from rag.llm.providers import (
    LLMProvider,
    default_model_for_provider,
    generate_structured_output,
    generate_structured_output_with_fallback,
)

load_dotenv()

SYNTHETIC_SUMMARY_FORMAT_NOTE = (
    "case_fields.Synthetic_summary may be a JSON array encoded as a string. "
    "Each array item is a separate first-person story for one defendant. "
    "Map each story to the defendant named inside that story and keep per-defendant facts separate."
)

# Keep this in sync with reasoning_act_eval's mandatory supporting articles.
ALWAYS_CHECK_DIEU = MANDATORY_SUPPORTING_DIEU


class PredictedLawClauseNoFacts(BaseModel):
    Dieu: str | None = Field(default=None, description="Law article number (Dieu), e.g. '173'.")
    Khoan: str | None = Field(default=None, description="Law clause number (Khoan) within the Dieu.")
    Diem: str | None = Field(default=None, description="Law point letter (Diem), typically lowercase letter like 'a'.")
    Bo_Luat_Va_Van_Ban_Khac: str | None = Field(
        default=None,
        description="Legal source name/code (e.g. BLHS) for the cited clause.",
    )


class PredictedDefendantNoAnalysis(BaseModel):
    Bi_Cao: str = Field(description="Defendant name exactly matching the case context.")
    Toi_Danh: str | None = Field(default=None, description="Offense/crime name for the defendant.")
    Applied_Law_Clauses: list[PredictedLawClauseNoFacts] = Field(
        default_factory=list,
        description="List of applicable legal clauses supporting the verdict for this defendant.",
    )
    Phat_Tu: str | None = Field(
        default=None,
        description="Final concrete imprisonment verdict text, not a sentencing range.",
    )
    Phat_Tien: str | None = Field(
        default=None,
        description="Monetary fine verdict text for the defendant when applicable; keep null if not imposed.",
    )
    Trach_Nhiem_Dan_Su: str | None = Field(
        default=None,
        description="Civil liability decision for the defendant when applicable.",
    )


class GenerationOutputNoAnalysis(BaseModel):
    defendants: list[PredictedDefendantNoAnalysis] = Field(description="Per-defendant structured predictions.")
    Xu_Ly_Vat_Chung: str | None = Field(
        default=None,
        description="Decision on handling/seizure/disposal of physical evidence related to the case.",
    )


def _restore_generation_output(prediction: GenerationOutput | GenerationOutputNoAnalysis) -> GenerationOutput:
    if isinstance(prediction, GenerationOutput):
        return prediction
    return GenerationOutput(
        defendants=[
            PredictedDefendant(
                Bi_Cao=item.Bi_Cao,
                Phan_Tich_Phap_Ly=None,
                Toi_Danh=item.Toi_Danh,
                Applied_Law_Clauses=[
                    PredictedLawClause(
                        Dieu=clause.Dieu,
                        Khoan=clause.Khoan,
                        Diem=clause.Diem,
                        Tinh_tiet_ap_dung=None,
                        Bo_Luat_Va_Van_Ban_Khac=clause.Bo_Luat_Va_Van_Ban_Khac,
                    )
                    for clause in item.Applied_Law_Clauses
                ],
                Phat_Tu=item.Phat_Tu,
                Phat_Tien=item.Phat_Tien,
                Trach_Nhiem_Dan_Su=item.Trach_Nhiem_Dan_Su,
            )
            for item in prediction.defendants
        ],
        Xu_Ly_Vat_Chung=prediction.Xu_Ly_Vat_Chung,
    )


def _build_one_call_prompt(
    *,
    doc_id: str,
    case_payload: dict[str, str],
    omit_phan_tich_phap_ly: bool,
) -> tuple[str, str]:
    system_prompt = (
        "You are a Vietnamese criminal judgment prediction assistant. "
        "Return only valid JSON matching the requested schema."
    )
    payload = {
        "doc_id": doc_id,
        "case_fields": case_payload,
        "input_format": {
            "Synthetic_summary": SYNTHETIC_SUMMARY_FORMAT_NOTE,
            "per_defendant_requirement": (
                "Generate one defendants entry for each defendant story and use only that defendant's own story plus shared case facts."
            ),
        },
        "task": [
            "Predict the final judgment directly from case_fields in one response.",
            "For each defendant, predict Toi_Danh, Applied_Law_Clauses, Phat_Tu, Phat_Tien, and Trach_Nhiem_Dan_Su.",
            "Preserve defendant names exactly as provided.",
            f"Always check these BLHS articles when deciding applicable clauses: {', '.join(ALWAYS_CHECK_DIEU)}.",
            "Applied_Law_Clauses must contain only clauses inferable from case_fields.",
            "Do not include checked-but-not-applicable articles in Applied_Law_Clauses.",
            "Phat_Tu must be a single concrete verdict statement, not a sentencing range.",
            "Do not ask for or rely on external law text, retrieved cases, or other context.",
        ],
        "output_schema": build_output_schema_instruction(
            GenerationOutputNoAnalysis if omit_phan_tich_phap_ly else GenerationOutput
        ),
    }
    if not omit_phan_tich_phap_ly:
        payload["task"].insert(
            6,
            "For each Applied_Law_Clauses item, fill Tinh_tiet_ap_dung with concise supporting facts from case_fields.",
        )
    return system_prompt, json.dumps(payload, ensure_ascii=False, indent=2)


def run_one_call_generation(
    *,
    data: dict[str, Any],
    doc_id: str,
    provider: LLMProvider | str,
    model_name: str,
    use_provider_fallback: bool = True,
    input_fields: list[str] | None = None,
    omit_phan_tich_phap_ly: bool = False,
) -> dict[str, Any]:
    input_fields = input_fields or ["THONG_TIN_CHUNG.Thong_Tin_Bi_Cao", "Synthetic_summary_2"]
    case_payload = _extract_input_payload(data, input_fields)
    if not case_payload:
        raise ValueError("empty_input_payload")

    output_model = GenerationOutputNoAnalysis if omit_phan_tich_phap_ly else GenerationOutput
    system_prompt, user_prompt = _build_one_call_prompt(
        doc_id=doc_id,
        case_payload=case_payload,
        omit_phan_tich_phap_ly=omit_phan_tich_phap_ly,
    )
    if use_provider_fallback:
        prediction, usage = generate_structured_output_with_fallback(
            preferred_provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
        )
    else:
        prediction, usage = generate_structured_output(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
        )

    return {
        "prediction": _restore_generation_output(prediction),
        "usage": usage,
        "llm_input_payload": case_payload,
        "omit_phan_tich_phap_ly": omit_phan_tich_phap_ly,
    }


def safe_run_one_call_generation(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return run_one_call_generation(**kwargs), None
    except (ValidationError, json.JSONDecodeError) as exc:
        return None, f"parse_error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"generation_error: {exc}"


def _evaluate_single_doc(
    *,
    path: Path,
    data: dict[str, Any],
    input_fields: list[str],
    provider: LLMProvider,
    model_name: str,
    only_blhs: bool,
    use_provider_fallback: bool,
    omit_phan_tich_phap_ly: bool,
) -> dict[str, Any]:
    doc_id = _extract_doc_id(data, path.stem)
    gt_defendants = _extract_gt_defendants(data, only_blhs=only_blhs)

    baseline_result, error = safe_run_one_call_generation(
        data=data,
        doc_id=doc_id,
        provider=provider,
        model_name=model_name,
        use_provider_fallback=use_provider_fallback,
        input_fields=input_fields,
        omit_phan_tich_phap_ly=omit_phan_tich_phap_ly,
    )

    if baseline_result is None:
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

    pred_output = baseline_result["prediction"]
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

    usage = baseline_result.get("usage", {})
    return {
        "doc_id": doc_id,
        "source_file": path.name,
        "status": "processed",
        "reason": "ok",
        "llm_input_payload": baseline_result.get("llm_input_payload", {}),
        "prediction_raw": pred_output.model_dump(),
        "trace": {
            "mode": "one_call_generic",
            "agentic_reasoning": False,
            "llm_calls": 1,
            "law_retrieval": False,
            "similar_case_retrieval": False,
            "omit_phan_tich_phap_ly": omit_phan_tich_phap_ly,
        },
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
        "_usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one-call generic LLM judgment prediction on test cases."
    )
    parser.add_argument("--test-dir", default="chunk/test")
    parser.add_argument("--results-out", default="output/reasoning_one_call_eval/results.json")
    parser.add_argument("--provider", choices=[p.value for p in LLMProvider], default="openrouter")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--input-fields",
        default="THONG_TIN_CHUNG.Thong_Tin_Bi_Cao,Synthetic_summary_2",
        help=(
            "Comma-separated fields passed to the one-call LLM prompt. "
            "Synthetic_summary may be a list of separate first-person defendant stories."
        ),
    )
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--only-blhs", action="store_true", default=True)
    parser.add_argument("--include-non-blhs", action="store_false", dest="only_blhs")
    parser.add_argument("--disable-provider-fallback", action="store_true", default=False)
    parser.add_argument(
        "--omit-phan-tich-phap-ly",
        action="store_true",
        default=False,
        help="Remove Phan_Tich_Phap_Ly and Tinh_tiet_ap_dung from the requested LLM output schema and task.",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    results_out = Path(args.results_out)

    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    input_fields = _parse_fields(args.input_fields)
    provider = LLMProvider(args.provider)
    model_name = args.model or default_model_for_provider(provider)
    use_provider_fallback = not args.disable_provider_fallback

    files = sorted(test_dir.glob("*.json"))
    if args.first_n is not None:
        if args.first_n < 1:
            raise ValueError("--first-n must be >= 1")
        files = files[: args.first_n]

    print(f"Found {len(files)} test files")
    print(f"Provider={provider.value} | Model={model_name}")
    print(f"Provider fallback enabled={use_provider_fallback}")
    print(f"Input fields={input_fields}")
    print(f"Synthetic_summary format={SYNTHETIC_SUMMARY_FORMAT_NOTE}")
    print(f"Always-checked Dieu={list(ALWAYS_CHECK_DIEU)}")
    print(f"Omit Phan_Tich_Phap_Ly={args.omit_phan_tich_phap_ly}")
    print("One-call baseline: no agentic loop, no law retrieval, no similar-case retrieval.")

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
        "mode": "one_call_generic",
        "test_dir": str(test_dir),
        "provider": provider.value,
        "model": model_name,
        "provider_fallback": use_provider_fallback,
        "input_fields": input_fields,
        "synthetic_summary_format": SYNTHETIC_SUMMARY_FORMAT_NOTE,
        "always_check_dieu": list(ALWAYS_CHECK_DIEU),
        "omit_phan_tich_phap_ly": args.omit_phan_tich_phap_ly,
        "agentic_reasoning": False,
        "llm_calls_per_doc": 1,
        "law_retrieval": False,
        "similar_case_retrieval": False,
        "only_blhs": args.only_blhs,
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
            input_fields=input_fields,
            provider=provider,
            model_name=model_name,
            only_blhs=args.only_blhs,
            use_provider_fallback=use_provider_fallback,
            omit_phan_tich_phap_ly=args.omit_phan_tich_phap_ly,
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

#uv run python -m rag.evaluation.reasoning_one_call_eval   --first-n 10   --omit-phan-tich-phap-ly   --results-out output/bench/one_call_first10_no_analysis.json
