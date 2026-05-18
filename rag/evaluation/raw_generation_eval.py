"""Evaluate raw LLM verdict generation without any legal-clause retrieval context.

Workflow per test case:
1. Build an input payload from only raw case fields (default: Synthetic_summary + defendant info).
2. Prompt an LLM to predict final verdict fields for each defendant.
3. Compare predictions against ground-truth verdict fields.
4. Save a JSON report with per-document details and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from rag.config import LEGAL_SOURCE_FIELD, VERDICT_FIELD
from rag.core.sentencing import extract_imprisonment_months
from rag.core.verdict_labels import is_blhs_legal_source, split_multi_value
from rag.generation.schemas import GenerationOutput
from rag.generation.schemas import build_output_schema_instruction
from rag.llm.providers import (
    LLMProvider,
    default_model_for_provider,
    generate_structured_output,
    generate_structured_output_with_fallback,
)

load_dotenv()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _name_key(name: str) -> str:
    folded = _strip_accents(_normalize_space(name)).lower()
    return re.sub(r"[^a-z0-9]+", "", folded)


def _norm_text(text: str | None) -> str:
    return _normalize_space(text or "").lower()


def _safe_float(num: float) -> float:
    return round(float(num), 6)


def _set_prf(pred: set[str], gt: set[str]) -> dict[str, float | int]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "f1": _safe_float(f1),
    }


def _to_dieu_set(signatures: set[str] | list[str]) -> set[str]:
    out: set[str] = set()
    for sig in signatures:
        raw = str(sig or "").strip()
        if not raw:
            continue
        dieu = raw.split("-")[0].strip()
        if dieu:
            out.add(dieu)
    return out


def _macro_mean(values: list[float]) -> float:
    return _safe_float(mean(values)) if values else 0.0


def _extract_doc_id(data: dict[str, Any], fallback: str) -> str:
    thong_tin = data.get("THONG_TIN_CHUNG") or {}
    if not isinstance(thong_tin, dict):
        thong_tin = {}
    value = thong_tin.get("Ma_Ban_An") or data.get("Ma_Ban_An") or fallback
    return str(value).strip() or fallback


def _extract_input_payload(data: dict[str, Any], fields: list[str]) -> dict[str, str]:
    def _resolve_field_value(field: str) -> Any:
        if field in {"Defendant_info", "defendant_info", "Thong_Tin_Bi_Cao"}:
            info = data.get("THONG_TIN_CHUNG")
            if isinstance(info, dict):
                return info.get("Thong_Tin_Bi_Cao")
            return None

        if "." in field:
            cur: Any = data
            for part in field.split("."):
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(part)
            return cur

        return data.get(field)

    payload: dict[str, str] = {}
    for field in fields:
        value = _resolve_field_value(field)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
        else:
            text = json.dumps(value, ensure_ascii=False)
        if text:
            payload[field] = text
    return payload


def _norm_token(token: Any, *, lowercase: bool) -> str:
    text = str(token or "").strip()
    text = re.sub(r"^(dieu|điều|khoan|khoản|diem|điểm)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" .")
    text = re.sub(r"\s+", "", text)
    return text.lower() if lowercase else text


def _build_signatures_from_basis_item(item: dict[str, Any]) -> set[str]:
    dieu_tokens = split_multi_value(item.get("Dieu"), lowercase=False)
    khoan_tokens = split_multi_value(item.get("Khoan"), lowercase=False)
    diem_tokens = split_multi_value(item.get("Diem"), lowercase=True)

    out: set[str] = set()
    for dieu in dieu_tokens:
        dieu_norm = _norm_token(dieu, lowercase=False)
        if not dieu_norm:
            continue
        if khoan_tokens and diem_tokens:
            for khoan in khoan_tokens:
                khoan_norm = _norm_token(khoan, lowercase=False)
                for diem in diem_tokens:
                    diem_norm = _norm_token(diem, lowercase=True)
                    if khoan_norm and diem_norm:
                        out.add(f"{dieu_norm}-{khoan_norm}-{diem_norm}")
        elif khoan_tokens:
            for khoan in khoan_tokens:
                khoan_norm = _norm_token(khoan, lowercase=False)
                if khoan_norm:
                    out.add(f"{dieu_norm}-{khoan_norm}")
        elif diem_tokens:
            for diem in diem_tokens:
                diem_norm = _norm_token(diem, lowercase=True)
                if diem_norm:
                    out.add(f"{dieu_norm}-{diem_norm}")
        else:
            out.add(dieu_norm)
    return out


def _extract_gt_defendants(data: dict[str, Any], *, only_blhs: bool) -> list[dict[str, Any]]:
    verdict_items = data.get(VERDICT_FIELD)
    if not isinstance(verdict_items, list):
        return []

    out: list[dict[str, Any]] = []
    for item in verdict_items:
        if not isinstance(item, dict):
            continue
        can_cu = item.get("Can_Cu_Dieu_Luat")
        signatures: set[str] = set()
        if isinstance(can_cu, list):
            for basis_item in can_cu:
                if not isinstance(basis_item, dict):
                    continue
                if only_blhs and not is_blhs_legal_source(basis_item.get(LEGAL_SOURCE_FIELD)):
                    continue
                signatures |= _build_signatures_from_basis_item(basis_item)

        out.append(
            {
                "Bi_Cao": _normalize_space(str(item.get("Bi_Cao") or "")),
                "Phan_Tich_Phap_Ly": "",
                "Toi_Danh": _normalize_space(str(item.get("Pham_Toi") or "")),
                "Applied_Law_Clauses": sorted(signatures),
                "Applied_Law_Clauses_Detailed": [],
                "Phat_Tu": _normalize_space(str(item.get("Phat_Tu") or "")),
                "Phat_Tien": _normalize_space(str(item.get("Phat_Tien") or "")),
                "Trach_Nhiem_Dan_Su": _normalize_space(str(item.get("Trach_Nhiem_Dan_Su") or "")),
                "Xu_Ly_Vat_Chung": _normalize_space(str(item.get("Xu_Ly_Vat_Chung") or "")),
            }
        )
    return out


def _extract_pred_defendants(pred: GenerationOutput, *, only_blhs: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for defendant in pred.defendants:
        signatures: set[str] = set()
        clause_details: list[dict[str, str]] = []
        for clause in defendant.Applied_Law_Clauses:
            if only_blhs and not is_blhs_legal_source(clause.Bo_Luat_Va_Van_Ban_Khac):
                continue
            clause_signatures = _build_signatures_from_basis_item(
                {
                    "Dieu": clause.Dieu,
                    "Khoan": clause.Khoan,
                    "Diem": clause.Diem,
                }
            )
            signatures |= clause_signatures
            tinh_tiet = _normalize_space(clause.Tinh_tiet_ap_dung or "")
            for signature in sorted(clause_signatures):
                clause_details.append(
                    {
                        "signature": signature,
                        "Tinh_tiet_ap_dung": tinh_tiet,
                    }
                )

        out.append(
            {
                "Bi_Cao": _normalize_space(defendant.Bi_Cao),
                "Phan_Tich_Phap_Ly": _normalize_space(defendant.Phan_Tich_Phap_Ly),
                "Toi_Danh": _normalize_space(defendant.Toi_Danh or ""),
                "Applied_Law_Clauses": sorted(signatures),
                "Applied_Law_Clauses_Detailed": clause_details,
                "Phat_Tu": _normalize_space(defendant.Phat_Tu or ""),
                "Phat_Tien": _normalize_space(defendant.Phat_Tien or ""),
                "Trach_Nhiem_Dan_Su": _normalize_space(defendant.Trach_Nhiem_Dan_Su or ""),
                "Xu_Ly_Vat_Chung": _normalize_space(defendant.Xu_Ly_Vat_Chung or ""),
            }
        )
    return out


def _extract_phat_tu_months(text: str | None) -> int:
    return extract_imprisonment_months(text)


def _build_prompts(*, doc_id: str, case_payload: dict[str, str]) -> tuple[str, str]:
    system_prompt = (
        "You are an expert Vietnamese criminal judgment assistant. "
        "Return only valid JSON. Infer verdict outcomes strictly from provided case facts."
    )

    requirements = [
        "For each defendant, provide Phan_Tich_Phap_Ly first, then conclude with Applied_Law_Clauses, Toi_Danh, Phat_Tu, and Trach_Nhiem_Dan_Su.",
        "For each defendant, predict applied law clauses, Toi_Danh, Phat_Tu, and Trach_Nhiem_Dan_Su.",
        "For each Applied_Law_Clauses item, fill Tinh_tiet_ap_dung with concise case facts that justify applying that clause.",
        "Use only the provided case_fields content.",
        "Do not rely on or ask for any external legal clause context.",
    ]
    constraints = [
        "No markdown, no extra explanation.",
        "Do not invent defendant names not supported by case_fields.",
        "Phan_Tich_Phap_Ly should be a concise legal reasoning based only on provided facts.",
        "Phat_Tu must be a single concrete verdict statement; do not output a range like 'từ X đến Y năm tù'.",
        "Applied_Law_Clauses must reflect only clauses inferable from case_fields.",
        "Tinh_tiet_ap_dung must cite only facts present in case_fields; do not invent facts.",
    ]

    input_payload = {
        "doc_id": doc_id,
        "case_fields": case_payload,
        "task": {
            "requirement": requirements,
            "reasoning_instruction": [
                "Use the factual timeline in case_fields to identify offense behavior.",
                "Use defendant information in case_fields to reflect role and sentencing context.",
                "Infer a concrete verdict for each defendant without legal retrieval references.",
            ],
            "output_schema": build_output_schema_instruction(GenerationOutput),
            "constraints": constraints,
        },
    }
    user_prompt = json.dumps(input_payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def _evaluate_single_doc(
    *,
    path: Path,
    data: dict[str, Any],
    input_fields: list[str],
    provider: LLMProvider,
    model_name: str,
    only_blhs: bool,
    use_provider_fallback: bool,
) -> dict[str, Any]:
    doc_id = _extract_doc_id(data, path.stem)
    case_payload = _extract_input_payload(data, input_fields)

    if not case_payload:
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "skipped",
            "reason": "empty_input_payload",
            "ground_truth": {"defendants": _extract_gt_defendants(data, only_blhs=only_blhs)},
        }

    gt_defendants = _extract_gt_defendants(data, only_blhs=only_blhs)

    system_prompt, user_prompt = _build_prompts(
        doc_id=doc_id,
        case_payload=case_payload,
    )

    usage: dict[str, Any] = {}
    parse_error = None
    generation_error = None
    pred_output: GenerationOutput | None = None
    llm_used_provider = provider.value
    llm_used_model = model_name

    try:
        if use_provider_fallback:
            pred_output, usage = generate_structured_output_with_fallback(
                preferred_provider=provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=GenerationOutput,
            )
        else:
            pred_output, usage = generate_structured_output(
                provider=provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=GenerationOutput,
            )
    except (ValidationError, json.JSONDecodeError) as exc:
        parse_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        generation_error = str(exc)

    if usage:
        llm_used_provider = str(usage.get("provider") or llm_used_provider)
        llm_used_model = str(usage.get("model") or llm_used_model)

    if pred_output is None:
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "failed",
            "reason": "parse_error" if parse_error else "generation_error",
            "llm_input_payload": case_payload,
            "defendants": [
                {
                    "Bi_Cao": item.get("Bi_Cao", ""),
                    "ground_truth": item,
                    "prediction": None,
                }
                for item in gt_defendants
            ],
            "llm": {
                "requested_provider": provider.value,
                "requested_model": model_name,
                "used_provider": llm_used_provider,
                "used_model": llm_used_model,
                "provider_fallback_enabled": use_provider_fallback,
            },
            "error": parse_error or generation_error,
            "_usage": usage,
        }

    pred_defendants = _extract_pred_defendants(pred_output, only_blhs=only_blhs)

    gt_by_name = {_name_key(item["Bi_Cao"]): item for item in gt_defendants if item.get("Bi_Cao")}
    pred_by_name = {_name_key(item["Bi_Cao"]): item for item in pred_defendants if item.get("Bi_Cao")}
    all_keys = sorted(set(gt_by_name) | set(pred_by_name))
    matched_keys = sorted(set(gt_by_name) & set(pred_by_name))
    gt_only = sorted(set(gt_by_name) - set(pred_by_name))
    pred_only = sorted(set(pred_by_name) - set(gt_by_name))

    toi_danh_exact_values: list[float] = []
    trach_nhiem_exact_values: list[float] = []
    clause_precision_values: list[float] = []
    clause_recall_values: list[float] = []
    clause_f1_values: list[float] = []
    phat_tu_sq_err_values: list[float] = []
    defendants: list[dict[str, Any]] = []

    for key in all_keys:
        gt_item = gt_by_name.get(key)
        pred_item = pred_by_name.get(key)

        gt_toi = _norm_text((gt_item or {}).get("Toi_Danh"))
        pred_toi = _norm_text((pred_item or {}).get("Toi_Danh"))
        toi_danh_exact = float(gt_toi == pred_toi and bool(gt_toi or pred_toi))
        toi_danh_exact_values.append(toi_danh_exact)

        gt_trach = _norm_text((gt_item or {}).get("Trach_Nhiem_Dan_Su"))
        pred_trach = _norm_text((pred_item or {}).get("Trach_Nhiem_Dan_Su"))
        trach_nhiem_exact = float(gt_trach == pred_trach and bool(gt_trach or pred_trach))
        trach_nhiem_exact_values.append(trach_nhiem_exact)

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
                "ground_truth": gt_item,
                "prediction": pred_item,
                "metrics": {
                    "toi_danh_exact": bool(toi_danh_exact),
                    "trach_nhiem_dan_su_exact": bool(trach_nhiem_exact),
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

    return {
        "doc_id": doc_id,
        "source_file": path.name,
        "status": "processed",
        "reason": "ok",
        "llm_input_payload": case_payload,
        "llm": {
            "requested_provider": provider.value,
            "requested_model": model_name,
            "used_provider": llm_used_provider,
            "used_model": llm_used_model,
            "provider_fallback_enabled": use_provider_fallback,
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
            "toi_danh_exact_macro": _macro_mean(toi_danh_exact_values),
            "trach_nhiem_dan_su_exact_macro": _macro_mean(trach_nhiem_exact_values),
            "law_clause_precision_macro": _macro_mean(clause_precision_values),
            "law_clause_recall_macro": _macro_mean(clause_recall_values),
            "law_clause_f1_macro": _macro_mean(clause_f1_values),
            "phat_tu_rmse_months": phat_tu_rmse_months,
            "n_defendants_scored": len(all_keys),
        },
        "_usage": usage,
    }


def _parse_fields(raw: str) -> list[str]:
    fields = [part.strip() for part in raw.split(",") if part.strip()]
    if not fields:
        raise ValueError("At least one field must be provided")
    return fields


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    processed = [item for item in results if item.get("status") == "processed"]
    failed = [item for item in results if item.get("status") == "failed"]
    skipped = [item for item in results if item.get("status") == "skipped"]

    toi_danh_exact = [float(item["doc_metrics"]["toi_danh_exact_macro"]) for item in processed]
    trach_nhiem_exact = [float(item["doc_metrics"]["trach_nhiem_dan_su_exact_macro"]) for item in processed]
    clause_p = [float(item["doc_metrics"]["law_clause_precision_macro"]) for item in processed]
    clause_r = [float(item["doc_metrics"]["law_clause_recall_macro"]) for item in processed]
    clause_f1 = [float(item["doc_metrics"]["law_clause_f1_macro"]) for item in processed]
    rmse_months = [float(item["doc_metrics"]["phat_tu_rmse_months"]) for item in processed]

    return {
        "n_total": len(results),
        "n_processed": len(processed),
        "n_failed": len(failed),
        "n_skipped": len(skipped),
        "metrics": {
            "toi_danh_exact_macro": _macro_mean(toi_danh_exact),
            "trach_nhiem_dan_su_exact_macro": _macro_mean(trach_nhiem_exact),
            "law_clause_set_precision_macro": _macro_mean(clause_p),
            "law_clause_set_recall_macro": _macro_mean(clause_r),
            "law_clause_set_f1_macro": _macro_mean(clause_f1),
            "phat_tu_rmse_months_macro": _macro_mean(rmse_months),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw verdict generation from case facts only "
            "(no retrieval and no legal-clause context)."
        )
    )
    parser.add_argument("--test-dir", default="chunk/test", help="Directory with test JSON files")
    parser.add_argument("--results-out", default="output/generation_eval/raw_verdict_generation_eval.json")
    parser.add_argument("--provider", choices=[p.value for p in LLMProvider], default="openrouter")
    parser.add_argument("--model", default=None, help="Provider model override")
    parser.add_argument(
        "--input-fields",
        default="Synthetic_summary,THONG_TIN_CHUNG.Thong_Tin_Bi_Cao",
        help="Comma-separated fields passed to the LLM payload",
    )
    parser.add_argument("--first-n", type=int, default=None, help="Process only first N files")
    parser.add_argument(
        "--only-blhs",
        action="store_true",
        default=True,
        help="When enabled, keep only BLHS clauses in ground truth and predictions",
    )
    parser.add_argument(
        "--include-non-blhs",
        action="store_false",
        dest="only_blhs",
        help="Include non-BLHS clauses in judging (overrides default BLHS-only filtering)",
    )
    parser.add_argument(
        "--disable-provider-fallback",
        action="store_true",
        default=False,
        help="Disable automatic fallback (OpenRouter free -> AI Studio -> OpenRouter standard)",
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
    print(f"BLHS-only clause filtering={args.only_blhs}")
    print("Raw-generation mode: no retrieval context and no legal clause context are supplied.")

    per_doc: list[dict[str, Any]] = []
    for path in files:
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
        )
        per_doc.append(result)
        print(f"{result['status']}: {path.name} ({result.get('reason', '')})")

    summary = _aggregate(per_doc)
    output = {
        "config": {
            "test_dir": str(test_dir),
            "provider": provider.value,
            "model": model_name,
            "provider_fallback": use_provider_fallback,
            "input_fields": input_fields,
            "only_blhs": args.only_blhs,
        },
        "summary": summary,
        "per_doc": per_doc,
    }

    results_out.parent.mkdir(parents=True, exist_ok=True)
    with open(results_out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print("DONE")
    print(f"Saved: {results_out}")


if __name__ == "__main__":
    main()
