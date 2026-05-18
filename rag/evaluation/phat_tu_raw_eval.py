"""Evaluate verdict reasoning when full court analysis and law text are provided.

Workflow per test case:
1. Extract NHAN_DINH_CUA_TOA_AN and defendant information from the case JSON.
2. Extract ground-truth applied legal clauses (Can_Cu_Dieu_Luat) from PHAN_QUYET_CUA_TOA_SO_THAM.
3. Retrieve detailed legal article text for those clauses from law_doc.json.
4. Prompt the LLM to generate final verdict outputs using GenerationOutput schema.
5. Compare predictions to ground truth and save aggregate/per-doc metrics.
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

from rag.config import DEFAULT_LAW_DOC_PATH, LEGAL_SOURCE_FIELD, VERDICT_FIELD
from rag.core.law_retriever import LawClauseRetriever
from rag.core.sentencing import extract_imprisonment_months
from rag.core.verdict_labels import is_blhs_legal_source, split_multi_value
from rag.generation.schemas import GenerationOutput, build_output_schema_instruction
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


def _macro_mean(values: list[float]) -> float:
    return _safe_float(mean(values)) if values else 0.0


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


def _extract_doc_id(data: dict[str, Any], fallback: str) -> str:
    thong_tin = data.get("THONG_TIN_CHUNG") or {}
    if not isinstance(thong_tin, dict):
        thong_tin = {}
    value = thong_tin.get("Ma_Ban_An") or data.get("Ma_Ban_An") or fallback
    return str(value).strip() or fallback


def _extract_defendant_names(data: dict[str, Any]) -> list[str]:
    thong_tin = data.get("THONG_TIN_CHUNG") or {}
    if not isinstance(thong_tin, dict):
        return []
    raw_people = thong_tin.get("Thong_Tin_Bi_Cao")
    if not isinstance(raw_people, list):
        return []

    names: list[str] = []
    for person in raw_people:
        if not isinstance(person, dict):
            continue
        name = _normalize_space(str(person.get("Ho_Ten") or ""))
        if name:
            names.append(name)

    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        key = _name_key(name)
        if key and key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped


def _extract_nhan_dinh_text(data: dict[str, Any]) -> str:
    section = data.get("NHAN_DINH_CUA_TOA_AN")
    if isinstance(section, str):
        return _normalize_space(section)
    if isinstance(section, dict):
        parts: list[str] = []
        for key in sorted(section.keys(), key=str):
            value = section.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                parts.append(f"{key}: {text}")
        return "\n\n".join(parts).strip()
    return ""


def _extract_gt_defendants(data: dict[str, Any], *, only_blhs: bool) -> list[dict[str, Any]]:
    verdict_items = data.get(VERDICT_FIELD)
    if not isinstance(verdict_items, list):
        return []

    out: list[dict[str, Any]] = []
    for item in verdict_items:
        if not isinstance(item, dict):
            continue

        signatures: set[str] = set()
        can_cu = item.get("Can_Cu_Dieu_Luat")
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
                "Toi_Danh": _normalize_space(str(item.get("Pham_Toi") or "")),
                "Applied_Law_Clauses": sorted(signatures),
                "Phat_Tu": _normalize_space(str(item.get("Phat_Tu") or "")),
                "Phat_Tien": _normalize_space(str(item.get("Phat_Tien") or "")),
                "Trach_Nhiem_Dan_Su": _normalize_space(str(item.get("Trach_Nhiem_Dan_Su") or "")),
                "Xu_Ly_Vat_Chung": _normalize_space(str(item.get("Xu_Ly_Vat_Chung") or "")),
                "Hinh_Phat_Bo_Sung": _normalize_space(str(item.get("Hinh_Phat_Bo_Sung") or "")),
                "An_Phi": _normalize_space(str(item.get("An_Phi") or "")),
            }
        )

    return out


def _extract_pred_defendants(pred: GenerationOutput, *, only_blhs: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for defendant in pred.defendants:
        signatures: set[str] = set()
        for clause in defendant.Applied_Law_Clauses:
            if only_blhs and not is_blhs_legal_source(clause.Bo_Luat_Va_Van_Ban_Khac):
                continue
            signatures |= _build_signatures_from_basis_item(
                {
                    "Dieu": clause.Dieu,
                    "Khoan": clause.Khoan,
                    "Diem": clause.Diem,
                }
            )

        out.append(
            {
                "Bi_Cao": _normalize_space(defendant.Bi_Cao),
                "Toi_Danh": _normalize_space(defendant.Toi_Danh or ""),
                "Applied_Law_Clauses": sorted(signatures),
                "Phat_Tu": _normalize_space(defendant.Phat_Tu or ""),
                "Phat_Tien": _normalize_space(defendant.Phat_Tien or ""),
                "Trach_Nhiem_Dan_Su": _normalize_space(defendant.Trach_Nhiem_Dan_Su or ""),
                "Xu_Ly_Vat_Chung": _normalize_space(defendant.Xu_Ly_Vat_Chung or ""),
            }
        )

    return out


def _build_law_context(signatures: list[str], retriever: LawClauseRetriever) -> dict[str, Any]:
    retrieved_clause_texts: list[dict[str, Any]] = []
    missing_signatures: list[str] = []
    warnings: list[str] = []

    for result in retriever.retrieve_many(signatures):
        signature = str(result.get("query") or "").strip()
        found = bool(result.get("found", False))
        if not found:
            if signature:
                missing_signatures.append(signature)
            reason = str(result.get("reason") or "not_found")
            warnings.append(f"{reason}:{signature}")
            continue

        retrieved_clause_texts.append(
            {
                "signature": result.get("normalized") or signature,
                "query": signature,
                "retrieval_level": result.get("level"),
                "dieu": result.get("dieu"),
                "khoan": result.get("khoan"),
                "diem": result.get("diem"),
                "context_text": str(result.get("text") or "").strip(),
            }
        )

    return {
        "signature_inputs": signatures,
        "retrieved_clause_texts": retrieved_clause_texts,
        "missing_signatures": missing_signatures,
        "warnings": warnings,
    }


def _extract_phat_tu_months(text: str | None) -> int:
    return extract_imprisonment_months(text)


def _build_prompts(
    *,
    doc_id: str,
    nhan_dinh_text: str,
    defendant_names: list[str],
    applied_clause_signatures: list[str],
    law_context: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert Vietnamese criminal judgment assistant. "
        "Return only valid JSON and follow the required output schema exactly. "
        "You must predict concrete final verdict outputs, not summarize source text."
    )

    requirements = [
        "Predict final verdict outcomes per defendant.",
        "Use only provided NHAN_DINH_CUA_TOA_AN, applied clause list, and retrieved article text.",
        "Applied_Law_Clauses must be supported by retrieved law context.",
        "Phat_Tu must be a concrete final verdict statement with a specific imprisonment duration (months/years), not a range.",
        "Never output uncertainty or meta-commentary such as 'khong duoc neu', 'khong ro', 'khong xac dinh', 'phu hop', or 'can nhac'.",
        "Do not summarize case narrative; infer and output adjudicated results for each field.",
        "Trach_Nhiem_Dan_Su must state a final civil-liability disposition in verdict style, not a recap sentence from the analysis section.",
        "Do not add any defendant not present in defendant_names.",
        "Populate fields available in GenerationOutput only.",
    ]

    output_style = {
        "Phat_Tu": [
            "Use verdict-style text, e.g. '15 nam tu' or '24 thang tu cho huong an treo, thoi gian thu thach 48 thang'.",
            "Never use ambiguous ranges such as 'tu 01 nam den 05 nam' unless a single final value is also explicitly provided.",
        ],
        "Trach_Nhiem_Dan_Su": [
            "Output the court disposition, not factual recap.",
            "Prefer concise verdict forms: 'Khong xem xet', 'Buoc bi cao boi thuong ... dong', or 'Ghi nhan da boi thuong ... dong, khong con yeu cau'.",
            "Do not copy narrative clauses like 'bi hai da nhan du tien boi thuong ...' without converting to a verdict disposition.",
        ],
    }

    payload = {
        "doc_id": doc_id,
        "defendant_names": defendant_names,
        "nhan_dinh_cua_toa_an": nhan_dinh_text,
        "applied_clause_signatures": applied_clause_signatures,
        "law_context": {
            "retrieved_clause_texts": law_context.get("retrieved_clause_texts", []),
            "missing_signatures": law_context.get("missing_signatures", []),
        },
        "task": {
            "requirements": requirements,
            "output_style": output_style,
            "constraints": [
                "Output JSON only.",
                "No markdown and no extra explanations.",
                "No external legal knowledge beyond provided inputs.",
                "No placeholder text, no hedging, and no references to missing information.",
            ],
            "output_schema": build_output_schema_instruction(GenerationOutput),
        },
    }

    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def _evaluate_single_doc(
    *,
    path: Path,
    data: dict[str, Any],
    provider: LLMProvider,
    model_name: str,
    only_blhs: bool,
    use_provider_fallback: bool,
    law_retriever: LawClauseRetriever,
) -> dict[str, Any]:
    doc_id = _extract_doc_id(data, path.stem)
    defendant_names = _extract_defendant_names(data)
    nhan_dinh_text = _extract_nhan_dinh_text(data)
    gt_defendants = _extract_gt_defendants(data, only_blhs=only_blhs)

    gt_clause_signatures = sorted(
        {
            sig
            for defendant in gt_defendants
            for sig in defendant.get("Applied_Law_Clauses", [])
            if str(sig).strip()
        }
    )
    law_context = _build_law_context(gt_clause_signatures, law_retriever)

    llm_input_payload = {
        "doc_id": doc_id,
        "defendant_names": defendant_names,
        "nhan_dinh_cua_toa_an": nhan_dinh_text,
        "applied_clause_signatures": gt_clause_signatures,
        "law_context": {
            "retrieved_clause_texts": law_context.get("retrieved_clause_texts", []),
            "missing_signatures": law_context.get("missing_signatures", []),
        },
    }

    if not nhan_dinh_text:
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "skipped",
            "reason": "missing_nhan_dinh_cua_toa_an",
            "llm_input_payload": llm_input_payload,
            "ground_truth": {"defendants": gt_defendants},
        }

    if not gt_defendants:
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "skipped",
            "reason": "missing_ground_truth_verdict",
            "llm_input_payload": llm_input_payload,
            "ground_truth": {"defendants": gt_defendants},
        }

    if not gt_clause_signatures:
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "skipped",
            "reason": "empty_ground_truth_clause_signatures",
            "llm_input_payload": llm_input_payload,
            "ground_truth": {"defendants": gt_defendants},
        }

    if not law_context.get("retrieved_clause_texts"):
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "skipped",
            "reason": "missing_law_context",
            "llm_input_payload": llm_input_payload,
            "ground_truth": {"defendants": gt_defendants},
            "law_context_warnings": law_context.get("warnings", []),
        }

    system_prompt, user_prompt = _build_prompts(
        doc_id=doc_id,
        nhan_dinh_text=nhan_dinh_text,
        defendant_names=defendant_names,
        applied_clause_signatures=gt_clause_signatures,
        law_context=law_context,
    )

    parse_error = None
    generation_error = None
    usage: dict[str, Any] = {}
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
            "llm_input_payload": llm_input_payload,
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
    xu_ly_vat_chung_exact_values: list[float] = []
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

        gt_xlvc = _norm_text((gt_item or {}).get("Xu_Ly_Vat_Chung"))
        pred_xlvc = _norm_text((pred_item or {}).get("Xu_Ly_Vat_Chung"))
        xu_ly_vat_chung_exact = float(gt_xlvc == pred_xlvc and bool(gt_xlvc or pred_xlvc))
        xu_ly_vat_chung_exact_values.append(xu_ly_vat_chung_exact)

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
                    "xu_ly_vat_chung_exact": bool(xu_ly_vat_chung_exact),
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
        "llm_input_payload": llm_input_payload,
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
        "law_context": {
            "n_retrieved": len(law_context.get("retrieved_clause_texts", [])),
            "n_missing": len(law_context.get("missing_signatures", [])),
            "warnings": law_context.get("warnings", []),
        },
        "defendants": defendants,
        "doc_metrics": {
            "toi_danh_exact_macro": _macro_mean(toi_danh_exact_values),
            "trach_nhiem_dan_su_exact_macro": _macro_mean(trach_nhiem_exact_values),
            "xu_ly_vat_chung_exact_macro": _macro_mean(xu_ly_vat_chung_exact_values),
            "law_clause_precision_macro": _macro_mean(clause_precision_values),
            "law_clause_recall_macro": _macro_mean(clause_recall_values),
            "law_clause_f1_macro": _macro_mean(clause_f1_values),
            "phat_tu_rmse_months": phat_tu_rmse_months,
            "n_defendants_scored": len(all_keys),
        },
        "_usage": usage,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    processed = [item for item in results if item.get("status") == "processed"]
    failed = [item for item in results if item.get("status") == "failed"]
    skipped = [item for item in results if item.get("status") == "skipped"]

    toi_danh_exact = [float(item["doc_metrics"]["toi_danh_exact_macro"]) for item in processed]
    trach_nhiem_exact = [float(item["doc_metrics"]["trach_nhiem_dan_su_exact_macro"]) for item in processed]
    xu_ly_vat_chung_exact = [float(item["doc_metrics"]["xu_ly_vat_chung_exact_macro"]) for item in processed]
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
            "xu_ly_vat_chung_exact_macro": _macro_mean(xu_ly_vat_chung_exact),
            "law_clause_set_precision_macro": _macro_mean(clause_p),
            "law_clause_set_recall_macro": _macro_mean(clause_r),
            "law_clause_set_f1_macro": _macro_mean(clause_f1),
            "phat_tu_rmse_months_macro": _macro_mean(rmse_months),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate verdict reasoning from NHAN_DINH_CUA_TOA_AN and ground-truth law context "
            "(no embedding retrieval)."
        )
    )
    parser.add_argument("--test-dir", default="chunk/test", help="Directory with test JSON files")
    parser.add_argument("--results-out", default="output/generation_eval/phat_tu_raw_eval.json")
    parser.add_argument("--provider", choices=[p.value for p in LLMProvider], default="openrouter")
    parser.add_argument("--model", default=None, help="Provider model override")
    parser.add_argument("--law-doc-path", default=DEFAULT_LAW_DOC_PATH, help="Path to law_doc.json")
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
    law_doc_path = Path(args.law_doc_path)

    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    if not law_doc_path.exists():
        raise FileNotFoundError(f"Missing law doc path: {law_doc_path}")

    provider = LLMProvider(args.provider)
    model_name = args.model or default_model_for_provider(provider)
    use_provider_fallback = not args.disable_provider_fallback

    law_retriever = LawClauseRetriever(law_doc_path)

    files = sorted(test_dir.glob("*.json"))
    if args.first_n is not None:
        if args.first_n < 1:
            raise ValueError("--first-n must be >= 1")
        files = files[: args.first_n]

    print(f"Found {len(files)} test files")
    print(f"Provider={provider.value} | Model={model_name}")
    print(f"Provider fallback enabled={use_provider_fallback}")
    print(f"Law doc path={law_doc_path}")
    print(f"BLHS-only clause filtering={args.only_blhs}")
    print("Reasoning mode: NHAN_DINH_CUA_TOA_AN + ground-truth applied clauses + detailed law text")

    per_doc: list[dict[str, Any]] = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        result = _evaluate_single_doc(
            path=path,
            data=data,
            provider=provider,
            model_name=model_name,
            only_blhs=args.only_blhs,
            use_provider_fallback=use_provider_fallback,
            law_retriever=law_retriever,
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
            "law_doc_path": str(law_doc_path),
            "only_blhs": args.only_blhs,
            "target_schema": "GenerationOutput",
            "notes": [
                "No embedding retrieval is used in this evaluation.",
                "An_Phi is excluded because GenerationOutput does not include this field.",
            ],
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
