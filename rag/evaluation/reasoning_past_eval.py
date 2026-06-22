"""Evaluate verdict generation and law-clause filtering in one pipeline.

Workflow per test case:
1. Build a query text from selected case fields.
2. Retrieve top-k similar past cases from the train index.
3. Union their law clauses, aggregate to unique Dieu list, and fetch explicit Dieu text.
4. Load the narrative text and judicial reasoning of the top 2 retrieved past cases.
5. Prompt an LLM to predict, per defendant:
   - applied law clauses
   - Toi_Danh
   - Phat_Tu
   - Trach_Nhiem_Dan_Su
6. Compare predictions against ground truth verdict fields.
7. Save a single JSON report with per-document details and aggregate metrics.
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

from rag.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MODEL_NAME,
    LEGAL_SOURCE_FIELD,
    VERDICT_FIELD,
)
from rag.core.law_retriever import LawClauseRetriever
from rag.core.sentencing import extract_imprisonment_months
from rag.core.verdict_labels import is_blhs_legal_source, split_multi_value
from rag.evaluation.eval_utils import load_articles_index, save_eval_results
from rag.llm.providers import (
    LLMProvider,
    default_model_for_provider,
    generate_structured_output,
    generate_structured_output_with_fallback,
)
from rag.core.embeddings import load_chroma, run_pipeline
from rag.generation.reasoning_act import DEFAULT_REASON_ACT_TRAIN_FIELDS, MANDATORY_SUPPORTING_DIEU
from rag.generation.schemas import GenerationOutput, build_output_schema_instruction
from rag.runtime.retrieval import RetrievalRuntime, RetrievalRuntimeConfig

load_dotenv()

# Keep this in sync with reasoning_act_eval's mandatory supporting articles.
ALWAYS_INCLUDE_DIEU = MANDATORY_SUPPORTING_DIEU
DEFAULT_TRAIN_EMBEDDING_FIELDS = DEFAULT_REASON_ACT_TRAIN_FIELDS + [
    "PHAN_QUYET_CUA_TOA_SO_THAM.Tang_nang",
    "PHAN_QUYET_CUA_TOA_SO_THAM.Giam_nhe",
]
MAX_FINAL_CASES_PER_ISSUE = 5


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _name_key(name: str) -> str:
    folded = _strip_accents(_normalize_space(name)).lower()
    return re.sub(r"[^a-z0-9]+", "", folded)


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


def _build_query_text(data: dict[str, Any], query_fields: list[str]) -> str:
    payload = _extract_input_payload(data, query_fields)
    parts = [f"[{k}]\n{v}" for k, v in payload.items()]
    return "\n\n".join(parts).strip()


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
        name = str(item.get("Bi_Cao") or "").strip()
        toi_danh = str(item.get("Pham_Toi") or "").strip()
        phat_tu = str(item.get("Phat_Tu") or "").strip()
        phat_tien = str(item.get("Phat_Tien") or "").strip()
        trach_nhiem = str(item.get("Trach_Nhiem_Dan_Su") or "").strip()
        xu_ly_vat_chung = str(data.get("Xu_Ly_Vat_Chung") or "").strip()

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
                "Bi_Cao": name,
                "Phan_Tich_Phap_Ly": "",
                "Toi_Danh": toi_danh,
                "Phat_Tu": phat_tu,
                "Phat_Tien": phat_tien,
                "Trach_Nhiem_Dan_Su": trach_nhiem,
                "Xu_Ly_Vat_Chung": xu_ly_vat_chung,
                "Applied_Law_Clauses": sorted(signatures),
                "Applied_Law_Clauses_Detailed": [],
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
                "Phan_Tich_Phap_Ly": _normalize_space(defendant.Phan_Tich_Phap_Ly or ""),
                "Toi_Danh": _normalize_space(defendant.Toi_Danh or ""),
                "Phat_Tu": _normalize_space(defendant.Phat_Tu or ""),
                "Phat_Tien": _normalize_space(defendant.Phat_Tien or ""),
                "Trach_Nhiem_Dan_Su": _normalize_space(defendant.Trach_Nhiem_Dan_Su or ""),
                "Xu_Ly_Vat_Chung": _normalize_space(pred.Xu_Ly_Vat_Chung or ""),
                "Applied_Law_Clauses": sorted(signatures),
                "Applied_Law_Clauses_Detailed": clause_details,
            }
        )
    return out


def _compact_defendant_item(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    clauses = list(item.get("Applied_Law_Clauses") or [])
    return {
        "Bi_Cao": item.get("Bi_Cao", ""),
        "Phan_Tich_Phap_Ly": item.get("Phan_Tich_Phap_Ly", ""),
        "Toi_Danh": item.get("Toi_Danh", ""),
        "Phat_Tu": item.get("Phat_Tu", ""),
        "Phat_Tien": item.get("Phat_Tien", ""),
        "Trach_Nhiem_Dan_Su": item.get("Trach_Nhiem_Dan_Su", ""),
        "Xu_Ly_Vat_Chung": item.get("Xu_Ly_Vat_Chung", ""),
        "Applied_Law_Clauses": clauses,
        "Applied_Law_Clauses_flat": ", ".join(clauses),
        "Applied_Law_Clauses_Detailed": list(item.get("Applied_Law_Clauses_Detailed") or []),
    }


def _macro_mean(values: list[float]) -> float:
    return _safe_float(mean(values)) if values else 0.0


def _extract_phat_tu_months(text: str | None) -> int:
    return extract_imprisonment_months(text)


def _retrieve_similar_case_doc_ids(
    runtime: RetrievalRuntime,
    *,
    query_text: str,
    exclude_doc_id: str,
    top_k_case: int,
) -> list[str]:
    top_k_case = min(top_k_case, MAX_FINAL_CASES_PER_ISSUE)
    if top_k_case <= 0:
        return []

    n_fetch = max(top_k_case * 8, 64)
    cap = max(top_k_case * 200, 800)

    while True:
        results = runtime.query_train(
            query_text=query_text,
            top_k=n_fetch,
            exclude_doc_id=exclude_doc_id,
            include=["metadatas", "distances"],
        )

        case_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for meta in results.get("metadatas", [[]])[0]:
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
            if len(case_doc_ids) >= top_k_case:
                break

        if len(case_doc_ids) >= top_k_case or n_fetch >= cap:
            return case_doc_ids

        n_fetch = min(n_fetch * 2, cap)


def _build_past_cases_context(
    case_doc_ids: list[str],
    train_dir: Path,
    max_cases: int = MAX_FINAL_CASES_PER_ISSUE
) -> list[dict[str, Any]]:
    """Load court reasoning and verdict of top retrieved cases for reasoning examples."""
    past_cases = []
    for doc_id in case_doc_ids[:max_cases]:
        file_path = train_dir / f"{doc_id}.json"
        if not file_path.exists():
            continue
        
        try:
            with open(file_path, encoding="utf-8") as fh:
                data = json.load(fh)
                
            past_cases.append({
                "doc_id": doc_id,
                "Court_Reasoning": data.get("NHAN_DINH_CUA_TOA_AN", {}),
                "Verdict": data.get("PHAN_QUYET_CUA_TOA_SO_THAM", [])
            })
        except Exception:
            continue
            
    return past_cases


def _select_reference_court_reasoning(past_cases_context: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in past_cases_context:
        court_reasoning = item.get("Court_Reasoning")
        if court_reasoning:
            return {
                "doc_id": item.get("doc_id", ""),
                "NHAN_DINH_CUA_TOA_AN": court_reasoning,
            }
    return None


def _build_similar_case_clause_context(
    *,
    case_doc_ids: list[str],
    train_articles_index: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    by_case: list[dict[str, Any]] = []
    union_clauses: set[str] = set()

    for rid in case_doc_ids:
        labels = train_articles_index.get(rid)
        clauses = sorted(labels["full_signature"]) if labels else []
        by_case.append(
            {
                "doc_id": rid,
                "law_clause_set": clauses,
            }
        )
        union_clauses |= set(clauses)

    return {
        "similar_case_doc_ids": case_doc_ids,
        "similar_case_law_clause_set": sorted(union_clauses),
        "similar_case_law_clause_by_doc": by_case,
    }


def _retrieve_explicit_law_by_dieu(
    *,
    law_signatures: set[str],
    law_retriever: LawClauseRetriever,
) -> list[dict[str, str]]:
    dieu_ids = sorted({str(sig).split("-")[0].strip() for sig in law_signatures if str(sig).strip()})
    explicit: list[dict[str, str]] = []
    for dieu in dieu_ids:
        if not dieu:
            continue
        result = law_retriever.retrieve(dieu)
        if not result.get("found"):
            continue
        text = str(result.get("text") or "").strip()
        if not text:
            continue
        explicit.append({
            "dieu": dieu,
            "signature": dieu,
            "text": text,
        })
    return explicit


def _compact_explicit_law_clauses(explicit_law_clauses: list[dict[str, str]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in explicit_law_clauses:
        text = str(item.get("text") or "")
        compact.append(
            {
                "dieu": item.get("dieu", ""),
                "signature": item.get("signature", ""),
                "text_chars": len(text),
            }
        )
    return compact


def _build_prompts(
    *,
    doc_id: str,
    case_payload: dict[str, str],
    defendant_info: str,
    explicit_law_clauses: list[dict[str, str]],
    past_cases_context: list[dict[str, Any]],
    reference_court_reasoning: dict[str, Any] | None,
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert Vietnamese criminal judgment assistant. "
        "Return only valid JSON. Predict legal outcomes strictly from provided facts, law clauses, and judicial precedents. "
    )

    requirements = [
        "For each defendant, provide Phan_Tich_Phap_Ly as concise legal reasoning.",
        "For each defendant, list applied law clauses in Applied_Law_Clauses.",
        "For each Applied_Law_Clauses item, fill Tinh_tiet_ap_dung with concise case facts that justify applying that clause.",
        "State Toi_Danh.",
        "Predict a concrete final Phat_Tu verdict for each defendant.",
        "Predict Trach_Nhiem_Dan_Su when applicable.",
        "Use explicit_law_clauses (retrieved at Dieu level from similar past cases) as legal references and select only applicable clauses.",
        "Before producing output, reason from the provided case_fields, Defendant_info, and explicit_law_clauses.",
        "Use reference_court_reasoning.NHAN_DINH_CUA_TOA_AN as a judicial reasoning reference from one retrieved similar case. Treat it as persuasive precedent only, not as ground truth for the current case.",
        "MANDATORY: You must evaluate and include applicable general sentencing provisions from the 50s articles (e.g., Article 51 for mitigating factors, Article 52 for aggravating factors, Articles 55/56 for sentence synthesis) in Applied_Law_Clauses.",
        "MANDATORY: You must compare the current case facts against the provided 'past_cases_context'. Use the judicial reasoning (NHAN_DINH_CUA_TOA_AN) of past cases to determine the correct Toi_Danh and Phat_Tu."
    ]
    constraints = [
        "No markdown, no extra explanation.",
        "Do not invent defendant names not supported by case fields.",
        "Applied_Law_Clauses should prioritize selected clauses from explicit_law_clauses.",
        "Tinh_tiet_ap_dung must cite only facts present in case_fields or Defendant_info; do not invent facts.",
        "Phat_Tu must be a single concrete verdict statement; do not output a range like 'từ X đến Y năm tù'.",
    ]

    input_payload = {
        "doc_id": doc_id,
        "case_fields": case_payload,
        "Defendant_info": defendant_info,
        "explicit_law_clauses": explicit_law_clauses,
        "past_cases_context": past_cases_context,
        "reference_court_reasoning": reference_court_reasoning,
        "task": {
            "requirement": requirements,
            "reasoning_instruction": [
                "Use the factual timeline in case_fields to identify offense behavior.",
                "Use Defendant_info for circumstances that affect sentencing and mitigating/aggravating factors.",
                "Select applicable clauses from explicit_law_clauses before deciding Toi_Danh and Phat_Tu.",
                "Use reference_court_reasoning as an example of court legal reasoning style and factor weighing. Apply only the parts that match the current case facts.",
                "When calculating the final Phat_Tu, you must explicitly adjust the sentence length based on the presence of mitigating factors (Article 51), aggravating factors (Article 52), or previous convictions/synthesis (Articles 55/56) found in Defendant_info.",
                "Analyze 'past_cases_context'. Compare weapons, intent, and severity in the past cases to the current case.",
                "If the current facts closely mirror a past case, align your Toi_Danh and Phat_Tu predictions with that past case, adjusting for specific aggravating/mitigating factors."
            ],
            "output_schema": build_output_schema_instruction(GenerationOutput),
            "constraints": constraints,
        },
    }
    user_prompt = json.dumps(input_payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def _call_llm(
    *,
    provider: LLMProvider,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_model: type[Any],
    use_provider_fallback: bool,
) -> tuple[Any, dict[str, Any]]:
    if use_provider_fallback:
        return generate_structured_output_with_fallback(
            preferred_provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
        )
    return generate_structured_output(
        provider=provider,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=output_model,
    )


def _evaluate_single_doc(
    *,
    path: Path,
    data: dict[str, Any],
    case_runtime: RetrievalRuntime,
    train_articles_index: dict[str, dict[str, set[str]]],
    law_retriever: LawClauseRetriever,
    input_fields: list[str],
    query_fields: list[str],
    top_k_case: int,
    provider: LLMProvider,
    model_name: str,
    only_blhs: bool,
    use_provider_fallback: bool,
    train_dir: Path,
) -> dict[str, Any]:
    doc_id = _extract_doc_id(data, path.stem)
    case_payload = _extract_input_payload(data, input_fields)
    defendant_info = _extract_input_payload(data, ["THONG_TIN_CHUNG.Thong_Tin_Bi_Cao"]).get(
        "THONG_TIN_CHUNG.Thong_Tin_Bi_Cao", ""
    )
    llm_input_payload = dict(case_payload)
    llm_input_payload["Defendant_info"] = defendant_info
    query_text = _build_query_text(data, query_fields)

    if not query_text:
        return {
            "doc_id": doc_id,
            "source_file": path.name,
            "status": "skipped",
            "reason": "empty_query_text",
            "ground_truth": {"defendants": _extract_gt_defendants(data, only_blhs=only_blhs)},
        }

    similar_case_doc_ids = _retrieve_similar_case_doc_ids(
        case_runtime,
        query_text=query_text,
        exclude_doc_id=doc_id,
        top_k_case=top_k_case,
    )
    
    similar_case_context = _build_similar_case_clause_context(
        case_doc_ids=similar_case_doc_ids,
        train_articles_index=train_articles_index,
    )

    # Fetch narrative context for the top retrieved cases.
    past_cases_context = _build_past_cases_context(
        case_doc_ids=similar_case_doc_ids,
        train_dir=train_dir,
        max_cases=MAX_FINAL_CASES_PER_ISSUE
    )
    reference_court_reasoning = _select_reference_court_reasoning(past_cases_context)

    forced_clause_set = set(similar_case_context.get("similar_case_law_clause_set", []))
    forced_clause_set |= set(ALWAYS_INCLUDE_DIEU)
    similar_case_context["similar_case_law_clause_set"] = sorted(forced_clause_set)

    explicit_law_clauses = _retrieve_explicit_law_by_dieu(
        law_signatures=forced_clause_set,
        law_retriever=law_retriever,
    )
    explicit_law_clause_log = _compact_explicit_law_clauses(explicit_law_clauses)

    gt_defendants = _extract_gt_defendants(data, only_blhs=only_blhs)
    gt_union: set[str] = set()
    for item in gt_defendants:
        gt_union |= set(item["Applied_Law_Clauses"])

    system_prompt, user_prompt = _build_prompts(
        doc_id=doc_id,
        case_payload=case_payload,
        defendant_info=defendant_info,
        explicit_law_clauses=explicit_law_clauses,
        past_cases_context=past_cases_context,
        reference_court_reasoning=reference_court_reasoning,
    )

    usage: dict[str, Any] = {}
    parse_error = None
    generation_error = None
    pred_output: GenerationOutput | None = None
    llm_used_provider = provider.value
    llm_used_model = model_name

    try:
        pred_output, usage = _call_llm(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=GenerationOutput,
            use_provider_fallback=use_provider_fallback,
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
            "similar_case_context": {
                "similar_case_doc_ids": similar_case_context.get("similar_case_doc_ids", []),
                "similar_case_law_clause_set": similar_case_context.get("similar_case_law_clause_set", []),
                "reference_court_reasoning_doc_id": (reference_court_reasoning or {}).get("doc_id", ""),
            },
            "explicit_law_clauses": explicit_law_clause_log,
            "defendants": [
                {
                    "Bi_Cao": item.get("Bi_Cao", ""),
                    "ground_truth": _compact_defendant_item(item),
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

    per_defendant: list[dict[str, Any]] = []
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

    return {
        "doc_id": doc_id,
        "source_file": path.name,
        "status": "processed",
        "reason": "ok",
        "llm_input_payload": llm_input_payload,
        "similar_case_context": {
            "similar_case_doc_ids": similar_case_context.get("similar_case_doc_ids", []),
            "similar_case_law_clause_set": similar_case_context.get("similar_case_law_clause_set", []),
            "reference_court_reasoning_doc_id": (reference_court_reasoning or {}).get("doc_id", ""),
        },
        "explicit_law_clauses": explicit_law_clause_log,
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
            "law_clause_set_precision_macro": _macro_mean(clause_p),
            "law_clause_set_recall_macro": _macro_mean(clause_r),
            "law_clause_set_f1_macro": _macro_mean(clause_f1),
            "phat_tu_rmse_months_macro": _macro_mean(rmse_months),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate verdict generation and law-clause filtering using test cases, "
            "law embedding retrieval, fixed system-law priors, and LLM generation."
        )
    )
    parser.add_argument("--test-dir", default="chunk/test", help="Directory with test JSON files")
    parser.add_argument("--train-dir", default="chunk/train", help="Directory with train JSON files")
    parser.add_argument("--law-json", default="raw_law.json", help="Path to raw law JSON used for embedding")
    parser.add_argument("--case-db-dir", default="output/generation_eval/case_db", help="Case Chroma DB directory")
    parser.add_argument("--law-db-dir", default="output/generation_eval/law_db", help="Law Chroma DB directory")
    parser.add_argument("--results-out", default="output/generation_eval/verdict_generation_system_eval.json")
    parser.add_argument("--provider", choices=[p.value for p in LLMProvider], default="openrouter")
    parser.add_argument("--model", default=None, help="Provider model override")
    parser.add_argument("--embed-model", default=DEFAULT_MODEL_NAME, help="Embedding model for law retrieval")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Embedding device")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--top-k-law",
        type=int,
        default=0,
        help="Deprecated (law text now comes from similar-case clause union via law_retriever)",
    )
    parser.add_argument(
        "--top-k-case",
        type=int,
        default=5,
        help="Top-k similar past train cases used to build the law-clause union",
    )
    parser.add_argument("--law-id", default="BLHS")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS)
    parser.add_argument(
        "--input-fields",
        default="THONG_TIN_CHUNG.Thong_Tin_Bi_Cao,Synthetic_summary_2",
        help="Comma-separated fields to pass to the LLM payload",
    )
    parser.add_argument(
        "--query-fields",
        default="Synthetic_summary_2,THONG_TIN_CHUNG.Thong_Tin_Bi_Cao",
        help="Comma-separated fields used to form embedding retrieval query",
    )
    parser.add_argument(
        "--train-embedding-fields",
        default=",".join(DEFAULT_TRAIN_EMBEDDING_FIELDS),
        help="Comma-separated fields to embed for train case retrieval index",
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

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    law_json = Path(args.law_json)
    results_out = Path(args.results_out)
    case_db_dir = Path(args.case_db_dir)
    law_db_dir = Path(args.law_db_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    if not law_json.exists():
        raise FileNotFoundError(f"Missing law json: {law_json}")
    if args.top_k_law < 0:
        raise ValueError("--top-k-law must be >= 0")
    if args.top_k_case < 1:
        raise ValueError("--top-k-case must be >= 1")

    input_fields = _parse_fields(args.input_fields)
    query_fields = _parse_fields(args.query_fields)
    train_embedding_fields = _parse_fields(args.train_embedding_fields)

    provider = LLMProvider(args.provider)
    model_name = args.model or default_model_for_provider(provider)
    use_provider_fallback = not args.disable_provider_fallback

    if args.top_k_law > 0:
        print("Note: --top-k-law is deprecated and ignored in this flow.")

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
    try:
        load_chroma(str(case_db_dir), collection_name=args.collection_name, create=False)
    except Exception as exc:
        fields = ", ".join(train_embedding_fields)
        raise RuntimeError(
            "Train case retrieval collection was not created. "
            f"No chunks may have been produced from --train-embedding-fields={fields!r}. "
            "Choose fields that exist in chunk/train, for example "
            f"{','.join(DEFAULT_TRAIN_EMBEDDING_FIELDS)!r}."
        ) from exc

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
    print(f"Always-included Dieu={list(ALWAYS_INCLUDE_DIEU)}")
    print("Past-case retrieval enabled for law candidate mining and context (passed as narrative context to LLM)")
    print(f"Law retriever dieu index size={len(getattr(law_retriever, '_dieu_index', {}))}")
    print(f"Train label index size={len(train_articles_index)} (skipped={len(train_skipped)})")

    config = {
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "law_json": str(law_json),
        "case_db_dir": str(case_db_dir),
        "law_db_dir": str(law_db_dir),
        "provider": provider.value,
        "model": model_name,
        "provider_fallback": use_provider_fallback,
        "embedding_model": args.embed_model,
        "device": args.device,
        "collection_name": args.collection_name,
        "top_k_law": args.top_k_law,
        "top_k_case": args.top_k_case,
        "input_fields": input_fields,
        "query_fields": query_fields,
        "train_embedding_fields": train_embedding_fields,
        "always_include_dieu": list(ALWAYS_INCLUDE_DIEU),
        "only_blhs": args.only_blhs,
        "n_train_label_index": len(train_articles_index),
        "n_train_label_skipped": len(train_skipped),
    }

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
            train_articles_index=train_articles_index,
            law_retriever=law_retriever,
            input_fields=input_fields,
            query_fields=query_fields,
            top_k_case=args.top_k_case,
            provider=provider,
            model_name=model_name,
            only_blhs=args.only_blhs,
            use_provider_fallback=use_provider_fallback,
            train_dir=train_dir,
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
