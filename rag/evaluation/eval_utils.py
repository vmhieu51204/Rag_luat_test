"""Shared helpers for generation evaluation scripts."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any

from rag.config import ID_FIELD, LEGAL_SOURCE_FIELD, VERDICT_FIELD
from rag.core.sentencing import extract_imprisonment_months
from rag.core.verdict_labels import extract_label_sets_from_verdict, is_blhs_legal_source, split_multi_value
from rag.generation.schemas import GenerationOutput


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


def _macro_mean(values: list[float]) -> float:
    return _safe_float(mean(values)) if values else 0.0


def _extract_phat_tu_months(text: str | None) -> int:
    return extract_imprisonment_months(text)


def _extract_doc_id(data: dict[str, Any], fallback: str) -> str:
    thong_tin = data.get("THONG_TIN_CHUNG") or {}
    if not isinstance(thong_tin, dict):
        thong_tin = {}
    value = thong_tin.get("Ma_Ban_An") or data.get("Ma_Ban_An") or fallback
    return str(value).strip() or fallback


def _resolve_field_value(data: dict[str, Any], field: str) -> Any:
    if field in {"Defendant_info", "defendant_info", "Thong_Tin_Bi_Cao"}:
        info = data.get("THONG_TIN_CHUNG")
        return info.get("Thong_Tin_Bi_Cao") if isinstance(info, dict) else None

    if "." not in field:
        return data.get(field)

    cur: Any = data
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _extract_input_payload(data: dict[str, Any], fields: list[str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for field in fields:
        value = _resolve_field_value(data, field)
        if value is None:
            continue
        text = value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if text.strip():
            payload[field] = text.strip()
    return payload


def _build_query_text(data: dict[str, Any], query_fields: list[str]) -> str:
    payload = _extract_input_payload(data, query_fields)
    return "\n\n".join(f"[{key}]\n{value}" for key, value in payload.items()).strip()


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
                "Phat_Tu": _normalize_space(str(item.get("Phat_Tu") or "")),
                "Phat_Tien": _normalize_space(str(item.get("Phat_Tien") or "")),
                "Trach_Nhiem_Dan_Su": _normalize_space(str(item.get("Trach_Nhiem_Dan_Su") or "")),
                "Xu_Ly_Vat_Chung": _normalize_space(str(data.get("Xu_Ly_Vat_Chung") or "")),
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
                clause_details.append({"signature": signature, "Tinh_tiet_ap_dung": tinh_tiet})

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


def save_eval_results(
    path: Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any] | None,
    per_doc: list[dict[str, Any]],
) -> None:
    """Atomically persist an evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"config": config, "summary": summary, "per_doc": per_doc},
            fh,
            ensure_ascii=False,
            indent=2,
        )
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


def load_articles_index(raw_dir: Path) -> tuple[dict[str, dict[str, set[str]]], list[dict[str, Any]]]:
    """Build train index: doc_id -> {'dieu_only', 'full_signature'}."""
    index: dict[str, dict[str, set[str]]] = {}
    skipped: list[dict[str, Any]] = []

    for path in sorted(raw_dir.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        doc_id = data.get(ID_FIELD, path.stem)
        label_sets, stats, errors = extract_label_sets_from_verdict(data)
        if errors:
            skipped.append(
                {
                    "doc_id": doc_id,
                    "file": path.name,
                    "stage": "train_index",
                    "reasons": sorted(set(errors)),
                    "stats": stats,
                }
            )
            continue

        index[str(doc_id)] = label_sets

    return index, skipped
