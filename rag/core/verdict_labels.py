"""Shared verdict-label extraction logic used by evaluation modules."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from rag.config import LEGAL_BASIS_FIELD, LEGAL_SOURCE_FIELD, VERDICT_FIELD


def normalize_token(token: str, *, lowercase: bool = True) -> str:
    token = str(token).strip()
    token = re.sub(r"^(điều|dieu|khoản|khoan|điểm|diem)\s+", "", token, flags=re.IGNORECASE)
    token = token.strip(" .")
    token = re.sub(r"\s+", "", token)
    return token.lower() if lowercase else token


def split_multi_value(raw_value: Any, *, lowercase: bool = True) -> list[str]:
    if raw_value is None:
        return []
    raw = str(raw_value).strip()
    if not raw:
        return []
    parts = re.split(r",|;|/|\bvà\b|\bva\b|\band\b", raw, flags=re.IGNORECASE)
    tokens = [normalize_token(p, lowercase=lowercase) for p in parts if p.strip()]
    return [t for t in tokens if t]


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def is_blhs_legal_source(raw_value: Any) -> bool:
    if raw_value is None:
        return False
    source = str(raw_value).strip()
    if not source:
        return False
    source_folded = strip_accents(source).lower()
    return "blhs" in source_folded or "bo luat hinh su" in source_folded


def extract_label_sets_from_verdict(
    data: dict,
    *,
    verdict_field: str = VERDICT_FIELD,
    legal_basis_field: str = LEGAL_BASIS_FIELD,
    legal_source_field: str = LEGAL_SOURCE_FIELD,
    only_blhs: bool = True,
) -> tuple[dict[str, set[str]], dict[str, int], list[str]]:
    """Extract legal-basis labels from verdict data.

    Returns:
      - label_sets: {'dieu_only': set[str], 'full_signature': set[str]}
      - stats: extraction counters
      - errors: non-empty list indicates strict-mode parsing failure
    """
    errors: list[str] = []
    stats = {"n_verdict_items": 0, "n_cancu_items": 0, "n_cancu_blhs_items": 0}
    label_sets = {"dieu_only": set(), "full_signature": set()}

    verdict_items = data.get(verdict_field)
    if not isinstance(verdict_items, list):
        return label_sets, stats, [f"{verdict_field}_invalid_type"]
    if not verdict_items:
        return label_sets, stats, [f"{verdict_field}_empty"]

    stats["n_verdict_items"] = len(verdict_items)
    for verdict in verdict_items:
        if not isinstance(verdict, dict):
            errors.append("verdict_item_not_object")
            continue

        legal_basis = verdict.get(legal_basis_field)
        if not isinstance(legal_basis, list):
            errors.append(f"{legal_basis_field}_invalid_type")
            continue
        if not legal_basis:
            errors.append(f"{legal_basis_field}_empty")
            continue

        stats["n_cancu_items"] += len(legal_basis)
        for basis_item in legal_basis:
            if not isinstance(basis_item, dict):
                errors.append("basis_item_not_object")
                continue

            if only_blhs and not is_blhs_legal_source(basis_item.get(legal_source_field)):
                continue

            if only_blhs:
                stats["n_cancu_blhs_items"] += 1

            dieu_tokens = split_multi_value(basis_item.get("Dieu"), lowercase=False)
            khoan_tokens = split_multi_value(basis_item.get("Khoan"), lowercase=False)
            diem_tokens = split_multi_value(basis_item.get("Diem"), lowercase=True)

            if not dieu_tokens:
                errors.append("missing_dieu")
                continue

            for dieu in dieu_tokens:
                label_sets["dieu_only"].add(dieu)

                if khoan_tokens and diem_tokens:
                    for khoan in khoan_tokens:
                        for diem in diem_tokens:
                            label_sets["full_signature"].add(f"{dieu}-{khoan}-{diem}")
                elif khoan_tokens:
                    for khoan in khoan_tokens:
                        label_sets["full_signature"].add(f"{dieu}-{khoan}")
                elif diem_tokens:
                    for diem in diem_tokens:
                        label_sets["full_signature"].add(f"{dieu}-{diem}")
                else:
                    label_sets["full_signature"].add(dieu)

    if not label_sets["dieu_only"]:
        errors.append("no_labels_extracted")

    return label_sets, stats, errors
