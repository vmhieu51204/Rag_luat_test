"""Utilities for extracting custodial sentence duration from Vietnamese text."""

from __future__ import annotations

import re
import unicodedata


LIFE_IMPRISONMENT_MONTHS = 360


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _cleanup_parenthetical_text(text: str) -> str:
    # Parentheses usually contain spelled-out numbers (e.g., "12 (mười hai)")
    # and may break naive regex patterns.
    return re.sub(r"\([^)]*\)", " ", text)


def _months_from_year_month_fragment(fragment: str) -> int | None:
    year_match = re.search(r"(\d{1,3})\s*nam\b", fragment)
    month_match = re.search(r"(\d{1,3})\s*thang\b", fragment)

    if not year_match and not month_match:
        return None

    years = int(year_match.group(1)) if year_match else 0
    months = int(month_match.group(1)) if month_match else 0
    return years * 12 + months


def extract_imprisonment_months(text: str | None) -> int:
    """Extract imprisonment months from a Vietnamese "Phat_Tu" string.

    Rules:
    - Non-custodial outcomes (e.g., suspended sentence) map to 0.
    - If a "tong hop" final sentence exists, prefer it.
    - Sentence ranges are converted to midpoint in months.
    - Combined year-month forms are parsed before unit-only forms.
    """
    raw = _normalize_space(text or "")
    if not raw:
        return 0

    lowered = _strip_accents(raw).lower()
    cleaned = _normalize_space(_cleanup_parenthetical_text(lowered))

    non_custodial_markers = [
        "cai tao khong giam giu",
        "an treo",
        "phat tien",
        "canh cao",
        "mien hinh phat",
        "khong phai chap hanh hinh phat tu",
        "khong phai di tu",
        "khong tu",
    ]
    if any(marker in cleaned for marker in non_custodial_markers):
        return 0

    # 1) Prefer final combined sentence after "tong hop".
    for m in re.finditer(r"tong hop[^.;,:\n]*", cleaned):
        frag = m.group(0)
        combined = _months_from_year_month_fragment(frag)
        if combined is not None:
            return combined
        if "chung than" in frag:
            return LIFE_IMPRISONMENT_MONTHS

    # 2) Range forms.
    year_range = re.search(r"tu\s*(\d{1,3})\s*nam\s*den\s*(\d{1,3})\s*nam", cleaned)
    if year_range:
        lo = int(year_range.group(1)) * 12
        hi = int(year_range.group(2)) * 12
        return int(round((lo + hi) / 2))

    month_range = re.search(r"tu\s*(\d{1,3})\s*thang\s*den\s*(\d{1,3})\s*thang", cleaned)
    if month_range:
        lo = int(month_range.group(1))
        hi = int(month_range.group(2))
        return int(round((lo + hi) / 2))

    # 3) Explicit custodial duration with both year and month.
    explicit_combined = re.search(r"(\d{1,3})\s*nam\s*(\d{1,3})\s*thang\s*tu\b", cleaned)
    if explicit_combined:
        years = int(explicit_combined.group(1))
        months = int(explicit_combined.group(2))
        return years * 12 + months

    # 4) Explicit single-unit custodial duration.
    explicit_year_only = re.search(r"(\d{1,3})\s*nam\s*tu\b", cleaned)
    if explicit_year_only:
        return int(explicit_year_only.group(1)) * 12

    explicit_month_only = re.search(r"(\d{1,3})\s*thang\s*tu\b", cleaned)
    if explicit_month_only:
        return int(explicit_month_only.group(1))

    if "chung than" in cleaned:
        return LIFE_IMPRISONMENT_MONTHS

    # 5) Generic fallback with year-month before single unit.
    generic_combined = re.search(r"(\d{1,3})\s*nam\s*(\d{1,3})\s*thang", cleaned)
    if generic_combined:
        years = int(generic_combined.group(1))
        months = int(generic_combined.group(2))
        return years * 12 + months

    generic_year = re.search(r"(\d{1,3})\s*nam", cleaned)
    if generic_year:
        return int(generic_year.group(1)) * 12

    generic_month = re.search(r"(\d{1,3})\s*thang", cleaned)
    if generic_month:
        return int(generic_month.group(1))

    return 0
