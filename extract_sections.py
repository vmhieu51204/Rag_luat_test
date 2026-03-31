#!/usr/bin/env python
"""
extract_sections.py
───────────────────
Extract structured sections from Vietnamese court verdict .md files into JSON.

Sections:
  1. Bi_cao         : Trial-date line → "Người có quyền lợi liên quan"
  2. Lien_quan      : "Người có quyền lợi liên quan" → "NỘI DUNG VỤ ÁN"
  3. NOI_DUNG_VU_AN : "NỘI DUNG VỤ ÁN" → "NHẬN ĐỊNH CỦA TÒA ÁN"
  4. NHAN_DINH      : "NHẬN ĐỊNH CỦA TÒA ÁN" → "QUYẾT ĐỊNH"
  5. QUYET_DINH     : "QUYẾT ĐỊNH" → end of file

Robust against OCR diacritic errors (e.g. ĐỊNH→ĐINH/ĐÌNH, TÒA→TOÀ).
Reports any files with missing fields.

Usage:
    python extract_sections.py
    python extract_sections.py --file 28-10-2025-Nghe_An-2ta2024366t1cvn.md
    python extract_sections.py --input-dir ./ocr_marker_fixed --output-dir ./extracted_json

Requires: Python 3.10+  (no external deps)
"""

import json
import re
import sys
import argparse
import unicodedata
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────
INPUT_DIR  = Path(__file__).parent / "ocr_marker_fixed"
OUTPUT_DIR = Path(__file__).parent / "extracted_json"


# ═════════════════════════════════════════════════════════════════════════════
#  TEXT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics / tone marks, keep base latin chars.
    Also converts Đ/đ → D/d (stroke is not a combining mark in Unicode)."""
    nfkd = unicodedata.normalize("NFD", text)
    result = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return result.replace("Đ", "D").replace("đ", "d")


def norm(line: str) -> str:
    """Normalize a line for matching: strip markdown + diacritics + uppercase."""
    s = re.sub(r"^#{1,6}\s*", "", line.strip())      # heading markers
    s = s.replace("*", "").replace("\\", "").strip()  # bold / italic / escape
    return strip_diacritics(s).upper()


def is_heading_style(line: str, max_content_len: int = 40) -> bool:
    """True if line looks like a standalone section header (heading or short)."""
    stripped = line.strip()
    # Markdown heading
    if re.match(r"^\s*#{1,6}\s", stripped):
        return True
    # Short bold line
    content = re.sub(r"^#{1,6}\s*", "", stripped)
    content = content.replace("*", "").replace("\\", "").strip()
    content_n = strip_diacritics(content).upper().rstrip(":").strip()
    return len(content_n) <= max_content_len


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION BOUNDARY FINDERS
# ═════════════════════════════════════════════════════════════════════════════

def find_trial_date(lines: list[str]) -> int:
    """Find the trial-session line: 'Ngày DD tháng MM năm YYYY … xét xử …'
    Also handles 'Trong ngày', 'Trong các ngày', 'Từ ngày', and 'DD/MM/YYYY'.
    """
    for i, line in enumerate(lines):
        n = norm(line)
        # Must mention trial-related keywords
        if not any(kw in n for kw in ("XET XU", "TAI TRU SO", "THU LY")):
            continue
        # Standard: Ngày DD tháng MM năm YYYY
        if re.search(r"NGAY\s+\d{1,2}\s+THANG\s+\d{1,2}\s+NAM\s+\d{4}", n):
            return i
        # Numeric: Ngày DD/MM/YYYY
        if re.search(r"NGAY\s+\d{1,2}/\d{1,2}/\d{4}", n):
            return i
        # Prefix variants: Trong ngày / Trong các ngày / Từ ngày
        if re.search(r"(?:TRONG\s+(?:CAC\s+)?|TU\s+)NGAY", n):
            return i
    return -1


def find_lien_quan(lines: list[str], start: int = 0) -> int:
    """Find 'Người có quyền lợi [,/và] nghĩa vụ liên quan', 'Bị hại', or other parties."""
    for i in range(start, len(lines)):
        n = norm(lines[i])
        if "QUYEN LOI" in n and "LIEN QUAN" in n:
            return i
        if re.match(r"^(?:-\s*)?(?:BI|NGUOI\s+BI)\s+HAI\b", n):
            return i
        if re.match(r"^(?:-\s*)?NGUYEN\s+DON\s+DAN\s+SU\b", n):
            return i
        if re.match(r"^(?:-\s*)?BI\s+DON\s+DAN\s+SU\b", n):
            return i
        if re.match(r"^(?:-\s*)?NGUOI\s+LAM\s+CHUNG\b", n):
            return i
    return -1


def _find_header(lines: list[str], keywords: list[str],
                 start: int = 0, require_heading: bool = False) -> int:
    """Find the first line containing ALL *keywords* (diacritic-stripped).

    If *require_heading* is True, only accept lines that look like standalone
    section headers (markdown headings ``#`` or short lines ≤ 40 chars).
    """
    for i in range(start, len(lines)):
        n = norm(lines[i])
        if all(kw in n for kw in keywords):
            if not require_heading or is_heading_style(lines[i]):
                return i
    return -1


def find_noi_dung(lines: list[str], start: int = 0) -> int:
    """Find 'NỘI DUNG VỤ ÁN'."""
    return _find_header(lines, ["NOI DUNG", "VU AN"], start)


def find_nhan_dinh(lines: list[str], start: int = 0) -> int:
    """Find 'NHẬN ĐỊNH CỦA TÒA ÁN' (or 'CỦA HỘI ĐỒNG XÉT XỬ')."""
    # Primary: "NHẬN ĐỊNH CỦA TÒA ÁN"
    idx = _find_header(lines, ["NHAN DINH", "TOA AN"], start)
    if idx >= 0:
        return idx
    # Variant: "NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ"
    idx = _find_header(lines, ["NHAN DINH", "HOI DONG"], start)
    if idx >= 0:
        return idx
    # Fallback: standalone "NHẬN ĐỊNH" heading
    idx = _find_header(lines, ["NHAN DINH"], start, require_heading=True)
    if idx >= 0:
        return idx
    # Last resort: inline "nhận định như sau"
    for i in range(start, len(lines)):
        n = norm(lines[i])
        if "NHAN DINH NHU SAU" in n:
            return i
    return -1


def find_quyet_dinh(lines: list[str], start: int = 0) -> int:
    """Find the 'QUYẾT ĐỊNH' *section header* (not body-text mentions).

    After stripping diacritics, QUYẾT ĐỊNH / ĐINH / ĐÌNH all become QUYET DINH.
    Only match lines that look like standalone headings where "QUYET DINH"
    is the primary content — not embedded in longer phrases like
    "căn cứ quyết định hình phạt".
    """
    for i in range(start, len(lines)):
        n = norm(lines[i])
        if "QUYET DINH" not in n:
            continue
        if not is_heading_style(lines[i]):
            continue
        # Strip numbering, brackets, "VI CAC LE TREN" preambles
        cleaned = re.sub(r'^\s*[\[\(]?\s*\d+\s*[\]\)]?\s*\.?\s*', '', n)
        cleaned = re.sub(r'^VE\s+', '', cleaned)
        cleaned = cleaned.strip(': .')
        # Accept only if the cleaned content starts with "QUYET DINH"
        # This rejects lines like "CAN CU QUYET DINH HINH PHAT"
        if cleaned.startswith("QUYET DINH"):
            return i
    return -1


# ═════════════════════════════════════════════════════════════════════════════
#  EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _extract(lines: list[str], start: int, end: int) -> str | None:
    """Join lines[start:end], strip outer whitespace. None if invalid bounds."""
    if start < 0 or end < 0 or start >= end:
        return None
    text = "\n".join(lines[start:end]).strip()
    return text or None


def process_file(filepath: Path) -> dict:
    """Extract all five sections from one .md file."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    missing: list[str] = []

    # ── Locate boundaries ────────────────────────────────────────────────
    i_trial = find_trial_date(lines)
    i_nd    = find_noi_dung(lines, max(i_trial, 0))
    i_nh    = find_nhan_dinh(lines, max(i_nd, 0) if i_nd >= 0 else 0)
    i_qd    = find_quyet_dinh(lines, max(i_nh, 0) if i_nh >= 0 else 0)

    # Lien_quan is optional (only present when there are third-party interests)
    i_lq = find_lien_quan(lines, max(i_trial, 0)) if i_trial >= 0 else -1
    # Only valid if it sits between trial-date and NỘI DUNG
    if i_lq >= 0 and i_nd >= 0 and not (i_trial < i_lq < i_nd):
        i_lq = -1

    # ── Record missing fields ────────────────────────────────────────────
    if i_trial < 0:
        missing.append("Bi_cao (trial date line not found)")
    if i_nd < 0:
        missing.append("NOI_DUNG_VU_AN")
    if i_nh < 0:
        missing.append("NHAN_DINH")
    if i_qd < 0:
        missing.append("QUYET_DINH")

    # ── Build record ─────────────────────────────────────────────────────
    # Bi_cao ends at Lien_quan (if present), otherwise at NỘI DUNG
    bi_cao_end = i_lq if i_lq >= 0 else i_nd

    record = {
        "filename":       filepath.name,
        "Bi_cao":         _extract(lines, i_trial, bi_cao_end)
                              if i_trial >= 0 and bi_cao_end is not None and bi_cao_end >= 0
                              else None,
        "Lien_quan":      _extract(lines, i_lq, i_nd)
                              if i_lq >= 0 and i_nd >= 0
                              else None,
        "NOI_DUNG_VU_AN": _extract(lines, i_nd, i_nh)
                              if i_nd >= 0 and i_nh >= 0
                              else None,
        "NHAN_DINH":      _extract(lines, i_nh, i_qd)
                              if i_nh >= 0 and i_qd >= 0
                              else None,
        "QUYET_DINH":     _extract(lines, i_qd, len(lines))
                              if i_qd >= 0
                              else None,
        "_missing":       missing,
        "_lines":         {
            "trial_date": i_trial,
            "lien_quan":  i_lq,
            "noi_dung":   i_nd,
            "nhan_dinh":  i_nh,
            "quyet_dinh": i_qd,
        },
    }
    return record


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract sections from court-verdict .md files → JSON")
    ap.add_argument("--input-dir",  type=Path, default=INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--file", help="Process a single file (name only)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [args.input_dir / args.file]
    else:
        files = sorted(args.input_dir.glob("*.md"))

    print(f"Processing {len(files)} file(s) from {args.input_dir}\n")

    records: list[dict] = []
    issues:  list[tuple[str, list[str]]] = []

    for fp in files:
        if not fp.exists():
            print(f"  SKIP (not found): {fp.name}")
            continue
        rec = process_file(fp)
        records.append(rec)
        if rec["_missing"]:
            issues.append((fp.name, rec["_missing"]))

    # ── Save individual JSON files (same name as .md → .json) ──────────
    # Only save files that have no missing fields (successful extractions)
    saved = 0
    for rec in records:
        if rec["_missing"]:
            continue  # skip failures — don't output JSON
        json_name = Path(rec["filename"]).stem + ".json"
        out_path = args.output_dir / json_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        saved += 1

    # ── Report ───────────────────────────────────────────────────────────
    total = len(records)
    if total == 0:
        print("No files processed.")
        return 0

    ok   = sum(1 for r in records if not r["_missing"])
    fail = total - ok
    fail_pct = (fail / total) * 100
    
    print(f"Saved {saved} JSON files → {args.output_dir} (skipped {total - saved} failures)")
    print(f"{'=' * 60}")
    print(f"  PASS  : {ok} ({(ok/total)*100:.1f}%)")
    print(f"  FAIL  : {fail} ({fail_pct:.1f}%)")
    print(f"  TOTAL : {total}")
    print(f"{'=' * 60}")

    failed_by_field: dict[str, list[str]] = {}
    for fname, fields in issues:
        for f in fields:
            failed_by_field.setdefault(f, []).append(fname)

    if total > 0:
        print(f"\n{'─' * 60}")
        print("FIELD EXTRACTION SUCCESS RATES")
        print(f"{'─' * 60}")
        print(f"{'Field':<45s} {'Success':>8s} {'Total':>8s} {'Rate':>8s}")
        print(f"{'─' * 45} {'─' * 8} {'─' * 8} {'─' * 8}")
        
        mandatory_fields = [
            "Bi_cao (trial date line not found)",
            "NOI_DUNG_VU_AN",
            "NHAN_DINH",
            "QUYET_DINH"
        ]
        for field in mandatory_fields:
            fail_count = len(failed_by_field.get(field, []))
            succ_count = total - fail_count
            rate = f"{100*succ_count/total:.1f}%"
            print(f"{field:<45s} {succ_count:>8d} {total:>8d} {rate:>8s}")

    if issues:
        print(f"\n{'─' * 60}")
        print("FAILED FILES PER FIELD (showing max 10)")
        print(f"{'─' * 60}")
        
        for field, fnames in failed_by_field.items():
            print(f"\n  {field} ({len(fnames)} failures):")
            for fn in fnames[:10]:
                print(f"    ✗ {fn}")
            if len(fnames) > 10:
                print(f"    ... and {len(fnames) - 10} more")
        print()
    else:
        print("\nAll fields found in all files.")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
