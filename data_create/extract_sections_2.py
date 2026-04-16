#!/usr/bin/env python
"""
test_extraction.py
──────────────────
Rule-based field extraction test for Vietnamese caselaw JSON files.

Loads JSON files produced by extract_sections.py and attempts to extract:
  - Danh_sach_bi_cao           (defendant list — raw text)
  - Noi_dung_vu_an
      ├── Qua_trinh_dieu_tra   (investigation process)
      └── Ket_luan_cac_ben     (conclusions of related parties)
  - Nhan_dinh_cua_toa_an       (court's assessment)
    - QUYET_DINH                 (original verdict text)

Usage:
    python test_extraction.py
    python test_extraction.py --input-dir ./extracted_json --output-dir ./extracted_fields
    python test_extraction.py --file 02-12-2025-Ninh_Binh-2ta2032065t1cvn.json

Requires: Python 3.10+
"""

import json
import re
import sys
import argparse
import unicodedata
from pathlib import Path
from typing import Optional

# ── Defaults ─────────────────────────────────────────────────────────────────
INPUT_DIR  = Path(__file__).parent / "extracted_json"
OUTPUT_DIR = Path(__file__).parent / "extracted_fields"

# ═════════════════════════════════════════════════════════════════════════════
#  TEXT HELPERS (reused from extract_sections.py)
# ═════════════════════════════════════════════════════════════════════════════

def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics/tone marks, keep base latin chars."""
    nfkd = unicodedata.normalize("NFD", text)
    result = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return result.replace("Đ", "D").replace("đ", "d")


def norm(text: str) -> str:
    """Normalize for matching: strip markdown + diacritics + uppercase."""
    s = re.sub(r"^#{1,6}\s*", "", text.strip())
    s = s.replace("*", "").replace("\\", "").strip()
    return strip_diacritics(s).upper()


def norm_lower(text: str) -> str:
    """Normalize but keep lowercase (for case-insensitive regex on original)."""
    return strip_diacritics(text).lower()


# ═════════════════════════════════════════════════════════════════════════════
#  CASE TYPE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_case_type(bi_cao: Optional[str], noi_dung: Optional[str] = None) -> str:
    """Detect whether this is a 'phuc_tham' (appeal) or 'so_tham' (first instance)."""
    if not bi_cao:
        # Fallback: check NOI_DUNG for embedded sơ thẩm verdict
        if noi_dung:
            n_nd = norm(noi_dung[:3000])  # Check first 3k chars
            if "BAN AN" in n_nd and "SO THAM" in n_nd:
                return "phuc_tham"
        return "unknown"
    n = norm(bi_cao)
    if "PHUC THAM" in n:
        return "phuc_tham"
    if "SO THAM" in n:
        return "so_tham"
    # Fallback: look for appeal indicators (kháng cáo in the case header/intro)
    if "KHANG CAO" in n or "KHANG NGHI" in n:
        return "phuc_tham"
    return "unknown"


# ═════════════════════════════════════════════════════════════════════════════
#  DEFENDANT LIST EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_danh_sach_bi_cao(bi_cao: Optional[str]) -> Optional[list[str]]:
    """Extract defendant list from Bi_cao section as a list of individual defendants.
    
    Strips the opening trial-date preamble (everything before first defendant).
    Defendants are identified by: numbered list (1., 2.) or bold name (**Name**)
    or 'sinh ngày/sinh năm' pattern.
    Returns a list where each element is one defendant's full info text.
    """
    if not bi_cao:
        return None

    lines = bi_cao.split("\n")
    start_idx = 0

    # Find where defendants start
    for i, line in enumerate(lines):
        stripped = line.strip()
        n = norm(stripped)
        # Numbered defendant: "1. Name" or "- 1. Name"
        if re.match(r"^[-\s]*\d+\.\s+", stripped):
            # Check if it's a defendant (has birth info nearby)
            context = "\n".join(lines[i:min(i+3, len(lines))])
            if re.search(r"sinh\s+(?:ngày|năm)", context, re.IGNORECASE):
                start_idx = i
                break
        # Bold name: **Name**
        if re.match(r"^[-\s]*\*\*[^*]+\*\*", stripped):
            context = "\n".join(lines[i:min(i+3, len(lines))])
            if re.search(r"sinh\s+(?:ngày|năm)", context, re.IGNORECASE):
                start_idx = i
                break
        # Direct defendant mention pattern (single defendant, no numbering)
        if re.search(r"(?:đối với|bị cáo)[:\s]", stripped, re.IGNORECASE):
            if re.search(r"sinh\s+(?:ngày|năm)", stripped, re.IGNORECASE):
                start_idx = i
                break
        # "Các bị cáo kháng cáo:" pattern
        if "BI CAO" in n and ("KHANG CAO" in n or "SINH" in n):
            start_idx = i
            break

    full_text = "\n".join(lines[start_idx:]).strip()
    if not full_text:
        return None

    # ── Split into individual defendants by numbered pattern ──
    # Matches lines starting with: "1. ", "- 1. ", "**1. ", "**1.** ", etc.
    defendant_pattern = r'^[-\s]*(?:\*\*)?\d+\s*[.\)]\s*'

    # Find start positions of each numbered defendant block
    starts = []
    for m in re.finditer(defendant_pattern, full_text, re.MULTILINE):
        # Verify birth info exists within the next ~500 chars
        end_check = min(m.start() + 500, len(full_text))
        context = full_text[m.start():end_check]
        if re.search(r"sinh\s+(?:ngày|năm)", context, re.IGNORECASE):
            starts.append(m.start())

    if len(starts) >= 2:
        # Multiple defendants — split at each start position
        defendants = []
        for idx, pos in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(full_text)
            chunk = full_text[pos:end].strip()
            # Strip trailing collective attendance lines
            chunk = re.sub(
                r'\n\s*\(\s*[Cc]ác\s+bị\s+cáo\s+.*?\)\s*$', '', chunk
            ).strip()
            if chunk:
                defendants.append(chunk)
        return defendants if defendants else None
    elif len(starts) == 1:
        # Single numbered defendant
        chunk = full_text[starts[0]:].strip()
        chunk = re.sub(
            r'\n\s*\(\s*[Cc]ác\s+bị\s+cáo\s+.*?\)\s*$', '', chunk
        ).strip()
        return [chunk] if chunk else None
    else:
        # No numbered pattern — return full text as single defendant
        full_text = re.sub(
            r'\n\s*\(\s*[Cc]ác\s+bị\s+cáo\s+.*?\)\s*$', '', full_text
        ).strip()
        return [full_text] if full_text else None


# ═════════════════════════════════════════════════════════════════════════════
#  NOI_DUNG_VU_AN SUB-FIELD EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _find_cao_trang_pos(text: str) -> int:
    """Find position of the indictment marker ('bản cáo trạng số' or similar)."""
    patterns = [
        r'(?:tại\s+)?(?:bản\s+)?cáo\s+trạng(?:\s+số)?\s*[:\.]?\s*\d',
        r'(?:tại\s+)?(?:bản\s+)?cáo\s+trạng(?:\s+của)?\s+viện\s+kiểm\s+sát',
        r'đại\s+di[ệễ]n\s+viện\s+kiểm\s+sát.*?(?:phát\s+biểu|thực\s+hành|có\s+ý\s+kiến|giữ\s+nguyên|rút|tham\s+gia)',
        r'tại\s+phiên\s+tò?a(?:.*?đại\s+di[ệễ]n)?\s+viện\s+kiểm\s+sát',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.start()
            
    t_norm = norm_lower(text)
    alt_patterns = [
        r'(?:tai\s+)?(?:ban\s+)?cao\s+trang(?:\s+so)?\s*[:\. ]*\d',
        r'(?:tai\s+)?(?:ban\s+)?cao\s+trang(?:\s+cua)?\s+vien\s+kiem\s+sat',
        r'dai\s+dien\s+vien\s+kiem\s+sat.*?(?:phat\s+bieu|thuc\s+hanh|co\s+y\s+kien|giu\s+nguyen|rut|tham\s+gia)',
        r'tai\s+phien\s+toa(?:.*?dai\s+dien)?\s+vien\s+kiem\s+sat',
    ]
    for pat in alt_patterns:
        m = re.search(pat, t_norm, re.DOTALL)
        if m:
            return m.start()
            
    return -1


def extract_noi_dung_sub_fields(noi_dung: Optional[str], case_type: str) -> dict:
    result = {
        "Qua_trinh_dieu_tra": None,
        "Ket_luan_cac_ben": None,
    }
    if not noi_dung:
        return result

    text = noi_dung
    header_match = re.match(r'^(?:#{1,6}\s*)?(?:\*{1,2})?N[ỘÔO]I\s+DUNG\s+V[ỤU]\s+[AÁ]N\s*:?\s*(?:\*{1,2})?\s*\n?', 
                            text, re.IGNORECASE)
    if header_match:
        text = text[header_match.end():]

    cao_trang_pos = _find_cao_trang_pos(text)

    if cao_trang_pos >= 0:
        result["Qua_trinh_dieu_tra"] = text[:cao_trang_pos].strip() or None
        result["Ket_luan_cac_ben"] = text[cao_trang_pos:].strip() or None
    else:
        result["Qua_trinh_dieu_tra"] = text.strip() or None

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  NHAN_DINH SECTION EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_nhan_dinh_points(nhan_dinh: Optional[str]) -> Optional[dict[str, str]]:
    """Split NHAN_DINH into numbered points keyed by "[number]".

    A point starts at line-begin markers like:
      - [1] ...
      [2] ...
      * [3] ...
            #### [4] ...
            **[5]** ...
    """
    if not nhan_dinh:
        return None

    text = nhan_dinh.strip()
    if not text:
        return None

    point_re = re.compile(
        r'(?m)^\s*(?:#{1,6}\s*)?(?:[-*+]\s*)?(?:\*\*)?\[\s*(\d+)\s*\](?:\*\*)?\s*(?:[.)\-:]\s*)?'
    )
    matches = list(point_re.finditer(text))

    if not matches:
        # Fallback to a single bucket to preserve content.
        return {"[1]": text}

    result: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key = f"[{m.group(1)}]"
        value = text[start:end].strip()

        if not value:
            continue
        if key in result:
            # Handle duplicated indices by appending content, avoiding data loss.
            result[key] = f"{result[key]}\n\n{value}".strip()
        else:
            result[key] = value

    return result or None


# ═════════════════════════════════════════════════════════════════════════════
#  VERDICT SUB-FIELD EXTRACTION (shared for sơ thẩm & phúc thẩm)
# ═════════════════════════════════════════════════════════════════════════════

def extract_verdict_sub_fields(text: str) -> dict:
    """Extract sub-fields from a verdict text block (QUYET_DINH or sơ thẩm verdict).
    
    Returns: {Can_cu_dieu_luat, Pham_toi, Phat_tu, Phat_tien, An_phi}
    """
    result = {
        "Can_cu_dieu_luat": None,
        "Pham_toi": None,
        "Phat_tu": None,
        "Phat_tien": None,
        "An_phi": None,
    }

    if not text:
        return result

    # ── Căn cứ điều luật ─────────────────────────────────────────────
    # Look for "Căn cứ" or "Áp dụng" followed by legal article references
    legal_basis_patterns = [
        r'([Cc]ăn\s+cứ[:*\s]*(?:vào\s+)?.*?(?:[Đđ]iều\s+\d+.*?)(?:Bộ\s+luật|BLHS|BLTTHS)[\w\s]*)',
        r'([Áá]p\s+dụng[:*\s]*.*?(?:[Đđ]iều\s+\d+.*?)(?:Bộ\s+luật|BLHS|BLTTHS)[\w\s]*)',
    ]
    # Collect ALL legal basis mentions
    all_bases = []
    for pat in legal_basis_patterns:
        for m in re.finditer(pat, text, re.DOTALL):
            candidate = m.group(1).strip()
            # Limit to reasonable length (avoid capturing entire paragraphs)
            if len(candidate) < 1000:
                all_bases.append(candidate)
    if all_bases:
        result["Can_cu_dieu_luat"] = "\n".join(all_bases)

    # ── Phạm tội ─────────────────────────────────────────────────────
    # Extract quoted crime names (handles various quote styles + colon)
    crime_patterns = [
        # Quoted with optional colon: phạm tội: "Tên tội" or phạm tội "Tên tội"
        r'(?:phạm\s+tội|về\s+tội)\s*:?\s*["“”\u201c\u201d]([^"“”\u201c\u201d]+)["“”\u201c\u201d]',
        r'(?:tội)\s*["“”\u201c\u201d]([^"“”\u201c\u201d]+)["“”\u201c\u201d]',
        # Unquoted after colon: phạm tội: Tên tội
        r'phạm\s+tội\s*:\s*([A-ZÀ-Ỹ][^.;\n]{3,80})',
        # Unquoted no colon: phạm tội Tên tội (capitalized, ends at period/comma)
        r'phạm\s+tội\s+([A-ZÀ-Ỹ][a-zà-ỹ\s]{2,}(?:[a-zà-ỹ]+\s*){1,12}[a-zà-ỹ]+)',
    ]
    crimes = set()
    for pat in crime_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            crime_name = m.group(1).strip().rstrip(',. ')
            if crime_name and len(crime_name) > 2:
                crimes.add(crime_name)
    if crimes:
        result["Pham_toi"] = "; ".join(sorted(crimes))

    # ── Phạt tù ──────────────────────────────────────────────────────
    sentence_patterns = [
        # "Xử phạt bị cáo X ... NN năm/tháng tù"
        r'[Xx]ử\s+phạt[:\s]+.*?(\d+\s*\([^)]+\)\s*(?:năm|tháng)\s*(?:\d+\s*\([^)]+\)\s*(?:năm|tháng))?\s*tù(?:\s+(?:giam|nhưng\s+cho\s+hưởng\s+án\s+treo))?)',
        # Simpler: "NN năm tù" or "NN tháng tù"
        r'[Xx]ử\s+phạt.*?(\d+[\s\(]*[^)]*[\)]*\s*(?:năm|tháng)(?:\s+\d+[\s\(]*[^)]*[\)]*\s*(?:năm|tháng))?\s*tù)',
        # Non-custodial: "NN năm/tháng cải tạo không giam giữ"
        r'[Xx]ử\s+phạt.*?(\d+[\s\(]*[^)]*[\)]*\s*(?:năm|tháng)(?:\s+\d+[\s\(]*[^)]*[\)]*\s*(?:năm|tháng))?\s*cải\s+tạo\s+không\s+giam\s+giữ)',
        # "tù chung thân"
        r'[Xx]ử\s+phạt.*?(tù\s+chung\s+thân)',
        # "tử hình"
        r'[Xx]ử\s+phạt.*?(tử\s+hình)',
    ]
    sentences = []
    for pat in sentence_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE | re.DOTALL):
            sent = m.group(1).strip()
            if sent and sent not in sentences:
                sentences.append(sent)
        if sentences:
            break  # Use first matching pattern group
    if sentences:
        result["Phat_tu"] = "; ".join(sentences)

    # ── Phạt tiền ────────────────────────────────────────────────────
    fine_patterns = [
        r'(?:[Pp]hạt\s+(?:bổ\s+sung\s+)?(?:tiền|mỗi\s+bị\s+cáo)|[Pp]hạt\s+tiền).*?(\d[\d.,]+)\s*(?:đồng|VND)',
        r'(?:phạt\s+bổ\s+sung).*?số\s+tiền\s+(\d[\d.,]+)\s*(?:đồng|VND)',
    ]
    fines = []
    for pat in fine_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            fines.append(m.group(0).strip())
    if fines:
        result["Phat_tien"] = "; ".join(fines)

    # ── Án phí ───────────────────────────────────────────────────────
    an_phi_patterns = [
        # "án phí ... NNN đồng" (án phí appears before the amount)
        r'(?:[Áá]n\s+phí[^.\n]{0,200}?(?:\d[\d.,]+\s*(?:đồng|VND|đ)[^.\n]*))',
        # "NNN đồng ... án phí" (amount appears before án phí, same line/sentence)
        r'(?:\d[\d.,]+\s*(?:\([^)]*\)\s*)?(?:đồng|đ)\s*[^.\n]{0,150}?án\s+phí[^.\n]*)',
        # "phải chịu/nộp NNN ... đồng ... án phí" (verb + amount + án phí)
        r'(?:phải\s+(?:chịu|nộp)\s+\d[\d.,]+\s*(?:\([^)]*\)\s*)?(?:đồng|đ)[^.\n]{0,150}?án\s+phí[^.\n]*)',
        # "án phí ... không phải nộp/chịu" / "được miễn"
        r'(?:[Áá]n\s+phí[^.\n]{0,200}?(?:không\s+phải\s+(?:nộp|chịu)|được\s+miễn)[^.\n]*)',
        # "không phải chịu/nộp án phí"
        r'(?:không\s+phải\s+(?:chịu|nộp)\s+[^.\n]{0,150}?án\s+phí[^.\n]*)',
        # "không phải án phí" (shorthand exemption)
        r'(?:không\s+phải\s+án\s+phí[^.\n]*)',
        # "miễn ... án phí"
        r'(?:miễn[^.\n]{0,150}?án\s+phí[^.\n]*)',
    ]
    an_phi_matches = []
    captured_spans = []  # list of (start, end) tuples
    for pat in an_phi_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            ms, me = m.start(), m.end()
            # Skip if this match overlaps with any already-captured span
            if any(not (me <= cs or ms >= ce) for cs, ce in captured_spans):
                continue
            captured_spans.append((ms, me))
            an_phi_matches.append(m.group(0).strip())
    if an_phi_matches:
        result["An_phi"] = "; ".join(an_phi_matches)

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  QUYET_DINH SECTION CLEANUP
# ═════════════════════════════════════════════════════════════════════════════

def clean_quyet_dinh(quyet_dinh: Optional[str]) -> Optional[str]:
    """Strip trailing 'Nơi nhận' / signatory block from QUYET_DINH."""
    if not quyet_dinh:
        return None
    
    # Find and truncate at signatory / receipient sections
    cut_patterns = [
        r'\n\s*(?:#{1,6}\s*)?(?:\*{0,2})?(?:Nơi\s+nhận|NOI\s+NHAN)\s*[:\.]',
        r'\n\s*(?:#{1,6}\s*)?TM\.\s+H[ỘO]I\s+[ĐD][ỒO]NG',
    ]
    
    text = quyet_dinh
    earliest_cut = len(text)
    for pat in cut_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.start() < earliest_cut:
            earliest_cut = m.start()

    if earliest_cut < len(text):
        text = text[:earliest_cut].strip()
    
    return text if text else None


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def extract_fields(record: dict) -> dict:
    """Extract all target fields from one caselaw JSON record."""
    
    bi_cao = record.get("Bi_cao")
    lien_quan = record.get("Lien_quan")
    noi_dung = record.get("NOI_DUNG_VU_AN")
    nhan_dinh = record.get("NHAN_DINH")
    quyet_dinh = record.get("QUYET_DINH")
    
    # 1. Case type
    case_type = detect_case_type(bi_cao, noi_dung)
    
    # 2. Defendant list
    danh_sach_bi_cao = extract_danh_sach_bi_cao(bi_cao)
    
    # 3. NOI_DUNG sub-fields
    noi_dung_fields = extract_noi_dung_sub_fields(noi_dung, case_type)
    
    # 6. Court's assessment split by numbered points
    nhan_dinh_text = extract_nhan_dinh_points(nhan_dinh)
    
    return {
        "filename": record.get("filename"),
        "case_type": case_type,
        "Danh_sach_bi_cao": danh_sach_bi_cao,
        "Nguoi_tham_gia": lien_quan,
        "Noi_dung_vu_an": {
            "Qua_trinh_dieu_tra": noi_dung_fields["Qua_trinh_dieu_tra"],
            "Ket_luan_cac_ben": noi_dung_fields["Ket_luan_cac_ben"],
        },
        "Nhan_dinh_cua_toa_an": nhan_dinh_text,
        "QUYET_DINH": quyet_dinh,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def _preview(text, max_len: int = 120) -> str:
    """Return a short preview of text for reporting."""
    if text is None:
        return "NULL"
    if isinstance(text, dict):
        s = json.dumps(text, ensure_ascii=False)
    else:
        s = str(text)
    s = s.replace("\n", " ").strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def generate_report(results: list[dict]) -> str:
    """Generate summary report over all extraction results."""
    total = len(results)
    if total == 0:
        return "No files processed."

    # Count case types
    type_counts = {}
    for r in results:
        ct = r["case_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    # Field success counts
    field_stats = {
        "Danh_sach_bi_cao": 0,
        "Noi_dung.Qua_trinh_dieu_tra": 0,
        "Noi_dung.Ket_luan_cac_ben": 0,
        "Nhan_dinh_cua_toa_an": 0,
        "QUYET_DINH": 0,
    }
    
    # Track failed files per field
    failed_files = {k: [] for k in field_stats}
    

    for r in results:
        fname = r["filename"]
        ct = r["case_type"]
        
        # Universal fields
        if r["Danh_sach_bi_cao"]:
            field_stats["Danh_sach_bi_cao"] += 1
        else:
            failed_files["Danh_sach_bi_cao"].append(fname)

        if r["Noi_dung_vu_an"]["Qua_trinh_dieu_tra"]:
            field_stats["Noi_dung.Qua_trinh_dieu_tra"] += 1
        else:
            failed_files["Noi_dung.Qua_trinh_dieu_tra"].append(fname)

        if r["Noi_dung_vu_an"]["Ket_luan_cac_ben"]:
            field_stats["Noi_dung.Ket_luan_cac_ben"] += 1
        else:
            failed_files["Noi_dung.Ket_luan_cac_ben"].append(fname)

        if r["Nhan_dinh_cua_toa_an"]:
            field_stats["Nhan_dinh_cua_toa_an"] += 1
        else:
            failed_files["Nhan_dinh_cua_toa_an"].append(fname)

        if r.get("QUYET_DINH"):
            field_stats["QUYET_DINH"] += 1
        else:
            failed_files["QUYET_DINH"].append(fname)

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("CASELAW FIELD EXTRACTION — TEST REPORT")
    lines.append("=" * 80)
    lines.append(f"\nTotal files processed: {total}")
    lines.append(f"\nCase type distribution:")
    for ct, cnt in sorted(type_counts.items()):
        lines.append(f"  {ct:15s} : {cnt:3d} ({100*cnt/total:.1f}%)")

    lines.append(f"\n{'─' * 80}")
    lines.append("FIELD EXTRACTION SUCCESS RATES")
    lines.append(f"{'─' * 80}")
    lines.append(f"{'Field':<45s} {'Success':>8s} {'Total':>8s} {'Rate':>8s}")
    lines.append(f"{'─' * 45} {'─' * 8} {'─' * 8} {'─' * 8}")

    for field, count in field_stats.items():
        denom = total
        label = field

        rate = f"{100*count/denom:.1f}%" if denom > 0 else "N/A"
        lines.append(f"{label:<45s} {count:>8d} {denom:>8d} {rate:>8s}")

    lines.append(f"\n{'─' * 80}")
    lines.append("FAILED FILES PER FIELD (showing max 10)")
    lines.append(f"{'─' * 80}")

    for field, fnames in failed_files.items():
        if not fnames:
            continue
        relevant_fnames = fnames
        
        if not relevant_fnames:
            continue
            
        lines.append(f"\n  {field} ({len(relevant_fnames)} failures):")
        for fn in relevant_fnames:
            lines.append(f"    ✗ {fn}")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Test rule-based field extraction from caselaw JSON")
    ap.add_argument("--input-dir",  type=Path, default=INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--file", help="Process a single file (name only)")
    ap.add_argument("--report-only", action="store_true",
                    help="Only print report, don't save extracted JSON files")
    args = ap.parse_args()

    if not args.report_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [args.input_dir / args.file]
    else:
        files = sorted(args.input_dir.glob("*.json"))

    print(f"Processing {len(files)} file(s) from {args.input_dir}\n")

    results = []
    for fp in files:
        if not fp.exists():
            print(f"  SKIP (not found): {fp.name}")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            record = json.load(f)
        extracted = extract_fields(record)
        results.append(extracted)
        
        if not args.report_only:
            # Check if this is a failure (any key field is null)
            has_failure = (
                not extracted["Danh_sach_bi_cao"]
                or not extracted["Noi_dung_vu_an"]["Qua_trinh_dieu_tra"]
                or not extracted["Noi_dung_vu_an"]["Ket_luan_cac_ben"]
                or not extracted["Nhan_dinh_cua_toa_an"]
                or not extracted.get("QUYET_DINH")
            )
            if not has_failure:
                out_path = args.output_dir / fp.name
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(extracted, f, ensure_ascii=False, indent=2)

    # Generate and print report
    report = generate_report(results)
    print(report)

    # Save report
    report_path = args.input_dir.parent / "extraction_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    if not args.report_only:
        print(f"Extracted JSON files saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
