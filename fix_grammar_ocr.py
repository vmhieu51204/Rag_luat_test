"""
fix_grammar_ocr.py
──────────────────
Correct OCR-induced grammar / spelling mistakes in **header lines only**
(lines starting with '#') of Vietnamese markdown files, using the local
HuggingFace model bmd1905/vietnamese-correction-v2 (mbart, 0.4B).

After grammar correction, page-break markers ({N}----…) are removed from
the output files.

**Conservative word-level merge**: the model's suggestions are accepted only
when they look like genuine spelling / diacritic fixes.  Changes that drop
words, add punctuation, or alter the structure are silently rejected.

This preserves:
  - Original casing of every word
  - All markdown formatting (headings, lists, bold, italic)
  - Legal identifiers (case numbers, IDs, phone numbers, etc.)

Usage:
    python fix_grammar_ocr.py                          # all files
    python fix_grammar_ocr.py --file <name>.md         # single file
    python fix_grammar_ocr.py --dry-run                # preview only
    python fix_grammar_ocr.py --output-dir ./out       # custom output

Requires:
    pip install transformers torch sentencepiece protobuf
"""

import sys
import re
import argparse
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_DIR = Path(__file__).parent / "ocr_marker"
OUTPUT_DIR = Path(__file__).parent / "ocr_marker_fixed"
MODEL_NAME = "bmd1905/vietnamese-correction-v2"

MAX_SEG_CHARS = 300   # max chars per segment sent to the model
MAX_NEW_TOKENS = 512  # model max output tokens
BATCH_SIZE = 8        # segments per inference batch


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT NORMALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics / tone marks, keep base latin chars."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _strip_md(word: str) -> str:
    """Remove ALL markdown-emphasis characters from a word."""
    word = word.replace("\\*", "").replace("*", "").replace("\\", "")
    return word.strip("\"'\u201c\u201d")


def _norm(word: str) -> str:
    """Normalise a word for alignment: strip markdown + diacritics + lower."""
    return _remove_diacritics(_strip_md(word)).lower()


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,       # delete
                curr[j] + 1,           # insert
                prev[j] + (ca != cb),  # replace
            ))
        prev = curr
    return prev[-1]


# ═══════════════════════════════════════════════════════════════════════════════
#  CASE HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def _case_type(word: str) -> str:
    alpha = [c for c in word if c.isalpha()]
    if not alpha:
        return "none"
    if all(c.isupper() for c in alpha):
        return "upper"
    if all(c.islower() for c in alpha):
        return "lower"
    if alpha[0].isupper() and all(c.islower() for c in alpha[1:]):
        return "title"
    return "mixed"


def _apply_case(word: str, ct: str) -> str:
    if ct == "upper":
        return word.upper()
    if ct == "lower":
        return word.lower()
    if ct == "title":
        result = []
        first = True
        for c in word:
            if c.isalpha() and first:
                result.append(c.upper()); first = False
            elif c.isalpha():
                result.append(c.lower())
            else:
                result.append(c)
        return "".join(result)
    return word


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSERVATIVE WORD-LEVEL MERGE
# ═══════════════════════════════════════════════════════════════════════════════

def _is_valid_correction(orig_clean: str, corr_clean: str) -> bool:
    """Decide whether *corr_clean* is a genuine spelling fix of *orig_clean*.

    Both arguments should already have markdown chars stripped.

    Accepts:
      - Pure diacritic / tone-mark changes  (trăng -> trắng)
      - Small edit-distance base-form changes (đan ông -> đàn ông)
      - Missing-space merges are handled at word-alignment level

    Rejects:
      - Added / changed punctuation only
      - Dramatically different words
      - Empty corrections
    """
    if orig_clean == corr_clean:
        return False

    # Reject if the only diff is trailing punctuation
    o = orig_clean.rstrip(".,;:!?")
    c = corr_clean.rstrip(".,;:!?")
    if o == c:
        return False
    if not o or not c:
        return False

    o_base = _remove_diacritics(o).lower()
    c_base = _remove_diacritics(c).lower()

    # Same base form, different diacritics -> valid
    if o_base == c_base:
        return True

    # Reject if correction introduces punctuation / symbols not in original
    o_nonalnum = set(c for c in o if not c.isalnum())
    c_nonalnum = set(c for c in c if not c.isalnum())
    if c_nonalnum - o_nonalnum:          # new punctuation chars added
        return False

    # Close base forms (edit distance <= 2 AND length ratio reasonable)
    dist = _edit_distance(o_base, c_base)
    max_len = max(len(o_base), len(c_base))
    if max_len == 0:
        return False
    if dist <= 2 and dist / max_len < 0.4:
        return True

    return False


def _is_word_split(orig_word: str, corr_words: list[str]) -> bool:
    """Check if *corr_words* are a valid split of *orig_word*.

    E.g. ``QUYÉTĐỊNH`` → ``['QUYẾT', 'ĐỊNH']``.
    Returns True when the concatenated base forms are close enough.
    """
    o_base = _remove_diacritics(_strip_md(orig_word)).lower()
    c_base = _remove_diacritics("".join(_strip_md(w) for w in corr_words)).lower()
    if not o_base or not c_base:
        return False
    # Exact base match (only diacritics / space differ)
    if o_base == c_base:
        return True
    # Allow small edit distance for OCR noise
    dist = _edit_distance(o_base, c_base)
    max_len = max(len(o_base), len(c_base))
    return dist <= 2 and dist / max_len < 0.35


def _apply_case_to_words(words: list[str], case_type: str) -> list[str]:
    """Apply *case_type* uniformly to a list of words."""
    if case_type in ("none", "mixed"):
        return words
    return [_apply_case(w, case_type) for w in words]


def conservative_merge(original: str, corrected: str) -> str:
    """Merge *corrected* into *original* at word level.

    Strategy: start from the original text; only accept individual word
    replacements that pass ``_is_valid_correction``.  Also handles word
    splits (one merged OCR word → multiple corrected words).
    Always preserve original casing and markdown characters.
    """
    if original == corrected:
        return original

    orig_words = original.split()
    corr_words = corrected.split()

    if not orig_words or not corr_words:
        return original

    # Reject if model output is drastically shorter (word-level)
    if len(corr_words) < len(orig_words) * 0.6:
        return original

    sm = SequenceMatcher(
        None,
        [_norm(w) for w in orig_words],
        [_norm(w) for w in corr_words],
    )

    result = list(orig_words)  # always start from original
    # Track offset shifts caused by word splits (inserts extra words)
    offset = 0

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            # Even "equal" pairs may differ in diacritics (since _norm
            # strips them).  Accept pure diacritic fixes here.
            for k in range(i2 - i1):
                ow = orig_words[i1 + k]
                cw = corr_words[j1 + k]
                ow_clean = _strip_md(ow)
                cw_clean = _strip_md(cw)
                if ow_clean != cw_clean and _is_valid_correction(ow_clean, cw_clean):
                    md_prefix = ow[: len(ow) - len(ow.lstrip("*\\"))]
                    stripped_right = ow.rstrip("*\\\"")
                    md_suffix = ow[len(stripped_right):] if stripped_right != ow else ""

                    orig_trail = ""
                    for ch in reversed(ow_clean):
                        if ch in ".,;:!?)\"'""":
                            orig_trail = ch + orig_trail
                        else:
                            break
                    core_orig = ow_clean[: len(ow_clean) - len(orig_trail)] if orig_trail else ow_clean

                    core_corr = cw_clean.rstrip(".,;:!?)\"'""")
                    if not core_corr:
                        continue

                    ct = _case_type(core_orig)
                    if ct not in ("none", "mixed"):
                        core_corr = _apply_case(core_corr, ct)

                    result[i1 + k + offset] = md_prefix + core_corr + orig_trail + md_suffix
            continue

        orig_span = i2 - i1
        corr_span = j2 - j1

        if op == "replace" and orig_span == corr_span:
            # One-to-one word replacements — evaluate each pair
            for k in range(orig_span):
                ow = orig_words[i1 + k]
                cw = corr_words[j1 + k]

                ow_clean = _strip_md(ow)
                cw_clean = _strip_md(cw)

                if _is_valid_correction(ow_clean, cw_clean):
                    md_prefix = ow[: len(ow) - len(ow.lstrip("*\\"))]
                    stripped_right = ow.rstrip("*\\\"")
                    md_suffix = ow[len(stripped_right):] if stripped_right != ow else ""

                    orig_trail = ""
                    for ch in reversed(ow_clean):
                        if ch in ".,;:!?)\"'""":
                            orig_trail = ch + orig_trail
                        else:
                            break
                    core_orig = ow_clean[: len(ow_clean) - len(orig_trail)] if orig_trail else ow_clean

                    core_corr = cw_clean.rstrip(".,;:!?)\"'""")
                    if not core_corr:
                        continue

                    ct = _case_type(core_orig)
                    if ct not in ("none", "mixed"):
                        core_corr = _apply_case(core_corr, ct)

                    result[i1 + k + offset] = md_prefix + core_corr + orig_trail + md_suffix

        elif op == "replace" and orig_span == 1 and corr_span > 1:
            # One original word → multiple corrected words  (word split)
            ow = orig_words[i1]
            cws = corr_words[j1:j2]
            if _is_word_split(ow, cws):
                ow_clean = _strip_md(ow)
                md_prefix = ow[: len(ow) - len(ow.lstrip("*\\"))]
                stripped_right = ow.rstrip("*\\\"")
                md_suffix = ow[len(stripped_right):] if stripped_right != ow else ""

                # Preserve trailing punctuation from original word
                orig_trail = ""
                for ch in reversed(ow_clean):
                    if ch in ".,;:!?)\"'""":
                        orig_trail = ch + orig_trail
                    else:
                        break

                ct = _case_type(ow_clean)
                fixed = _apply_case_to_words(
                    [_strip_md(w).rstrip(".,;:!?)\"'""") for w in cws], ct
                )
                # Re-attach md markers and trailing punct to the group
                replacement = md_prefix + " ".join(fixed) + orig_trail + md_suffix
                idx = i1 + offset
                result[idx:idx + 1] = [replacement]
                # No net offset change — we replaced 1 slot with 1 joined string

        elif op == "replace" and orig_span > 1 and corr_span == 1:
            # Multiple original words → one corrected word (word merge)
            # Reject: never merge words
            pass

        # For "insert" / "delete" — DO NOTHING (keep original words)

    return " ".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  LINE / SEGMENT SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

_SKIP_LINE_RE = re.compile(r"^\s*$|^\{?\d+\}?-{3,}\s*$")

_MD_PREFIX_RE = re.compile(
    r"^("
    r"#{1,6}\s+"
    r"|(?:\s*[-*+]\s+)+"
    r"|\s*\d+\.\s+"
    r"|\s*>\s+"
    r")?"
)


def _split_line(line: str) -> tuple[str, str]:
    m = _MD_PREFIX_RE.match(line)
    if m and m.group(0):
        return m.group(0), line[len(m.group(0)):]
    return "", line


def _segment_text(text: str, max_chars: int = MAX_SEG_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.;!?])\s+", text)
    segs: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 1 > max_chars and buf:
            segs.append(buf.strip())
            buf = p
        else:
            buf = (buf + " " + p) if buf else p
    if buf.strip():
        segs.append(buf.strip())
    final: list[str] = []
    for seg in segs:
        if len(seg) <= max_chars:
            final.append(seg)
        else:
            sub = re.split(r"(?<=,)\s+", seg)
            buf2 = ""
            for s in sub:
                if len(buf2) + len(s) + 1 > max_chars and buf2:
                    final.append(buf2.strip())
                    buf2 = s
                else:
                    buf2 = (buf2 + " " + s) if buf2 else s
            if buf2.strip():
                final.append(buf2.strip())
    return final if final else [text]


def _clean_for_model(text: str) -> str:
    """Strip inline markdown characters before sending to model."""
    # Remove \* and * used as markdown emphasis (but keep * inside words if any)
    text = text.replace("\\*", "")
    # Remove markdown emphasis markers: *text* or **text**
    text = re.sub(r"\*{1,3}", "", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def init_model():
    from transformers import pipeline as hf_pipeline
    print(f"   Loading model: {MODEL_NAME} ...")
    corrector = hf_pipeline("text2text-generation", model=MODEL_NAME)
    print("   Model loaded.")
    return corrector


def correct_segments(corrector, segments: list[str], batch_size: int = BATCH_SIZE) -> list[str]:
    if not segments:
        return []
    results: list[str] = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        preds = corrector(batch, max_new_tokens=MAX_NEW_TOKENS, batch_size=len(batch))
        for pred in preds:
            results.append(pred["generated_text"])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(corrector, filepath: Path, output_dir: Path,
                 dry_run: bool = False, batch_size: int = BATCH_SIZE) -> bool:
    print(f"\n  {filepath.name}")
    text = filepath.read_text(encoding="utf-8")
    if not text.strip():
        print("   (empty, skipping)")
        return False

    lines = text.split("\n")

    # ── Collect segments ──────────────────────────────────────────────────
    # (line_idx, seg_idx, prefix, original_segment, cleaned_for_model)
    tasks: list[tuple[int, int, str, str, str]] = []

    for li, line in enumerate(lines):
        if _SKIP_LINE_RE.match(line):
            continue
        # Only correct header lines (lines starting with '#')
        if not line.lstrip().startswith('#'):
            continue
        prefix, content = _split_line(line)
        if not content.strip():
            continue
        segments = _segment_text(content.strip())
        for si, seg in enumerate(segments):
            cleaned = _clean_for_model(seg)
            if cleaned.strip():
                tasks.append((li, si, prefix, seg, cleaned))

    if not tasks:
        print("   (no correctable content)")
        return False

    print(f"   {len(tasks)} segment(s) across {len(lines)} lines")

    # ── Batch correct ─────────────────────────────────────────────────────
    cleaned_texts = [t[4] for t in tasks]
    corrected_texts = correct_segments(corrector, cleaned_texts, batch_size)

    # ── Conservative merge per segment ────────────────────────────────────
    merged: list[str] = []
    changes = 0
    for (li, si, prefix, orig_seg, cleaned), corrected in zip(tasks, corrected_texts):
        m = conservative_merge(orig_seg, corrected)
        if m != orig_seg:
            changes += 1
        merged.append(m)

    print(f"   {changes}/{len(tasks)} segments had accepted corrections")

    # ── Reassemble lines ──────────────────────────────────────────────────
    line_parts: dict[int, list[tuple[int, str, str]]] = {}
    for (li, si, prefix, _o, _c), m in zip(tasks, merged):
        line_parts.setdefault(li, []).append((si, prefix, m))

    output_lines = list(lines)
    for li, parts in line_parts.items():
        parts.sort(key=lambda x: x[0])
        prefix = parts[0][1]
        joined = " ".join(p[2] for p in parts)
        output_lines[li] = prefix + joined

    fixed_text = "\n".join(output_lines)

    # ── Remove page breaks ({N}----…) ─────────────────────────────────────
    fixed_text = re.sub(r'\n*\s*\{?\d+\}?-{3,}\s*\n*', '\n\n', fixed_text)
    fixed_text = fixed_text.strip() + '\n'

    # ── Output ────────────────────────────────────────────────────────────
    ratio = len(fixed_text) / max(len(text), 1)
    if ratio < 0.7:
        print(f"   Warning: output is {ratio:.0%} of original — review!")

    if dry_run:
        print(f"   [DRY-RUN] {changes} corrections would be applied")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filepath.name
        out_path.write_text(fixed_text, encoding="utf-8")
        print(f"   Saved -> {out_path}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fix OCR grammar in Vietnamese legal .md files "
                    "using bmd1905/vietnamese-correction-v2 (local)."
    )
    parser.add_argument("--file", type=str, default=None,
                        help="Process only this filename")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing")
    parser.add_argument("--input-dir", type=str, default=None,
                        help=f"Input directory (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    if args.file:
        target = input_dir / args.file
        if not target.exists():
            sys.exit(f"ERROR: File not found: {target}")
        files = [target]
    else:
        files = sorted(input_dir.glob("*.md"))

    if not files:
        sys.exit(f"ERROR: No .md files in {input_dir}")

    print("OCR Grammar Fixer (local model, conservative merge)")
    print(f"   Model      : {MODEL_NAME}")
    print(f"   Input dir  : {input_dir}")
    print(f"   Output dir : {output_dir}")
    print(f"   Files      : {len(files)}")
    print(f"   Batch size : {args.batch_size}")
    print(f"   Dry run    : {args.dry_run}")

    corrector = init_model()

    success = 0
    failed = 0
    for f in files:
        try:
            if process_file(corrector, f, output_dir,
                            dry_run=args.dry_run, batch_size=args.batch_size):
                success += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            break
        except Exception as e:
            print(f"   Error: {f.name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Done.  Processed: {success}  |  Failed: {failed}  |  Total: {len(files)}")


if __name__ == "__main__":
    main()
