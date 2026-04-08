"""
chunk_json.py
=============
Reads each JSON file in an input folder and produces one chunk file per
non-empty field (excluding the ID field).  Each chunk is saved as a
separate JSON file so it can be fed directly to an embedding / RAG pipeline.

Output file naming:
    <Ma_Ban_An>__<field_name>__<chunk_index>.json

Example:
    08-11-2024-Dien_Bien-2ta1687131t1cvn__Summary__0.json
"""

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fields that carry textual content to be chunked (order matters for output)
CONTENT_FIELDS = ["Summary", "Tang_nang", "Giam_nhe"]

# The field that acts as the document identifier
ID_FIELD = "Ma_Ban_An"

# Maximum character length of a single chunk.
# Set to None to keep each field as one chunk (no further splitting).
MAX_CHUNK_CHARS = 1500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_text(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters.
    Tries to break on sentence boundaries (. ! ?) when possible.
    """
    if not text or not text.strip():
        return []

    if max_chars is None or len(text) <= max_chars:
        return [text.strip()]

    sentence_end = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end.split(text)

    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence is longer than max_chars, hard-split it
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars].strip())
                sentence = sentence[max_chars:]
            current = sentence.strip()

    if current:
        chunks.append(current)

    return chunks


def chunk_file(input_path: Path, output_dir: Path) -> list[dict]:
    """
    Read one JSON file, produce chunk records, write them to *output_dir*.
    Returns the list of chunk metadata dicts (for test/reporting purposes).
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get(ID_FIELD, input_path.stem)
    produced = []

    for field in CONTENT_FIELDS:
        raw_text = data.get(field, "") or ""
        raw_text = raw_text.strip()
        if not raw_text:
            continue  # skip empty fields

        sub_chunks = split_text(raw_text, MAX_CHUNK_CHARS)

        for idx, chunk_text in enumerate(sub_chunks):
            chunk_record = {
                "doc_id":      doc_id,
                "source_file": input_path.name,
                "field":       field,
                "chunk_index": idx,
                "total_chunks_for_field": len(sub_chunks),
                "text":        chunk_text,
            }

            safe_id = re.sub(r'[^A-Za-z0-9_\-]', '_', doc_id)
            out_name = f"{safe_id}__{field}__{idx:03d}.json"
            out_path = output_dir / out_name

            with open(out_path, "w", encoding="utf-8") as fout:
                json.dump(chunk_record, fout, ensure_ascii=False, indent=2)

            produced.append(chunk_record)

    return produced


def process_folder(input_folder: str, output_folder: str) -> None:
    input_dir  = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    total_files  = 0
    total_chunks = 0
    errors       = []

    for jf in json_files:
        try:
            chunks = chunk_file(jf, output_dir)
            total_files  += 1
            total_chunks += len(chunks)
            print(f"  ✓ {jf.name:55s} → {len(chunks):3d} chunk(s)")
        except Exception as exc:
            errors.append((jf.name, str(exc)))
            print(f"  ✗ {jf.name} — ERROR: {exc}")

    print("\n" + "="*60)
    print(f"Files processed : {total_files}")
    print(f"Total chunks    : {total_chunks}")
    if errors:
        print(f"Errors          : {len(errors)}")
        for name, msg in errors:
            print(f"   {name}: {msg}")
    print("="*60)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 3:
        inp, out = sys.argv[1], sys.argv[2]
    else:
        # Defaults for quick testing
        inp = "chunk/Chuong_XXII_test_chunked"
        out = "chunk/Chuong_XXII_test_chunked_2"

    print(f"\nInput  : {inp}")
    print(f"Output : {out}\n")
    process_folder(inp, out)