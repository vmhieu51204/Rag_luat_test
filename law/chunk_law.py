import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def make_dieu_id(chuong_index: str, dieu_index: str) -> str:
    """e.g. 'chuong_I_dieu_10'"""
    return f"chuong_{chuong_index}_dieu_{dieu_index}"


def make_khoan_id(dieu_id: str, khoan_index: str) -> str:
    """e.g. 'chuong_I_dieu_10_khoan_1'"""
    return f"{dieu_id}_khoan_{khoan_index}"


def make_diem_id(khoan_id: str, diem_index: str) -> str:
    """e.g. 'chuong_I_dieu_10_khoan_1_diem_a'"""
    return f"{khoan_id}_diem_{diem_index}"


# ---------------------------------------------------------------------------
# Text builders  – each level prepends its ancestors' context
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Text builders  – each level prepends its ancestors' context (WITHOUT children)
# ---------------------------------------------------------------------------

def _get_chuong_header(chuong: dict) -> str:
    """Helper to safely build the Chapter header, ignoring dummy 'Unknown' chapters."""
    if chuong.get("index") == "Unknown":
        return ""
    
    header = f"Chương {chuong['index']}"
    if chuong.get("text"):
        header += f" – {chuong['text'].strip()}"
    return header

def build_dieu_text(chuong: dict, dieu: dict) -> str:
    """
    Self-contained text for a Điều.
    Contains ONLY the Chapter header and the Article's preamble/title.
    Does NOT append the text of its child Khoản/Điểm to prevent RAG duplication.
    """
    lines = []
    
    chuong_header = _get_chuong_header(chuong)
    if chuong_header:
        lines.append(chuong_header)

    lines.append(f"Điều {dieu['index']}. {dieu['text'].strip()}")
    
    return "\n".join(lines)


def build_khoan_text(chuong: dict, dieu: dict, khoan: dict) -> str:
    """
    Self-contained text for a Khoản.
    Contains Chapter, Article preamble, and this Clause's text.
    Does NOT append the text of its child Điểm.
    """
    lines = []

    chuong_header = _get_chuong_header(chuong)
    if chuong_header:
        lines.append(chuong_header)

    lines.append(f"Điều {dieu['index']}. {dieu['text'].strip()}")
    lines.append(f"  {khoan['index']}. {khoan['text'].strip()}")

    return "\n".join(lines)


def build_diem_text(chuong: dict, dieu: dict, khoan: dict, diem: dict) -> str:
    """
    Self-contained text for a Điểm (Leaf node).
    Contains Chapter, Article preamble, Clause preamble, and this Point's text.
    """
    lines = []

    chuong_header = _get_chuong_header(chuong)
    if chuong_header:
        lines.append(chuong_header)

    lines.append(f"Điều {dieu['index']}. {dieu['text'].strip()}")
    lines.append(f"  {khoan['index']}. {khoan['text'].strip()}")
    lines.append(f"    {diem['index']}) {diem['text'].strip()}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def law_to_chunks(document_data: list) -> list[dict]:
    """
    Convert the nested parse_law.py output into a flat list of chunks.

    Chunk levels
    ────────────
    • Điều   – one chunk per Điều (contains full Điều text + all descendants)
    • Khoản  – one chunk per Khoản (contains parent headers + this Khoản + its Điểm)
    • Điểm   – one chunk per Điểm (contains parent headers + just this Điểm)

    Each chunk dict has:
        chunk_id        – unique, human-readable ID
        chunk_type      – "Điều" | "Khoản" | "Điểm"
        text            – self-contained text ready for embedding / retrieval
        metadata        – rich metadata dict (see below)
    """
    chunks: list[dict] = []

    for chuong in document_data:
        chuong_index = chuong.get("index", "Unknown")
        chuong_text  = chuong.get("text", "").strip()

        for dieu in chuong.get("Điều", []):
            dieu_index = dieu["index"]
            dieu_id    = make_dieu_id(chuong_index, dieu_index)

            # ── Điều-level chunk ──────────────────────────────────────────
            dieu_chunk = {
                "chunk_id":   dieu_id,
                "chunk_type": "Điều",
                "text":       build_dieu_text(chuong, dieu),
                "metadata": {
                    "chunk_id":      dieu_id,
                    "chunk_type":    "Điều",
                    "parent_id":     None,          # Điều is the top retrieval level
                    "chuong_index":  chuong_index,
                    "chuong_title":  chuong_text,
                    "dieu_index":    dieu_index,
                    "dieu_title":    dieu["text"].strip(),
                    "khoan_index":   None,
                    "diem_index":    None,
                    "mentions":      dieu.get("mentions", []),
                    "child_khoans":  [k["index"] for k in dieu.get("Khoản", [])],
                },
            }
            chunks.append(dieu_chunk)

            for khoan in dieu.get("Khoản", []):
                khoan_index = khoan["index"]
                khoan_id    = make_khoan_id(dieu_id, khoan_index)

                # ── Khoản-level chunk ─────────────────────────────────────
                khoan_chunk = {
                    "chunk_id":   khoan_id,
                    "chunk_type": "Khoản",
                    "text":       build_khoan_text(chuong, dieu, khoan),
                    "metadata": {
                        "chunk_id":      khoan_id,
                        "chunk_type":    "Khoản",
                        "parent_id":     dieu_id,
                        "chuong_index":  chuong_index,
                        "chuong_title":  chuong_text,
                        "dieu_index":    dieu_index,
                        "dieu_title":    dieu["text"].strip(),
                        "khoan_index":   khoan_index,
                        "diem_index":    None,
                        "mentions":      khoan.get("mentions", []),
                        "child_diems":   [d["index"] for d in khoan.get("Điểm", [])],
                    },
                }
                chunks.append(khoan_chunk)

                for diem in khoan.get("Điểm", []):
                    diem_index = diem["index"]
                    diem_id    = make_diem_id(khoan_id, diem_index)

                    # ── Điểm-level chunk ──────────────────────────────────
                    diem_chunk = {
                        "chunk_id":   diem_id,
                        "chunk_type": "Điểm",
                        "text":       build_diem_text(chuong, dieu, khoan, diem),
                        "metadata": {
                            "chunk_id":     diem_id,
                            "chunk_type":   "Điểm",
                            "parent_id":    khoan_id,
                            "chuong_index": chuong_index,
                            "chuong_title": chuong_text,
                            "dieu_index":   dieu_index,
                            "dieu_title":   dieu["text"].strip(),
                            "khoan_index":  khoan_index,
                            "diem_index":   diem_index,
                            "mentions":     diem.get("mentions", []),
                        },
                    }
                    chunks.append(diem_chunk)

    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a parsed Vietnamese law JSON (from parse_law.py) into retrieval chunks."
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default='C:\\Users\\hungn\\Downloads\\darkin\\stage3_final.json',
        help="Path to the parsed JSON file produced by parse_law.py",
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        default=None,
        help="Path for the output chunks JSON (defaults to <input>_chunks.json)",
    )
    args = parser.parse_args()

    input_path  = Path(args.input_json)
    output_path = Path(args.output_json) if args.output_json else input_path.with_stem(input_path.stem + "_chunks")

    with open(input_path, encoding="utf-8") as f:
        document_data = json.load(f)

    chunks = law_to_chunks(document_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    type_counts = {}
    for c in chunks:
        type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1

    print(f"✓ {len(chunks)} chunks written to {output_path}")
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n}")
