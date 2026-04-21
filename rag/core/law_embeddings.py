"""Chunk and embed nested law JSON structures into ChromaDB."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rag.core.embeddings import load_chroma, load_model, split_text


def _norm_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"\s+", "", token)
    return token


def _first_line(text: str) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _mentions_to_signatures(mentions: Any, *, default_dieu: str = "", default_khoan: str = "") -> list[str]:
    if not isinstance(mentions, list):
        return []
    out: list[str] = []
    for item in mentions:
        if not isinstance(item, dict):
            continue
        m_type = _norm_token(item.get("type"))
        idx = _norm_token(item.get("index"))
        if not idx:
            continue
        if m_type == "dieu":
            out.append(idx)
        elif m_type == "khoan" and default_dieu:
            out.append(f"{default_dieu}-{idx}")
        elif m_type == "diem" and default_dieu and default_khoan:
            out.append(f"{default_dieu}-{default_khoan}-{idx}")
    return out


def build_law_chunks(
    raw_law_path: str | Path,
    *,
    law_id: str = "blhs",
    max_chunk_chars: int = 1500,
) -> list[dict[str, Any]]:
    """Create hierarchy-aware chunk records from nested raw_law JSON."""
    with open(raw_law_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("raw_law.json must be a top-level list")

    source_file = Path(raw_law_path).name
    doc_id = f"LAW::{law_id}"
    chunks: list[dict[str, Any]] = []

    def next_chunk_id(level: str, signature: str, chapter: str, chunk_idx: int) -> str:
        # raw_law can contain repeated signatures, so include a running ordinal.
        return f"law::{law_id}::{level}::{signature}::chuong-{chapter}::{chunk_idx:02d}::{len(chunks):06d}"

    for chapter in data:
        if not isinstance(chapter, dict):
            continue

        chapter_idx = str(chapter.get("index", "")).strip()
        chapter_title = str(chapter.get("text", "")).strip()
        chapter_refs = _mentions_to_signatures(chapter.get("mentions"))

        for dieu in chapter.get("Điều", []) or []:
            if not isinstance(dieu, dict):
                continue

            dieu_idx = _norm_token(dieu.get("index"))
            if not dieu_idx:
                continue

            dieu_heading = _first_line(dieu.get("text", ""))
            dieu_text = str(dieu.get("text", "")).strip()
            dieu_sig = dieu_idx
            dieu_refs = _mentions_to_signatures(dieu.get("mentions"), default_dieu=dieu_idx)

            khoan_items = dieu.get("Khoản", []) or []
            if not khoan_items:
                payload = (
                    f"[LAW:{law_id}] [Chuong {chapter_idx}] [Dieu {dieu_idx}]\n"
                    f"{chapter_title}\n{dieu_text}"
                ).strip()
                parts = split_text(payload, max_chunk_chars)
                for chunk_idx, part in enumerate(parts):
                    chunks.append(
                        {
                            "id": next_chunk_id("dieu", dieu_sig, chapter_idx, chunk_idx),
                            "text": part,
                            "doc_id": doc_id,
                            "source_file": source_file,
                            "source_type": "law",
                            "law_id": law_id,
                            "law_level": "dieu",
                            "chapter_index": chapter_idx,
                            "dieu": dieu_idx,
                            "khoan": "",
                            "diem": "",
                            "law_signature_full": dieu_sig,
                            "law_dieu": dieu_idx,
                            "mentions": "|".join(sorted(set(chapter_refs + dieu_refs))),
                        }
                    )

            for khoan in khoan_items:
                if not isinstance(khoan, dict):
                    continue

                khoan_idx = _norm_token(khoan.get("index"))
                if not khoan_idx:
                    continue

                khoan_text = str(khoan.get("text", "")).strip()
                khoan_sig = f"{dieu_idx}-{khoan_idx}"
                khoan_refs = _mentions_to_signatures(
                    khoan.get("mentions"),
                    default_dieu=dieu_idx,
                    default_khoan=khoan_idx,
                )
                payload = (
                    f"[LAW:{law_id}] [Chuong {chapter_idx}] [Dieu {dieu_idx}] [Khoan {khoan_idx}]\n"
                    f"{chapter_title}\n{dieu_heading}\n{khoan_text}"
                ).strip()
                parts = split_text(payload, max_chunk_chars)
                for chunk_idx, part in enumerate(parts):
                    chunks.append(
                        {
                            "id": next_chunk_id("khoan", khoan_sig, chapter_idx, chunk_idx),
                            "text": part,
                            "doc_id": doc_id,
                            "source_file": source_file,
                            "source_type": "law",
                            "law_id": law_id,
                            "law_level": "khoan",
                            "chapter_index": chapter_idx,
                            "dieu": dieu_idx,
                            "khoan": khoan_idx,
                            "diem": "",
                            "law_signature_full": khoan_sig,
                            "law_dieu": dieu_idx,
                            "mentions": "|".join(sorted(set(chapter_refs + dieu_refs + khoan_refs))),
                        }
                    )

                for diem in khoan.get("Điểm", []) or []:
                    if not isinstance(diem, dict):
                        continue

                    diem_idx = _norm_token(diem.get("index"))
                    if not diem_idx:
                        continue

                    diem_text = str(diem.get("text", "")).strip()
                    diem_sig = f"{dieu_idx}-{khoan_idx}-{diem_idx}"
                    diem_refs = _mentions_to_signatures(
                        diem.get("mentions"),
                        default_dieu=dieu_idx,
                        default_khoan=khoan_idx,
                    )
                    payload = (
                        f"[LAW:{law_id}] [Chuong {chapter_idx}] [Dieu {dieu_idx}] "
                        f"[Khoan {khoan_idx}] [Diem {diem_idx}]\n"
                        f"{chapter_title}\n{dieu_heading}\n{khoan_text}\n{diem_text}"
                    ).strip()
                    parts = split_text(payload, max_chunk_chars)
                    for chunk_idx, part in enumerate(parts):
                        chunks.append(
                            {
                                "id": next_chunk_id("diem", diem_sig, chapter_idx, chunk_idx),
                                "text": part,
                                "doc_id": doc_id,
                                "source_file": source_file,
                                "source_type": "law",
                                "law_id": law_id,
                                "law_level": "diem",
                                "chapter_index": chapter_idx,
                                "dieu": dieu_idx,
                                "khoan": khoan_idx,
                                "diem": diem_idx,
                                "law_signature_full": diem_sig,
                                "law_dieu": dieu_idx,
                                "mentions": "|".join(
                                    sorted(set(chapter_refs + dieu_refs + khoan_refs + diem_refs))
                                ),
                            }
                        )

    return chunks


def embed_law_chunks(
    *,
    raw_law_path: str,
    db_dir: str,
    model_name: str,
    device: str,
    batch_size: int,
    collection_name: str,
    law_id: str,
    max_chunk_chars: int,
) -> int:
    """Chunk and embed law data into an existing ChromaDB collection."""
    chunks = build_law_chunks(
        raw_law_path,
        law_id=law_id,
        max_chunk_chars=max_chunk_chars,
    )
    if not chunks:
        print("  [WARN] No law chunks were produced.")
        return 0

    model = load_model(model_name=model_name, device=device)
    collection = load_chroma(db_dir, collection_name=collection_name, create=True)

    ids = [chunk["id"] for chunk in chunks]
    existing = set(collection.get(ids=ids).get("ids", []))
    new_chunks = [chunk for chunk in chunks if chunk["id"] not in existing]
    if not new_chunks:
        print(f"  All {len(chunks)} law chunks already present in DB.")
        return 0

    print(f"  Embedding {len(new_chunks)} new law chunks ({len(existing)} already present) ...")

    ids_new = [chunk["id"] for chunk in new_chunks]
    docs_new = [chunk["text"] for chunk in new_chunks]
    metas_new = [
        {
            "doc_id": chunk["doc_id"],
            "source_file": chunk["source_file"],
            "source_type": chunk["source_type"],
            "law_id": chunk["law_id"],
            "law_level": chunk["law_level"],
            "chapter_index": chunk["chapter_index"],
            "dieu": chunk["dieu"],
            "khoan": chunk["khoan"],
            "diem": chunk["diem"],
            "law_signature_full": chunk["law_signature_full"],
            "law_dieu": chunk["law_dieu"],
            "mentions": chunk["mentions"],
        }
        for chunk in new_chunks
    ]

    all_vecs: list[list[float]] = []
    for start in range(0, len(docs_new), batch_size):
        batch = docs_new[start:start + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.extend(vecs.tolist())

    chroma_batch = 500
    for start in range(0, len(ids_new), chroma_batch):
        sl = slice(start, start + chroma_batch)
        collection.upsert(
            ids=ids_new[sl],
            embeddings=all_vecs[sl],
            documents=docs_new[sl],
            metadatas=metas_new[sl],
        )

    print(f"  Added {len(ids_new)} law chunks to '{collection_name}'.")
    return len(ids_new)
