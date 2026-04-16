"""
Retrieve legal clause text from law_doc.json using signatures like:
- 174-4-a  -> Dieu 174, Khoan 4, Diem a
- 51-2     -> Dieu 51, Khoan 2
- 51       -> Dieu 51
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIEU_KEY = "Điều"
KHOAN_KEY = "Khoản"
DIEM_KEY = "Điểm"


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_lookup_token(token: Any) -> str:
    value = str(token).strip().lower()
    value = _strip_accents(value).replace("đ", "d")
    value = re.sub(r"\s+", "", value)
    return value


@dataclass(frozen=True)
class ClauseSignature:
    dieu: str
    khoan: str | None = None
    diem: str | None = None


def parse_clause_signature(signature: str) -> ClauseSignature:
    """
    Parse flexible clause references into (dieu, khoan, diem).

    Supported forms:
    - "174-4-a"
    - "51-2"
    - "51"
    - "Dieu 174, khoan 4, diem a"
    """
    if signature is None:
        raise ValueError("signature is None")

    raw = str(signature).strip()
    if not raw:
        raise ValueError("signature is empty")

    folded = _normalize_lookup_token(raw)
    folded = re.sub(r"(dieu|khoan|diem)", "-", folded)
    folded = re.sub(r"[^0-9a-z-]+", "-", folded)
    parts = [part for part in folded.split("-") if part]

    if not parts:
        raise ValueError(f"invalid signature: {signature!r}")
    if not parts[0].isdigit():
        raise ValueError(f"invalid dieu in signature: {signature!r}")
    if len(parts) > 3:
        raise ValueError(f"too many parts in signature: {signature!r}")

    dieu = parts[0]
    khoan = parts[1] if len(parts) >= 2 else None
    diem = parts[2] if len(parts) == 3 else None
    return ClauseSignature(dieu=dieu, khoan=khoan, diem=diem)


class LawClauseRetriever:
    def __init__(self, law_doc_path: str | Path = "law_doc.json") -> None:
        self.law_doc_path = Path(law_doc_path)
        self._dieu_index: dict[str, dict[str, Any]] = {}
        self._khoan_index: dict[tuple[str, str], dict[str, Any]] = {}
        self._diem_index: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        with open(self.law_doc_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError("law_doc.json must be a top-level list")

        for chapter in data:
            if not isinstance(chapter, dict):
                continue

            for dieu in chapter.get(DIEU_KEY, []) or []:
                if not isinstance(dieu, dict):
                    continue

                dieu_idx = _normalize_lookup_token(dieu.get("index", ""))
                if not dieu_idx:
                    continue

                self._dieu_index[dieu_idx] = dieu

                for khoan in dieu.get(KHOAN_KEY, []) or []:
                    if not isinstance(khoan, dict):
                        continue

                    khoan_idx = _normalize_lookup_token(khoan.get("index", ""))
                    if not khoan_idx:
                        continue

                    self._khoan_index[(dieu_idx, khoan_idx)] = khoan

                    for diem in khoan.get(DIEM_KEY, []) or []:
                        if not isinstance(diem, dict):
                            continue

                        diem_idx = _normalize_lookup_token(diem.get("index", ""))
                        if not diem_idx:
                            continue

                        self._diem_index[(dieu_idx, khoan_idx, diem_idx)] = diem

    @staticmethod
    def _extract_text(node: dict[str, Any]) -> str:
        full_text = str(node.get("full_text", "")).strip()
        if full_text:
            return full_text
        return str(node.get("text", "")).strip()

    def retrieve(self, signature: str) -> dict[str, Any]:
        """
        Retrieve one clause reference.

        Returns:
            {
              "query": "174-4-a",
              "normalized": "174-4-a",
              "found": True/False,
              "level": "dieu"|"khoan"|"diem"|None,
              "text": "...",
              "reason": "..."  # only when found=False
            }
        """
        try:
            parsed = parse_clause_signature(signature)
        except ValueError as exc:
            return {
                "query": signature,
                "normalized": None,
                "found": False,
                "level": None,
                "text": None,
                "reason": str(exc),
            }

        dieu_key = _normalize_lookup_token(parsed.dieu)
        normalized = parsed.dieu
        if parsed.khoan is not None:
            normalized += f"-{parsed.khoan}"
        if parsed.diem is not None:
            normalized += f"-{parsed.diem}"

        dieu = self._dieu_index.get(dieu_key)
        if dieu is None:
            return {
                "query": signature,
                "normalized": normalized,
                "found": False,
                "level": None,
                "text": None,
                "reason": "dieu_not_found",
            }

        if parsed.khoan is None:
            return {
                "query": signature,
                "normalized": normalized,
                "found": True,
                "level": "dieu",
                "text": self._extract_text(dieu),
                "dieu": parsed.dieu,
                "khoan": None,
                "diem": None,
            }

        khoan_key = _normalize_lookup_token(parsed.khoan)
        khoan = self._khoan_index.get((dieu_key, khoan_key))
        if khoan is None:
            return {
                "query": signature,
                "normalized": normalized,
                "found": False,
                "level": None,
                "text": None,
                "reason": "khoan_not_found",
                "dieu": parsed.dieu,
                "khoan": parsed.khoan,
                "diem": parsed.diem,
            }

        if parsed.diem is None:
            return {
                "query": signature,
                "normalized": normalized,
                "found": True,
                "level": "khoan",
                "text": self._extract_text(khoan),
                "dieu": parsed.dieu,
                "khoan": parsed.khoan,
                "diem": None,
            }

        diem_key = _normalize_lookup_token(parsed.diem)
        diem = self._diem_index.get((dieu_key, khoan_key, diem_key))
        if diem is None:
            return {
                "query": signature,
                "normalized": normalized,
                "found": False,
                "level": None,
                "text": None,
                "reason": "diem_not_found",
                "dieu": parsed.dieu,
                "khoan": parsed.khoan,
                "diem": parsed.diem,
            }

        return {
            "query": signature,
            "normalized": normalized,
            "found": True,
            "level": "diem",
            "text": str(diem.get("text", "")).strip(),
            "dieu": parsed.dieu,
            "khoan": parsed.khoan,
            "diem": parsed.diem,
        }

    def retrieve_many(self, signatures: list[str]) -> list[dict[str, Any]]:
        return [self.retrieve(sig) for sig in signatures]


def retrieve_law_clause(signature: str, law_doc_path: str | Path = "law_doc.json") -> str | None:
    """Convenience helper that returns only text (or None if not found)."""
    retriever = LawClauseRetriever(law_doc_path)
    result = retriever.retrieve(signature)
    return result["text"] if result["found"] else None


def retrieve_law_clauses(signatures: list[str], law_doc_path: str | Path = "law_doc.json") -> dict[str, str | None]:
    """
    Convenience helper for batch retrieval.
    Returns mapping: signature -> text (or None if not found).
    """
    retriever = LawClauseRetriever(law_doc_path)
    output: dict[str, str | None] = {}
    for sig in signatures:
        result = retriever.retrieve(sig)
        output[sig] = result["text"] if result["found"] else None
    return output


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve BLHS clause text from law_doc.json")
    parser.add_argument(
        "--law_doc",
        default="law_doc.json",
        help="Path to law_doc.json",
    )
    parser.add_argument(
        "--clauses",
        nargs="+",
        required=True,
        help="Clause signatures, e.g. 174-4-a 51-2 51",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    retriever = LawClauseRetriever(args.law_doc)
    results = retriever.retrieve_many(args.clauses)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
