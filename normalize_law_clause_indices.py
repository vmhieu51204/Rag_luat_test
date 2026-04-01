#!/usr/bin/env python3
"""Normalize legal clause index fields in filled JSON outputs.

This script fixes cases where LLM outputs redundant text in index fields, e.g.:
  {"Dieu": "Điều 347"} -> {"Dieu": "347"}
  {"Khoan": "Khoản 1"} -> {"Khoan": "1"}
  {"Diem": "Điểm s"} -> {"Diem": "s"}

By default it scans JSON files in `filled_Chuong_XII`.
Only keys named `Dieu`, `Khoan`, `Diem` are normalized.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TARGET_KEYS = {"Dieu", "Khoan", "Diem"}


def normalize_dieu(value: str) -> str:
    v = value.strip()
    # Remove leading label like "Điều", "Dieu", optional punctuation.
    v = re.sub(r"(?i)^\s*(?:điều|dieu)\s*[:\.-]*\s*", "", v)
    m = re.search(r"\d+[a-zA-Z]?", v)
    return m.group(0) if m else v


def normalize_khoan(value: str) -> str:
    v = value.strip()
    v = re.sub(r"(?i)^\s*(?:khoản|khoan)\s*[:\.-]*\s*", "", v)
    m = re.search(r"\d+[a-zA-Z]?", v)
    return m.group(0) if m else v


def normalize_diem(value: str) -> str:
    v = value.strip()
    v = re.sub(r"(?i)^\s*(?:điểm|diem)\s*[:\.-]*\s*", "", v)
    # Keep first letter token, including Vietnamese "đ".
    m = re.search(r"[a-zA-ZđĐ]", v)
    return m.group(0).lower() if m else v


def normalize_value(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value

    if key == "Dieu":
        return normalize_dieu(value)
    if key == "Khoan":
        return normalize_khoan(value)
    if key == "Diem":
        return normalize_diem(value)
    return value


def normalize_object(obj: Any) -> tuple[Any, int]:
    """Recursively normalize keys Dieu/Khoan/Diem.

    Returns:
      (normalized_obj, change_count)
    """
    changes = 0

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in TARGET_KEYS and isinstance(v, str):
                nv = normalize_value(k, v)
                out[k] = nv
                if nv != v:
                    changes += 1
            else:
                nv, child_changes = normalize_object(v)
                out[k] = nv
                changes += child_changes
        return out, changes

    if isinstance(obj, list):
        out_list = []
        for item in obj:
            ni, child_changes = normalize_object(item)
            out_list.append(ni)
            changes += child_changes
        return out_list, changes

    return obj, 0


def process_file(path: Path, dry_run: bool = False) -> tuple[bool, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, 0

    normalized, change_count = normalize_object(data)
    if change_count == 0:
        return True, 0

    if not dry_run:
        path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return True, change_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Dieu/Khoan/Diem index values in JSON files, "
            "e.g. 'Điều 347' -> '347'."
        )
    )
    parser.add_argument(
        "--folder",
        default="filled_Chuong_XXII",
        help="Folder containing target JSON files (default: filled_Chuong_XII)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    total_files = 0
    changed_files = 0
    total_changes = 0
    invalid_json = 0

    for file_path in sorted(folder.glob("*.json")):
        if file_path.name == "law_doc.json":
            continue
        total_files += 1

        ok, change_count = process_file(file_path, dry_run=args.dry_run)
        if not ok:
            invalid_json += 1
            continue

        if change_count > 0:
            changed_files += 1
            total_changes += change_count

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"[{mode}] folder={folder}")
    print(f"[{mode}] total_files={total_files}")
    print(f"[{mode}] changed_files={changed_files}")
    print(f"[{mode}] total_field_changes={total_changes}")
    print(f"[{mode}] invalid_json={invalid_json}")


if __name__ == "__main__":
    main()
