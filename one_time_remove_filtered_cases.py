"""Remove quality-check filtered cases from chunk files and act_full.json.

This one-time script reads case filenames from qualitycheck_eliminated_files.txt,
removes matching JSON files under chunk/, removes matching docs from act_full.json
through remove_case_eval.py, then runs recalculate_saved_metrics.py.

Preview first:
    uv run python one_time_remove_filtered_cases.py

Apply removals:
    uv run python one_time_remove_filtered_cases.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_FILTER_LOG = ROOT / "qualitycheck_eliminated_files.txt"
DEFAULT_CHUNK_ROOT = ROOT / "chunk"
DEFAULT_RESULTS_PATH = ROOT / "act_full.json"
REMOVE_CASE_SCRIPT = ROOT / "remove_case_eval.py"
RECALCULATE_SCRIPT = ROOT / "rag" / "evaluation" / "recalculate_saved_metrics.py"


def load_filtered_case_ids(filter_log: Path) -> set[str]:
    case_ids: set[str] = set()
    with filter_log.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" in line and "\t" not in line:
                continue
            path_text = line.split("\t", 1)[0].strip()
            if not path_text:
                continue
            case_ids.add(Path(path_text).stem)
    return case_ids


def find_chunk_files(chunk_root: Path, case_ids: set[str]) -> list[Path]:
    return sorted(
        path
        for path in chunk_root.rglob("*.json")
        if path.is_file() and path.stem in case_ids
    )


def find_eval_doc_ids(results_path: Path, case_ids: set[str]) -> list[str]:
    with results_path.open(encoding="utf-8") as fh:
        report = json.load(fh)

    doc_ids: list[str] = []
    for doc in report.get("per_doc", []):
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("doc_id") or "")
        source_file = str(doc.get("source_file") or "")
        source_stem = Path(source_file).stem if source_file else ""
        if doc_id in case_ids or source_stem in case_ids:
            doc_ids.append(doc_id or source_stem)
    return sorted(set(doc_ids))


def backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.before_filtered_cleanup_{timestamp}.bak")
    shutil.copy2(path, backup_path)
    return backup_path


def run_command(args: list[str], apply: bool) -> None:
    display = " ".join(args)
    if not apply:
        print(f"DRY RUN command: {display}")
        return
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-log", type=Path, default=DEFAULT_FILTER_LOG)
    parser.add_argument("--chunk-root", type=Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--apply", action="store_true", help="Actually remove files and rewrite act_full.json")
    args = parser.parse_args()

    filter_log = args.filter_log.resolve()
    chunk_root = args.chunk_root.resolve()
    results_path = args.results.resolve()

    if not filter_log.exists():
        raise FileNotFoundError(f"Missing filter log: {filter_log}")
    if not chunk_root.exists():
        raise FileNotFoundError(f"Missing chunk root: {chunk_root}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results JSON: {results_path}")

    case_ids = load_filtered_case_ids(filter_log)
    chunk_files = find_chunk_files(chunk_root, case_ids)
    eval_doc_ids = find_eval_doc_ids(results_path, case_ids)

    print(f"Filtered case IDs loaded: {len(case_ids)}")
    print(f"Matching chunk files: {len(chunk_files)}")
    print(f"Matching act_full.json docs: {len(eval_doc_ids)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")

    if chunk_files:
        print("\nChunk files to remove:")
        for path in chunk_files[:100]:
            print(f"- {path.relative_to(ROOT)}")
        if len(chunk_files) > 100:
            print(f"... and {len(chunk_files) - 100} more")

    if eval_doc_ids:
        print("\nact_full.json doc IDs to remove:")
        for doc_id in eval_doc_ids[:100]:
            print(f"- {doc_id}")
        if len(eval_doc_ids) > 100:
            print(f"... and {len(eval_doc_ids) - 100} more")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to remove files and update act_full.json.")
        return

    backup_path = backup_file(results_path)
    print(f"\nBackup: {backup_path.relative_to(ROOT)}")

    removed_chunk_count = 0
    for path in chunk_files:
        path.unlink()
        removed_chunk_count += 1
    print(f"Removed chunk files: {removed_chunk_count}")

    for doc_id in eval_doc_ids:
        run_command(
            [sys.executable, str(REMOVE_CASE_SCRIPT), str(results_path), doc_id],
            apply=True,
        )

    run_command(
        [
            sys.executable,
            str(RECALCULATE_SCRIPT),
            str(results_path),
            "--no-backup",
        ],
        apply=True,
    )
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
