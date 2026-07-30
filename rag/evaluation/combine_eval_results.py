"""Combine two saved evaluation JSON reports and recalculate metrics.

Usage:
    python -m rag.evaluation.combine_eval_results <file1.json> <file2.json> [--output combined.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.evaluation.eval_utils import _aggregate
from rag.evaluation.recalculate_saved_metrics import _recalculate_doc_metrics


def combine_and_recalculate(path1: Path, path2: Path, output_path: Path) -> None:
    with open(path1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    
    with open(path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    # Combine configurations (defaulting to the first one)
    combined_config = data1.get("config", {})
    
    # Merge per_doc lists
    per_doc1 = data1.get("per_doc", [])
    per_doc2 = data2.get("per_doc", [])
    
    # Optional: deduplicate by doc_id or source_file if necessary
    seen_ids = set()
    combined_per_doc = []
    
    for doc in per_doc1 + per_doc2:
        if not isinstance(doc, dict):
            continue
        
        doc_id = doc.get("doc_id") or doc.get("source_file")
        if doc_id in seen_ids:
            print(f"Warning: Duplicate case found and skipped: {doc_id}")
            continue
        
        seen_ids.add(doc_id)
        
        # Recalculate metrics just to be safe
        _recalculate_doc_metrics(doc)
        combined_per_doc.append(doc)
        
    new_summary = _aggregate(combined_per_doc)
    
    combined_report = {
        "config": combined_config,
        "summary": new_summary,
        "per_doc": combined_per_doc
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, ensure_ascii=False, indent=2)
        f.write("\n")
        
    print(f"Combined {len(per_doc1)} docs from {path1.name} and {len(per_doc2)} docs from {path2.name}.")
    print(f"Total unique docs in combined report: {len(combined_per_doc)}")
    print(f"Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine two evaluation JSONs and recalculate metrics.")
    parser.add_argument("file1", type=Path, help="Path to the first JSON report")
    parser.add_argument("file2", type=Path, help="Path to the second JSON report")
    parser.add_argument("--output", type=Path, default=Path("combined_results.json"), help="Output path")
    args = parser.parse_args()

    combine_and_recalculate(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()
