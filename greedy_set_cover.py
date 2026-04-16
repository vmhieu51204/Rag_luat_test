#!/usr/bin/env python3
"""Greedy set cover for law-clause matching.

For each test case:
- Universe U: clause signatures from that test case's Cac_Dieu_Quyet_Dinh
- Candidate subsets: clause signatures from every train case
- Selection: fixed top-k greedy picks maximizing newly covered clauses
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ID_FIELD = "Ma_Ban_An"
ARTICLES_FIELD = "PHAN_QUYET_CUA_TOA_SO_THAM"

DEFAULT_TRAIN_DIR = "chunk/train"
DEFAULT_TEST_DIR = "chunk/test"
DEFAULT_OUTPUT = "/home/hieujayce/Downloads/complete_repo/output/greedy_set_cover_results.json"


def precision_recall_f1(predicted: set[str], ground_truth: set[str]) -> tuple[float, float, float]:
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def extract_clause_signatures(articles: Any) -> set[str]:
    """Extract unique clause signatures from PHAN_QUYET_CUA_TOA_SO_THAM.

    Expected structure per case (new train/test data):
      PHAN_QUYET_CUA_TOA_SO_THAM: [
        {
          "Can_Cu_Dieu_Luat": [
            {"Dieu": "174", "Khoan": "4", "Diem": "a", ...},
            ...
          ],
          ...
        },
        ...
      ]

    Signature format: "Dieu-Khoan-Diem" with 0 fallback for missing khoan/diem.
    """
    signatures: set[str] = set()

    def _normalize_part(value: Any, default: str = "0") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    def _add_from_rule(rule: dict[str, Any]) -> None:
        dieu = _normalize_part(rule.get("Dieu"), default="")
        if not dieu:
            # Ignore references without "Dieu" (e.g., Nghị quyết lines).
            return
        khoan = _normalize_part(rule.get("Khoan"), default="0")
        diem = _normalize_part(rule.get("Diem"), default="0").lower()
        signatures.add(f"{dieu}-{khoan}-{diem}")

    def _add_item(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            # New shape: verdict item with Can_Cu_Dieu_Luat list
            can_cu = item.get("Can_Cu_Dieu_Luat")
            if isinstance(can_cu, list):
                for rule in can_cu:
                    if isinstance(rule, dict):
                        _add_from_rule(rule)
                return

            # Direct single-rule dict fallback
            if "Dieu" in item:
                _add_from_rule(item)
                return

            # Last-resort fallback for unexpected dict shape
            signatures.add(str(item))
            return

        if isinstance(item, list):
            signatures.add("-".join(str(part) for part in item))
        else:
            signatures.add(str(item))

    if isinstance(articles, dict):
        for value in articles.values():
            if isinstance(value, list):
                for item in value:
                    _add_item(item)
            else:
                _add_item(value)
    elif isinstance(articles, list):
        for item in articles:
            _add_item(item)
    elif articles:
        _add_item(articles)

    return signatures


def load_case_clause_sets(folder: Path) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Load case_id -> clause-signature set from all JSON files in folder."""
    case_to_clauses: dict[str, set[str]] = {}
    stats = {
        "files_seen": 0,
        "loaded_cases": 0,
        "skipped_missing_articles": 0,
        "skipped_empty_clause_set": 0,
        "json_errors": 0,
        "duplicate_case_ids": 0,
    }

    for json_file in sorted(folder.glob("*.json")):
        stats["files_seen"] += 1

        try:
            with open(json_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            stats["json_errors"] += 1
            print(f"  [WARN] Could not parse JSON: {json_file}")
            continue

        case_id = data.get(ID_FIELD, json_file.stem)
        articles = data.get(ARTICLES_FIELD)

        if not articles:
            stats["skipped_missing_articles"] += 1
            continue

        clauses = extract_clause_signatures(articles)
        if not clauses:
            stats["skipped_empty_clause_set"] += 1
            continue

        if case_id in case_to_clauses:
            stats["duplicate_case_ids"] += 1
            case_to_clauses[case_id] |= clauses
        else:
            case_to_clauses[case_id] = clauses

    stats["loaded_cases"] = len(case_to_clauses)
    return case_to_clauses, stats


def greedy_select_top_k(
    query_clauses: set[str],
    train_case_clauses: dict[str, set[str]],
    top_k: int,
) -> dict[str, Any]:
    """Select top-k train cases greedily by maximum newly covered query clauses."""
    uncovered = set(query_clauses)
    covered: set[str] = set()
    remaining_case_ids = set(train_case_clauses.keys())

    selected_case_ids: list[str] = []
    iterations: list[dict[str, Any]] = []
    zero_gain_started_at: int | None = None

    if not query_clauses:
        return {
            "selected_train_case_ids": selected_case_ids,
            "iterations": iterations,
            "final_covered_clauses": [],
            "final_uncovered_clauses": [],
            "final_coverage": 1.0,
            "zero_gain_started_at_iteration": zero_gain_started_at,
        }

    for iteration in range(1, top_k + 1):
        if not remaining_case_ids:
            break

        best_case_id: str | None = None
        best_gain_set: set[str] = set()
        best_gain = -1
        best_subset_size = 0

        for case_id in remaining_case_ids:
            subset = train_case_clauses[case_id]
            gain_set = uncovered & subset
            gain = len(gain_set)
            subset_size = len(subset)

            if best_case_id is None:
                best_case_id = case_id
                best_gain_set = gain_set
                best_gain = gain
                best_subset_size = subset_size
                continue

            if gain > best_gain:
                best_case_id = case_id
                best_gain_set = gain_set
                best_gain = gain
                best_subset_size = subset_size
                continue

            if gain == best_gain and subset_size < best_subset_size:
                best_case_id = case_id
                best_gain_set = gain_set
                best_gain = gain
                best_subset_size = subset_size
                continue

            if (
                gain == best_gain
                and subset_size == best_subset_size
                and case_id < best_case_id
            ):
                best_case_id = case_id
                best_gain_set = gain_set
                best_gain = gain
                best_subset_size = subset_size

        assert best_case_id is not None

        remaining_case_ids.remove(best_case_id)
        selected_case_ids.append(best_case_id)

        covered |= best_gain_set
        uncovered -= best_gain_set

        if best_gain == 0 and zero_gain_started_at is None:
            zero_gain_started_at = iteration

        coverage = len(covered) / len(query_clauses)
        iterations.append(
            {
                "iteration": iteration,
                "train_case_id": best_case_id,
                "subset_size": len(train_case_clauses[best_case_id]),
                "newly_covered_clauses": sorted(best_gain_set),
                "gain": best_gain,
                "cumulative_covered_count": len(covered),
                "cumulative_coverage": round(coverage, 6),
                "remaining_uncovered_count": len(uncovered),
            }
        )

    final_coverage = len(covered) / len(query_clauses)
    return {
        "selected_train_case_ids": selected_case_ids,
        "iterations": iterations,
        "final_covered_clauses": sorted(covered),
        "final_uncovered_clauses": sorted(uncovered),
        "final_coverage": round(final_coverage, 6),
        "zero_gain_started_at_iteration": zero_gain_started_at,
    }


def run_greedy_set_cover(
    train_dir: Path,
    test_dir: Path,
    top_k: int,
    output_path: Path,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    print("Loading train cases...")
    train_case_clauses, train_stats = load_case_clause_sets(train_dir)
    print(f"  Loaded {len(train_case_clauses)} train case(s)")

    print("Loading test cases...")
    test_case_clauses, test_stats = load_case_clause_sets(test_dir)
    print(f"  Loaded {len(test_case_clauses)} test case(s)")

    if not train_case_clauses:
        raise RuntimeError("No train cases with valid clause sets were loaded")
    if not test_case_clauses:
        raise RuntimeError("No test cases with valid clause sets were loaded")

    per_test: list[dict[str, Any]] = []
    total_query_clauses = 0
    total_covered_clauses = 0

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    print(f"Running greedy set cover for {len(test_case_clauses)} test case(s)...")
    for idx, test_case_id in enumerate(sorted(test_case_clauses.keys()), start=1):
        query_clauses = test_case_clauses[test_case_id]
        selection = greedy_select_top_k(query_clauses, train_case_clauses, top_k=top_k)

        covered_count = len(selection["final_covered_clauses"])
        total_query_clauses += len(query_clauses)
        total_covered_clauses += covered_count

        predicted_clauses: set[str] = set()
        for selected_case_id in selection["selected_train_case_ids"]:
            predicted_clauses |= train_case_clauses[selected_case_id]

        p, r, f1 = precision_recall_f1(predicted_clauses, query_clauses)
        sum_precision += p
        sum_recall += r
        sum_f1 += f1

        per_test.append(
            {
                "test_case_id": test_case_id,
                "query_clause_count": len(query_clauses),
                "query_clauses": sorted(query_clauses),
                "selected_train_case_ids": selection["selected_train_case_ids"],
                "iterations": selection["iterations"],
                "selected_count": len(selection["selected_train_case_ids"]),
                "predicted_clause_count": len(predicted_clauses),
                "predicted_clauses": sorted(predicted_clauses),
                "final_covered_count": covered_count,
                "final_uncovered_count": len(selection["final_uncovered_clauses"]),
                "final_covered_clauses": selection["final_covered_clauses"],
                "final_uncovered_clauses": selection["final_uncovered_clauses"],
                "final_coverage": selection["final_coverage"],
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "zero_gain_started_at_iteration": selection[
                    "zero_gain_started_at_iteration"
                ],
            }
        )

        if idx % 10 == 0 or idx == len(test_case_clauses):
            print(f"  Processed {idx}/{len(test_case_clauses)} test case(s)")

    macro_coverage = sum(item["final_coverage"] for item in per_test) / len(per_test)
    macro_precision = sum_precision / len(per_test)
    macro_recall = sum_recall / len(per_test)
    macro_f1 = sum_f1 / len(per_test)
    micro_coverage = (
        total_covered_clauses / total_query_clauses if total_query_clauses else 1.0
    )

    output = {
        "config": {
            "algorithm": "greedy_set_cover",
            "clause_granularity": "full_signature",
            "selection_rule": "fixed_top_k",
            "top_k": top_k,
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
        },
        "train_stats": train_stats,
        "test_stats": test_stats,
        "summary": {
            "n_train_cases": len(train_case_clauses),
            "n_test_cases": len(test_case_clauses),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_final_coverage": round(macro_coverage, 6),
            "micro_final_coverage": round(micro_coverage, 6),
            "total_query_clauses": total_query_clauses,
            "total_covered_clauses": total_covered_clauses,
            "full_coverage_test_cases": sum(
                1 for item in per_test if item["final_coverage"] >= 1.0
            ),
        },
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_test": per_test,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Results written to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run greedy set cover where each test case defines a universe of law "
            "clause signatures and each train case defines a subset."
        )
    )
    parser.add_argument(
        "--train_dir",
        default=DEFAULT_TRAIN_DIR,
        help="Directory containing train case JSON files",
    )
    parser.add_argument(
        "--test_dir",
        default=DEFAULT_TEST_DIR,
        help="Directory containing test case JSON files",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Fixed number of train cases to select per test case",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON file path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_greedy_set_cover(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        top_k=args.top_k,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
