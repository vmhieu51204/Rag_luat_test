"""Case-only embedding strategy optimizer for dieu-level macro recall.

This module runs a reproducible experiment loop over local embedding models and
retrieval settings, using only past cases as the retrieval database. It stops
early once the target macro_recall_dieu is reached.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MODEL_NAME,
    QUERY_CONTENT_FIELDS,
    TEST_EMBED_CONTENT_FIELDS,
    TRAIN_EMBED_CONTENT_FIELDS,
)
from rag.evaluation.law_embedding_model_eval import evaluate_single_configuration


DEFAULT_MODEL_CANDIDATES = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-small",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]


def _slugify(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return slug or "run"


def _parse_csv(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    values = []
    for token in _parse_csv(raw):
        values.append(int(token))
    return values


def _ensure_fields(raw: str, fallback: list[str]) -> list[str]:
    fields = _parse_csv(raw)
    return fields if fields else list(fallback)


def _run_trial(
    *,
    model_name: str,
    top_k_case: int,
    train_fields: list[str],
    test_fields: list[str],
    query_fields: list[str],
    args: argparse.Namespace,
    run_root: Path,
    case_db_dir: str,
    test_db_dir: str,
) -> dict[str, Any]:
    model_slug = _slugify(model_name)
    train_slug = _slugify("-".join(train_fields))
    query_slug = _slugify("-".join(query_fields))
    trial_slug = f"{model_slug}__k{top_k_case}__train_{train_slug}__query_{query_slug}"
    trial_root = run_root / "trials" / trial_slug

    law_db_dir = str(trial_root / "db" / "law_unused")

    result = evaluate_single_configuration(
        model_name=model_name,
        device=args.device,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        case_db_dir=case_db_dir,
        law_db_dir=law_db_dir,
        test_db_dir=test_db_dir,
        top_k_case=top_k_case,
        top_k_law=0,
        max_chunk_chars=args.max_chunk_chars,
        batch_size=args.batch_size,
        collection_name=args.collection_name,
        train_embedding_fields=train_fields,
        test_embedding_fields=test_fields,
        query_content_fields=query_fields,
        embed_law=False,
        law_json=args.law_json,
        law_id=args.law_id,
    )

    recall = float(result.get("macro_recall_dieu_case_only", 0.0))
    f1 = float(result.get("macro_f1_dieu_case_only", 0.0))
    precision = float(result.get("macro_precision_dieu_case_only", 0.0))

    trial_output = {
        "trial_id": trial_slug,
        "model_name": model_name,
        "top_k_case": top_k_case,
        "train_embedding_fields": train_fields,
        "test_embedding_fields": test_fields,
        "query_content_fields": query_fields,
        "target_macro_recall_dieu": args.target_macro_recall_dieu,
        "achieved_macro_recall_dieu": recall,
        "achieved_macro_precision_dieu": precision,
        "achieved_macro_f1_dieu": f1,
        "n_docs": result.get("n_docs", 0),
        "n_train_skipped": result.get("n_train_skipped", 0),
        "n_test_skipped": result.get("n_test_skipped", 0),
        "metrics": {
            "macro_precision_case_only": result.get("macro_precision_case_only", 0.0),
            "macro_recall_case_only": result.get("macro_recall_case_only", 0.0),
            "macro_f1_case_only": result.get("macro_f1_case_only", 0.0),
            "macro_precision_dieu_case_only": result.get("macro_precision_dieu_case_only", 0.0),
            "macro_recall_dieu_case_only": recall,
            "macro_f1_dieu_case_only": f1,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    trial_root.mkdir(parents=True, exist_ok=True)
    with open(trial_root / "result.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": trial_output, "full_result": result}, fh, ensure_ascii=False, indent=2)

    return trial_output


def _sorted_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        trials,
        key=lambda t: (
            float(t.get("achieved_macro_recall_dieu", 0.0)),
            float(t.get("achieved_macro_f1_dieu", 0.0)),
            -int(t.get("top_k_case", 0)),
        ),
        reverse=True,
    )


def _write_scoreboard(*, run_root: Path, payload: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "scoreboard.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize case-only retrieval over past-case database by sweeping "
            "embedding models and retrieval settings until macro_recall_dieu target is met."
        )
    )
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--work_dir", default="output/case_only_optimizer")
    parser.add_argument("--run_name", default=None, help="Optional fixed run folder name")

    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_CANDIDATES),
        help="Comma-separated local embedding model names",
    )
    parser.add_argument("--top_k_values", default="3,5,8,10,15")

    parser.add_argument(
        "--train_field_sets",
        default="|".join([
            ",".join(TRAIN_EMBED_CONTENT_FIELDS),
            "Summary",
        ]),
        help=(
            "Pipe-separated train field sets; each set is comma-separated. "
            "Example: 'Summary,Tang_nang,Giam_nhe|Summary'"
        ),
    )
    parser.add_argument(
        "--query_field_sets",
        default="|".join([
            ",".join(QUERY_CONTENT_FIELDS),
            "Synthetic_summary,Summary",
        ]),
        help=(
            "Pipe-separated query field sets; each set is comma-separated. "
            "Example: 'Synthetic_summary|Synthetic_summary,Summary'"
        ),
    )
    parser.add_argument(
        "--test_embedding_fields",
        default=",".join(TEST_EMBED_CONTENT_FIELDS),
        help="Comma-separated fields used for test embedding index generation",
    )

    parser.add_argument("--target_macro_recall_dieu", type=float, default=0.9)
    parser.add_argument("--max_trials", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--strict_until_target",
        action="store_true",
        help="Exit with non-zero code if target is not reached",
    )

    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_chunk_chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS)
    parser.add_argument("--collection_name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--law_json", default="raw_law.json")
    parser.add_argument("--law_id", default="blhs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_names = _parse_csv(args.models) or [DEFAULT_MODEL_NAME]
    top_k_values = _parse_int_csv(args.top_k_values)
    if not top_k_values:
        raise ValueError("--top_k_values must contain at least one integer")

    test_fields = _ensure_fields(args.test_embedding_fields, TEST_EMBED_CONTENT_FIELDS)

    raw_train_sets = [s.strip() for s in str(args.train_field_sets).split("|") if s.strip()]
    train_field_sets = [_ensure_fields(s, TRAIN_EMBED_CONTENT_FIELDS) for s in raw_train_sets]
    if not train_field_sets:
        train_field_sets = [list(TRAIN_EMBED_CONTENT_FIELDS)]

    raw_query_sets = [s.strip() for s in str(args.query_field_sets).split("|") if s.strip()]
    query_field_sets = [_ensure_fields(s, QUERY_CONTENT_FIELDS) for s in raw_query_sets]
    if not query_field_sets:
        query_field_sets = [list(QUERY_CONTENT_FIELDS)]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"case_only_{stamp}"
    run_root = Path(args.work_dir) / run_name

    summary: dict[str, Any] = {
        "run_name": run_name,
        "work_dir": str(Path(args.work_dir)),
        "run_root": str(run_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_macro_recall_dieu": args.target_macro_recall_dieu,
        "search_space": {
            "models": model_names,
            "top_k_values": top_k_values,
            "train_field_sets": train_field_sets,
            "query_field_sets": query_field_sets,
            "test_embedding_fields": test_fields,
        },
        "trials": [],
        "best": None,
        "target_reached": False,
    }

    max_trials = int(args.max_trials)
    trial_count = 0

    print("== Case-only optimizer started ==")
    print(f"Target macro_recall_dieu >= {args.target_macro_recall_dieu:.4f}")
    print(f"Run root: {run_root}")

    stop = False
    for model_name in model_names:
        if stop:
            break
        for train_fields in train_field_sets:
            if stop:
                break
            model_slug = _slugify(model_name)
            train_slug = _slugify("-".join(train_fields))
            test_slug = _slugify("-".join(test_fields))
            shared_case_db_dir = str(run_root / "shared" / model_slug / f"train_{train_slug}" / "case")
            shared_test_db_dir = str(run_root / "shared" / model_slug / f"test_{test_slug}" / "test")
            for query_fields in query_field_sets:
                if stop:
                    break
                for top_k_case in top_k_values:
                    if max_trials > 0 and trial_count >= max_trials:
                        stop = True
                        break

                    trial_count += 1
                    print(
                        f"\n[Trial {trial_count}] model={model_name} | k={top_k_case} "
                        f"| train={train_fields} | query={query_fields}"
                    )
                    trial = _run_trial(
                        model_name=model_name,
                        top_k_case=top_k_case,
                        train_fields=train_fields,
                        test_fields=test_fields,
                        query_fields=query_fields,
                        args=args,
                        run_root=run_root,
                        case_db_dir=shared_case_db_dir,
                        test_db_dir=shared_test_db_dir,
                    )
                    summary["trials"].append(trial)

                    ranked = _sorted_trials(summary["trials"])
                    summary["best"] = ranked[0] if ranked else None
                    best_recall = float(summary["best"].get("achieved_macro_recall_dieu", 0.0)) if summary["best"] else 0.0
                    summary["target_reached"] = best_recall >= float(args.target_macro_recall_dieu)
                    summary["updated_at"] = datetime.now().isoformat(timespec="seconds")

                    print(
                        "  -> result: "
                        f"R_dieu={trial['achieved_macro_recall_dieu']:.4f}, "
                        f"P_dieu={trial['achieved_macro_precision_dieu']:.4f}, "
                        f"F1_dieu={trial['achieved_macro_f1_dieu']:.4f}"
                    )
                    print(
                        "  -> best so far: "
                        f"{summary['best']['model_name']} k={summary['best']['top_k_case']} "
                        f"R_dieu={best_recall:.4f}"
                    )

                    _write_scoreboard(run_root=run_root, payload=summary)

                    if summary["target_reached"]:
                        print("\nTarget reached. Stopping early.")
                        stop = True
                        break

    ranked = _sorted_trials(summary["trials"])
    summary["ranking"] = ranked
    summary["best"] = ranked[0] if ranked else None
    summary["target_reached"] = (
        bool(summary["best"])
        and float(summary["best"].get("achieved_macro_recall_dieu", 0.0)) >= float(args.target_macro_recall_dieu)
    )
    summary["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _write_scoreboard(run_root=run_root, payload=summary)

    if summary["best"]:
        best = summary["best"]
        print("\n== Final best ==")
        print(
            f"model={best['model_name']} | k={best['top_k_case']} | "
            f"R_dieu={best['achieved_macro_recall_dieu']:.4f} | "
            f"F1_dieu={best['achieved_macro_f1_dieu']:.4f}"
        )
    else:
        print("\nNo successful trials were produced.")

    print(f"Scoreboard saved: {run_root / 'scoreboard.json'}")

    if args.strict_until_target and not summary["target_reached"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
