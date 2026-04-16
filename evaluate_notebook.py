"""
evaluate_notebook.py
====================
Interactive evaluation notebook (notebook-style .py with cell markers).
Runs the same retrieval evaluation as evaluate.py but produces richer
analysis: per-clause statistics, error patterns, score distributions, etc.

Convert to Jupyter:
    pip install jupytext
    jupytext --to notebook evaluate_notebook.py
"""

# %% [markdown]
# # Legal Retrieval Evaluation — Detailed Analysis
#
# This notebook embeds train/test JSON folders into ChromaDB, runs retrieval
# queries, and produces a comprehensive evaluation report with:
#
# - **Macro & micro metrics** (Precision / Recall / F1)
# - **Score distributions** (histograms, box plots)
# - **Top-N most missed clauses** (frequently in ground truth but not retrieved)
# - **Top-N most redundant clauses** (frequently retrieved but not in ground truth)
# - **Per-clause retrieval rate** (how reliably each clause is found)
# - **Worst & best performing documents**
# - **Retrieval overlap heatmap** (ground truth vs predicted)
# - **Error-pattern co-occurrence** (which missed clauses tend to appear together)

# %% [markdown]
# ## 0. Configuration
# Edit the variables below to match your setup.

# %%
# ── User configuration ─────────────────────────────────────────────────────
TRAIN_DIR       = "./chunk/train"
TEST_DIR        = "./chunk/test"
TRAIN_DB_DIR    = "./output/chroma_db_train"
TEST_DB_DIR     = "./output/chroma_db_test"

TOP_K           = 5
RESULTS_OUT     = "./output/eval_results_notebook.json"

# Set to True to use existing ChromaDB (skip embedding step)
SKIP_EMBEDDING  = True

# Embedding / chunking parameters
TRAIN_EMBED_CONTENT_FIELDS = ["Summary"]
TEST_EMBED_CONTENT_FIELDS = ["Summary"]
QUERY_CONTENT_FIELDS = ["Summary"]
MODEL_NAME           = "BAAI/bge-m3"
DEVICE               = "cuda"
MAX_CHUNK_CHARS      = 1500
BATCH_SIZE           = 32
COLLECTION_NAME      = "legal_chunks_vn"

# Analysis parameters
TOP_N = 15  # number of items to show in "top-N" tables

# %% [markdown]
# ## 1. Imports & helpers

# %%
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Local modules
from embedding_store import (
    run_pipeline,
    load_model,
    load_chroma,
)

# Constants from evaluate.py (duplicated here so the notebook is self-contained)
ID_FIELD = "Ma_Ban_An"
VERDICT_FIELD = "PHAN_QUYET_CUA_TOA_SO_THAM"
LEGAL_BASIS_FIELD = "Can_Cu_Dieu_Luat"


def _normalize_token(token: str, *, lowercase: bool = True) -> str:
    token = token.strip()
    token = re.sub(r"^(điều|dieu|khoản|khoan|điểm|diem)\s+", "", token, flags=re.IGNORECASE)
    token = token.strip(" .")
    token = re.sub(r"\s+", "", token)
    return token.lower() if lowercase else token


def _split_multi_value(raw_value, *, lowercase: bool = True) -> list[str]:
    if raw_value is None:
        return []
    raw = str(raw_value).strip()
    if not raw:
        return []
    parts = re.split(r",|;|/|\bvà\b|\bva\b|\band\b", raw, flags=re.IGNORECASE)
    tokens = [_normalize_token(p, lowercase=lowercase) for p in parts if p.strip()]
    return [t for t in tokens if t]


def extract_label_sets_from_verdict(data: dict) -> tuple[dict[str, set[str]], dict[str, int], list[str]]:
    errors: list[str] = []
    stats = {"n_verdict_items": 0, "n_cancu_items": 0}
    label_sets = {"dieu_only": set(), "full_signature": set()}

    verdict_items = data.get(VERDICT_FIELD)
    if not isinstance(verdict_items, list):
        return label_sets, stats, [f"{VERDICT_FIELD}_invalid_type"]
    if not verdict_items:
        return label_sets, stats, [f"{VERDICT_FIELD}_empty"]

    stats["n_verdict_items"] = len(verdict_items)
    for verdict in verdict_items:
        if not isinstance(verdict, dict):
            errors.append("verdict_item_not_object")
            continue

        legal_basis = verdict.get(LEGAL_BASIS_FIELD)
        if not isinstance(legal_basis, list):
            errors.append(f"{LEGAL_BASIS_FIELD}_invalid_type")
            continue
        if not legal_basis:
            errors.append(f"{LEGAL_BASIS_FIELD}_empty")
            continue

        stats["n_cancu_items"] += len(legal_basis)
        for basis_item in legal_basis:
            if not isinstance(basis_item, dict):
                errors.append("basis_item_not_object")
                continue

            dieu_tokens = _split_multi_value(basis_item.get("Dieu"), lowercase=False)
            khoan_tokens = _split_multi_value(basis_item.get("Khoan"), lowercase=False)
            diem_tokens = _split_multi_value(basis_item.get("Diem"), lowercase=True)

            if not dieu_tokens:
                errors.append("missing_dieu")
                continue

            for dieu in dieu_tokens:
                label_sets["dieu_only"].add(dieu)
                if khoan_tokens and diem_tokens:
                    for khoan in khoan_tokens:
                        for diem in diem_tokens:
                            label_sets["full_signature"].add(f"{dieu}-{khoan}-{diem}")
                elif khoan_tokens:
                    for khoan in khoan_tokens:
                        label_sets["full_signature"].add(f"{dieu}-{khoan}")
                elif diem_tokens:
                    for diem in diem_tokens:
                        label_sets["full_signature"].add(f"{dieu}-{diem}")
                else:
                    label_sets["full_signature"].add(dieu)

    if not label_sets["dieu_only"]:
        errors.append("no_labels_extracted")
    return label_sets, stats, errors


def load_articles_index(raw_dir: Path) -> tuple[dict[str, dict[str, set[str]]], list[dict]]:
    """doc_id -> {'dieu_only': set[str], 'full_signature': set[str]}"""
    index: dict[str, dict[str, set[str]]] = {}
    skipped: list[dict] = []
    for f in sorted(raw_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        doc_id = data.get(ID_FIELD, f.stem)
        labels, stats, errors = extract_label_sets_from_verdict(data)
        if errors:
            skipped.append({
                "doc_id": doc_id,
                "file": f.name,
                "stage": "train_index",
                "reasons": sorted(set(errors)),
                "stats": stats,
            })
            continue
        index[doc_id] = labels
    return index, skipped


def load_test_docs(test_dir: Path, query_fields: list[str]) -> tuple[list[dict], list[dict]]:
    """Load test docs with strict verdict parsing and strict query-field checks."""
    test_docs: list[dict] = []
    skipped: list[dict] = []
    for f in sorted(test_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        doc_id = data.get(ID_FIELD, f.stem)
        labels, stats, errors = extract_label_sets_from_verdict(data)
        if errors:
            skipped.append({
                "doc_id": doc_id,
                "file": f.name,
                "stage": "test_ground_truth",
                "reasons": sorted(set(errors)),
                "stats": stats,
            })
            continue

        parts = []
        empty_fields = []
        for field in query_fields:
            val = (data.get(field) or "").strip()
            if val:
                parts.append(val)
            else:
                empty_fields.append(field)

        query_text = "\n\n".join(parts).strip()
        if not query_text:
            skipped.append({
                "doc_id": doc_id,
                "file": f.name,
                "stage": "test_query",
                "reasons": ["empty_query_text"],
                "query_fields": query_fields,
                "empty_fields": empty_fields,
            })
            continue

        test_docs.append({
            "doc_id": doc_id,
            "query_text": query_text,
            "ground_truth": labels["dieu_only"],
            "ground_truth_full": labels["full_signature"],
            "gt_stats": stats,
        })

    return test_docs, skipped


def print_skip_report(skipped: list[dict], title: str) -> None:
    if not skipped:
        return
    print(f"  [INFO] {title}: skipped {len(skipped)} file(s)")
    for item in skipped[:20]:
        reasons = ",".join(item.get("reasons", []))
        print(f"    - {item.get('doc_id')} ({item.get('stage')}): {reasons}")
    if len(skipped) > 20:
        print(f"    ... and {len(skipped) - 20} more")


def precision_recall_f1(predicted: set[str], ground_truth: set[str]):
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    p = tp / len(predicted)
    r = tp / len(ground_truth)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

# %% [markdown]
# ## 2. Embedding (skip if DBs already exist)

# %%
if not SKIP_EMBEDDING:
    print("── Embedding train folder ──────────────────────────────────")
    run_pipeline(
        TRAIN_DIR, TRAIN_DB_DIR,
        content_fields=TRAIN_EMBED_CONTENT_FIELDS,
        model_name=MODEL_NAME, device=DEVICE,
        max_chunk_chars=MAX_CHUNK_CHARS, batch_size=BATCH_SIZE,
        collection_name=COLLECTION_NAME,
    )
    print("\n── Embedding test folder ───────────────────────────────────")
    run_pipeline(
        TEST_DIR, TEST_DB_DIR,
        content_fields=TEST_EMBED_CONTENT_FIELDS,
        model_name=MODEL_NAME, device=DEVICE,
        max_chunk_chars=MAX_CHUNK_CHARS, batch_size=BATCH_SIZE,
        collection_name=COLLECTION_NAME,
    )
else:
    print("⏭️  Skipping embedding (SKIP_EMBEDDING=True)")

# %% [markdown]
# ## 3. Load model & data

# %%
print("Loading model ...")
model = load_model(model_name=MODEL_NAME, device=DEVICE)

print("Loading train ChromaDB ...")
collection = load_chroma(TRAIN_DB_DIR, collection_name=COLLECTION_NAME, create=False)
print(f"  Train collection has {collection.count()} documents")

train_path = Path(TRAIN_DIR)
test_path  = Path(TEST_DIR)

train_articles_index, train_skipped = load_articles_index(train_path)
test_docs, test_skipped = load_test_docs(test_path, QUERY_CONTENT_FIELDS)
print_skip_report(train_skipped, "Train label index")
print_skip_report(test_skipped, "Test loading")
print(f"  {len(test_docs)} evaluable test documents")

if not test_docs:
    raise RuntimeError("No evaluable test documents found after strict filtering")

# %% [markdown]
# ## 4. Run retrieval & collect per-document results

# %%
per_doc_results = []

for doc in test_docs:
    doc_id       = doc["doc_id"]
    query_text   = doc["query_text"]
    ground_truth = doc["ground_truth"]
    ground_truth_full = doc["ground_truth_full"]

    vec = model.encode([query_text], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=vec,
        n_results=TOP_K,
        include=["metadatas", "distances"],
        where={"doc_id": {"$ne": doc_id}},
    )

    # Unique retrieved doc_ids in rank order
    retrieved_doc_ids, seen = [], set()
    distances_per_doc = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        rid = meta["doc_id"]
        if rid not in seen:
            seen.add(rid)
            retrieved_doc_ids.append(rid)
            distances_per_doc[rid] = dist  # keep first (best) distance

    # Union of articles from retrieved docs
    predicted_articles: set[str] = set()
    predicted_articles_full: set[str] = set()
    for rid in retrieved_doc_ids:
        label_sets = train_articles_index.get(rid)
        if not label_sets:
            continue
        predicted_articles |= label_sets["dieu_only"]
        predicted_articles_full |= label_sets["full_signature"]

    p, r, f1 = precision_recall_f1(predicted_articles, ground_truth)
    p_full, r_full, f1_full = precision_recall_f1(predicted_articles_full, ground_truth_full)

    per_doc_results.append({
        "doc_id":             doc_id,
        "ground_truth":       sorted(ground_truth),
        "ground_truth_full":  sorted(ground_truth_full),
        "retrieved_doc_ids":  retrieved_doc_ids,
        "distances":          distances_per_doc,
        "predicted_articles": sorted(predicted_articles),
        "predicted_articles_full": sorted(predicted_articles_full),
        "matched_articles":   sorted(predicted_articles & ground_truth),
        "missed_articles":    sorted(ground_truth - predicted_articles),
        "extra_articles":     sorted(predicted_articles - ground_truth),
        "matched_articles_full": sorted(predicted_articles_full & ground_truth_full),
        "missed_articles_full": sorted(ground_truth_full - predicted_articles_full),
        "extra_articles_full": sorted(predicted_articles_full - ground_truth_full),
        "precision":          round(p, 4),
        "recall":             round(r, 4),
        "f1":                 round(f1, 4),
        "precision_full":     round(p_full, 4),
        "recall_full":        round(r_full, 4),
        "f1_full":            round(f1_full, 4),
        "n_ground_truth":     len(ground_truth),
        "n_ground_truth_full": len(ground_truth_full),
        "n_predicted":        len(predicted_articles),
        "n_predicted_full":   len(predicted_articles_full),
        "n_matched":          len(predicted_articles & ground_truth),
        "n_matched_full":     len(predicted_articles_full & ground_truth_full),
        "n_missed":           len(ground_truth - predicted_articles),
        "n_missed_full":      len(ground_truth_full - predicted_articles_full),
        "n_extra":            len(predicted_articles - ground_truth),
        "n_extra_full":       len(predicted_articles_full - ground_truth_full),
        "gt_stats":           doc["gt_stats"],
    })

print(f"✅ Evaluated {len(per_doc_results)} documents")

# %% [markdown]
# ## 5. Macro & Micro Metrics

# %%
n = len(per_doc_results)
macro_p  = sum(r["precision"] for r in per_doc_results) / n
macro_r  = sum(r["recall"]    for r in per_doc_results) / n
macro_f1 = sum(r["f1"]        for r in per_doc_results) / n
macro_p_full = sum(r["precision_full"] for r in per_doc_results) / n
macro_r_full = sum(r["recall_full"] for r in per_doc_results) / n
macro_f1_full = sum(r["f1_full"] for r in per_doc_results) / n

# Micro: pool all TP/FP/FN across documents
total_matched = sum(r["n_matched"] for r in per_doc_results)
total_pred    = sum(r["n_predicted"] for r in per_doc_results)
total_gt      = sum(r["n_ground_truth"] for r in per_doc_results)
total_missed  = sum(r["n_missed"] for r in per_doc_results)
total_extra   = sum(r["n_extra"] for r in per_doc_results)
total_matched_full = sum(r["n_matched_full"] for r in per_doc_results)
total_pred_full = sum(r["n_predicted_full"] for r in per_doc_results)
total_gt_full = sum(r["n_ground_truth_full"] for r in per_doc_results)
total_missed_full = sum(r["n_missed_full"] for r in per_doc_results)
total_extra_full = sum(r["n_extra_full"] for r in per_doc_results)

micro_p  = total_matched / total_pred if total_pred else 0
micro_r  = total_matched / total_gt   if total_gt   else 0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
micro_p_full = total_matched_full / total_pred_full if total_pred_full else 0
micro_r_full = total_matched_full / total_gt_full if total_gt_full else 0
micro_f1_full = (
    2 * micro_p_full * micro_r_full / (micro_p_full + micro_r_full)
    if (micro_p_full + micro_r_full) else 0
)

print("=" * 60)
print(f"  Documents evaluated : {n}")
print(f"  top_k               : {TOP_K}")
print(f"  Model               : {MODEL_NAME}")
print(f"  Query fields        : {QUERY_CONTENT_FIELDS}")
print("=" * 60)
print(f"  MACRO  Precision    : {macro_p:.4f}")
print(f"  MACRO  Recall       : {macro_r:.4f}")
print(f"  MACRO  F1           : {macro_f1:.4f}")
print(f"  MACRO  Precision(full): {macro_p_full:.4f}")
print(f"  MACRO  Recall(full)   : {macro_r_full:.4f}")
print(f"  MACRO  F1(full)       : {macro_f1_full:.4f}")
print("-" * 60)
print(f"  MICRO  Precision    : {micro_p:.4f}")
print(f"  MICRO  Recall       : {micro_r:.4f}")
print(f"  MICRO  F1           : {micro_f1:.4f}")
print(f"  MICRO  Precision(full): {micro_p_full:.4f}")
print(f"  MICRO  Recall(full)   : {micro_r_full:.4f}")
print(f"  MICRO  F1(full)       : {micro_f1_full:.4f}")
print("-" * 60)
print(f"  Total ground truth clauses  : {total_gt}")
print(f"  Total predicted clauses     : {total_pred}")
print(f"  Total matched (TP)          : {total_matched}")
print(f"  Total missed  (FN)          : {total_missed}")
print(f"  Total extra   (FP)          : {total_extra}")
print(f"  Total ground truth full     : {total_gt_full}")
print(f"  Total predicted full        : {total_pred_full}")
print(f"  Total matched full          : {total_matched_full}")
print(f"  Total missed full           : {total_missed_full}")
print(f"  Total extra full            : {total_extra_full}")
print("=" * 60)

# %% [markdown]
# ## 6. Score Distributions (Precision / Recall / F1)

# %%
precisions = [r["precision"] for r in per_doc_results]
recalls    = [r["recall"]    for r in per_doc_results]
f1s        = [r["f1"]        for r in per_doc_results]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, data, label, color in zip(
    axes,
    [precisions, recalls, f1s],
    ["Precision", "Recall", "F1"],
    ["#4C72B0", "#55A868", "#C44E52"],
):
    ax.hist(data, bins=15, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(data), color="black", linestyle="--", linewidth=1.2,
               label=f"mean={np.mean(data):.3f}")
    ax.axvline(np.median(data), color="gray", linestyle=":", linewidth=1.2,
               label=f"median={np.median(data):.3f}")
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("# Documents")
    ax.legend(fontsize=9)

fig.suptitle("Per-Document Score Distributions", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()

# %%
# Box plot comparison
fig, ax = plt.subplots(figsize=(8, 4.5))
bp = ax.boxplot(
    [precisions, recalls, f1s],
    labels=["Precision", "Recall", "F1"],
    patch_artist=True,
    boxprops=dict(facecolor="#e8e8e8"),
    medianprops=dict(color="#C44E52", linewidth=2),
)
colors = ["#4C72B0", "#55A868", "#C44E52"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)
ax.set_ylabel("Score")
ax.set_title("Score Distribution (Box Plot)", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Top-N Most Missed Clauses
# Clauses that appear in ground truth but are **not** retrieved.
# High frequency here means the retrieval system systematically fails to find these.

# %%
missed_counter = Counter()
missed_doc_map = {}  # clause -> list of doc_ids that missed it
for r in per_doc_results:
    for clause in r["missed_articles"]:
        missed_counter[clause] += 1
        missed_doc_map.setdefault(clause, []).append(r["doc_id"])

print(f"{'Rank':<5} {'Clause':<15} {'Times Missed':<15} {'% of Docs':<12} Example Doc IDs")
print("─" * 95)
for rank, (clause, count) in enumerate(missed_counter.most_common(TOP_N), 1):
    pct = count / n * 100
    examples = ", ".join(missed_doc_map[clause][:3])
    if len(missed_doc_map[clause]) > 3:
        examples += " ..."
    print(f"{rank:<5} {clause:<15} {count:<15} {pct:<12.1f} {examples}")

# %%
# Bar chart — top missed clauses
top_missed = missed_counter.most_common(TOP_N)
if top_missed:
    fig, ax = plt.subplots(figsize=(12, 5))
    clauses_m = [c for c, _ in top_missed]
    counts_m  = [cnt for _, cnt in top_missed]
    bars = ax.barh(clauses_m[::-1], counts_m[::-1], color="#C44E52", edgecolor="white")
    ax.set_xlabel("Times Missed (FN)")
    ax.set_title(f"Top {TOP_N} Most Missed Clauses", fontsize=13, fontweight="bold")
    for bar, cnt in zip(bars, counts_m[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(cnt), va="center", fontsize=9)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Top-N Most Redundant (Extra) Clauses
# Clauses that are retrieved but are **not** in the ground truth.
# High frequency here means these clauses "leak" into results from similar documents.

# %%
extra_counter = Counter()
extra_doc_map = {}  # clause -> list of doc_ids
for r in per_doc_results:
    for clause in r["extra_articles"]:
        extra_counter[clause] += 1
        extra_doc_map.setdefault(clause, []).append(r["doc_id"])

print(f"{'Rank':<5} {'Clause':<15} {'Times Extra':<15} {'% of Docs':<12} Example Doc IDs")
print("─" * 95)
for rank, (clause, count) in enumerate(extra_counter.most_common(TOP_N), 1):
    pct = count / n * 100
    examples = ", ".join(extra_doc_map[clause][:3])
    if len(extra_doc_map[clause]) > 3:
        examples += " ..."
    print(f"{rank:<5} {clause:<15} {count:<15} {pct:<12.1f} {examples}")

# %%
# Bar chart — top redundant clauses
top_extra = extra_counter.most_common(TOP_N)
if top_extra:
    fig, ax = plt.subplots(figsize=(12, 5))
    clauses_e = [c for c, _ in top_extra]
    counts_e  = [cnt for _, cnt in top_extra]
    bars = ax.barh(clauses_e[::-1], counts_e[::-1], color="#E5A84B", edgecolor="white")
    ax.set_xlabel("Times Redundant (FP)")
    ax.set_title(f"Top {TOP_N} Most Redundant Clauses", fontsize=13, fontweight="bold")
    for bar, cnt in zip(bars, counts_e[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(cnt), va="center", fontsize=9)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Per-Clause Retrieval Rate
# For each clause that appears in **any** ground truth, what fraction of the
# time is it correctly retrieved?

# %%
clause_gt_count   = Counter()  # how many docs have this clause in GT
clause_hit_count  = Counter()  # how many of those docs successfully retrieved it

for r in per_doc_results:
    gt_set  = set(r["ground_truth"])
    hit_set = set(r["matched_articles"])
    for clause in gt_set:
        clause_gt_count[clause] += 1
        if clause in hit_set:
            clause_hit_count[clause] += 1

clause_retrieval_rate = {}
for clause in clause_gt_count:
    total   = clause_gt_count[clause]
    hits    = clause_hit_count.get(clause, 0)
    misses  = total - hits
    rate    = hits / total if total else 0
    clause_retrieval_rate[clause] = {
        "total": total, "hits": hits, "misses": misses, "rate": rate
    }

# Sort by retrieval rate (ascending = hardest to retrieve first)
sorted_by_rate = sorted(clause_retrieval_rate.items(), key=lambda x: (x[1]["rate"], -x[1]["total"]))

print(f"{'Clause':<15} {'GT Count':<10} {'Hits':<8} {'Misses':<8} {'Retrieval Rate'}")
print("─" * 60)
for clause, info in sorted_by_rate[:TOP_N]:
    print(f"{clause:<15} {info['total']:<10} {info['hits']:<8} {info['misses']:<8} "
          f"{info['rate']:.1%}")

print(f"\n... showing {min(TOP_N, len(sorted_by_rate))} of {len(sorted_by_rate)} unique clauses "
      f"(sorted by retrieval rate, ascending)")

# %%
# Plot: retrieval rate per clause (all clauses, sorted)
if clause_retrieval_rate:
    sorted_all = sorted(clause_retrieval_rate.items(),
                        key=lambda x: x[1]["rate"], reverse=True)
    clause_labels = [c for c, _ in sorted_all]
    rates = [info["rate"] for _, info in sorted_all]
    sizes = [info["total"] for _, info in sorted_all]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors_rate = ["#55A868" if r >= 0.8 else "#E5A84B" if r >= 0.5 else "#C44E52"
                   for r in rates]
    ax.bar(range(len(rates)), rates, color=colors_rate, edgecolor="white", width=0.8)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(clause_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Retrieval Rate")
    ax.set_title("Per-Clause Retrieval Rate (green ≥80%, yellow ≥50%, red <50%)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.4)
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 10. Worst & Best Performing Documents

# %%
sorted_by_f1 = sorted(per_doc_results, key=lambda r: r["f1"])

print("=" * 90)
print("  WORST performing documents (lowest F1)")
print("=" * 90)
print(f"{'Rank':<5} {'Doc ID':<50} {'P':<8} {'R':<8} {'F1':<8} Missed")
print("─" * 90)
for rank, r in enumerate(sorted_by_f1[:TOP_N], 1):
    missed_str = ", ".join(r["missed_articles"][:5])
    if len(r["missed_articles"]) > 5:
        missed_str += f" ... (+{len(r['missed_articles'])-5})"
    print(f"{rank:<5} {r['doc_id']:<50} {r['precision']:<8.4f} "
          f"{r['recall']:<8.4f} {r['f1']:<8.4f} {missed_str}")

print()
print("=" * 90)
print("  BEST performing documents (highest F1)")
print("=" * 90)
print(f"{'Rank':<5} {'Doc ID':<50} {'P':<8} {'R':<8} {'F1':<8} #GT  #Pred")
print("─" * 90)
for rank, r in enumerate(sorted(per_doc_results, key=lambda r: -r["f1"])[:TOP_N], 1):
    print(f"{rank:<5} {r['doc_id']:<50} {r['precision']:<8.4f} "
          f"{r['recall']:<8.4f} {r['f1']:<8.4f} "
          f"{r['n_ground_truth']:<5}{r['n_predicted']}")

# %% [markdown]
# ## 11. Precision vs Recall Scatter Plot

# %%
fig, ax = plt.subplots(figsize=(8, 7))

f1_vals = np.array(f1s)
scatter = ax.scatter(
    recalls, precisions,
    c=f1_vals, cmap="RdYlGn", s=60, edgecolors="white", linewidth=0.5,
    vmin=0, vmax=1, alpha=0.85,
)
cbar = fig.colorbar(scatter, ax=ax, label="F1 Score")

# Draw iso-F1 curves
for f1_target in [0.3, 0.5, 0.7, 0.9]:
    r_range = np.linspace(0.01, 1.0, 200)
    p_iso = (f1_target * r_range) / (2 * r_range - f1_target)
    mask = (p_iso >= 0) & (p_iso <= 1)
    ax.plot(r_range[mask], p_iso[mask], "--", color="gray", alpha=0.3, linewidth=0.8)
    # label at right edge
    valid_r = r_range[mask]
    valid_p = p_iso[mask]
    if len(valid_r):
        ax.text(valid_r[-1] + 0.01, valid_p[-1], f"F1={f1_target}",
                fontsize=7, color="gray", alpha=0.6)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision vs Recall (colored by F1)", fontsize=14, fontweight="bold")
ax.set_xlim(-0.03, 1.08)
ax.set_ylim(-0.03, 1.08)
ax.grid(alpha=0.2)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Ground-Truth Size vs Recall
# Do documents with more clauses tend to have lower recall?

# %%
gt_sizes  = [r["n_ground_truth"] for r in per_doc_results]
doc_recalls = [r["recall"] for r in per_doc_results]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(gt_sizes, doc_recalls, alpha=0.6, s=40, color="#4C72B0", edgecolors="white")
# trend line
z = np.polyfit(gt_sizes, doc_recalls, 1)
p_line = np.poly1d(z)
xs = np.linspace(min(gt_sizes), max(gt_sizes), 100)
ax.plot(xs, p_line(xs), "--", color="#C44E52", linewidth=1.5,
        label=f"trend: y={z[0]:.4f}x + {z[1]:.4f}")
ax.set_xlabel("# Ground Truth Clauses", fontsize=12)
ax.set_ylabel("Recall", fontsize=12)
ax.set_title("Ground-Truth Size vs Recall", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Missed-Clause Co-occurrence
# Which clauses tend to be missed **together**? Highlights systematic gaps.

# %%
from itertools import combinations

co_occur = Counter()
for r in per_doc_results:
    missed = sorted(r["missed_articles"])
    if len(missed) >= 2:
        for pair in combinations(missed, 2):
            co_occur[pair] += 1

if co_occur:
    print(f"{'Rank':<5} {'Clause Pair':<30} {'Co-occurrences':<15}")
    print("─" * 55)
    for rank, (pair, count) in enumerate(co_occur.most_common(TOP_N), 1):
        print(f"{rank:<5} {str(pair):<30} {count}")
else:
    print("No missed-clause co-occurrences found (all documents have ≤1 miss).")

# %% [markdown]
# ## 14. Retrieval Distance Statistics
# How close are the retrieved documents in embedding space?

# %%
all_best_dists = []
all_worst_dists = []

for r in per_doc_results:
    dists = list(r["distances"].values())
    if dists:
        all_best_dists.append(min(dists))
        all_worst_dists.append(max(dists))

if all_best_dists:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].hist(all_best_dists, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(all_best_dists), color="black", linestyle="--",
                    label=f"mean={np.mean(all_best_dists):.4f}")
    axes[0].set_title("Best (Closest) Retrieval Distance", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Cosine Distance")
    axes[0].set_ylabel("# Documents")
    axes[0].legend()

    axes[1].hist(all_worst_dists, bins=20, color="#E5A84B", edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(all_worst_dists), color="black", linestyle="--",
                    label=f"mean={np.mean(all_worst_dists):.4f}")
    axes[1].set_title("Worst (Farthest) Retrieval Distance", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Cosine Distance")
    axes[1].set_ylabel("# Documents")
    axes[1].legend()

    fig.suptitle("Retrieval Distance Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()

    # Correlation: distance vs F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(all_best_dists, f1s, alpha=0.6, s=40, color="#55A868", edgecolors="white")
    z = np.polyfit(all_best_dists, f1s, 1)
    p_line = np.poly1d(z)
    xs = np.linspace(min(all_best_dists), max(all_best_dists), 100)
    ax.plot(xs, p_line(xs), "--", color="#C44E52", linewidth=1.5,
            label=f"trend: y={z[0]:.4f}x + {z[1]:.4f}")
    ax.set_xlabel("Best Retrieval Distance (lower = closer match)", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Retrieval Distance vs F1", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 15. Per-Document # Retrieved Unique Docs

# %%
n_retrieved = [len(r["retrieved_doc_ids"]) for r in per_doc_results]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(n_retrieved, bins=range(0, max(n_retrieved) + 2), color="#8172B2",
        edgecolor="white", alpha=0.85, align="left")
ax.set_xlabel("# Unique Retrieved Documents")
ax.set_ylabel("# Test Documents")
ax.set_title("Distribution of Retrieved Unique Documents per Query", fontsize=13, fontweight="bold")
ax.axvline(np.mean(n_retrieved), color="black", linestyle="--",
           label=f"mean={np.mean(n_retrieved):.2f}")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 16. Perfect Recall vs Imperfect Recall Breakdown

# %%
perfect_recall_docs   = [r for r in per_doc_results if r["recall"] == 1.0]
imperfect_recall_docs = [r for r in per_doc_results if r["recall"] < 1.0]
zero_recall_docs      = [r for r in per_doc_results if r["recall"] == 0.0]

print(f"  Perfect recall (R=1.0)   : {len(perfect_recall_docs):>4}  ({len(perfect_recall_docs)/n*100:.1f}%)")
print(f"  Imperfect recall (R<1.0) : {len(imperfect_recall_docs):>4}  ({len(imperfect_recall_docs)/n*100:.1f}%)")
print(f"  Zero recall (R=0.0)      : {len(zero_recall_docs):>4}  ({len(zero_recall_docs)/n*100:.1f}%)")

if perfect_recall_docs:
    avg_precision_perfect = np.mean([r["precision"] for r in perfect_recall_docs])
    print(f"\n  Among perfect-recall docs: avg precision = {avg_precision_perfect:.4f}")
    print(f"    → avg {np.mean([r['n_extra'] for r in perfect_recall_docs]):.1f} extra clauses per doc")

if imperfect_recall_docs:
    avg_recall_imperfect = np.mean([r["recall"] for r in imperfect_recall_docs])
    avg_missed = np.mean([r["n_missed"] for r in imperfect_recall_docs])
    print(f"\n  Among imperfect-recall docs: avg recall = {avg_recall_imperfect:.4f}")
    print(f"    → avg {avg_missed:.1f} missed clauses per doc")

# %% [markdown]
# ## 17. Save full results to JSON

# %%
output = {
    "config": {
        "train_dir":    TRAIN_DIR,
        "test_dir":     TEST_DIR,
        "train_db_dir": TRAIN_DB_DIR,
        "test_db_dir":  TEST_DB_DIR,
        "top_k":        TOP_K,
        "model_name":   MODEL_NAME,
        "device":       DEVICE,
        "max_chunk_chars":     MAX_CHUNK_CHARS,
        "collection_name":     COLLECTION_NAME,
        "train_embedding_fields": TRAIN_EMBED_CONTENT_FIELDS,
        "test_embedding_fields": TEST_EMBED_CONTENT_FIELDS,
        "query_content_fields": QUERY_CONTENT_FIELDS,
    },
    "summary": {
        "n_docs":           n,
        "macro_precision":  round(macro_p, 4),
        "macro_recall":     round(macro_r, 4),
        "macro_f1":         round(macro_f1, 4),
        "macro_precision_full": round(macro_p_full, 4),
        "macro_recall_full": round(macro_r_full, 4),
        "macro_f1_full": round(macro_f1_full, 4),
        "micro_precision":  round(micro_p, 4),
        "micro_recall":     round(micro_r, 4),
        "micro_f1":         round(micro_f1, 4),
        "micro_precision_full": round(micro_p_full, 4),
        "micro_recall_full": round(micro_r_full, 4),
        "micro_f1_full": round(micro_f1_full, 4),
        "total_gt":         total_gt,
        "total_predicted":  total_pred,
        "total_matched":    total_matched,
        "total_missed":     total_missed,
        "total_extra":      total_extra,
        "total_gt_full": total_gt_full,
        "total_predicted_full": total_pred_full,
        "total_matched_full": total_matched_full,
        "total_missed_full": total_missed_full,
        "total_extra_full": total_extra_full,
        "perfect_recall_count":   len(perfect_recall_docs),
        "imperfect_recall_count": len(imperfect_recall_docs),
        "zero_recall_count":      len(zero_recall_docs),
        "n_train_skipped": len(train_skipped),
        "n_test_skipped": len(test_skipped),
    },
    "train_skipped": train_skipped,
    "test_skipped": test_skipped,
    "top_missed_clauses": [
        {"clause": c, "count": cnt, "pct": round(cnt / n * 100, 2)}
        for c, cnt in missed_counter.most_common(TOP_N)
    ],
    "top_extra_clauses": [
        {"clause": c, "count": cnt, "pct": round(cnt / n * 100, 2)}
        for c, cnt in extra_counter.most_common(TOP_N)
    ],
    "clause_retrieval_rates": {
        clause: {
            "gt_count": info["total"],
            "hits": info["hits"],
            "misses": info["misses"],
            "rate": round(info["rate"], 4),
        }
        for clause, info in sorted(clause_retrieval_rate.items(),
                                    key=lambda x: x[1]["rate"])
    },
    "per_doc": [
        {k: v for k, v in r.items() if k != "distances"}
        for r in per_doc_results
    ],
}

out_path = Path(RESULTS_OUT)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(output, fh, ensure_ascii=False, indent=2)

print(f"\n✅ Full results saved to: {out_path}")

# %% [markdown]
# ## 18. Summary
#
# | Metric | Macro | Micro |
# |--------|-------|-------|
# | Precision | `macro_p` | `micro_p` |
# | Recall | `macro_r` | `micro_r` |
# | F1 | `macro_f1` | `micro_f1` |
#
# See the saved JSON for full per-document results, per-clause retrieval
# rates, and error analysis data.
