"""Visualize embedding evaluation results from retrieval_eval.py output JSON files.

Each section is delimited by a '# %%' comment so this file can be converted to
a Jupyter notebook using:
    jupytext --to notebook visualize_eval.py
or opened directly in VS Code / JupyterLab as a paired notebook.
"""

# %% [markdown]
# # Embedding Retrieval Evaluation — Visual Analysis
#
# This notebook loads one or two `eval_results_*.json` files produced by
# `retrieval_eval.py` and renders diagnostic charts:
#
# | Section | What it shows |
# |---------|---------------|
# | 1 | Global macro metrics (P / R / F1) comparison across runs |
# | 2 | Per-document P / R / F1 distribution |
# | 3 | Top-N **missed** clauses (in ground truth but never retrieved) |
# | 4 | Top-N **redundant / noisy** clauses (retrieved but never in ground truth) |
# | 5 | Top-N **popular** clauses in the ground truth corpus |
# | 6 | Recall heat-map by điều (article number) |
# | 7 | Scatter: recall vs precision per document (coloured by F1) |
# | 8 | Ground-truth set-size distribution |

# %%
# ── Imports ─────────────────────────────────────────────────────────────────
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# %%
# ── Configuration — edit these paths / constants ─────────────────────────────

RESULTS_FILES = {
    "top_k=5":  "output/eval_results_top5.json",
    "top_k=10": "output/eval_results_top10.json",
}

# Which run to use for clause-level charts (sections 3-6)
PRIMARY_RUN = "top_k=5"

# How many top entries to show in bar charts
TOP_N = 20

# Figure output directory (set to None to skip saving)
FIG_DIR = Path("output/figures")

PALETTE = {
    "top_k=5":  "#5B8FF9",
    "top_k=10": "#F6BD16",
    "missed":   "#E8684A",
    "redundant":"#5AD8A6",
    "popular":  "#9270CA",
    "neutral":  "#6B6B6B",
}

# %%
# ── Helper utilities ──────────────────────────────────────────────────────────

def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_fig(fig: plt.Figure, name: str) -> None:
    if FIG_DIR is None:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved → {FIG_DIR / name}.png")


def style_ax(ax: plt.Axes, *, xlabel="", ylabel="", title="") -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def per_doc_df(data: dict) -> pd.DataFrame:
    rows = []
    for doc in data["per_doc"]:
        rows.append({
            "doc_id":           doc["doc_id"],
            "precision":        doc["precision"],
            "recall":           doc["recall"],
            "f1":               doc["f1"],
            "precision_dieu":   doc["precision_dieu"],
            "recall_dieu":      doc["recall_dieu"],
            "f1_dieu":          doc["f1_dieu"],
            "n_gt":             len(doc["ground_truth"]),
            "n_predicted":      len(doc["predicted_articles"]),
            "n_matched":        len(doc["matched_articles"]),
            "n_missed":         len(doc["missed_articles"]),
            "n_extra":          len(doc["extra_articles"]),
        })
    return pd.DataFrame(rows)


def clause_counters(data: dict) -> tuple[Counter, Counter, Counter]:
    """Return (missed_counter, extra_counter, gt_popular_counter)."""
    missed_ctr:  Counter = Counter()
    extra_ctr:   Counter = Counter()
    popular_ctr: Counter = Counter()
    for doc in data["per_doc"]:
        missed_ctr.update(doc["missed_articles"])
        extra_ctr.update(doc["extra_articles"])
        popular_ctr.update(doc["ground_truth"])
    return missed_ctr, extra_ctr, popular_ctr


# %%
# ── Load data ─────────────────────────────────────────────────────────────────

results = {label: load_result(path) for label, path in RESULTS_FILES.items()
           if Path(path).exists()}

if not results:
    raise FileNotFoundError(
        "No result files found. Check RESULTS_FILES paths and run retrieval_eval.py first."
    )

dfs = {label: per_doc_df(data) for label, data in results.items()}

primary_data = results[PRIMARY_RUN]
primary_df   = dfs[PRIMARY_RUN]
missed_ctr, extra_ctr, popular_ctr = clause_counters(primary_data)

print(f"Loaded {len(results)} run(s): {list(results)}")
for label, data in results.items():
    print(f"  [{label}] n_docs={data['n_docs']}  "
          f"P={data['macro_precision']:.4f}  "
          f"R={data['macro_recall']:.4f}  "
          f"F1={data['macro_f1']:.4f}")

# %% [markdown]
# ## Section 1 — Global macro metrics comparison

# %%
metric_labels  = ["Precision", "Recall", "F1", "Precision(điều)", "Recall(điều)", "F1(điều)"]
metric_keys    = ["macro_precision", "macro_recall", "macro_f1",
                  "macro_precision_dieu", "macro_recall_dieu", "macro_f1_dieu"]

n_runs  = len(results)
n_items = len(metric_keys)
x       = np.arange(n_items)
width   = 0.8 / max(n_runs, 1)

fig, ax = plt.subplots(figsize=(13, 5))

for i, (label, data) in enumerate(results.items()):
    vals   = [data[k] for k in metric_keys]
    offset = (i - (n_runs - 1) / 2) * width
    bars   = ax.bar(x + offset, vals, width=width * 0.9,
                    label=label, color=PALETTE.get(label, f"C{i}"),
                    edgecolor="white", linewidth=0.6, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
style_ax(ax, ylabel="Score", title="Macro Metrics Comparison Across Runs")
plt.tight_layout()
save_fig(fig, "01_macro_metrics")
plt.show()

# %% [markdown]
# ## Section 3 — Top-N missed clauses
#
# Clauses that appear in the ground truth but were **never** retrieved across the most test cases.
# High frequency here means the retriever systematically cannot recall those articles.

# %%
top_missed = missed_ctr.most_common(TOP_N)
labels_m, counts_m = zip(*top_missed) if top_missed else ([], [])

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(list(labels_m)[::-1], list(counts_m)[::-1],
               color=PALETTE["missed"], edgecolor="white", linewidth=0.5, zorder=3)
for bar, cnt in zip(bars, list(counts_m)[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", fontsize=9)
ax.set_xlim(0, max(counts_m) * 1.12 if counts_m else 1)
style_ax(ax,
         xlabel="# Test Cases Where Clause Was Missed",
         title=f"Top {TOP_N} Most Frequently Missed Clauses  [{PRIMARY_RUN}]")
ax.set_yticks(range(len(labels_m)))
ax.set_yticklabels(list(labels_m)[::-1], fontsize=9)
plt.tight_layout()
save_fig(fig, "03_top_missed_clauses")
plt.show()

print(f"\nTop {TOP_N} missed clauses (full_signature level):")
for clause, cnt in top_missed:
    pct = 100 * cnt / primary_data["n_docs"]
    print(f"  {clause:<20}  missed in {cnt:>3} / {primary_data['n_docs']} docs  ({pct:.1f}%)")

# %% [markdown]
# ## Section 4 — Top-N redundant (extra) clauses
#
# Clauses that appear in predictions but **never** match the ground truth.
# High frequency = systematic noise injected by the retriever.

# %%
top_extra = extra_ctr.most_common(TOP_N)
labels_e, counts_e = zip(*top_extra) if top_extra else ([], [])

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(list(labels_e)[::-1], list(counts_e)[::-1],
               color=PALETTE["redundant"], edgecolor="white", linewidth=0.5, zorder=3)
for bar, cnt in zip(bars, list(counts_e)[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", fontsize=9)
ax.set_xlim(0, max(counts_e) * 1.12 if counts_e else 1)
style_ax(ax,
         xlabel="# Test Cases Where Clause Was Redundant",
         title=f"Top {TOP_N} Most Frequently Redundant (Extra) Clauses  [{PRIMARY_RUN}]")
ax.set_yticks(range(len(labels_e)))
ax.set_yticklabels(list(labels_e)[::-1], fontsize=9)
plt.tight_layout()
save_fig(fig, "04_top_redundant_clauses")
plt.show()

print(f"\nTop {TOP_N} redundant clauses (full_signature level):")
for clause, cnt in top_extra:
    pct = 100 * cnt / primary_data["n_docs"]
    print(f"  {clause:<20}  extra in  {cnt:>3} / {primary_data['n_docs']} docs  ({pct:.1f}%)")

# %% [markdown]
# ## Section 5 — Top-N most popular ground-truth clauses
#
# These are the most common clauses courts actually apply. Cross-reference with
# the missed chart to understand if popular clauses are also hard to retrieve.

# %%
top_popular = popular_ctr.most_common(TOP_N)
labels_p, counts_p = zip(*top_popular) if top_popular else ([], [])

# Colour each bar by missed-frequency (darker = missed more often)
max_miss = max(missed_ctr.values()) if missed_ctr else 1
miss_ratio = [missed_ctr.get(c, 0) / max_miss for c in list(labels_p)[::-1]]
cmap = LinearSegmentedColormap.from_list("miss", ["#9270CA", "#E8684A"])
bar_colors = [cmap(r) for r in miss_ratio]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(list(labels_p)[::-1], list(counts_p)[::-1],
               color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
for bar, cnt in zip(bars, list(counts_p)[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", fontsize=9)
ax.set_xlim(0, max(counts_p) * 1.12 if counts_p else 1)
style_ax(ax,
         xlabel="# Test Cases Containing Clause in Ground Truth",
         title=f"Top {TOP_N} Most Popular Clauses in Ground Truth  [{PRIMARY_RUN}]\n"
               "(colour = miss frequency: purple→low, red→high)")
ax.set_yticks(range(len(labels_p)))
ax.set_yticklabels(list(labels_p)[::-1], fontsize=9)
# Colour bar legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_miss))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="How often clause is missed", shrink=0.6, pad=0.01)
plt.tight_layout()
save_fig(fig, "05_top_popular_clauses")
plt.show()
