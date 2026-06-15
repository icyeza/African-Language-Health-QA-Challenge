"""
Exploratory Data Analysis for the Multilingual Health QA dataset.

Running this module (``python -m healthqa.eda``) produces:
  * a set of PNG figures in reports/figures/
  * a machine- and human-readable stats summary (reports/eda_summary.md + .json)

All figure labels are in English/ASCII so the plots render correctly without
needing Ethiopic/African-language fonts installed. The analysis is purely
descriptive and has no side effects on the data used for training.
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless / Colab-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .preprocessing import (
    extract_instruction_style,
    extract_topic,
    normalized_question_key,
    normalize_whitespace,
)

# A restrained, consistent visual style.
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)
PALETTE = plt.get_cmap("tab10").colors


def _save(fig, name: str) -> str:
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return str(path)


# ==========================================================================
# Individual figures
# ==========================================================================
def fig_subset_distribution(df: pd.DataFrame) -> str:
    counts = df[config.SUBSET_COL].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index, counts.values, color=PALETTE[0])
    for i, v in enumerate(counts.values):
        ax.text(v + max(counts) * 0.01, i, f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of records")
    ax.set_title("Training records per language-country subset")
    ax.set_xlim(0, max(counts) * 1.12)
    return _save(fig, "01_subset_distribution.png")


def fig_output_length_by_subset(df: pd.DataFrame) -> str:
    order = df.groupby(config.SUBSET_COL)["out_words"].median().sort_values().index
    data = [df.loc[df[config.SUBSET_COL] == s, "out_words"].values for s in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(list(order))
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xlabel("Answer length (words)")
    ax.set_title("Reference answer length varies ~5x across subsets")
    return _save(fig, "02_output_length_by_subset.png")


def fig_length_distributions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["in_words"], bins=40, color=PALETTE[2])
    axes[0].set_title("Question length")
    axes[0].set_xlabel("words")
    axes[0].set_ylabel("count")
    axes[1].hist(
        df.loc[df["out_words"] < df["out_words"].quantile(0.99), "out_words"],
        bins=40,
        color=PALETTE[3],
    )
    axes[1].set_title("Answer length (<99th pct)")
    axes[1].set_xlabel("words")
    fig.suptitle("Input vs. output length distributions", y=1.02)
    return _save(fig, "03_length_distributions.png")


def fig_question_types(df: pd.DataFrame) -> str:
    eng = df[df[config.SUBSET_COL].str.startswith("Eng")]
    starters = Counter()
    for q in eng[config.INPUT_COL].astype(str):
        w = q.strip().split()
        if w:
            starters[w[0].lower().strip("?,.")] += 1
    top = dict(starters.most_common(10))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(top.keys()), list(top.values()), color=PALETTE[4])
    ax.set_title("Question types (English subsets, first word)")
    ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, "04_question_types.png")


def fig_style_and_topic(df: pd.DataFrame) -> str:
    style = df["style_tag"].value_counts()
    topics = df.loc[df["topic"] != "", "topic"].value_counts().head(10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(style.index, style.values, color=PALETTE[5])
    axes[0].set_title("Instruction-style tag (parsed from question)")
    axes[0].set_ylabel("count")
    axes[1].barh(topics.index[::-1], topics.values[::-1], color=PALETTE[6])
    axes[1].set_title("Topic (parsed from answer prefix)")
    axes[1].set_xlabel("count")
    return _save(fig, "05_style_and_topic.png")


def fig_duplication_summary(df: pd.DataFrame, stats: dict) -> str:
    labels = ["Total\nrows", "Distinct\nquestions", "Repeated\nquestions",
              "Exact dup\n(q,a) pairs"]
    values = [
        stats["total_rows"],
        stats["distinct_questions"],
        stats["repeated_question_rows"],
        stats["exact_duplicate_pairs"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=PALETTE[7])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Duplication & leakage risk")
    ax.set_ylabel("count")
    return _save(fig, "06_duplication_summary.png")


# ==========================================================================
# Stats summary
# ==========================================================================
def compute_stats(df: pd.DataFrame) -> dict:
    q_keys = df[config.INPUT_COL].map(normalized_question_key)
    vc = q_keys.value_counts()
    repeated = vc[vc > 1]
    # repeated questions whose answers differ
    tmp = df.assign(_k=q_keys)
    diff_ans = tmp.groupby("_k")[config.OUTPUT_COL].nunique()
    cross_subset = tmp.groupby("_k")[config.SUBSET_COL].nunique()

    stats = {
        "total_rows": int(len(df)),
        "subsets": df[config.SUBSET_COL].value_counts().to_dict(),
        "distinct_questions": int(q_keys.nunique()),
        "repeated_questions": int(len(repeated)),
        "repeated_question_rows": int(repeated.sum()),
        "repeated_with_differing_answers": int((diff_ans > 1).sum()),
        "repeated_across_subsets": int((cross_subset > 1).sum()),
        "exact_duplicate_pairs": int(
            df.duplicated(subset=[config.INPUT_COL, config.OUTPUT_COL]).sum()
        ),
        "blank_inputs": int((df[config.INPUT_COL].str.strip().str.len() == 0).sum()),
        "input_words": _describe(df["in_words"]),
        "output_words": _describe(df["out_words"]),
        "output_words_by_subset": df.groupby(config.SUBSET_COL)["out_words"]
        .median()
        .round(1)
        .to_dict(),
        "style_tag_counts": df["style_tag"].value_counts().to_dict(),
        "topic_counts": df.loc[df["topic"] != "", "topic"]
        .value_counts()
        .to_dict(),
    }
    return stats


def _describe(s: pd.Series) -> dict:
    return {
        "mean": round(float(s.mean()), 1),
        "median": float(s.median()),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
        "max": int(s.max()),
    }


def write_markdown(stats: dict, fig_paths: list[str]) -> str:
    lines = ["# EDA Summary — Multilingual Health QA\n"]
    lines.append(f"- **Total rows:** {stats['total_rows']:,}")
    lines.append(f"- **Distinct questions:** {stats['distinct_questions']:,}")
    lines.append(
        f"- **Repeated questions:** {stats['repeated_questions']:,} "
        f"(covering {stats['repeated_question_rows']:,} rows; "
        f"{stats['repeated_with_differing_answers']:,} have differing answers; "
        f"{stats['repeated_across_subsets']:,} span >1 subset)"
    )
    lines.append(f"- **Exact duplicate (q,a) pairs:** {stats['exact_duplicate_pairs']:,}")
    lines.append(f"- **Blank inputs:** {stats['blank_inputs']:,}\n")

    lines.append("## Answer length by subset (median words)\n")
    lines.append("| Subset | Median answer words |")
    lines.append("|---|---|")
    for s, v in sorted(
        stats["output_words_by_subset"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {s} | {v} |")

    lines.append("\n## Instruction-style tags (parsed from question)\n")
    for k, v in stats["style_tag_counts"].items():
        lines.append(f"- `{k}`: {v:,}")

    lines.append("\n## Figures\n")
    for p in fig_paths:
        name = p.split("/")[-1]
        lines.append(f"![{name}](figures/{name})")

    text = "\n".join(lines)
    out = config.REPORTS_DIR / "eda_summary.md"
    out.write_text(text)
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(stats, indent=2))
    print(f"  wrote {out}")
    return str(out)


# ==========================================================================
# Entrypoint
# ==========================================================================
def run(raw_path=None) -> dict:
    """Full EDA: load raw, add light features, render figures + summary."""
    config.set_global_seed()
    raw_path = raw_path or config.RAW_TRAIN_PATH
    df = pd.read_csv(raw_path)

    # Light feature columns needed for plots (does NOT alter training data).
    df[config.INPUT_COL] = df[config.INPUT_COL].map(normalize_whitespace)
    df[config.OUTPUT_COL] = df[config.OUTPUT_COL].map(normalize_whitespace)
    parsed = df[config.INPUT_COL].map(extract_instruction_style)
    df["style_tag"] = parsed.map(lambda t: t[1])
    df["topic"] = df[config.OUTPUT_COL].map(extract_topic)
    df["in_words"] = df[config.INPUT_COL].str.split().map(len)
    df["out_words"] = df[config.OUTPUT_COL].str.split().map(len)

    stats = compute_stats(df)
    print("Rendering figures...")
    fig_paths = [
        fig_subset_distribution(df),
        fig_output_length_by_subset(df),
        fig_length_distributions(df),
        fig_question_types(df),
        fig_style_and_topic(df),
        fig_duplication_summary(df, stats),
    ]
    write_markdown(stats, fig_paths)
    return stats


if __name__ == "__main__":
    run()
