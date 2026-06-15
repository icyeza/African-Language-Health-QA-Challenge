"""
Experiment tracker.

The rubric rewards *systematic* experiment progression: each change recorded
with its reasoning, configuration, and measured effect. This module is a tiny,
dependency-free logger that appends one row per experiment to
``reports/experiments.csv`` (and a richer JSON), so the final report's
experiment-progression table and learning-curve plots build themselves from a
single source of truth.

Usage
-----
    from healthqa.experiment_tracker import log_experiment
    log_experiment(
        exp_id="01_retrieval_baseline",
        description="TF-IDF nearest-train-question answer copying, per-subset",
        hypothesis="Repeated questions make copied answers a strong lexical floor.",
        config={"retrieval": "char_wb 2-5", "per_subset": True},
        result=eval_result,                 # an evaluation.EvalResult
        notes="Floor every fine-tuned model must beat.",
    )
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config

LOG_CSV = config.REPORTS_DIR / "experiments.csv"
LOG_JSON = config.REPORTS_DIR / "experiments.json"


def log_experiment(
    exp_id: str,
    description: str,
    result,                      # evaluation.EvalResult
    hypothesis: str = "",
    config: Optional[dict] = None,   # noqa: A002 (shadow is fine here)
    notes: str = "",
    tokenizer_mode: Optional[str] = None,
) -> dict:
    """Append one experiment record and return it. Idempotent per exp_id+mode."""
    mode = tokenizer_mode or getattr(result, "tokenizer_mode", "leaderboard")
    rw = result.overall_reweighted
    rm = result.overall_mean

    record = {
        "exp_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": description,
        "hypothesis": hypothesis,
        "tokenizer_mode": mode,
        # Leaderboard estimate (test-reweighted) is the headline number.
        "lb_rouge1_f1": round(rw["rouge1_f1"], 4),
        "lb_rougeL_f1": round(rw["rougeL_f1"], 4),
        "lb_rouge_weighted": round(rw["rouge_weighted"], 4),
        # Row-mean on the eval set (for reference).
        "val_rouge1_f1": round(rm["rouge1_f1"], 4),
        "val_rougeL_f1": round(rm["rougeL_f1"], 4),
        "val_rouge_weighted": round(rm["rouge_weighted"], 4),
        "config": json.dumps(config or {}),
        "notes": notes,
        "by_subset": result.by_subset["rouge1_f1"].round(4).to_dict(),
    }

    # CSV (flat) — for the report table; drop nested by_subset.
    flat = {k: v for k, v in record.items() if k != "by_subset"}
    df_row = pd.DataFrame([flat])
    if LOG_CSV.exists():
        prev = pd.read_csv(LOG_CSV)
        prev = prev[~((prev["exp_id"] == exp_id) & (prev["tokenizer_mode"] == mode))]
        df_row = pd.concat([prev, df_row], ignore_index=True)
    df_row.to_csv(LOG_CSV, index=False)

    # JSON (rich) — keeps the per-subset breakdown.
    records = []
    if LOG_JSON.exists():
        records = json.loads(LOG_JSON.read_text())
    records = [
        r for r in records
        if not (r["exp_id"] == exp_id and r["tokenizer_mode"] == mode)
    ]
    records.append(record)
    LOG_JSON.write_text(json.dumps(records, indent=2))

    return record


def progression_table(tokenizer_mode: str = "leaderboard") -> pd.DataFrame:
    """Return the experiment-progression table (headline leaderboard estimates)."""
    if not LOG_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOG_CSV)
    df = df[df["tokenizer_mode"] == tokenizer_mode].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["delta_vs_prev"] = df["lb_rouge_weighted"].diff().round(4)
    cols = [
        "exp_id", "description", "lb_rouge1_f1", "lb_rougeL_f1",
        "lb_rouge_weighted", "delta_vs_prev",
    ]
    return df[cols]
