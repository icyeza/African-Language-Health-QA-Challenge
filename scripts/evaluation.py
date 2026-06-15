"""
Local evaluation harness for the Multilingual Health QA challenge.

Why this module exists
----------------------
You cannot measure the *effect* of an experiment without (a) a scoring function
and (b) a baseline. This module is (a): it reproduces the leaderboard's ROUGE
components locally on the labelled validation set so every experiment produces a
comparable number without spending a leaderboard submission.

Two things make this non-trivial and are handled explicitly:

1. **Tokenizer choice changes the score dramatically for non-Latin scripts.**
   The official ``rouge-score`` default tokenizer strips non-ASCII characters,
   so *identical* Amharic text scores 0.0. We expose two modes:
     - ``leaderboard`` (default): mimic the official library so our local number
       tracks the real leaderboard, even though it under-credits Amharic.
     - ``multilingual``: a Unicode-aware tokenizer that scores all scripts, used
       to understand true per-language quality during analysis.

2. **The validation subset mix differs from the test mix.** Row-averaging val
   over-weights the (easy, short) Ethiopia subsets, which are only ~4.6% of the
   test set. We therefore report a per-subset breakdown and a test-proportion
   *reweighted* aggregate that better estimates the leaderboard.

The LLM-as-a-Judge component (0.26 of the official score) is not reproducible
offline, so the primary local proxy is the ROUGE-only weighted score; the full
formula is exposed with a pluggable judge term for when an estimate is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from .. import config

# --------------------------------------------------------------------------
# Test-set subset proportions (from Test.csv) used to reweight local val.
# --------------------------------------------------------------------------
TEST_SUBSET_PROPORTIONS = {
    "Eng_Uga": 744,
    "Aka_Gha": 492,
    "Eng_Gha": 491,
    "Lug_Uga": 374,
    "Swa_Ken": 229,
    "Eng_Ken": 167,
    "Amh_Eth": 61,
    "Eng_Eth": 60,
}
_TEST_TOTAL = sum(TEST_SUBSET_PROPORTIONS.values())
TEST_SUBSET_WEIGHTS = {k: v / _TEST_TOTAL for k, v in TEST_SUBSET_PROPORTIONS.items()}


# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------
class WhitespaceTokenizer(Tokenizer):
    """
    The OFFICIAL leaderboard tokenizer (replicated from the competition starter
    notebook): split on whitespace, no stemming, no lowercasing. It is
    language-agnostic and safe for African scripts — Amharic uses spaces
    between words, so it is scored normally (unlike rouge-score's Latin-only
    default tokenizer, which would erase it).
    """

    def tokenize(self, text: str) -> list[str]:
        if text is None:
            return []
        return str(text).strip().split()


class DefaultRougeTokenizer(Tokenizer):
    """rouge-score's built-in default (Latin-only, lowercased). Kept only for
    side-by-side comparison; it under-credits non-Latin scripts and is NOT the
    leaderboard metric."""

    def tokenize(self, text: str) -> list[str]:
        from rouge_score import tokenize as _rs_tok

        return _rs_tok.tokenize(text or "", None)


def make_scorer(mode: str = "leaderboard") -> rouge_scorer.RougeScorer:
    """
    Build a RougeScorer.

    mode='leaderboard' -> OFFICIAL metric: whitespace tokenizer, no stemmer,
                          no lowercasing. Matches the Zindi leaderboard.
    mode='default'     -> rouge-score's Latin-only default + stemmer. For
                          comparison only; under-credits non-Latin text.
    """
    if mode == "leaderboard":
        return rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=False, tokenizer=WhitespaceTokenizer()
        )
    if mode == "default":
        return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    raise ValueError(f"Unknown tokenizer mode: {mode!r}")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Holds per-row scores and aggregate tables for one evaluation run."""

    per_row: pd.DataFrame          # columns: subset, rouge1_f1, rougeL_f1
    by_subset: pd.DataFrame        # index: subset; mean r1, rL, n
    overall_mean: dict             # simple row-mean (val-native)
    overall_reweighted: dict       # test-proportion reweighted (leaderboard est.)
    tokenizer_mode: str

    def summary(self) -> str:
        lines = [f"Tokenizer mode: {self.tokenizer_mode}"]
        lines.append("\nPer-subset (mean F1):")
        lines.append(self.by_subset.to_string())
        m, r = self.overall_mean, self.overall_reweighted
        lines.append(
            f"\nRow-mean      : R1={m['rouge1_f1']:.4f}  RL={m['rougeL_f1']:.4f}"
            f"  ROUGE-weighted={m['rouge_weighted']:.4f}"
        )
        lines.append(
            f"Test-reweighted: R1={r['rouge1_f1']:.4f}  RL={r['rougeL_f1']:.4f}"
            f"  ROUGE-weighted={r['rouge_weighted']:.4f}   <-- leaderboard estimate"
        )
        return "\n".join(lines)


def _rouge_weighted(r1: float, rL: float, llm_judge: Optional[float] = None) -> float:
    """
    Official score = 0.37*R1 + 0.37*RL + 0.26*LLM.
    Offline, LLM-judge is unavailable, so the default proxy renormalizes over the
    two ROUGE terms. Pass `llm_judge` to compute the full estimate.
    """
    w = config.METRIC_WEIGHTS
    if llm_judge is None:
        denom = w["rouge1_f1"] + w["rougeL_f1"]
        return (w["rouge1_f1"] * r1 + w["rougeL_f1"] * rL) / denom
    return w["rouge1_f1"] * r1 + w["rougeL_f1"] * rL + w["llm_judge"] * llm_judge


def evaluate(
    predictions: Sequence[str],
    references: Sequence[str],
    subsets: Sequence[str],
    tokenizer_mode: str = "leaderboard",
    llm_judge_by_subset: Optional[dict] = None,
) -> EvalResult:
    """
    Score predictions against references.

    Parameters
    ----------
    predictions, references, subsets : equal-length sequences.
    tokenizer_mode : 'leaderboard' or 'multilingual'.
    llm_judge_by_subset : optional {subset: judge_score in [0,1]} to fold the
        0.26 LLM term into the reweighted aggregate (else ROUGE-only proxy).
    """
    assert len(predictions) == len(references) == len(subsets), "length mismatch"
    scorer = make_scorer(tokenizer_mode)

    r1, rL = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(str(ref), str(pred))  # (target, prediction) order
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

    per_row = pd.DataFrame(
        {"subset": list(subsets), "rouge1_f1": r1, "rougeL_f1": rL}
    )

    by_subset = (
        per_row.groupby("subset")[["rouge1_f1", "rougeL_f1"]]
        .mean()
        .assign(n=per_row.groupby("subset").size())
        .round(4)
    )

    # Row-mean aggregate (native to whatever set you passed in).
    m1, mL = per_row["rouge1_f1"].mean(), per_row["rougeL_f1"].mean()
    overall_mean = {
        "rouge1_f1": float(m1),
        "rougeL_f1": float(mL),
        "rouge_weighted": float(_rouge_weighted(m1, mL)),
    }

    # Test-proportion reweighted aggregate (leaderboard estimate).
    rw1 = rwL = 0.0
    judge_total = 0.0
    covered = 0.0
    for sub, w in TEST_SUBSET_WEIGHTS.items():
        if sub in by_subset.index:
            rw1 += w * by_subset.loc[sub, "rouge1_f1"]
            rwL += w * by_subset.loc[sub, "rougeL_f1"]
            covered += w
            if llm_judge_by_subset and sub in llm_judge_by_subset:
                judge_total += w * llm_judge_by_subset[sub]
    if covered > 0:  # renormalize if some subsets absent
        rw1, rwL = rw1 / covered, rwL / covered
    judge = (judge_total / covered) if (llm_judge_by_subset and covered) else None
    overall_reweighted = {
        "rouge1_f1": float(rw1),
        "rougeL_f1": float(rwL),
        "rouge_weighted": float(_rouge_weighted(rw1, rwL, judge)),
    }

    return EvalResult(
        per_row=per_row,
        by_subset=by_subset,
        overall_mean=overall_mean,
        overall_reweighted=overall_reweighted,
        tokenizer_mode=tokenizer_mode,
    )


def evaluate_df(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    ref_col: str = config.OUTPUT_COL,
    subset_col: str = config.SUBSET_COL,
    tokenizer_mode: str = "leaderboard",
) -> EvalResult:
    """Convenience wrapper to score a DataFrame that has prediction + reference."""
    return evaluate(
        df[pred_col].tolist(),
        df[ref_col].tolist(),
        df[subset_col].tolist(),
        tokenizer_mode=tokenizer_mode,
    )
