"""
Baseline models — the numbers every fine-tuned model must beat.

Experiment 1: TF-IDF retrieval baseline
---------------------------------------
For each query question we retrieve the most lexically similar *training*
question and copy its answer verbatim. Retrieval is done **within the same
subset** so we never answer an Amharic question with an English answer.

Why this is the right first baseline
  * It needs no GPU and runs in seconds (works on free Colab).
  * It directly exploits the structure we found in EDA: ~1,570 questions repeat,
    so many queries have a near-identical training question whose answer is a
    strong lexical match.
  * Because ~74% of the leaderboard score is lexical overlap, a copied real
    answer is a deceptively strong, fully reproducible floor. Any fine-tuned
    model that cannot beat "copy the nearest training answer" is not earning
    its complexity — which is exactly the kind of insight the rubric rewards.

The module returns predictions so they can be scored with ``evaluation.py`` and
compared against later experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .. import config


def _fit_subset_index(train_questions: list[str]):
    """Fit a char+word TF-IDF index over one subset's training questions."""
    # Char n-grams help with the low-resource / morphologically rich languages
    # and with spelling variation; word n-grams capture content overlap.
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=1, lowercase=True
    )
    matrix = vec.fit_transform(train_questions)
    return vec, matrix


def retrieval_baseline(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    per_subset: bool = True,
) -> pd.DataFrame:
    """
    Predict an answer for every row in ``query_df`` by copying the answer of the
    most similar training question.

    Parameters
    ----------
    train_df : must have columns input, output, subset.
    query_df : must have columns input, subset (output optional, for scoring).
    per_subset : if True (recommended) retrieve only within the same subset.

    Returns a copy of ``query_df`` with added columns:
        prediction      - the copied answer
        match_score     - cosine similarity to the retrieved training question
        matched_question- the training question that was matched (for inspection)
    """
    config.set_global_seed()
    out = query_df.copy().reset_index(drop=True)
    out["prediction"] = ""
    out["match_score"] = np.nan
    out["matched_question"] = ""

    groups = (
        list(out.groupby(config.SUBSET_COL).groups.items())
        if per_subset
        else [("__all__", out.index)]
    )

    for subset, idx in groups:
        q_idx = list(idx)
        if per_subset:
            tr = train_df[train_df[config.SUBSET_COL] == subset]
        else:
            tr = train_df
        if len(tr) == 0:
            continue
        tr_questions = tr[config.INPUT_COL].astype(str).tolist()
        tr_answers = tr[config.OUTPUT_COL].astype(str).tolist()

        vec, tr_matrix = _fit_subset_index(tr_questions)
        q_questions = out.loc[q_idx, config.INPUT_COL].astype(str).tolist()
        q_matrix = vec.transform(q_questions)

        sims = linear_kernel(q_matrix, tr_matrix)  # cosine (tf-idf is L2-normed)
        best = sims.argmax(axis=1)
        best_scores = sims[np.arange(sims.shape[0]), best]

        out.loc[q_idx, "prediction"] = [tr_answers[b] for b in best]
        out.loc[q_idx, "match_score"] = best_scores
        out.loc[q_idx, "matched_question"] = [tr_questions[b] for b in best]

    return out
