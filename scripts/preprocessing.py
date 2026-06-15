"""
Preprocessing pipeline for the Multilingual Health QA dataset.

Design principles
-----------------
1. **Every cleaning step is a toggle.** Because ~74% of the leaderboard score
   is lexical overlap (ROUGE-1 + ROUGE-L F1) with the reference answers,
   "cleaning" that moves the training targets away from the test reference
   distribution can actively *lower* your score. So nothing is mutated
   silently: cleaning is opt-in and each option is something you can A/B test
   as one of your required experiments.

2. **Leakage-safe splitting.** ~1,570 questions repeat in the data (3,170 rows),
   sometimes with different answers and sometimes across subsets. A naive
   random split would leak the same question into both train and validation,
   inflating your local scores. We split by *normalized question group* and
   stratify by subset.

3. **Feature extraction, not destruction.** Instruction suffixes
   ("please answer in detail") and answer-side topic prefixes
   ("This is a question about, HPV.") are parsed into columns so you can use
   them for prompting / analysis without losing the original text.

The module is import-clean (no side effects) and Colab-friendly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from .. import config


# ==========================================================================
# Text normalization
# ==========================================================================
_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip ends. Safe for all scripts."""
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", text).strip()


# ==========================================================================
# Feature extraction (parse, do not delete)
# ==========================================================================
_SUFFIX_RES = [
    (suf, re.compile(r",\s*" + re.escape(suf) + r"\s*$", re.IGNORECASE))
    for suf in config.INSTRUCTION_SUFFIXES
]
_PREFIX_RE = re.compile(config.ANSWER_METADATA_PREFIX_RE, re.IGNORECASE)
_TOPIC_CAPTURE_RE = re.compile(
    r"^\s*this is a question about[,:]\s*([^.]*)\.", re.IGNORECASE
)


def extract_instruction_style(question: str) -> tuple[str, str]:
    """
    Split a question into (core_question, style_tag).

    style_tag is one of: 'detailed', 'simple', 'none'.
    The core question has the recognized suffix removed; if no known suffix is
    present the question is returned unchanged with style 'none'.
    """
    q = question
    for suf, rgx in _SUFFIX_RES:
        if rgx.search(q):
            core = rgx.sub("", q).strip().rstrip(",").strip()
            tag = "detailed" if "detail" in suf else "simple"
            return core, tag
    return q, "none"


def extract_topic(answer: str) -> str:
    """Return the topic from a 'This is a question about, X.' prefix, else ''."""
    m = _TOPIC_CAPTURE_RE.match(answer or "")
    return m.group(1).strip() if m else ""


def strip_answer_metadata_prefix(answer: str) -> str:
    """Remove a leading 'This is a question about, X.' prefix from an answer."""
    return _PREFIX_RE.sub("", answer or "").strip()


def normalized_question_key(question: str) -> str:
    """Grouping key for leakage-safe splitting: lowercased, whitespace-normal."""
    return normalize_whitespace(question).lower()


# ==========================================================================
# Cleaning configuration
# ==========================================================================
@dataclass
class CleanConfig:
    """Toggle-able cleaning options. Defaults preserve the reference style."""

    normalize_ws: bool = True            # collapse whitespace (safe, recommended)
    drop_blank: bool = True              # drop rows with empty input or output
    drop_exact_duplicates: bool = True   # drop identical (input, output) rows
    # The next two CHANGE the target distribution -> off by default, treat as
    # experiments and A/B test their effect on validation ROUGE.
    strip_answer_prefix: bool = False    # remove "This is a question about, X."
    strip_instruction_suffix: bool = False  # remove ", please answer ..." suffix
    min_output_chars: int = 1            # drop degenerate ultra-short answers

    def as_dict(self) -> dict:
        return asdict(self)


# ==========================================================================
# Main cleaning routine
# ==========================================================================
def clean_dataframe(df: pd.DataFrame, cfg: CleanConfig | None = None) -> pd.DataFrame:
    """
    Apply the configured cleaning steps and add derived feature columns.

    Returns a new DataFrame with the original columns plus:
      core_question   - question with instruction suffix parsed out
      style_tag       - 'detailed' | 'simple' | 'none'
      topic           - disease/topic parsed from answer prefix (may be '')
      q_key           - normalized question key (for grouped splitting)
      in_words/out_words, in_chars/out_chars - length features
    """
    cfg = cfg or CleanConfig()
    out = df.copy()

    # --- normalize text -------------------------------------------------
    if cfg.normalize_ws:
        out[config.INPUT_COL] = out[config.INPUT_COL].map(normalize_whitespace)
        out[config.OUTPUT_COL] = out[config.OUTPUT_COL].map(normalize_whitespace)

    # --- drop blanks ----------------------------------------------------
    if cfg.drop_blank:
        before = len(out)
        out = out[
            (out[config.INPUT_COL].str.len() > 0)
            & (out[config.OUTPUT_COL].str.len() >= cfg.min_output_chars)
        ].copy()
        _log("drop_blank", before, len(out))

    # --- feature extraction (always parse) ------------------------------
    parsed = out[config.INPUT_COL].map(extract_instruction_style)
    out["core_question"] = parsed.map(lambda t: t[0])
    out["style_tag"] = parsed.map(lambda t: t[1])
    out["topic"] = out[config.OUTPUT_COL].map(extract_topic)

    # --- optional target-altering cleaning ------------------------------
    if cfg.strip_instruction_suffix:
        out[config.INPUT_COL] = out["core_question"]
    if cfg.strip_answer_prefix:
        out[config.OUTPUT_COL] = out[config.OUTPUT_COL].map(strip_answer_metadata_prefix)

    # --- drop exact duplicates ------------------------------------------
    if cfg.drop_exact_duplicates:
        before = len(out)
        out = out.drop_duplicates(subset=[config.INPUT_COL, config.OUTPUT_COL]).copy()
        _log("drop_exact_duplicates", before, len(out))

    # --- length + grouping features -------------------------------------
    out["q_key"] = out[config.INPUT_COL].map(normalized_question_key)
    out["in_chars"] = out[config.INPUT_COL].str.len()
    out["out_chars"] = out[config.OUTPUT_COL].str.len()
    out["in_words"] = out[config.INPUT_COL].str.split().map(len)
    out["out_words"] = out[config.OUTPUT_COL].str.split().map(len)

    return out.reset_index(drop=True)


def _log(step: str, before: int, after: int) -> None:
    if before != after:
        print(f"  [{step}] {before:,} -> {after:,}  ({before - after:,} removed)")


# ==========================================================================
# Leakage-safe, subset-stratified split
# ==========================================================================
def stratified_group_split(
    df: pd.DataFrame,
    val_size: float = 0.10,
    seed: int = config.SEED,
    group_col: str = "q_key",
    strata_col: str = config.SUBSET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into (train, val) so that:
      * no question group spans both sides  (prevents leakage), and
      * the subset distribution is preserved (stratification).

    Uses StratifiedGroupKFold with n_splits = round(1/val_size); one fold is
    held out as validation. Deterministic given `seed`.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1 / val_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y = df[strata_col].values
    groups = df[group_col].values
    train_idx, val_idx = next(sgkf.split(df, y, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    _assert_no_group_leakage(train_df, val_df, group_col)
    return train_df, val_df


def _assert_no_group_leakage(train_df, val_df, group_col) -> None:
    overlap = set(train_df[group_col]) & set(val_df[group_col])
    if overlap:
        raise AssertionError(
            f"Leakage detected: {len(overlap)} question groups in both splits."
        )


# ==========================================================================
# Prompt construction (used by both training and inference)
# ==========================================================================
def _parse_subset(subset: str) -> tuple[str, str]:
    """'Aka_Gha' -> ('Akan', 'Ghana'); robust to unseen tags."""
    lang_code, _, country_code = subset.partition("_")
    lang = config.LANG_NAME.get(lang_code, lang_code)
    country = config.COUNTRY_NAME.get(country_code, country_code)
    return lang, country


def build_prompt(question: str, subset: str, template: str = "tagged") -> str:
    """
    Format a model input. `template` selects a prompting strategy so you can
    treat the prompt format itself as an experiment.

    Templates
    ---------
    'plain'  : the question as-is.
    'tagged' : prepend an explicit language/country instruction.
    'qa'     : a light QA framing with the language/country tag.
    """
    lang, country = _parse_subset(subset)
    if template == "plain":
        return question
    if template == "tagged":
        return f"[{lang} | {country}] {question}"
    if template == "qa":
        return (
            f"Answer the following health question in {lang} "
            f"(context: {country}).\nQuestion: {question}\nAnswer:"
        )
    raise ValueError(f"Unknown template: {template!r}")


# ==========================================================================
# One-call convenience entrypoint
# ==========================================================================
def prepare(
    raw_path=None,
    clean_cfg: CleanConfig | None = None,
    val_size: float = 0.10,
    seed: int = config.SEED,
    save: bool = True,
) -> dict:
    """
    Load -> clean -> split, optionally writing parquet files to data/processed.

    Returns a dict with keys: train, val, clean_cfg, stats.
    """
    config.set_global_seed(seed)
    raw_path = raw_path or config.RAW_TRAIN_PATH
    raw = pd.read_csv(raw_path)
    print(f"Loaded {len(raw):,} raw rows from {raw_path}")

    cleaned = clean_dataframe(raw, clean_cfg)
    print(f"After cleaning: {len(cleaned):,} rows")

    train_df, val_df = stratified_group_split(cleaned, val_size=val_size, seed=seed)
    print(f"Split -> train {len(train_df):,} | val {len(val_df):,}")

    stats = {
        "raw_rows": len(raw),
        "clean_rows": len(cleaned),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "clean_cfg": (clean_cfg or CleanConfig()).as_dict(),
    }

    if save:
        train_df.to_parquet(config.PROCESSED_DIR / "train.parquet", index=False)
        val_df.to_parquet(config.PROCESSED_DIR / "val.parquet", index=False)
        print(f"Saved processed splits to {config.PROCESSED_DIR}")

    return {"train": train_df, "val": val_df, "clean_cfg": clean_cfg, "stats": stats}


if __name__ == "__main__":
    prepare()
