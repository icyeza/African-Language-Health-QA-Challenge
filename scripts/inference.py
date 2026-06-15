"""
Inference and submission generation.

Two responsibilities:
  1. Batched generation from a fine-tuned seq2seq model, with optional
     per-subset ``max_new_tokens`` calibration — the EDA showed reference
     length varies ~5x across subsets, and over-generating tanks ROUGE
     precision, so capping generation per subset is a cheap, high-ROI lever.
  2. Writing the exact submission format the leaderboard requires: columns
     ``ID, TargetRLF1, TargetR1F1, TargetLLM`` with the *same* generated answer
     in all three target columns.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .. import config
from .preprocessing import build_prompt

# Per-subset generation caps (in tokens) derived from training answer lengths.
# Roughly the 95th-percentile answer length per subset, with headroom; tune as
# an experiment. Subsets absent here fall back to ``default_max_new_tokens``.
DEFAULT_SUBSET_MAX_TOKENS = {
    "Amh_Eth": 64,
    "Eng_Eth": 80,
    "Eng_Ken": 256,
    "Swa_Ken": 256,
    "Eng_Gha": 256,
    "Lug_Uga": 320,
    "Eng_Uga": 320,
    "Aka_Gha": 384,
}


def generate_answers(
    model,
    tokenizer,
    df: pd.DataFrame,
    prompt_template: str = "tagged",
    num_beams: int = 4,
    default_max_new_tokens: int = 256,
    subset_max_tokens: Optional[dict] = None,
    batch_size: int = 16,
    max_input_len: int = 256,
) -> list[str]:
    """
    Generate one answer per row of ``df`` (needs columns input, subset).

    Generation is grouped by subset so each group uses its calibrated length cap
    while still batching efficiently.
    """
    import torch

    subset_max_tokens = subset_max_tokens or DEFAULT_SUBSET_MAX_TOKENS
    device = next(model.parameters()).device
    model.eval()

    answers = [""] * len(df)
    work = df.reset_index(drop=True)

    for subset, idx in work.groupby(config.SUBSET_COL).groups.items():
        idx = list(idx)
        max_new = subset_max_tokens.get(subset, default_max_new_tokens)
        prompts = [
            build_prompt(work.loc[i, config.INPUT_COL], subset, prompt_template)
            for i in idx
        ]
        for start in range(0, len(prompts), batch_size):
            chunk_ids = idx[start : start + batch_size]
            chunk = prompts[start : start + batch_size]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=max_input_len,
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    num_beams=num_beams,
                    max_new_tokens=max_new,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            dec = tokenizer.batch_decode(out, skip_special_tokens=True)
            for j, ans in zip(chunk_ids, dec):
                answers[j] = ans.strip()
    return answers


def write_submission(ids, answers, path, sample_submission_path=None) -> pd.DataFrame:
    """
    Write the multi-metric submission file.

    The leaderboard expects ID + three target columns, all holding the SAME
    generated answer (the platform computes ROUGE-1, ROUGE-L and LLM-judge from
    that single string).

    If ``sample_submission_path`` is given (or ``config.SAMPLE_SUBMISSION_PATH``
    exists), the output is aligned to the sample's exact ID order — robust to any
    reordering during generation. IDs missing a prediction get an empty string
    (with a warning); unexpected extra IDs are dropped.
    """
    id2ans = dict(zip([str(i) for i in ids], [str(a) for a in answers]))

    sample_path = sample_submission_path or config.SAMPLE_SUBMISSION_PATH
    try:
        sample = pd.read_csv(sample_path)
        ordered_ids = sample["ID"].astype(str).tolist()
        missing = [i for i in ordered_ids if i not in id2ans]
        if missing:
            print(f"WARNING: {len(missing)} sample IDs have no prediction; "
                  f"filling empty (e.g. {missing[:3]}).")
        final_ids = ordered_ids
        final_answers = [id2ans.get(i, "") for i in ordered_ids]
    except (FileNotFoundError, OSError):
        # No sample available -> use the provided order as-is.
        final_ids = [str(i) for i in ids]
        final_answers = [id2ans[i] for i in final_ids]

    sub = pd.DataFrame(
        {
            "ID": final_ids,
            "TargetRLF1": final_answers,
            "TargetR1F1": final_answers,
            "TargetLLM": final_answers,
        }
    )
    assert list(sub.columns) == config.SUBMISSION_COLS, "submission column mismatch"
    sub.to_csv(path, index=False)
    print(f"Wrote {len(sub):,} rows -> {path}")
    return sub
