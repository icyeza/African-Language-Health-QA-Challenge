"""
Central configuration for the Multilingual Health QA project.

Everything that another module or a notebook might need to know about
paths, the random seed, or dataset-specific constants lives here so the
pipeline stays reproducible and there are no magic strings scattered
around the codebase.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# Zindi requires that re-running your solution lands you at the same place on
# the leaderboard, so a single global seed is used everywhere.
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed every RNG we might touch. Call once at the top of any entrypoint."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy always present in practice, but stay defensive
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism vs. speed trade-off; enable when you need exact repro.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Paths  (override RAW_TRAIN_PATH via env var on Colab if your file differs)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_TRAIN_PATH = Path(
    os.environ.get("RAW_TRAIN_PATH", PROJECT_ROOT / "data" / "Train.csv")
)
RAW_TEST_PATH = Path(
    os.environ.get("RAW_TEST_PATH", PROJECT_ROOT / "data" / "Test.csv")
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset constants (verified against the provided Train.csv via EDA)
# --------------------------------------------------------------------------
# Column names in the raw files.
ID_COL = "ID"
INPUT_COL = "input"
OUTPUT_COL = "output"
SUBSET_COL = "subset"

# The eight language-country subsets observed in training.
# NOTE: the brief mentions *nine* configurations; verify whether the test set
# introduces a ninth (unseen) subset before relying on per-subset specialists.
SUBSETS = [
    "Aka_Gha",  # Akan        - Ghana
    "Amh_Eth",  # Amharic     - Ethiopia
    "Eng_Eth",  # English     - Ethiopia
    "Eng_Gha",  # English     - Ghana
    "Eng_Ken",  # English     - Kenya
    "Eng_Uga",  # English     - Uganda
    "Lug_Uga",  # Luganda     - Uganda
    "Swa_Ken",  # Swahili     - Kenya
]

# Human-readable language / country for prompt templates and reporting.
LANG_NAME = {
    "Aka": "Akan",
    "Amh": "Amharic",
    "Eng": "English",
    "Lug": "Luganda",
    "Swa": "Swahili",
}
COUNTRY_NAME = {
    "Gha": "Ghana",
    "Eth": "Ethiopia",
    "Ken": "Kenya",
    "Uga": "Uganda",
}

# Instruction-style suffixes that appear inside the questions themselves.
# These behave like style controls and almost certainly recur in the test set,
# so we PARSE them into a feature rather than treating them as noise.
INSTRUCTION_SUFFIXES = [
    "please answer in detail.",
    "please answer this using simple medical terms.",
]

# The metadata prefix some reference answers carry, e.g.
# "This is a question about, HPV. ..." — keep it by default because the test
# references may carry it too (74% of the score is lexical overlap).
ANSWER_METADATA_PREFIX_RE = r"^\s*this is a question about[,:]\s*[^.]*\.\s*"

# Submission format required by the multi-metric leaderboard.
SUBMISSION_COLS = ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]

# Optional reference submission to align row order / IDs against.
SAMPLE_SUBMISSION_PATH = Path(
    os.environ.get("SAMPLE_SUBMISSION_PATH", PROJECT_ROOT / "data" / "SampleSubmission.csv")
)

# Metric weights (for reference / local weighted-score computation).
METRIC_WEIGHTS = {"rouge1_f1": 0.37, "rougeL_f1": 0.37, "llm_judge": 0.26}
