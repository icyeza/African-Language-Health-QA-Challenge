"""
Fine-tuning driver for seq2seq Health QA models.

Wraps Hugging Face ``Seq2SeqTrainer`` with:
  * reproducible, hardware-aware configuration (via ``modeling.autoconfig``),
  * prompt construction shared with inference (via ``preprocessing.build_prompt``),
  * the OFFICIAL whitespace ROUGE computed in-loop so the validation metric you
    watch during training is the same one the leaderboard uses.

The actual training runs on Colab (GPU + model download required). A
``smoke_test`` flag trains on a tiny slice for one epoch so you can verify the
whole path end-to-end in ~2 minutes before committing to a full run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .. import config
from .preprocessing import build_prompt


@dataclass
class TrainConfig:
    model_name: str = "google/mt5-base"
    prompt_template: str = "tagged"       # plain | tagged | qa
    max_input_len: int = 256
    max_target_len: int = 320             # answers p99 ~272 words; 320 is safe-ish
    learning_rate: float = 3e-4           # full FT mT5; use ~1e-3 for LoRA
    epochs: float = 3.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_beams_eval: int = 4
    use_lora: bool = False
    seed: int = config.SEED
    smoke_test: bool = False              # tiny-slice sanity run
    output_dir: str = "outputs/mt5_base"


def _to_hf_dataset(df: pd.DataFrame, tmpl: str):
    """Build a HF Dataset of {prompt, target, subset} from a cleaned DataFrame."""
    from datasets import Dataset

    prompts = [
        build_prompt(q, s, template=tmpl)
        for q, s in zip(df[config.INPUT_COL], df[config.SUBSET_COL])
    ]
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "target": df[config.OUTPUT_COL].astype(str).tolist(),
            "subset": df[config.SUBSET_COL].tolist(),
        }
    )


def _make_tokenize_fn(tokenizer, cfg: TrainConfig):
    def fn(batch):
        model_inputs = tokenizer(
            batch["prompt"], max_length=cfg.max_input_len, truncation=True
        )
        labels = tokenizer(
            text_target=batch["target"], max_length=cfg.max_target_len, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return fn


def _make_compute_metrics(tokenizer):
    """In-loop metric == official whitespace ROUGE (matches the leaderboard)."""
    from .evaluation import make_scorer

    scorer = make_scorer("leaderboard")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        r1 = rL = 0.0
        for p, l in zip(dec_preds, dec_labels):
            s = scorer.score(l, p)
            r1 += s["rouge1"].fmeasure
            rL += s["rougeL"].fmeasure
        n = max(1, len(dec_preds))
        r1, rL = r1 / n, rL / n
        # weighted proxy over the two ROUGE terms (LLM-judge unavailable offline)
        weighted = (0.37 * r1 + 0.37 * rL) / 0.74
        return {"rouge1_f1": r1, "rougeL_f1": rL, "rouge_weighted": weighted}

    return compute_metrics


def train_seq2seq(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainConfig | None = None,
):
    """
    Fine-tune a seq2seq model and return (trainer, tokenizer, hw_config).

    Designed to run on Colab. On a free T4 with mT5-base this is a multi-hour
    run; use ``cfg.smoke_test=True`` first to validate the pipeline quickly.
    """
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from .modeling import autoconfig, load_seq2seq

    cfg = cfg or TrainConfig()
    config.set_global_seed(cfg.seed)
    hw = autoconfig(model_size_hint="small")
    print(f"Hardware: {hw.as_dict()}")

    if cfg.smoke_test:
        train_df = train_df.sample(min(200, len(train_df)), random_state=cfg.seed)
        val_df = val_df.sample(min(100, len(val_df)), random_state=cfg.seed)
        cfg.epochs = 1.0
        print("SMOKE TEST: 200 train / 100 val, 1 epoch")

    model, tokenizer = load_seq2seq(cfg.model_name, hw=hw, use_lora=cfg.use_lora)

    ds_train = _to_hf_dataset(train_df, cfg.prompt_template).map(
        _make_tokenize_fn(tokenizer, cfg), batched=True,
        remove_columns=["prompt", "target", "subset"],
    )
    ds_val = _to_hf_dataset(val_df, cfg.prompt_template).map(
        _make_tokenize_fn(tokenizer, cfg), batched=True,
        remove_columns=["prompt", "target", "subset"],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=hw.per_device_batch,
        per_device_eval_batch_size=hw.eval_batch,
        gradient_accumulation_steps=hw.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        predict_with_generate=True,
        generation_num_beams=cfg.num_beams_eval,
        generation_max_length=cfg.max_target_len,
        fp16=(hw.precision == "fp16"),
        bf16=(hw.precision == "bf16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge_weighted",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=_make_compute_metrics(tokenizer),
    )

    trainer.train()
    return trainer, tokenizer, hw
