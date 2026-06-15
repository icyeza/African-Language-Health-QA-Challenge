"""
Model loading and hardware auto-configuration.

This module is what makes the repo run end-to-end on a *free* Colab T4 while
still letting you scale up on Pro without touching the training code:

  * ``autoconfig()`` inspects the GPU and returns sensible precision / batch /
    accumulation settings (T4 -> fp16; A100/L4 -> bf16; small VRAM -> 4-bit).
  * ``load_seq2seq()`` loads any seq2seq backbone (mT5, NLLB, ...) and optionally
    wraps it in LoRA / QLoRA.

Defaults target the small seq2seq models recommended by the competition starter
(``google/mt5-base``, ``facebook/nllb-200-distilled-600M``) which *full* fine-tune
on a 16 GB T4. The LoRA / 4-bit paths exist for the larger "strong" config you
run on Pro.

Heavy deps (torch, transformers, peft, bitsandbytes) are imported lazily so the
rest of the package (EDA, preprocessing, evaluation, baselines) imports with no
ML stack installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HardwareConfig:
    """Resolved hardware-dependent training/inference settings."""

    device: str = "cpu"
    gpu_name: str = ""
    vram_gb: float = 0.0
    precision: str = "fp32"          # 'bf16' | 'fp16' | 'fp32'
    per_device_batch: int = 8
    grad_accum: int = 2
    eval_batch: int = 16
    use_4bit: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def autoconfig(model_size_hint: str = "small") -> HardwareConfig:
    """
    Inspect the GPU and return a HardwareConfig.

    model_size_hint : 'small' (<=1B seq2seq, full FT) or 'large' (>=7B decoder,
        forces 4-bit + small batch). Lets one notebook serve both repo tiers.
    """
    try:
        import torch
    except ImportError:
        return HardwareConfig()

    if not torch.cuda.is_available():
        return HardwareConfig(device="cpu", precision="fp32",
                              per_device_batch=2, grad_accum=1, eval_batch=4)

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    # bf16 is supported on Ampere+ (A100, A10, L4, ...); T4 is Turing -> fp16.
    bf16_ok = torch.cuda.is_bf16_supported()
    precision = "bf16" if bf16_ok else "fp16"

    cfg = HardwareConfig(
        device="cuda", gpu_name=name, vram_gb=round(vram, 1), precision=precision
    )

    if model_size_hint == "large":
        # 7-8B decoder via QLoRA.
        cfg.use_4bit = True
        cfg.per_device_batch = 4 if vram >= 30 else 2
        cfg.grad_accum = 4 if vram >= 30 else 8
        cfg.eval_batch = 8
    else:
        # small seq2seq, full fine-tune.
        if vram >= 30:           # A100 40GB
            cfg.per_device_batch, cfg.grad_accum, cfg.eval_batch = 32, 1, 64
        elif vram >= 20:         # L4 22GB / A10
            cfg.per_device_batch, cfg.grad_accum, cfg.eval_batch = 16, 1, 32
        else:                    # T4 16GB
            cfg.per_device_batch, cfg.grad_accum, cfg.eval_batch = 8, 2, 16
    return cfg


@dataclass
class LoraConfig:
    """LoRA hyper-parameters (used by both the seq2seq and decoder paths)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Sensible defaults per architecture; override for decoder models.
    target_modules: Optional[list] = None

    def to_peft(self, task_type: str):
        from peft import LoraConfig as PeftLoraConfig

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type=task_type,
        )


def load_seq2seq(
    model_name: str,
    hw: HardwareConfig | None = None,
    use_lora: bool = False,
    lora: LoraConfig | None = None,
):
    """
    Load a seq2seq backbone (+ tokenizer), optionally LoRA/QLoRA-wrapped.

    Returns (model, tokenizer). Always loads weights in fp32; mixed precision is
    applied by the Trainer (the starter notebook's recommendation — storing
    weights in fp16 breaks gradient computation for these models).
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    hw = hw or autoconfig()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"torch_dtype": torch.float32}
    if use_lora and hw.use_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if hw.precision == "bf16" else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)

    if use_lora:
        from peft import get_peft_model, prepare_model_for_kbit_training

        if hw.use_4bit:
            model = prepare_model_for_kbit_training(model)
        lora = lora or LoraConfig(target_modules=["q", "v"])  # mT5/NLLB attn proj
        model = get_peft_model(model, lora.to_peft(task_type="SEQ_2_SEQ_LM"))
        model.print_trainable_parameters()

    if hw.device == "cuda" and not hw.use_4bit:
        model = model.to("cuda")
    return model, tokenizer
