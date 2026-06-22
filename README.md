# Multilingual Health QA — African Language Health Question Answering

Competition solution for the Zindi *Multilingual Health Question Answering in
Low-Resource African Languages* challenge. The master notebook
(`health_qa_master_1.ipynb`) documents the full 15-experiment progression from
a TF-IDF retrieval baseline (LB 0.496) to the best submission (LB 0.621).

## Experiment Progression

| # | Experiment | Method | Result | Key Insight |
|---|-----------|--------|--------|-------------|
| 1 | Retrieval baseline | TF-IDF nearest-train-answer | LB 0.496 | Copy ceiling; Ghana ~0.23 |
| 2 | Retriever selection | semantic vs TF-IDF vs best-of-3 | LB 0.543 | No single retriever wins |
| 3 | mT5 generator | fine-tuned mT5-base | LB 0.346 | Weak seq2seq loses to copying |
| 4 | 7B QLoRA + routing | Qwen2.5-7B QLoRA, per-subset | LB 0.594 | Generator wins on novel subsets |
| 5 | Beam vs greedy | beam decoding | Offline: lost every subset | Greedy wins on ROUGE |
| 6 | Epochs 1 vs 2 | training length | Offline: epoch 2 > epoch 1 | No overfitting |
| 7 | Per-subset token caps | max_new_tokens per subset | Offline: small gain | Length must track reference |
| 8 | Routing granularity | per-question router | LB 0.577 (-0.017) | Per-question regressed |
| 9 | Scale 7B to 14B | Qwen2.5-14B | LB 0.614 (+0.020) | Scale helps gen, not Ghana |
| 10 | Use-all-data (train) | train+val (7B) | LB 0.607 (+0.013) | Validation set is usable signal |
| 11 | Use-all-data (retrieval) | train+val corpus, no retrain | **LB 0.621** (best) | Free gain |
| 12 | Cross-subset retrieval | pool English for Ghana | Offline: 0.21 to 0.22 | No copyable Ghana answers |
| 13 | Aya-101 Akan | Africa-capable model | Offline: ~0.21 < ~0.30 | Fluency does not equal ROUGE |
| 14 | Ghana prompt structure | structure prompts vs P0 | Offline: P0 best | Structure can't lift Ghana |
| 15 | Length-trim | trim to reference length | Offline: 0 rows | Self-calibrated |

**Best submission: Experiment 11 (LB 0.621)** — 14B generator + train+val retrieval corpus.

## Repository Structure

```
health_qa_master_1.ipynb          # Master notebook with all 15 experiments (code + results)
outputs/health_qa_master.ipynb    # Original EDA/preprocessing notebook (base for master)
health_qa_runall_(1).ipynb        # Evaluation notebook (Exp 5-8: beam, epochs, caps, routing)
probe_then_14b (1).ipynb          # Aya-101 probe + 14B QLoRA training (Exp 9, 13)
submission_14b (1).ipynb          # 14B submission builder + routing diagnostic (Exp 9)
train_submit_7b_alldata (2).ipynb # 7B all-data training + submission (Exp 10)
rebuild_and_aya (1).ipynb         # Train+val retrieval rebuild + Aya fine-tune (Exp 11, 13)
cross_subset_retrieval (1).ipynb  # Cross-subset retrieval diagnostic (Exp 12)
ghana_prompt_experiment (1).ipynb # Ghana prompt structure ablation (Exp 14)
length_trim (1).ipynb             # Length-trim experiment (Exp 15)
retrain_alldata_14b (1).ipynb     # 14B all-data retrain (supplementary)
scripts/
  assemble_master.py              # Script that assembles the master notebook from sources
  training.py                     # Training utilities
requirements.txt
```

## Dataset

- **29,815** training rows across **8** language-country subsets
- Subsets: Aka_Gha (Akan/Ghana), Amh_Eth (Amharic/Ethiopia), Eng_Eth, Eng_Gha, Eng_Ken, Eng_Uga, Lug_Uga (Luganda/Uganda), Swa_Ken (Swahili/Kenya)
- Reference answer length varies ~5x by subset (Amharic ~20 words, Akan ~106)
- Ghana subsets (Akan + Eng_Gha) = 37.6% of test — the main battleground

## Key Design Decisions

**Per-subset routing.** No single approach wins everywhere. The best submission
uses a 14B QLoRA generator for subsets where generation beats retrieval (Eng_Gha,
Eng_Uga, Aka_Gha, Eng_Eth) and best-of-3 retrieval (TF-IDF / MiniLM / LaBSE)
for templated subsets (Eng_Ken, Swa_Ken, Lug_Uga, Amh_Eth).

**Use all data.** Training on train+val and indexing the retrieval corpus with
train+val both gave free gains (Exp 10-11).

**Ghana is the ceiling.** Experiments 12-15 systematically tested cross-subset
retrieval, Aya-101 (Africa-capable model), prompt engineering, and length
trimming. None moved Ghana meaningfully — the ceiling is the reference's exact
wording, not model capability.

## Metric

The leaderboard scores `0.37*ROUGE-1 + 0.37*ROUGE-L + 0.26*LLM-judge` using a
whitespace tokenizer (no stemmer, no lowercasing) — safe for African scripts.
Offline experiments use the ROUGE-only weighted proxy since the LLM judge is not
reproducible locally.

## Quickstart

All experiments were run on Google Colab (free T4 for retrieval/evaluation,
A100 for QLoRA training). Upload the competition CSVs (`Train.csv`, `Val.csv`,
`Test.csv`, `SampleSubmission.csv`) and run the notebooks top-to-bottom.

```bash
pip install -r requirements.txt
```
