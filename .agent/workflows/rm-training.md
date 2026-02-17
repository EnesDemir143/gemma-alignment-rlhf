---
description: Bradley-Terry Reward Model training on pairwise preferences
---

# Faz 2: Bradley-Terry Reward Model

## Prerequisites
- Faz 1 completed (`checkpoints/sft_merged_model/` exists)
- Pairwise data ready (`data/processed/rm/rm_pairwise_train.jsonl`)

## Goal
Train a reward model that learns chosen > rejected ranking using Bradley-Terry loss.

## Dosyalar
- `src/train_rm_bt.py` — RM training script (henüz yazılmadı)
- `src/models/reward_model.py` — BradleyTerryRM class (henüz yazılmadı)

## Steps

### 1. Create train/valid split for RM
Split `rm_pairwise_train.jsonl` into train (90%) and valid (10%) with `seed=42`.
- Output: `data/processed/rm/rm_train.jsonl`, `data/processed/rm/rm_valid.jsonl`

### 2. Train Bradley-Terry RM
```bash
uv run python -m src.train_rm_bt \
    --model checkpoints/sft_merged_model \
    --train-data data/processed/rm/rm_train.jsonl \
    --valid-data data/processed/rm/rm_valid.jsonl \
    --batch-size 8 \
    --epochs 2 \
    --learning-rate 1e-4 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --max-seq-length 1024 \
    --output checkpoints/rm_model_bt
```

### 3. Verify RM quality
- Pairwise accuracy ≥ 0.70
- Expected Calibration Error (ECE) < 0.1
- If ECE high → apply Temperature Scaling

### 4. Sanity check
```bash
uv run python -m src.eval_rm \
    --model checkpoints/rm_model_bt \
    --test-data data/processed/rm/rm_valid.jsonl \
    --metrics accuracy ece
```

## Model Architecture
```
SFT Merged Gemma 2B (4-bit, frozen backbone)
    └── LoRA adapters (rank=8)
    └── Reward Head: Linear(2048, 1) → scalar r(x, y)
```

## Loss
```
P(chosen > rejected) = σ(r_chosen − r_rejected)
Loss = −log σ(r_chosen − r_rejected)
```

## Hyperparameters

| Param | Value |
|-------|-------|
| learning_rate | 1e-4 |
| batch_size | 8 |
| epochs | 2 |
| lora_rank | 8 |
| lora_alpha | 16 |
| max_seq_length | 1024 |
| quantize | 4bit |
