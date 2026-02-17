---
description: SFT fine-tuning with QLoRA on GenRM format data
---

# Faz 1: SFT (Supervised Fine-Tuning)

## Prerequisites
- Faz 0 completed (`data/processed/sft/sft_genrm_train.jsonl` exists)
- Gemma 2B model downloaded

## Goal
Train Gemma 2B as a Generative Reward Model (GenRM) — model learns to score response quality.

## Dosyalar
- `src/train_sft.py` — SFT training script (henüz yazılmadı)
- `src/fuse_model.py` — LoRA adapter merge script (henüz yazılmadı)

## Steps

### 1. Create train/valid split
Split `sft_genrm_train.jsonl` into train (90%) and valid (10%) sets with `seed=42`.
- Output: `data/processed/sft/sft_train.jsonl`, `data/processed/sft/sft_valid.jsonl`

### 2. Train SFT model with QLoRA
```bash
uv run python -m src.train_sft \
    --model google/gemma-2b-it \
    --train-data data/processed/sft/sft_train.jsonl \
    --valid-data data/processed/sft/sft_valid.jsonl \
    --iters 5000 \
    --batch-size 4 \
    --grad-accum 4 \
    --lora-layers 16 \
    --lora-rank 16 \
    --learning-rate 2e-4 \
    --warmup-steps 100 \
    --max-seq-length 512 \
    --adapter-path checkpoints/sft_adapter
```

### 3. Monitor training
- Watch perplexity decreasing (target: 3.5 → 2.8)
- Check `checkpoints/sft_adapter/` for saved adapter weights

### 4. Fuse LoRA adapter into base model
```bash
uv run python -m src.fuse_model \
    --model google/gemma-2b-it \
    --adapter-path checkpoints/sft_adapter \
    --save-path checkpoints/sft_merged_model
```

### 5. Verify merged model
- Output: `checkpoints/sft_merged_model/`
- This model becomes the **reference point** for both PPO and GRPO

## Hyperparameters

| Param | Value |
|-------|-------|
| learning_rate | 2e-4 |
| batch_size | 4 |
| grad_accum | 4 (effective=16) |
| epochs | 3 |
| warmup_steps | 100 |
| max_seq_length | 512 |
| lora_rank | 16 |
| lora_layers | 16 |
| quantize | 4bit |
