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
- `src/split_data.py` — Train/valid split script (mlx_lm API kullanmaz, basit JSONL splitter)
- `src/train_sft.py` — SFT training script (mlx_lm Python API ile custom)
- `src/fuse_model.py` — LoRA adapter merge script (mlx_lm Python API ile custom)

## Steps

### 1. Create train/valid split
```bash
uv run python -m src.split_data --max-seq-length 2048
```
- Input: `data/processed/sft/sft_genrm_train.jsonl`
- Output: `data/processed/sft/sft_train.jsonl`, `data/processed/sft/sft_valid.jsonl` (500 valid)
- Filters out samples exceeding `--max-seq-length` tokens (full chat-template sequence)


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
    --rank 16 \
    --learning-rate 2e-4 \
    --warmup-steps 100 \
    --max-seq-length 2048 \
    --adapter-path checkpoints/sft_adapter \
    --mask-prompt \
    --grad-checkpoint
```
> Script internally uses: `mlx_lm.load()`, `linear_to_lora_layers()`, `TrainingArgs`, `trainer.train()`, `CacheDataset`

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
> Script internally uses: `mlx_lm.load()` with adapter_path, `m.fuse()`, `mlx_lm.fuse.save()`

### 5. Verify merged model
- Output: `checkpoints/sft_merged_model/`
- This model becomes the **reference point** for both PPO and GRPO

## Hyperparameters

| Param | Value |
|-------|-------|
| learning_rate | 2e-4 |
| warmup_steps | 100 |
| batch_size | 4 |
| grad_accum | 4 (effective=16) |
| iters | 5000 |
| max_seq_length | 2048 |
| rank | 16 |
| lora_layers | 16 |
| mask_prompt | true |
| grad_checkpoint | true |
| optimizer | adam |
