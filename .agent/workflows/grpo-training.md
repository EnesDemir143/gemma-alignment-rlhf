---
description: GRPO training with group-relative advantage estimation
---

# Faz 3B: GRPO Training

## Prerequisites
- Faz 1 completed (`checkpoints/sft_merged_model/`)
- Faz 2 completed (`checkpoints/rm_model_bt/`)
- RL prompts ready (`data/processed/rl/rl_prompts.jsonl`)

## Goal
Train policy with GRPO — no Critic needed, uses group-relative advantage from K=6 sampled responses.

## Dosyalar
- `src/train_grpo.py` — GRPO training loop (henüz yazılmadı)
- `src/algorithms/grpo.py` — GRPO loss, group advantage, ScoreNormalizer (henüz yazılmadı)
- `src/utils/kl_controller.py` — Adaptive KL controller (PPO ile paylaşılır)

## Steps

### 1. Run GRPO training (3 seeds)
```bash
# Seed 42
uv run python -m src.train_grpo \
    --actor-model checkpoints/sft_merged_model \
    --rm-model checkpoints/rm_model_bt \
    --prompts data/processed/rl/rl_prompts.jsonl \
    --seed 42 --K 6 --num-iterations 800 \
    --clip-range 0.2 --kl-schedule phased \
    --output checkpoints/grpo_seed42

# Seed 123
uv run python -m src.train_grpo \
    --actor-model checkpoints/sft_merged_model \
    --rm-model checkpoints/rm_model_bt \
    --prompts data/processed/rl/rl_prompts.jsonl \
    --seed 123 --K 6 --num-iterations 800 \
    --clip-range 0.2 --kl-schedule phased \
    --output checkpoints/grpo_seed123

# Seed 777
uv run python -m src.train_grpo \
    --actor-model checkpoints/sft_merged_model \
    --rm-model checkpoints/rm_model_bt \
    --prompts data/processed/rl/rl_prompts.jsonl \
    --seed 777 --K 6 --num-iterations 800 \
    --clip-range 0.2 --kl-schedule phased \
    --output checkpoints/grpo_seed777
```

### 2. Monitor during training (every 50 iters)
- Perplexity (should decrease)
- KL Divergence (phased targets: 0.05 → 0.02 → 0.01)
- VRAM usage (should be ~2.5 GB, less than PPO)

### 3. Best checkpoint selection logic
Same as PPO: KL gate → PPL comparison.

## Key Components

| Component | Role |
|-----------|------|
| Actor | Policy being trained (4-bit + LoRA rank=16) |
| Reference | Frozen SFT model for KL anchor |
| BT RM | Frozen reward model, scores K responses |
| ScoreNormalizer | EMA-based drift prevention (α=0.99) |

## GRPO Advantage
```
Per prompt: generate K=6 responses → RM scores → Aᵢ = rᵢ - mean(r)
```

## GRPO Loss
```
L = L_CLIP   (no value loss, no entropy — just clipped surrogate)
```
- NO KL penalty in loss — adaptive controller handles it
- NO Critic — group-relative advantage replaces GAE

## Hyperparameters

| Param | Value |
|-------|-------|
| K (group size) | 6 |
| clip_range | 0.2 |
| target_kl | phased (0.05→0.02→0.01) |
| normalizer_alpha | 0.99 |
| max_grad_norm | 0.5 |
| estimated VRAM | ~2.5 GB |
| estimated time | ~12-15 hours (M2 Pro) |
