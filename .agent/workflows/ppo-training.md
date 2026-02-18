---
description: PPO training with Actor-Critic and GAE
---

# Faz 3A: PPO Training

## Prerequisites
- Faz 1 completed (`checkpoints/sft_merged_model/`)
- Faz 2 completed (`checkpoints/rm_model_bt/`)
- RL prompts ready (`data/processed/rl/rl_prompts.jsonl`)

## Goal
Train policy with PPO using Critic (value network) + GAE advantage estimation.

## Dosyalar
- `src/train_ppo.py` — PPO training loop (mlx_lm Python API ile custom, henüz yazılmadı)
- `src/models/critic.py` — GemmaCriticModel (henüz yazılmadı)
- `src/algorithms/ppo.py` — PPO loss, GAE, reward shaping (henüz yazılmadı)
- `src/utils/kl_controller.py` — Adaptive KL controller (henüz yazılmadı)

> Tüm scriptler `mlx_lm` Python API'sini doğrudan kullanır:
> `mlx_lm.load()`, `linear_to_lora_layers()`, `mlx.nn`, `mlx.optimizers`

## Steps

### 1. Run PPO training (3 seeds)
```bash
# Seed 42
uv run python -m src.train_ppo \
    --actor-model checkpoints/sft_merged_model \
    --rm-model checkpoints/rm_model_bt \
    --prompts data/processed/rl/rl_prompts.jsonl \
    --seed 42 --num-iterations 800 \
    --clip-range 0.2 --value-coef 0.5 --entropy-coef 0.01 \
    --kl-schedule phased \
    --output checkpoints/ppo_seed42

# Seed 123
uv run python -m src.train_ppo \
    --actor-model checkpoints/sft_merged_model \
    --rm-model checkpoints/rm_model_bt \
    --prompts data/processed/rl/rl_prompts.jsonl \
    --seed 123 --num-iterations 800 \
    --clip-range 0.2 --value-coef 0.5 --entropy-coef 0.01 \
    --kl-schedule phased \
    --output checkpoints/ppo_seed123

# Seed 777
uv run python -m src.train_ppo \
    --actor-model checkpoints/sft_merged_model \
    --rm-model checkpoints/rm_model_bt \
    --prompts data/processed/rl/rl_prompts.jsonl \
    --seed 777 --num-iterations 800 \
    --clip-range 0.2 --value-coef 0.5 --entropy-coef 0.01 \
    --kl-schedule phased \
    --output checkpoints/ppo_seed777
```
> Script internally uses: `mlx_lm.load()` for Actor/Reference, `linear_to_lora_layers()` for Actor LoRA, custom `GemmaCriticModel` with value head, custom PPO loss + GAE implementation in `mlx.core`

### 2. Monitor during training (every 50 iters)
- Perplexity (should decrease)
- KL Divergence (should stay within phased targets: 0.05 → 0.02 → 0.01)
- VRAM usage via `mx.metal.get_peak_memory()`

### 3. Best checkpoint selection logic
```
if eval_kl < target_kl:       # KL gate first
    if eval_ppl < best_ppl:   # then PPL comparison
        save_best_checkpoint()
```

## Key Components

| Component | Role |
|-----------|------|
| Actor | Policy being trained (4-bit + LoRA rank=16) |
| Critic | Value function V(s) for GAE (4-bit + value head) |
| Reference | Frozen SFT model for KL anchor |
| BT RM | Frozen reward model, scores responses |

## PPO Loss
```
L = L_CLIP + 0.5 * L_value - 0.01 * L_entropy
```
- NO KL penalty in loss — adaptive controller handles it externally
- EOS-only reward shaping for GAE compatibility

## Hyperparameters

| Param | Value |
|-------|-------|
| clip_range | 0.2 |
| value_coef | 0.5 |
| entropy_coef | 0.01 |
| ppo_epochs | 1 |
| gae_gamma | 0.99 |
| gae_lambda | 0.95 |
| target_kl | phased (0.05→0.02→0.01) |
| max_grad_norm | 0.5 |
| estimated VRAM | ~4-5 GB |
| estimated time | ~18-24 hours (M2 Pro) |
