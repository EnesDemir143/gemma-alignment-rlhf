---
description: Final comparison analysis of PPO vs GRPO results
---

# Faz 4: Comparison Analysis

## Prerequisites
- Faz 3A completed (PPO models for seeds 42, 123, 777)
- Faz 3B completed (GRPO models for seeds 42, 123, 777)
- OpenAI API key set for GPT-4o-mini evaluation

## Goal
Compare PPO and GRPO using statistical methods and GPT-4o-mini as judge.

## Dosyalar
- `src/eval_final.py` — GPT-4o-mini evaluation with position-swap (henüz yazılmadı)
- `src/compare_results.py` — Statistical comparison + report generation (henüz yazılmadı)

## Steps

### 1. Prepare 300 stratified test prompts
Split from original data (never seen in training):
- 100 factual, 100 instruction, 100 creative
- `seed=42`, held out before any training

### 2. Generate responses from all models
```bash
# Generate responses for each model
uv run python -m src.generate_responses \
    --models checkpoints/ppo_seed{42,123,777} checkpoints/grpo_seed{42,123,777} checkpoints/sft_merged_model \
    --prompts data/processed/rl/test_prompts.jsonl \
    --output results/responses/
```

### 3. GPT-4o-mini evaluation (position-swap debiased)
```bash
uv run python -m src.eval_final \
    --responses results/responses/ \
    --reference-model sft \
    --judge gpt-4o-mini \
    --output results/evaluations/
```

For each prompt:
1. `verdict_1 = judge(actor_resp, sft_resp)` — Actor first
2. `verdict_2 = judge(sft_resp, actor_resp)` — Actor second (swap)
3. Both agree → win/loss; disagree → tie

### 4. Compute statistics
```bash
uv run python -m src.compare_results \
    --ppo-logs logs/ppo_seed{42,123,777}.jsonl \
    --grpo-logs logs/grpo_seed{42,123,777}.jsonl \
    --evaluations results/evaluations/ \
    --output results/comparison_report.pdf
```

### 5. Report metrics

| Metric | Description |
|--------|-------------|
| Win Rate vs SFT | Position-swap debiased |
| Tie Rate | High (>30%) means judge unreliable |
| Win Rate 95% CI | Wilson Score Interval |
| Cohen's d | Primary comparison (effect size) |
| KL Divergence | Policy drift |
| Perplexity | Generation quality |
| Training Time | Hours on M2 Pro |
| Peak VRAM | `mx.metal.get_peak_memory()` |
| Avg Response Length | Verbosity bias check |
| Mean Reward Score | Reward inflation check |

### 6. Interpret Cohen's d

| Cohen's d | Meaning | 3-seed reliability |
|-----------|---------|-------------------|
| \|d\| < 0.2 | Negligible | ✅ Detectable |
| \|d\| 0.2–0.5 | Small | ⚠️ Insufficient power |
| \|d\| 0.5–0.8 | Medium | ⚠️ Borderline |
| \|d\| > 0.8 | Large | ✅ Sufficient power |

### 7. Danger signs to check
- **Verbosity Bias:** PPO response 2x longer than SFT but win rate same → fake success
- **Reward Inflation:** RM scores rising (3→9) but win rate flat → overoptimization
- **High Tie Rate:** >30% → judge can't distinguish quality differences
