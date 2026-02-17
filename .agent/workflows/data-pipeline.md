---
description: Download raw UltraFeedback data and process into training formats
---

# Faz 0: Data Pipeline

## Prerequisites
- `uv` installed
- Project dependencies installed (`uv sync`)

## Steps

### 1. Download raw data (stratified sampling)
// turbo
```bash
uv run python -m src.dataset.download_data --target-size 10000
```
- Output: `data/raw/ultrafeedback_stratified.parquet`
- 10K samples stratified by score_chosen bins

### 2. Process → SFT GenRM format (Faz 1)
// turbo
```bash
uv run python -m src.dataset.process_data --format genrm
```
- Output: `data/processed/sft/sft_genrm_train.jsonl`

### 3. Process → Pairwise format (Faz 2 RM)
// turbo
```bash
uv run python -m src.dataset.process_data --format pairwise
```
- Output: `data/processed/rm/rm_pairwise_train.jsonl`

### 4. Process → Prompts only (Faz 3 PPO/GRPO)
// turbo
```bash
uv run python -m src.dataset.process_data --format prompts
```
- Output: `data/processed/rl/rl_prompts.jsonl`

### 5. Verify outputs
// turbo
```bash
wc -l data/processed/sft/sft_genrm_train.jsonl data/processed/rm/rm_pairwise_train.jsonl data/processed/rl/rl_prompts.jsonl
```
- All three files should have 10,000 lines
