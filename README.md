# 🔬 Gemma Alignment: PPO vs GRPO Comparison Study

A controlled experiment comparing two RLHF alignment methods — **PPO** (Proximal Policy Optimization) and **GRPO** (Group Relative Policy Optimization) — through a full fine-tuning pipeline on `Gemma 2B-IT` using UltraFeedback preference data.

## 🎯 What This Project Does

This project implements a **complete RLHF pipeline from scratch** and runs both alignment algorithms under identical conditions to produce a fair, reproducible comparison.

## 🔗 Pipeline Overview

```text
Gemma 2B (4-bit)
    │
    ▼
 SFT (QLoRA) ──► sft_merged_model
    │
    ├──► Bradley-Terry RM ──► rm_model_bt
    │
    ├──► PPO  (Actor + Critic + GAE)
    │
    └──► GRPO (Group Sampling, K=6)
                │
                ▼
        📊 Comparison (Cohen's d + GPT-4o-mini)
```

### Pipeline Phases

| Phase | Technique | Purpose |
|-------|-----------|---------|
| **1. SFT** | QLoRA (4-bit, rank 16) | Warm-up fine-tuning on chosen responses |
| **2. Reward Model** | Bradley-Terry (pairwise preference) | Learn scalar reward from chosen/rejected pairs |
| **3A. PPO** | Actor-Critic + GAE + Clipped Surrogate | Alignment with learned value function |
| **3B. GRPO** | Group Sampling (K=6) + Clipped Surrogate | Critic-free alignment via group-relative advantages |

> [!NOTE]
> All phases are implemented via custom training loops (not CLI tools) to ensure full control over the PPO/GRPO logic and observability.

## 📦 Dataset

- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — A processed version of UltraFeedback containing binarized preference pairs (chosen/rejected).

### Data Transformation (`download_data.py`)

The `download_data.py` script downloads the `HuggingFaceH4/ultrafeedback_binarized` dataset, samples a subset, and transforms it into the GenRM-style JSONL format for SFT training:

```
HuggingFaceH4/ultrafeedback_binarized      train.jsonl
┌──────────────────────┐                   ┌──────────────────────────────────┐
│ prompt               │──┐                │ messages[0] (role: "user")       │
│ chosen[-1].content   │──┤── concat ──►   │   "User: {prompt}\n\n            │
│                      │  │                │    Assistant: {chosen}\n\n       │
│                      │  │                │    Analyze the quality..."       │
│ score_chosen         │──┘── format ──►   │ messages[1] (role: "assistant")  │
│                      │                   │   "Score: {score}/10. ..."       │
└──────────────────────┘                   └──────────────────────────────────┘
```

| `train.jsonl` Field | Source | Content |
|---------------------|--------|---------|
| `messages[0]` (user) | `prompt` + `chosen` response | Original prompt + chosen response + "Analyze the quality..." instruction |
| `messages[1]` (assistant) | `score_chosen` | `"Score: {score:.1f}/10. The response is helpful, harmless, and honest."` |
| Score | `score_chosen` only | Parsed via regex `Score:\s*([0-9]+(?:\.[0-9]+)?)/10` |

> [!NOTE]
> Only `score_chosen` is used — the rejected response score is not included in the training data. The "user" message contains both the prompt **and** the chosen response concatenated together.

## 🔑 Key Techniques

### Bradley-Terry Reward Model

Pairwise preference modeling trained on UltraFeedback `chosen` / `rejected` pairs. Learns a scalar reward function `r(prompt, response)` that scores any generation, providing the training signal for both PPO and GRPO.

> [!NOTE]
> **Supervised Training:** This phase trains the "Judge" (Score Model) using labeled data, preparing it to guide the "Player" (Actor Model) in the subsequent RL phases.

### PPO (Proximal Policy Optimization)

- **Actor-Critic** architecture with a separate value head for advantage estimation
- **GAE** (Generalized Advantage Estimation, λ=0.95) for low-variance advantage computation
- **EOS-only reward shaping** to bridge scalar RM output with per-token credit assignment
- **Adaptive KL controller** with phased schedule (0.05 → 0.02 → 0.01) to prevent policy drift

> [!TIP]
> **Why PPO?** It uses a "Critic" model to predict scores and "Clipping" to prevent dangerous policy updates, ensuring safe and stable learning.

### GRPO (Group Relative Policy Optimization)

- **Critic-free** — no value network needed, reducing memory overhead
- Generates **K=6** responses per prompt, computes group-relative advantages: `Aᵢ = rᵢ − mean(r)`

> [!TIP]
> **Why GRPO?** It removes the "Critic" model entirely, saving massive VRAM. Instead of predicting scores, it generates multiple responses and uses the *group average* as the baseline.
- **EMA-based score normalization** to prevent reward drift during training
- Same adaptive KL controller and clipped surrogate objective as PPO

## 📊 Comparison Design

Both methods are evaluated under **identical controlled conditions**:

- Same SFT base model and frozen Bradley-Terry RM
- Same training data, hyperparameters, and iteration count
- **3 random seeds** `[42, 123, 777]` per method → 6 total experiment runs
- **Cohen's d** as primary effect size metric (robust with small N)

> [!IMPORTANT]
> **Effect Size vs Significance:** With only 3 seeds, statistical significance (p-value) is hard to prove. **Cohen's d** measures the *magnitude* of the difference (Effect Size), telling us if the win is "meaningful" (d>0.8) or "noisy gain" (d<0.2).

### Evaluation Protocol

| Stage | Metrics |
|-------|---------|
| **During Training** (every 50 iter) | Perplexity, KL Divergence, **Response Length & Reward Score Analysis** |
| **Final Evaluation** | GPT-4o-mini win/loss/tie rate (position-swap debiased) + **95% Confidence Interval** |
| **Statistical Comparison** | Cohen's d across seeds |

> Position-swap debiasing: each comparison is judged in both A-B and B-A order; inconsistencies are counted as ties to eliminate position bias.

## 📈 Expected Trade-offs

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Stability** | ✅ High (GAE + Critic) | ⚠️ Medium (group variance) |
| **Memory** | ⚠️ ~4–5 GB (Critic overhead) | ✅ ~2.5 GB |
| **Complexity** | ⚠️ High (Critic + GAE) | ✅ Medium (group sampling) |
| **HP Sensitivity** | ⚠️ High (clip, value_coef, λ) | ✅ Medium (K, kl_coef) |
| **Final Quality** | TBD | TBD |

## 🗂️ Project Structure

```
├── src/                  # Training scripts & core modules
├── notebooks/            # EDA & analysis notebooks
├── docs/
│   └── pipeline.md       # Full technical specification
├── data/                 # Dataset cache
├── artifacts/            # Checkpoints & experiment logs
└── pyproject.toml
```

## � References

1. Schulman et al. — *Proximal Policy Optimization Algorithms* (2017)
2. Schulman et al. — *High-Dimensional Continuous Control Using GAE* (2016)
3. DeepSeek-Math — *GRPO* (2024)
4. Bradley & Terry — *Rank Analysis of Incomplete Block Designs* (1952)
5. Cui et al. — *UltraFeedback* (2023)
