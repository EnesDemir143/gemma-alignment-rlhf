---

# 🍎 MLX-Gemma Pipeline: PPO vs GRPO Comparison Study (2026)

**Hedef Donanım:** Apple Silicon (M1/M2/M3)  
**Kütüphane:** `mlx`, `mlx-lm` (Python)  
**Ana Model:** `google/gemma-2b-it` (4-bit Quantized)  
**Veri Seti:** `HuggingFaceH4/ultrafeedback_binarized`  
**Alignment Methods:** **PPO** (Proximal Policy Optimization) vs **GRPO** (Group Relative Policy Optimization)  
**Reward Model:** **Bradley-Terry** (Pairwise Preference, chosen/rejected ile eğitilir)

---

## 🔬 **Experimental Design: PPO vs GRPO**

Bu pipeline, PPO ve GRPO algoritmalarını kontrollü bir ortamda karşılaştırmak için tasarlanmıştır. Sonuçlar deneysel verilerle belirlenir.

### Kontrollü Değişkenler (Sabit Tutulanlar)

| Değişken | Değer |
|----------|-------|
| **SFT Base Model** | `sft_merged_model` (Faz 1 çıktısı) |
| **Reward Model** | `rm_model_bt` (Faz 2 çıktısı, Bradley-Terry) |
| **Training Data** | `HuggingFaceH4/ultrafeedback_binarized` |
| **Evaluation Data** | 300 stratified prompt (100 factual, 100 instruction, 100 creative) |
| **Random Seeds** | `[42, 123, 777]` (her yöntem için 3 run) |
| **Max Sequence Length** | 512 |
| **Base Learning Rate** | 5e-6 |
| **Batch Size** | 16 |
| **Num Iterations** | 800 |
| **KL Target (Phased)** | 0.05 → 0.02 → 0.01 (aşamalı) |

### Değişen Değişkenler (Yalnızca Bunlar)

| Değişken | PPO (Faz 3A) | GRPO (Faz 3B) |
|----------|-------------|--------------|
| **Algorithm** | PPO + Critic | GRPO (Group-based) |
| **Advantage Estimation** | GAE (λ=0.95, γ=0.99) | Group-relative |
| **Critic Model** | Var (Value function) | Yok |
| **clip_range** | 0.2 | 0.2 |
| **value_coef** | 0.5 | — |
| **entropy_coef** | 0.01 | — |
| **Group Size K** | — | 6 |

### Success Metrics

| Metrik | Açıklama | Ölçüm Yöntemi |
|--------|----------|---------------|
| **Win Rate vs SFT** | Final modelin SFT baseline'ı yenme oranı | **GPT-4o-mini** (position-swap debiased) |
| **Tie Rate** | Position-swap tutarsızlık oranı (yüksek → judge güvenilmez) | GPT-4o-mini (A-B ≠ B-A → tie) |
| **KL Divergence** | Policy drift (düşük = daha stabil) | Ortalama KL(actor ‖ reference) |
| **Perplexity** | Üretim kalitesi | Log-likelihood |
| **Training Time** | Toplam eğitim süresi | Saat (M2 Pro 16GB) |
| **Peak VRAM** | Maksimum bellek kullanımı | `mx.metal.get_peak_memory()` |
| **Convergence Speed** | Win rate %60'a ulaşma iterasyonu | Training curve |
| **Variance Across Runs** | 3 seed arasındaki std sapması | Std(win_rate) |
| **Cohen's d** | Effect size (birincil karşılaştırma metriği) | Standardized mean difference |

---

## 🏁 **FAZ 1: SFT (Supervised Fine-Tuning)**

Modelin talimatları anlaması ve UltraFeedback kalitesine alışması için yapılan ısınma turu. Bu faz her iki yöntem için **ortaktır**.

* **Girdi:** UltraFeedback veri setindeki `chosen` cevaplar.
* **Teknoloji:** **QLoRA** (Quantized Low-Rank Adaptation).

### 🧠 Model Durumu (Phase 1)

| Model | Tip | Durum | MLX Yapılandırması |
|-------|-----|-------|-------------------|
| **Gemma 2B** | Actor | 🔴 **EĞİTİLİYOR (LoRA)** | `--quantize 4bit`, `--rank 16`, `--lora-layers 16` |

### 📊 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| `learning_rate` | 2e-4 |
| `batch_size` | 4 |
| `gradient_accumulation_steps` | 4 (efektif batch = 16) |
| `epochs` | 3 |
| `warmup_steps` | 100 |
| `max_seq_length` | 512 |

> **Çıktı:** `sft_adapter.npz` → Base Gemma'ya merge edilerek `sft_merged_model` oluşur.

**⚠️ KRİTİK:** Bu model her iki alignment yöntemi için de **reference point**'tir. Aynı `sft_merged_model` hem PPO hem GRPO başlangıcında kullanılır.

---

## ⚖️ **FAZ 2: Bradley-Terry Reward Model Eğitimi**

> [!IMPORTANT]
> **Pairwise Preference Modeling.** UltraFeedback'teki `chosen` / `rejected` çiftleri kullanılarak Bradley-Terry modeli eğitilir. Eğitilen RM, training sırasında her üretilen cevaba scalar skor verir.

### 📦 Veri Formatı (Pairwise)

```json
{
  "prompt": "Python'da liste nasıl sıralanır?",
  "chosen": "list.sort() veya sorted(list) kullanabilirsiniz...",
  "rejected": "Python'da liste yok, sadece array var..."
}
```

### 🧠 Bradley-Terry Model Yapısı

Model, SFT checkpoint'inden başlayan bir Gemma backbone'u + scalar reward head'den oluşur:

```python
class BradleyTerryRM:
    def __init__(self, base_model_path):
        self.gemma = load_model_4bit(base_model_path)     # Shared backbone
        self.reward_head = nn.Linear(2048, 1)              # Scalar r(x, y)

    def __call__(self, tokens):
        hidden = self.gemma(tokens, output_hidden_states=True)
        return self.reward_head(hidden[-1][:, -1, :]).squeeze(-1)

    def get_reward(self, prompt, response) -> float:
        """Tek bir (prompt, response) çifti için scalar reward."""
        ...
```

### 🧩 Bradley-Terry Loss

```
P(chosen > rejected) = σ(r_chosen − r_rejected)
Loss = −log σ(r_chosen − r_rejected)
```

```python
def bradley_terry_loss(rm_model, prompt, chosen, rejected):
    r_chosen  = rm_model.get_reward(prompt, chosen)
    r_rejected = rm_model.get_reward(prompt, rejected)
    return -mx.log(mx.sigmoid(r_chosen - r_rejected) + 1e-8)
```

### 🔄 RM Training

`train_bradley_terry_rm(rm_model, dataset, epochs, lr, batch_size)` fonksiyonu ile UltraFeedback chosen/rejected çiftleri üzerinde eğitilir. Her epoch sonunda pairwise accuracy raporlanır.

### 📊 RM Hiperparametreleri

| Parametre | Değer |
|-----------|-------|
| `learning_rate` | 1e-4 |
| `batch_size` | 8 |
| `epochs` | 2 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `max_seq_length` | 1024 |

> **Çıktı:** `rm_model_bt` → Bradley-Terry RM. Eğitim sonrasında pairwise accuracy ≥ 0.70 beklenir.

---

## ⚖️ **FAZ 3A: PPO Implementation**

Proximal Policy Optimization, bir **Critic (value) network** kullanarak advantage tahminini öğrenen bir actor-critic yöntemidir. GAE ile varyansı kontrol altında tutar.

### 🎯 PPO Akışı

```
Prompt → Actor.generate(response) → BT RM → scalar reward
                                  → EOS-Only Reward Shaping (aşağıya bkz.)
                                  → Critic → V(s_t) (her token için)
                                  → GAE: A_t = Σ(γλ)^k · δ_{t+k}
                                  → PPO Clip: L = E[min(ratio·A, clip(ratio,1±ε)·A)]
```

### 🛠️ Critic Model

Actor ile aynı backbone'u paylaşan, ayrı bir value head'e sahip ağ. Her state'ten beklenen kümülatif ödülü tahmin eder:

```python
class GemmaCriticModel:
    def __init__(self, base_model_path):
        self.gemma = load_model_4bit(base_model_path)
        self.value_head = nn.Linear(2048, 1)  # V(s) tahmini

    def __call__(self, tokens) -> scalar:
        ...
```

### 🔢 Reward Shaping: EOS-Only Assignment

> [!IMPORTANT]
> **BT RM tüm response için tek bir scalar skor verir.** Ancak GAE, her token adımı (t) için ayrı bir `r_t` bekler. Bu uyumsuzluğu çözmek için standart yöntem kullanılır: **reward sadece son token'a (EOS) atanır**, ara token'larda `r_t = 0`.

```python
def shape_rewards_for_gae(rm_scalar_reward, response_length):
    """
    BT RM'nin tek scalar reward'ını GAE-uyumlu per-token reward dizisine çevirir.
    r_t = 0  (t < T-1)    — ara token'lar
    r_T = rm_reward        — EOS token
    """
    rewards = [0.0] * (response_length - 1) + [rm_scalar_reward]
    return rewards
```

Bu sayede GAE, EOS'a kadar olan tüm token'lar için advantage'ı geriye doğru yayar (temporal credit assignment).

### 🔢 GAE (Generalized Advantage Estimation)

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    A_t = δ_t + (γλ)·(1-done)·A_{t+1}
    Sonuç normalize edilir (mean=0, std=1).
    """
    ...
```

### 🔄 PPO Training Step

Her iterasyonda:
1. **Rollout:** Her prompt için 1 response üretilir, BT RM'den scalar reward alınır.
2. **Reward Shaping:** Scalar reward → EOS-only per-token reward dizisine çevrilir.
3. **Critic:** Her token pozisyonunda V(s_t) tahmin edilir.
4. **GAE:** Per-token rewards + values → advantages hesaplanır.
5. **Policy Update (mini-batch):**

```python
# PPO Clipped Surrogate Objective
ratio = exp(actor_logprobs - old_logprobs)
L_CLIP = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A).mean()

# Value Loss
L_value = ((V_current - returns)²).mean()

# Entropy Bonus
L_entropy = -(logp * exp(logp)).mean()

# Total (KL penalty loss'ta YOK — adaptive controller zaten β'yi dışarıdan ayarlıyor)
Loss = L_CLIP + 0.5 * L_value - 0.01 * L_entropy
```

> [!WARNING]
> **KL kontrolü yalnızca adaptive controller ile yapılır** (aşağıya bkz.). Loss'a ayrıca KL penalty terimi eklenmez — ikisi aynı anda çalışırsa policy neredeyse hiç hareket edemez (double suppression).

6. **Adaptive KL Controller:** Phased schedule (0.05 → 0.02 → 0.01) etrafında `kl_coef` dışarıdan ayarlanır. KL yüksekse learning rate düşürülür veya early stop uygulanır.
7. **Best Checkpoint (KL-Gate → PPL):**

> [!IMPORTANT]
> PPL ve KL tamamen farklı ölçeklerde (PPL: 2–5, KL: 0.01–0.05). İkisini toplamak anlamsız. Doğru mantık: **önce KL threshold'u geçip geçmediğine bak**, geçiyorsa o checkpoint'i atla, geçmiyorsa PPL'e göre kaydet.

```python
# Best checkpoint seçimi: KL gate THEN PPL comparison
if eval_kl < target_kl:          # 1. KL threshold'u geçiyor mu?
    if eval_ppl < best_ppl:      # 2. Evet → PPL daha iyi mi?
        best_ppl = eval_ppl
        save_best_checkpoint()
```

### 🧠 Model Durumları (Phase 3A — PPO)

| Model | Rolü | Durumu | Tahmini VRAM |
|-------|------|--------|-------------|
| **Actor** | Policy | 🔴 EĞİTİLİYOR (4-bit + LoRA rank=16) | ~1.5 GB |
| **Critic** | Value Function | 🔴 EĞİTİLİYOR (4-bit + Value Head) | ~1.5 GB |
| **Reference** | KL Anchor | 🧊 FROZEN (SFT checkpoint) | ~1.5 GB |
| **BT RM** | Reward | 🧊 FROZEN (Faz 2 çıktısı) | ~0.5 GB |

**Tahmini Toplam VRAM:** ~4–5 GB

### 📊 PPO Hiperparametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `clip_range` | 0.2 | ε — policy ratio clipping |
| `value_coef` | 0.5 | Value loss weight |
| `entropy_coef` | 0.01 | Entropy bonus |
| `ppo_epochs` | 1 | Single-use rollouts (GRPO ile eşit) |
| `gae_gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ |
| `target_kl` | phased | 0.05 → 0.02 → 0.01 (adaptive controller) |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `reward_shaping` | EOS-only | BT RM scalar → son token'a atanır |

---

## ⚖️ **FAZ 3B: GRPO Implementation**

Group Relative Policy Optimization, her prompt için bir grup cevap üretip bu grup içindeki **göreli karşılaştırma** ile advantage hesaplar. Critic ağı gerektirmez.

### 🎯 GRPO Akışı

```
Prompt → Actor.generate(K=6 responses) → BT RM → [r₁, r₂, ..., r₆]
                                        → Score Normalization (EMA)
                                        → Group-relative advantage: Aᵢ = rᵢ - mean(r)
                                        → Clipped Surrogate: L = -min(ratio·A, clip(ratio,1±ε)·A)
```

**Örnek:**
```
Responses:   [4.2, 3.8, 2.1, 1.3, 4.5, 2.8]  (RM scores)
Group Mean:  3.12
Advantages:  [+1.08, +0.68, -1.02, -1.82, +1.38, -0.32]
```

### 🔢 Score Normalization (Drift Önleme)

> [!CAUTION]
> RM skorları GRPO sırasında **drift edebilir**. EMA-based running normalization ile önlüyoruz.

```python
class ScoreNormalizer:
    """EMA ile running mean/std güncelleyerek score distribution'ı korur."""
    def __init__(self, alpha=0.99):
        self.running_mean, self.running_std = 3.0, 1.0
        self.alpha = alpha

    def normalize(self, scores) -> mx.array:
        # EMA update → normalize → rescale [1.0, 5.0] aralığına clip
        ...
```

### 🔄 GRPO Training Step

Her iterasyonda:
1. **Group Sampling:** Her prompt için K=6 response üretilir, old policy logprobs kaydedilir.
2. **Reward:** BT RM'den skorlar alınır, `ScoreNormalizer` ile normalize edilir.
3. **Group-Relative Advantage:** `Aᵢ = normalized_rᵢ - mean(normalized_r)`
4. **Clipped Surrogate Update (mini-batch):**

```python
# GRPO Clipped Surrogate (PPO-style, Critic yok)
ratio = exp(actor_logprobs - old_logprobs)
L_CLIP = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A).mean()

# KL penalty loss'ta YOK — adaptive controller dışarıdan yönetir
Loss = L_CLIP
```

> [!WARNING]
> PPO ile aynı şekilde, KL kontrolü **yalnızca adaptive controller** ile yapılır. Loss'a ayrıca KL terimi eklenmez (double suppression riski).

5. **Adaptive KL Controller:** PPO ile aynı phased schedule paylaşılır. KL > target ise learning rate düşürülür veya early stop.
6. **Best Checkpoint (KL-Gate → PPL):** PPO ile aynı mantık — önce `KL < target_kl` kontrolü, sonra PPL karşılaştırması.

### 🧠 Model Durumları (Phase 3B — GRPO)

| Model | Rolü | Durumu | Tahmini VRAM |
|-------|------|--------|-------------|
| **Actor** | Policy | 🔴 EĞİTİLİYOR (4-bit + LoRA rank=16) | ~1.5 GB |
| **Reference** | KL Anchor | 🧊 FROZEN (SFT checkpoint) | ~1.5 GB |
| **BT RM** | Reward | 🧊 FROZEN (Faz 2 çıktısı) | ~0.5 GB |

**Tahmini Toplam VRAM:** ~2.5 GB

### 📊 GRPO Hiperparametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `K` | 6 | Group size |
| `clip_range` | 0.2 | Policy ratio clipping |
| `target_kl` | phased | 0.05 → 0.02 → 0.01 (adaptive controller) |
| `normalizer_alpha` | 0.99 | EMA coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |

---

## 📊 **Comparison Protocol**

Her iki yöntem **eşdeğer koşullar** altında değerlendirilir.

> [!IMPORTANT]
> **Train/Test Split:** UltraFeedback, training başlamadan önce sabit seed (`seed=42`) ile split edilir. 300 test promptu training'de hiç kullanılmaz.

### Evaluation Schedule

- **Training sırası (her 50 iter):** Sadece **Perplexity** ve **KL Divergence** raporlanır.
- **Final eval (eğitim sonunda):** **GPT-4o-mini** ile **win rate**, **loss rate** ve **tie rate** birlikte raporlanır.
- 3 seed × 2 yöntem = **6 experiment run**.

### Final Evaluation (GPT-4o-mini — Position-Swap Debiased)

> [!IMPORTANT]
> Position bias önlemek için her karşılaştırma iki farklı sırayla yapılır (A-B ve B-A). Tutarsızlık varsa **TIE** sayılır.

```python
def final_evaluate_gpt4o(actor, reference, test_prompts):
    """
    Her prompt için:
      1. verdict_1 = gpt4o_judge(actor_resp, sft_resp)   # Actor=A
      2. verdict_2 = gpt4o_judge(sft_resp, actor_resp)   # Actor=B (swap)
      3. İkisi tutarlı → win/loss; tutarsız → tie
    Stratified kategorilere göre ayrı raporlama (factual, instruction, creative).
    Raporda win/loss/tie üçlüsü birlikte verilir — sadece win rate yanıltıcı olabilir.
    """
    ...
```

> [!NOTE]
> **Tie rate yüksekse** (>%30) GPT-4o-mini judge'ın ayrım gücü düşük demektir — bu durumda win rate yorumları temkinli yapılmalıdır.

### Statistical Significance

3 seed ile Welch t-test'in gücü düşük olduğundan **Cohen's d** birincil karşılaştırma kriteri olarak kullanılır. Ancak **3 seed yalnızca large effect size'ı güvenilir tespit eder**; medium ve small farklar için sonuçlar kesin değildir.

| Cohen's d | Yorum | 3 Seed ile Güvenilirlik |
|-----------|-------|------------------------|
| \|d\| < 0.2 | Negligible | ✅ Tespit edilebilir |
| \|d\| 0.2–0.5 | Small | ⚠️ Yetersiz güç, kesin değil |
| \|d\| 0.5–0.8 | Medium | ⚠️ Sınırda, temkinli yorumla |
| \|d\| > 0.8 | Large | ✅ Yeterli güç |

```python
def cohens_d(group1, group2):
    """Pooled std ile standardized mean difference."""
    ...
```

> [!WARNING]
> 3 seed ile istatistiksel güç düşüktür — yalnızca **large** effect size güvenilir tespit edilir. Sonuçlar **medium veya small** çıkarsa, seed sayısı artırılarak (5–10 seed) deney tekrarlanabilir.

---

## 🎯 **Expected Trade-offs Table**

Aşağıdaki tablo teorik beklentilere dayanmaktadır. Gerçek sonuçlar deneysel olarak belirlenecektir.

| Metrik | PPO | GRPO | Beklenen Avantaj |
|--------|-----|------|-----------------|
| **Variance (across runs)** | Düşük (Critic stabilize eder) | Orta-Yüksek (stochastic sampling) | PPO |
| **Training Stability** | Yüksek (GAE + value clipping) | Orta (group size'a bağlı) | PPO |
| **VRAM Kullanımı** | ~4–5 GB (Critic ekler) | ~2.5 GB (Critic yok) | GRPO |
| **Compute per Iteration** | Yüksek (Critic forward + backward) | Orta (K parallel samples) | GRPO |
| **Convergence Speed** | Belirsiz | Belirsiz | TBD |
| **Final Win Rate** | Belirsiz | Belirsiz | TBD |
| **Hyperparameter Sensitivity** | Yüksek (clip_range, value_coef, λ) | Orta (K, kl_coef) | GRPO |
| **Implementation Complexity** | Yüksek (Critic + GAE) | Orta (Group sampling) | GRPO |
| **Sample Efficiency** | Eşit (`ppo_epochs=1`, reuse disabled — fair comparison) | Eşit (rollouts tek kullanım) | Eşit |

---

## 💡 **M2 Pro & MLX Optimizasyonları**

| Optimizasyon | Açıklama |
|-------------|----------|
| **4-bit Quantization** | Tüm modeller 4-bit ile yüklenir |
| **Gradient Checkpointing** | `mx.checkpoint(model, checkpoints=8)` ile VRAM tasarrufu |
| **Dynamic Batch Sizing** | PPO: 8 (Critic VRAM), GRPO: 16 (available memory'ye göre) |
| **Memory Profiler** | Her 50 step'te `mx.metal.get_active_memory()` ile real-time monitoring |
| **LoRA** | Rank=16, sadece belirli layer'lar eğitilir |

---

## 🚀 **Başlangıç Komutları**

### Faz 1: SFT (Ortak)

```bash
python -m mlx_lm.lora \
    --model google/gemma-2b-it \
    --data HuggingFaceH4/ultrafeedback_binarized \
    --train --iters 5000 --batch-size 4 --lora-layers 16 \
    --rank 16 --learning-rate 2e-4 --quantize 4bit \
    --adapter-path ./sft_adapter

python -m mlx_lm.fuse \
    --model google/gemma-2b-it \
    --adapter-path ./sft_adapter \
    --save-path ./sft_merged_model
```

### Faz 2: RM Training (Bradley-Terry)

```bash
python train_rm_bt.py \
    --model ./sft_merged_model \
    --data HuggingFaceH4/ultrafeedback_binarized \
    --batch-size 8 --epochs 2 --learning-rate 1e-4 \
    --lora-rank 8 --quantize 4bit --output ./rm_model_bt
```

### Faz 3: PPO / GRPO Training

```bash
# PPO (seed=42)
python train_ppo.py \
    --actor-model ./sft_merged_model --rm-model ./rm_model_bt \
    --seed 42 --num-iterations 800 --clip-range 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --kl-schedule phased \
    --output ./ppo_model_seed42

# GRPO (seed=42)
python train_grpo.py \
    --actor-model ./sft_merged_model --rm-model ./rm_model_bt \
    --seed 42 --K 6 --num-iterations 800 --clip-range 0.2 \
    --kl-schedule phased --output ./grpo_model_seed42

# Diğer seed'ler için --seed ve --output değiştir: [42, 123, 777]
```

### Karşılaştırma Analizi

```bash
python compare_results.py \
    --ppo-logs  ./logs/ppo_seed{42,123,777}.jsonl \
    --grpo-logs ./logs/grpo_seed{42,123,777}.jsonl \
    --metrics win_rate_vs_sft kl_divergence perplexity training_time vram_peak \
    --output ./comparison_report.pdf
```

---

## 📈 **Beklenen Sonuçlar**

| Faz | Metrik | Beklenti |
|-----|--------|----------|
| **1. SFT** | Perplexity | 3.5 → 2.8 (~4 saat) |
| **2. RM** | Pairwise Accuracy | ≥ 0.70 (~2 saat) |
| **3. PPO** | Win Rate, KL, VRAM | TBD (~18–24 saat, ~4–5 GB VRAM) |
| **4. GRPO** | Win Rate, KL, VRAM | TBD (~12–15 saat, ~2.5 GB VRAM) |

> **Not:** Nihai karşılaştırma sonuçları, comparison protocol tamamlandıktan sonra eklenecektir.

---

## ⚠️ **Bilinen Limitasyonlar**

**Her iki yöntem için:**
- **Offline Data:** UltraFeedback statik, iterative data improvement sınırlı.
- **RM Bağımlılığı:** Bradley-Terry RM kalitesi her iki yöntemin başarısını belirler.
- **Score Drift:** Running normalization gerekli.

**PPO'ya özgü:**
- Critic overfitting riski (küçük batch)
- İki model koordinasyonu (Actor + Critic güncelleme sırası)

**GRPO'ya özgü:**
- Group size trade-off (K büyük → yavaş, K küçük → yüksek variance)
- Run-to-run tutarsızlık (stochastic group sampling)

---

## 🧩 **Pipeline Summary**

| Faz | Girdi | Teknik | Çıktı |
|-----|-------|--------|-------|
| **1. SFT** | UltraFeedback (Chosen) | QLoRA (4-bit, rank=16) | `sft_merged_model` |
| **2. RM** | UltraFeedback (Chosen/Rejected) | Bradley-Terry Loss | `rm_model_bt` |
| **3. PPO** | Prompts + RM | Actor-Critic + GAE + PPO Clip | `ppo_model_best` |
| **4. GRPO** | Prompts + RM | Group Sampling (K=6) + Clipped Surrogate | `grpo_model_best` |

### 🎯 Model Lineage

```
Base Gemma 2B (4-bit)
    ↓
[FAZ 1: SFT]
    ↓
SFT-Gemma ──────────┬──────────────────────────────────┐
    ↓               ↓                                  ↓
[FAZ 2: RM]    [FAZ 3: PPO]                     [FAZ 4: GRPO]
(BT Loss)      Actor + Critic (training)         Actor (training)
    ↓           Reference (frozen)                Reference (frozen)
  rm_model_bt   BT RM (frozen)                    BT RM (frozen)
                      ↓                                  ↓
                PPO Model (best)               GRPO Model (best)
                      ↓                                  ↓
                      └─────────────────┬────────────────┘
                                      ↓
                            📊 Comparison Analysis
                          (Cohen's d + GPT-4o-mini)
```

---

## 🎓 **Kaynaklar**

1. **PPO:** Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. **GAE:** Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
3. **GRPO:** DeepSeek-Math Paper (2024)
4. **Bradley-Terry:** Bradley & Terry, "Rank Analysis of Incomplete Block Designs" (1952)
5. **UltraFeedback:** Cui et al. "UltraFeedback" (2023)
6. **Cohen's d:** Cohen, "Statistical Power Analysis for the Behavioral Sciences" (1988)
7. **MLX Framework:** Apple MLX Documentation
8. **GPT-4o-mini:** OpenAI (2024)

---