"""SFT Training Metrics Callback — JSONL logging + live plot generation.

Plugs into mlx_lm's TrainingCallback to capture all training metrics,
compute derived metrics (PPL, token accuracy, entropy), and generate
live training curve plots.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.tuner.callbacks import TrainingCallback


class EarlyStopException(Exception):
    """Raised by callback to signal early stopping to the training loop."""
    pass

# ── plot config ──────────────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class SFTMetricsCallback(TrainingCallback):
    """Logs all SFT training metrics and generates live training curves.

    Args:
        log_dir: directory to write ``train_log.jsonl`` and ``training_curves.png``.
        model: the model being trained (for token accuracy / entropy eval).
        tokenizer: tokenizer object.
        val_dataset: validation dataset (list of dicts with ``input_ids``/``labels``).
        total_iters: total training iterations (for ETA calculation).
        target_ppl: target perplexity marker line on the plot.
        eval_batches: number of batches to use for token accuracy / entropy.
    """

    def __init__(
        self,
        log_dir: Path,
        model: nn.Module,
        tokenizer,
        val_dataset,
        total_iters: int,
        target_ppl: float = 2.8,
        eval_batches: int = -1,
        early_stop_patience: int = 3,
        early_stop_min_delta: float = 0.01,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train_log.jsonl"
        self.plot_file = self.log_dir / "training_curves.png"

        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.total_iters = total_iters
        self.target_ppl = target_ppl
        self.eval_batches = eval_batches

        # ── early stopping state ─────────────────────────────────────
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.best_val_loss: float = float("inf")
        self.early_stop_wait: int = 0
        self.early_stopped: bool = False

        # Find the token IDs for the assistant turn marker
        # so we can measure accuracy only on response tokens
        try:
            marker_ids = tokenizer.encode(
                "<start_of_turn>model\n", add_special_tokens=False
            )
            self._response_marker: list[int] | None = marker_ids
        except Exception:
            self._response_marker = None

        self.start_time = time.time()

        # ── metric history ───────────────────────────────────────────
        self.train_iters: list[int] = []
        self.train_losses: list[float] = []
        self.learning_rates: list[float] = []
        self.tokens_per_sec: list[float] = []
        self.peak_memories: list[float] = []

        self.val_iters: list[int] = []
        self.val_losses: list[float] = []
        self.val_ppls: list[float] = []
        self.val_token_accs: list[float] = []
        self.val_entropies: list[float] = []

    # ── helpers ──────────────────────────────────────────────────────

    def _append_log(self, entry: dict) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _elapsed(self) -> float:
        return time.time() - self.start_time

    def _eta(self, current_iter: int) -> float:
        elapsed = self._elapsed()
        if current_iter <= 0:
            return 0.0
        return elapsed / current_iter * (self.total_iters - current_iter)

    # ── token accuracy & entropy on small val batch ──────────────────

    def _find_response_start(self, tokens: mx.array) -> int:
        """Find where the assistant response starts in a token sequence.

        Looks for the last occurrence of the ``<start_of_turn>model`` marker.
        Returns the index of the first response token (after the marker),
        or 0 if no marker is found.
        """
        if self._response_marker is None:
            return 0

        tok_list = tokens.tolist()
        marker = self._response_marker
        mlen = len(marker)

        # search backwards for the last occurrence
        for i in range(len(tok_list) - mlen, -1, -1):
            if tok_list[i : i + mlen] == marker:
                return i + mlen  # first token AFTER the marker

        return 0

    def _compute_val_metrics(self) -> tuple[float, float]:
        """Compute token accuracy and avg entropy on assistant-only tokens."""
        if not self.val_dataset or len(self.val_dataset) == 0:
            return 0.0, 0.0

        correct = 0
        total = 0
        total_entropy = 0.0
        entropy_count = 0

        n_samples = len(self.val_dataset) if self.eval_batches < 0 else min(self.eval_batches, len(self.val_dataset))

        for i in range(n_samples):
            sample = self.val_dataset[i]

            # val_dataset items are arrays of token ids
            if isinstance(sample, dict):
                tokens = mx.array(sample.get("input_ids", sample.get("tokens", [])))
            elif isinstance(sample, (list, np.ndarray)):
                tokens = mx.array(sample)
            elif isinstance(sample, mx.array):
                tokens = sample
            else:
                continue

            if tokens.size <= 1:
                continue

            # find where assistant response starts
            resp_start = self._find_response_start(tokens)

            # ensure we have at least 1 response token to predict
            if resp_start >= tokens.size - 1:
                continue

            input_ids = tokens[:-1][None, :]  # (1, seq_len - 1)
            logits = self.model(input_ids)  # (1, seq_len-1, vocab)
            logits = logits.squeeze(0)  # (seq_len-1, vocab)

            # only measure on response tokens (resp_start onwards)
            resp_logits = logits[resp_start:]
            resp_targets = tokens[resp_start + 1:]  # shifted by 1

            if resp_targets.size == 0:
                continue

            # token accuracy (assistant response only)
            preds = mx.argmax(resp_logits, axis=-1)
            correct += (preds == resp_targets).sum().item()
            total += resp_targets.size

            # entropy (assistant response only)
            log_probs = resp_logits - mx.logsumexp(resp_logits, axis=-1, keepdims=True)
            probs = mx.exp(log_probs)
            token_entropy = -(probs * log_probs).sum(axis=-1).mean().item()
            total_entropy += token_entropy
            entropy_count += 1

            mx.eval(preds)  # force eval to free memory

        acc = correct / total if total > 0 else 0.0
        avg_entropy = total_entropy / entropy_count if entropy_count > 0 else 0.0
        return acc, avg_entropy

    # ── callback hooks ───────────────────────────────────────────────

    def on_train_loss_report(self, train_info: dict) -> None:
        it = train_info["iteration"]
        elapsed = self._elapsed()
        eta = self._eta(it)

        entry = {
            "type": "train",
            "iteration": it,
            "train_loss": round(train_info["train_loss"], 4),
            "lr": train_info["learning_rate"],
            "tokens_sec": round(train_info["tokens_per_second"], 1),
            "peak_mem_gb": round(train_info["peak_memory"], 3),
            "trained_tokens": train_info["trained_tokens"],
            "it_sec": round(train_info["iterations_per_second"], 3),
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(eta, 1),
        }
        self._append_log(entry)

        # cache for plotting
        self.train_iters.append(it)
        self.train_losses.append(train_info["train_loss"])
        self.learning_rates.append(train_info["learning_rate"])
        self.tokens_per_sec.append(train_info["tokens_per_second"])
        self.peak_memories.append(train_info["peak_memory"])

        # ── console output ───────────────────────────────────────────
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        print(
            f"Iter {it:>5}/{self.total_iters} | "
            f"Loss {train_info['train_loss']:.4f} | "
            f"LR {train_info['learning_rate']:.2e} | "
            f"Tok/s {train_info['tokens_per_second']:.0f} | "
            f"Mem {train_info['peak_memory']:.1f} GB | "
            f"ETA {eta_h}h {eta_m:02d}m",
            flush=True,
        )


    def on_val_loss_report(self, val_info: dict) -> None:
        it = val_info["iteration"]
        val_loss = val_info["val_loss"]
        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

        # compute token accuracy & entropy
        token_acc, avg_entropy = self._compute_val_metrics()

        entry = {
            "type": "val",
            "iteration": it,
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "val_time": round(val_info["val_time"], 2),
            "token_accuracy": round(token_acc, 4),
            "avg_entropy": round(avg_entropy, 4),
        }
        self._append_log(entry)

        # cache for plotting
        self.val_iters.append(it)
        self.val_losses.append(val_loss)
        self.val_ppls.append(val_ppl)
        self.val_token_accs.append(token_acc)
        self.val_entropies.append(avg_entropy)

        # update plots
        self._save_plots()

        # print extra metrics
        print(
            f"  → PPL {val_ppl:.2f} | "
            f"Token Acc {token_acc:.2%} | "
            f"Entropy {avg_entropy:.2f}",
            flush=True,
        )

        # ── early stopping check ─────────────────────────────────────
        if val_loss < self.best_val_loss - self.early_stop_min_delta:
            self.best_val_loss = val_loss
            self.early_stop_wait = 0
        else:
            self.early_stop_wait += 1
            print(
                f"  ⚠ Early stop: no improvement for "
                f"{self.early_stop_wait}/{self.early_stop_patience} evals "
                f"(best val_loss={self.best_val_loss:.4f})",
                flush=True,
            )
            if self.early_stop_wait >= self.early_stop_patience:
                self.early_stopped = True
                print(
                    f"\n🛑 Early stopping triggered at iter {it}! "
                    f"Best val_loss={self.best_val_loss:.4f}",
                    flush=True,
                )
                raise EarlyStopException(
                    f"Val loss did not improve for {self.early_stop_patience} "
                    f"consecutive evaluations."
                )

    # ── plotting ─────────────────────────────────────────────────────

    def _save_plots(self) -> None:
        if not HAS_MPL:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("SFT Training Curves", fontsize=14, fontweight="bold")

        # ① Train & Val Loss
        ax = axes[0, 0]
        ax.plot(self.train_iters, self.train_losses, "b-", alpha=0.7, label="Train")
        if self.val_iters:
            ax.plot(self.val_iters, self.val_losses, "ro-", markersize=4, label="Val")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ② Perplexity
        ax = axes[0, 1]
        if self.val_ppls:
            ax.plot(self.val_iters, self.val_ppls, "go-", markersize=4)
            ax.axhline(y=self.target_ppl, color="r", linestyle="--",
                       alpha=0.7, label=f"Target ({self.target_ppl})")
            ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity")
        ax.grid(True, alpha=0.3)

        # ③ Learning Rate
        ax = axes[0, 2]
        ax.plot(self.train_iters, self.learning_rates, "m-")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
        ax.grid(True, alpha=0.3)

        # ④ Token Accuracy
        ax = axes[1, 0]
        if self.val_token_accs:
            ax.plot(self.val_iters, self.val_token_accs, "co-", markersize=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Token Accuracy")
        ax.set_title("Token Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # ⑤ Entropy
        ax = axes[1, 1]
        if self.val_entropies:
            ax.plot(self.val_iters, self.val_entropies, "o-", color="orange", markersize=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Entropy")
        ax.set_title("Prediction Entropy")
        ax.grid(True, alpha=0.3)

        # ⑥ Throughput & Memory (twin y-axis)
        ax1 = axes[1, 2]
        if self.tokens_per_sec:
            ax1.plot(self.train_iters, self.tokens_per_sec, "b-", alpha=0.7,
                     label="Tokens/sec")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Tokens/sec", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        if self.peak_memories:
            ax2.plot(self.train_iters, self.peak_memories, "r-", alpha=0.5,
                     label="Peak Mem (GB)")
        ax2.set_ylabel("Peak Memory (GB)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax1.set_title("Throughput & Memory")
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
