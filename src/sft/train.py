"""SFT training with QLoRA using mlx_lm Python API directly.

Uses mlx_lm.load, linear_to_lora_layers, TrainingArgs, and trainer.train
instead of subprocess CLI calls.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load, save_config


def parse_args() -> argparse.Namespace:
    from src.config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description="SFT fine-tuning with QLoRA (mlx_lm Python API)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Linear warmup steps (default: 100)")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--adapter-path", type=str, default="checkpoints/sft_adapter")
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=cfg.seed,
                        help=f"Random seed (default: {cfg.seed} from config.yaml)")
    parser.add_argument("--mask-prompt", action="store_true", default=False,
                        help="Mask prompt tokens in loss")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Use gradient checkpointing to reduce memory")
    return parser.parse_args()


def _prepare_data_dir(train_path: Path, valid_path: Path) -> Path:
    """Create a temp directory with symlinks that mlx_lm load_dataset expects.

    mlx_lm expects {train,valid}.jsonl inside a single --data directory.
    """
    tmpdir = tempfile.mkdtemp(prefix="sft_data_")
    tmp_path = Path(tmpdir)
    (tmp_path / "train.jsonl").symlink_to(train_path.resolve())
    (tmp_path / "valid.jsonl").symlink_to(valid_path.resolve())
    return tmp_path


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # ── summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print("SFT Training — QLoRA (mlx_lm Python API)")
    print("=" * 60)
    print(f"  Model          : {args.model}")
    print(f"  Train data     : {args.train_data}")
    print(f"  Valid data     : {args.valid_data}")
    print(f"  Iterations     : {args.iters}")
    print(f"  Batch size     : {args.batch_size} "
          f"(× {args.grad_accum} grad accum = "
          f"{args.batch_size * args.grad_accum} effective)")
    print(f"  LoRA layers    : {args.lora_layers}")
    print(f"  LoRA rank      : {args.rank}")
    print(f"  Learning rate  : {args.learning_rate}")
    print(f"  Warmup steps   : {args.warmup_steps}")
    print(f"  Max seq length : {args.max_seq_length}")
    print(f"  Adapter path   : {args.adapter_path}")
    print(f"  Mask prompt    : {args.mask_prompt}")
    print(f"  Grad checkpoint: {args.grad_checkpoint}")
    print("=" * 60)

    # ── load model & tokenizer ───────────────────────────────────────
    print("\nLoading pretrained model...")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    # ── convert to LoRA ──────────────────────────────────────────────
    model.freeze()
    lora_config = {"rank": args.rank, "dropout": 0.0, "scale": 20.0}
    linear_to_lora_layers(model, args.lora_layers, lora_config)
    print_trainable_parameters(model)

    # ── load datasets ────────────────────────────────────────────────
    print("\nLoading datasets...")
    data_dir = _prepare_data_dir(args.train_data, args.valid_data)

    # Build a minimal args-like namespace for load_dataset
    dataset_args = argparse.Namespace(
        data=str(data_dir),
        train=True,
        test=False,
        mask_prompt=args.mask_prompt,
    )
    train_set, valid_set, _ = load_dataset(dataset_args, tokenizer)
    print(f"  Train samples  : {len(train_set):,}")
    print(f"  Valid samples  : {len(valid_set):,}")

    # ── training args ────────────────────────────────────────────────
    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path / "adapters.safetensors"

    save_config(
        {
            "model": args.model,
            "lora_layers": args.lora_layers,
            "rank": args.rank,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "iters": args.iters,
            "max_seq_length": args.max_seq_length,
            "mask_prompt": args.mask_prompt,
        },
        adapter_path / "adapter_config.json",
    )

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=-1,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=str(adapter_file),
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accum,
    )

    # ── optimizer (linear warmup + cosine decay) ──────────────────────
    # NOTE: both TrainingArgs.iters and cosine_decay use args.iters
    #       to keep them in sync. Do not set them independently.
    assert training_args.iters == args.iters, (
        f"iters mismatch: TrainingArgs={training_args.iters}, args={args.iters}"
    )
    if args.warmup_steps > 0:
        warmup = optim.linear_schedule(
            init=1e-7, end=args.learning_rate, steps=args.warmup_steps
        )
        cosine = optim.cosine_decay(
            init=args.learning_rate, decay_steps=args.iters - args.warmup_steps
        )
        lr_schedule = optim.join_schedules(
            [warmup, cosine], [args.warmup_steps]
        )
        optimizer = optim.Adam(learning_rate=lr_schedule)
    else:
        optimizer = optim.Adam(learning_rate=args.learning_rate)

    # ── metrics callback ──────────────────────────────────────────────
    from src.sft.callback import SFTMetricsCallback, EarlyStopException

    cached_train = CacheDataset(train_set)
    cached_valid = CacheDataset(valid_set)

    callback = SFTMetricsCallback(
        log_dir=adapter_path,
        model=model,
        tokenizer=tokenizer,
        val_dataset=valid_set,
        total_iters=args.iters,
    )

    # ── train ────────────────────────────────────────────────────────
    print("\nStarting training...")
    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=cached_train,
            val_dataset=cached_valid,
            args=training_args,
            training_callback=callback,
        )
    except EarlyStopException:
        print("Training ended early due to early stopping.")

    elapsed = callback._elapsed()
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nTraining complete! ({int(hours)}h {int(mins)}m {int(secs)}s)")
    print(f"Adapter saved to : {adapter_path}")
    print(f"Training log     : {callback.log_file}")
    print(f"Training curves  : {callback.plot_file}")


if __name__ == "__main__":
    main()
