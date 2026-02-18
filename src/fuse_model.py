"""Fuse LoRA adapter weights into the base model using mlx_lm Python API.

Uses mlx_lm.load with adapter_path, m.fuse() on LoRA layers,
and mlx_lm.fuse.save to write the merged model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx.utils import tree_unflatten

from mlx_lm.fuse import save
from mlx_lm.utils import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse LoRA adapter into base model (mlx_lm Python API)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--adapter-path", type=str, default="checkpoints/sft_adapter")
    parser.add_argument("--save-path", type=str, default="checkpoints/sft_merged_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Fusing LoRA Adapter → Base Model (mlx_lm Python API)")
    print("=" * 60)
    print(f"  Base model   : {args.model}")
    print(f"  Adapter      : {args.adapter_path}")
    print(f"  Output       : {args.save_path}")
    print("=" * 60)

    # ── load model with adapter ──────────────────────────────────────
    print("\nLoading pretrained model with adapter...")
    model, tokenizer, config = load(
        args.model,
        adapter_path=args.adapter_path,
        return_config=True,
    )

    # ── fuse LoRA layers (keep quantized) ────────────────────────────
    fused_linears = [
        (n, m.fuse(dequantize=False))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
        print(f"  Fused {len(fused_linears)} LoRA layers (quantized)")

    # ── save merged model ────────────────────────────────────────────
    save_path = Path(args.save_path)
    print(f"\nSaving merged model to: {save_path}")
    save(
        save_path,
        args.model,
        model,
        tokenizer,
        config,
        donate_model=False,
    )

    print("\nFuse complete!")


if __name__ == "__main__":
    main()
