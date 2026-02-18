"""Filter downloaded Parquet data by prompt token length.

Tokenizes each prompt with the Gemma tokenizer and drops rows
whose prompt exceeds MAX_TOKENS (default 2048).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import get_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_INPUT = RAW_DIR / "ultrafeedback_stratified.parquet"
DEFAULT_OUTPUT = RAW_DIR / "ultrafeedback_filtered.parquet"
MAX_TOKENS = 2048


def _extract_prompt(row) -> str:
    """Extract the user prompt text from a row."""
    # UltraFeedback 'prompt' column holds the raw user query
    return str(row["prompt"])


def filter_by_token_length(
    input_path: Path,
    output_path: Path,
    max_tokens: int,
    model_id: str,
) -> None:
    print(f"Loading data from {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  Total rows: {len(df):,}")

    # ── load tokenizer ──
    print(f"Loading tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ── tokenize & measure lengths ──
    print("Tokenizing prompts ...")
    token_lengths: list[int] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        prompt = _extract_prompt(row)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        token_lengths.append(len(tokens))

    df["prompt_token_len"] = token_lengths

    # ── filter ──
    before = len(df)
    df_filtered = df[df["prompt_token_len"] <= max_tokens].copy()
    after = len(df_filtered)
    removed = before - after

    print(f"\n── Filter Results ──")
    print(f"  Max tokens       : {max_tokens}")
    print(f"  Before filtering : {before:,}")
    print(f"  After filtering  : {after:,}")
    print(f"  Removed          : {removed:,}  ({removed / before * 100:.1f}%)")

    # ── stats ──
    lens = df_filtered["prompt_token_len"]
    print(f"\n── Token Length Stats (kept) ──")
    print(f"  min  : {lens.min()}")
    print(f"  max  : {lens.max()}")
    print(f"  mean : {lens.mean():.1f}")
    print(f"  p50  : {lens.median():.0f}")
    print(f"  p95  : {lens.quantile(0.95):.0f}")

    # ── drop helper column & save ──
    df_filtered = df_filtered.drop(columns=["prompt_token_len"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output_path, index=False)
    print(f"\nDone → {output_path}  ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Filter Parquet data by prompt token length using Gemma tokenizer.",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output parquet file (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"Max prompt token count to keep (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--model-id", type=str, default=cfg.tokenizer.model_id,
        help=f"HuggingFace tokenizer model ID (default: {cfg.tokenizer.model_id})",
    )
    return parser.parse_args(), cfg


if __name__ == "__main__":
    args, cfg = main()
    filter_by_token_length(
        input_path=args.input,
        output_path=args.output,
        max_tokens=args.max_tokens,
        model_id=args.model_id,
    )
