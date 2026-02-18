import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import get_config

# ── project root: two levels up from src/dataset/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAX_TOKENS = 2048


def download_and_sample(target_size: int, seed: int, max_tokens: int = MAX_TOKENS, model_id: str = "google/gemma-2-2b") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("UltraFeedback (Binary) dataset downloading...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    df = ds.to_pandas()
    print(f"Raw Data: {len(df):,} samples")

    # ── margin hesapla ──
    df["margin"] = df["score_chosen"] - df["score_rejected"]

    # ── stratify by score_chosen bins ──
    bins = [0, 3, 5, 7, 9, 10.01]
    labels = ["1-3", "3-5", "5-7", "7-9", "9-10"]
    df["score_bin"] = pd.cut(df["score_chosen"], bins=bins, labels=labels, right=False)

    selected_parts: list[pd.DataFrame] = []
    for bin_label, group in tqdm(
        df.groupby("score_bin", observed=True),
        desc="Stratified sampling",
    ):
        quota = max(1, int(target_size * (len(group) / len(df))))
        sampled = group.sample(n=min(quota, len(group)), random_state=seed)
        selected_parts.append(sampled)
        tqdm.write(f"   bin {bin_label}: {len(group):>6} total → {len(sampled):>5} selected")

    selected_df = pd.concat(selected_parts)

    # ── adjust to exact target_size ──
    if len(selected_df) < target_size:
        remaining = df.drop(selected_df.index).sort_values("margin", ascending=False)
        extra = remaining.head(target_size - len(selected_df))
        selected_df = pd.concat([selected_df, extra])
    elif len(selected_df) > target_size:
        selected_df = selected_df.sort_values("margin", ascending=False).head(target_size)

    selected_df = selected_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── summary stats ──
    print(f"\nSelected: {len(selected_df):,} samples")
    print(f"   margin range : {selected_df['margin'].min():.2f} – {selected_df['margin'].max():.2f}")
    print(f"   score_chosen  mean: {selected_df['score_chosen'].mean():.2f}")
    print(f"   score_rejected mean: {selected_df['score_rejected'].mean():.2f}")

    # ── token-length filtering ──
    print(f"\nFiltering prompts > {max_tokens} tokens (tokenizer: {model_id}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    token_lengths = [
        len(tokenizer.encode(str(p), add_special_tokens=False))
        for p in tqdm(selected_df["prompt"], desc="Tokenizing")
    ]
    selected_df["prompt_token_len"] = token_lengths

    before = len(selected_df)
    selected_df = selected_df[selected_df["prompt_token_len"] <= max_tokens].copy()
    removed = before - len(selected_df)

    print(f"  Before : {before:,}")
    print(f"  After  : {len(selected_df):,}")
    print(f"  Removed: {removed:,}  ({removed / before * 100:.1f}%)")

    lens = selected_df["prompt_token_len"]
    print(f"  Token len — min: {lens.min()}, max: {lens.max()}, "
          f"mean: {lens.mean():.0f}, p50: {lens.median():.0f}, p95: {lens.quantile(0.95):.0f}")

    selected_df = selected_df.drop(columns=["prompt_token_len"]).reset_index(drop=True)

    # ── save as Parquet ──
    output_file = RAW_DIR / "ultrafeedback_stratified.parquet"
    selected_df.to_parquet(output_file, index=False)
    print(f"\nDone → {output_file}  ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Download a stratified subset from UltraFeedback → Parquet.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=10_000,
        help="Total number of samples to keep (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.seed,
        help=f"Random seed (default: {cfg.seed} from config.yaml)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Drop prompts longer than this (default: {MAX_TOKENS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_sample(
        target_size=args.target_size,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )