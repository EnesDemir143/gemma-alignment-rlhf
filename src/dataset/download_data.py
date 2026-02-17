import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── project root: two levels up from src/dataset/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_and_sample(target_size: int) -> None:
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
        sampled = group.sample(n=min(quota, len(group)), random_state=42)
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

    selected_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── summary stats ──
    print(f"\nSelected: {len(selected_df):,} samples")
    print(f"   margin range : {selected_df['margin'].min():.2f} – {selected_df['margin'].max():.2f}")
    print(f"   score_chosen  mean: {selected_df['score_chosen'].mean():.2f}")
    print(f"   score_rejected mean: {selected_df['score_rejected'].mean():.2f}")

    # ── save as Parquet ──
    output_file = RAW_DIR / "ultrafeedback_stratified.parquet"
    selected_df.to_parquet(output_file, index=False)
    print(f"Done → {output_file}  ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a stratified subset from UltraFeedback → Parquet.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=10_000,
        help="Total number of samples to keep (default: 10000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_sample(target_size=args.target_size)