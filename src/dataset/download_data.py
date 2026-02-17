import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

# ── project root: two levels up from src/dataset/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def prepare_stratified_data(target_size: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 UltraFeedback (Binary) dataset is downloading...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    df = ds.to_pandas()

    print(f"📦 Raw Data: {len(df)} samples")

    df["margin"] = df["score_chosen"] - df["score_rejected"]

    # ── stratify by score_chosen bins, pick highest-margin within each ──
    bins = [0, 3, 5, 7, 9, 10.01]
    labels = ["1-3", "3-5", "5-7", "7-9", "9-10"]
    df["score_bin"] = pd.cut(df["score_chosen"], bins=bins, labels=labels, right=False)

    selected_parts = []
    for bin_label, group in df.groupby("score_bin", observed=True):
        quota = max(1, int(target_size * (len(group) / len(df))))
        sampled = group.sample(n=min(quota, len(group)), random_state=42)
        selected_parts.append(sampled)
        print(f"   bin {bin_label}: {len(group):>6} total → {len(sampled):>5} selected (random)")

    selected_df = pd.concat(selected_parts)

    # adjust to exact target_size
    if len(selected_df) < target_size:
        remaining = df.drop(selected_df.index).sort_values("margin", ascending=False)
        extra = remaining.head(target_size - len(selected_df))
        selected_df = pd.concat([selected_df, extra])
    elif len(selected_df) > target_size:
        selected_df = selected_df.sort_values("margin", ascending=False).head(target_size)

    selected_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n🎯 Selected: {len(selected_df)} samples")
    print(f"   margin range: {selected_df['margin'].min():.2f} – {selected_df['margin'].max():.2f}")
    print(f"   score_chosen  mean: {selected_df['score_chosen'].mean():.2f}")
    print(f"   score_rejected mean: {selected_df['score_rejected'].mean():.2f}")

    output_file = DATA_DIR / f"train.jsonl"
    print(f"📝 {output_file} writing...")

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in selected_df.iterrows():
            user_content = (
                f"User: {row['prompt']}\n\n"
                f"Assistant: {row['chosen'][-1]['content']}\n\n"
                f"Analyze the quality of this response."
            )

            score = row["score_chosen"]
            assistant_content = f"Score: {score:.1f}/10. The response is helpful, harmless, and honest."

            json_obj = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"✅ Done → {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a stratified subset from UltraFeedback and save as JSONL.",
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
    prepare_stratified_data(target_size=args.target_size)