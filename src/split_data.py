"""Split sft_genrm_train.jsonl into stratified train and valid sets.

Includes a full-sequence token length filter: tokenizes each sample with
the chat template and drops rows exceeding --max-seq-length to prevent
truncation warnings during training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import get_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "sft" / "sft_genrm_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sft"

# Score bins for stratification (same bins as download_data.py)
BINS = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 10.01)]
BIN_LABELS = ["0-3", "3-5", "5-7", "7-9", "9-10"]


def _extract_score(line: str, score_regex) -> float | None:
    """Extract score from a GenRM JSONL line using config regex."""
    try:
        obj = json.loads(line)
        assistant_msg = obj["messages"][-1]["content"]
        m = score_regex.search(assistant_msg)
        if m:
            return float(m.group(1))
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


def _score_to_bin(score: float) -> str:
    """Map a score to its bin label."""
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        if lo <= score < hi:
            return label
    return BIN_LABELS[-1]


def _filter_by_seq_length(
    lines: list[str],
    strata: list[str],
    max_seq_length: int,
    model_id: str,
) -> tuple[list[str], list[str]]:
    """Filter out lines whose full chat-template token count exceeds max_seq_length.

    Returns (filtered_lines, filtered_strata).
    """
    print(f"\n── Sequence length filter ──")
    print(f"  Loading tokenizer: {model_id} ...")
    from mlx_lm.utils import load
    _, tokenizer = load(model_id, tokenizer_config={"trust_remote_code": True})

    kept_lines: list[str] = []
    kept_strata: list[str] = []
    token_lengths: list[int] = []
    dropped = 0

    for line, stratum in tqdm(
        zip(lines, strata), total=len(lines), desc="Tokenizing"
    ):
        obj = json.loads(line)
        messages = obj["messages"]

        # Apply chat template → full formatted text → tokenize
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        if n_tokens <= max_seq_length:
            kept_lines.append(line)
            kept_strata.append(stratum)
            token_lengths.append(n_tokens)
        else:
            dropped += 1

    # ── stats ──
    arr = np.array(token_lengths)
    print(f"  Max seq length   : {max_seq_length}")
    print(f"  Before filtering : {len(lines):,}")
    print(f"  After filtering  : {len(kept_lines):,}")
    print(f"  Removed          : {dropped:,}  ({dropped / len(lines) * 100:.1f}%)")
    print(f"\n  Token length stats (kept):")
    print(f"    min  : {arr.min()}")
    print(f"    max  : {arr.max()}")
    print(f"    mean : {arr.mean():.0f}")
    print(f"    p50  : {int(np.median(arr))}")
    print(f"    p95  : {int(np.percentile(arr, 95))}")

    return kept_lines, kept_strata


def split(
    input_path: Path,
    output_dir: Path,
    valid_size: int = 500,
    seed: int = 42,
    max_seq_length: int = 2048,
    model_id: str = "google/gemma-2b-it",
) -> None:
    cfg = get_config()
    score_regex = cfg.score_regex

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines):,} lines from {input_path.name}")

    # ── extract score bins for stratification ────────────────────────
    strata = []
    for line in lines:
        score = _extract_score(line, score_regex)
        strata.append(_score_to_bin(score) if score is not None else "unknown")

    # ── filter by full-sequence token length ─────────────────────────
    lines, strata = _filter_by_seq_length(
        lines, strata, max_seq_length, model_id
    )

    # ── check strata distribution (catch sparse bins early) ──────────
    strata_counts = Counter(strata)
    print("\n  Strata distribution (after filter):")
    for label in BIN_LABELS + (["unknown"] if "unknown" in strata_counts else []):
        count = strata_counts.get(label, 0)
        if count > 0:
            pct = count / len(strata) * 100
            flag = " ⚠️ sparse" if count < 3 else ""
            print(f"    bin {label:>7}: {count:>5} ({pct:5.1f}%){flag}")

    # ── stratified split via sklearn ─────────────────────────────────
    train_lines, valid_lines = train_test_split(
        lines,
        test_size=valid_size,
        random_state=seed,
        stratify=strata,
    )

    # ── report distribution ──────────────────────────────────────────
    valid_bins = Counter(
        _score_to_bin(_extract_score(l, score_regex))
        for l in valid_lines
        if _extract_score(l, score_regex) is not None
    )
    print("\n  Stratified split (valid set):")
    for label in BIN_LABELS:
        if valid_bins.get(label):
            print(f"    bin {label:>5}: {valid_bins[label]:>4} samples")

    # ── write output ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "sft_train.jsonl"
    valid_path = output_dir / "sft_valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(valid_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    print(f"\n  Train : {len(train_lines):,} → {train_path}")
    print(f"  Valid : {len(valid_lines):,} → {valid_path}")


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Filter by seq length & stratified split JSONL into train/valid sets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-size", type=int, default=500,
                        help="Number of validation samples (default: 500)")
    parser.add_argument("--seed", type=int, default=cfg.seed,
                        help=f"Random seed (default: {cfg.seed} from config.yaml)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max full-sequence token count to keep (default: 2048)")
    parser.add_argument("--model-id", type=str, default="google/gemma-2b-it",
                        help="HuggingFace model ID for tokenizer (default: google/gemma-2b-it)")
    args = parser.parse_args()

    split(
        args.input,
        args.output_dir,
        args.valid_size,
        args.seed,
        args.max_seq_length,
        args.model_id,
    )


if __name__ == "__main__":
    main()

