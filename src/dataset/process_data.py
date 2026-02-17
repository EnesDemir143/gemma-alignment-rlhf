import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

# ── project root: two levels up from src/dataset/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_INPUT = RAW_DIR / "ultrafeedback_stratified.parquet"

# ── format → subfolder / filename mapping ──
FORMAT_OUTPUTS: dict[str, Path] = {
    "genrm": PROCESSED_DIR / "sft" / "sft_genrm_train.jsonl",
    "pairwise": PROCESSED_DIR / "rm" / "rm_pairwise_train.jsonl",
    "prompts": PROCESSED_DIR / "rl" / "rl_prompts.jsonl",
}

# ─────────────────────────────────────────────────────
# Row formatters — each returns a JSON string
# ─────────────────────────────────────────────────────

def _extract_content(messages) -> str:
    if isinstance(messages, list):
        return messages[-1]["content"]
    return str(messages)


def _row_to_genrm(row: dict) -> str:
    chosen_content = _extract_content(row["chosen"])

    user_content = (
        f"User: {row['prompt']}\n\n"
        f"Assistant: {chosen_content}\n\n"
        f"Analyze the quality of this response."
    )

    score = row["score_chosen"]
    assistant_content = f"Score: {score:.1f}/10. The response is helpful, harmless, and honest."

    obj = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    return json.dumps(obj, ensure_ascii=False)


def _to_native(val):
    """Convert numpy/non-serializable types to native Python."""
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def _row_to_pairwise(row: dict) -> str:
    obj = {
        "prompt": row["prompt"],
        "chosen": _to_native(row["chosen"]),
        "rejected": _to_native(row["rejected"]),
        "score_chosen": float(row["score_chosen"]),
        "score_rejected": float(row["score_rejected"]),
    }
    return json.dumps(obj, ensure_ascii=False)


def _row_to_prompts(row: dict) -> str:
    obj = {"prompt": row["prompt"]}
    return json.dumps(obj, ensure_ascii=False)


FORMATTERS: dict[str, Callable[[dict], str]] = {
    "genrm": _row_to_genrm,
    "pairwise": _row_to_pairwise,
    "prompts": _row_to_prompts,
}

# ─────────────────────────────────────────────────────
# Parallel processing engine
# ─────────────────────────────────────────────────────

_active_formatter: Callable[[dict], str] | None = None


def _init_worker(fmt: str) -> None:
    global _active_formatter
    _active_formatter = FORMATTERS[fmt]


def _process_chunk(rows: list[dict]) -> list[str]:
    return [_active_formatter(row) for row in rows]  # type: ignore[misc]


def process_data(
    input_path: Path,
    output_path: Path,
    fmt: str,
    workers: int,
    chunk_size: int = 500,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw veri bulunamadı: {input_path}\n"
            f"   Önce 'python -m src.dataset.download_data' çalıştırın."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Format  : {fmt}")
    print(f"Girdi   : {input_path}")
    print(f"Çıktı   : {output_path}")
    print(f"Workers : {workers}")

    df = pd.read_parquet(input_path)
    print(f"Loaded  : {len(df):,} rows")

    records = df.to_dict(orient="records")

    chunks: list[list[dict]] = [
        records[i : i + chunk_size] for i in range(0, len(records), chunk_size)
    ]

    results: list[str] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(fmt,),
    ) as executor:
        for chunk_result in tqdm(
            executor.map(_process_chunk, chunks),
            total=len(chunks),
            desc=f"Processing ({fmt})",
            unit="chunk",
        ):
            results.extend(chunk_result)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in tqdm(results, desc="Writing JSONL", unit="row"):
            f.write(line + "\n")

    print(f"\nDone → {output_path}")
    print(f"   Rows: {len(results):,}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def parse_args() -> argparse.Namespace:
    import os

    parser = argparse.ArgumentParser(
        description="Process raw UltraFeedback Parquet → training JSONL.",
    )
    parser.add_argument(
        "--format",
        choices=list(FORMATTERS.keys()),
        default="genrm",
        help="Output format: genrm (SFT), pairwise (RM), prompts (RL)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input Parquet path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: auto per format)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Rows per processing chunk (default: 500)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or FORMAT_OUTPUTS[args.format]
    process_data(
        input_path=args.input,
        output_path=output,
        fmt=args.format,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
