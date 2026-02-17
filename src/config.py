from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# ── dataclasses ──────────────────────────────────────────────────────────


@dataclass
class DataConfig:
    train_jsonl: str = "data/train.jsonl"


@dataclass
class TokenizerConfig:
    model_id: str = "google/gemma-2-2b"
    env_var: str = "MLX_GEMMA_MODEL"


@dataclass
class ScoreConfig:
    pattern: str = r"Score:\s*([0-9]+(?:\.[0-9]+)?)/10"


@dataclass
class PlottingConfig:
    style: List[str] = field(default_factory=lambda: ["science", "ieee"])


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    project_root: Path = field(default_factory=lambda: _DEFAULT_CONFIG_PATH.parent)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    # ── derived helpers ──────────────────────────────────────────────

    @property
    def train_jsonl_path(self) -> Path:
        return self.project_root / self.data.train_jsonl

    @property
    def score_regex(self) -> re.Pattern:
        return re.compile(self.score.pattern)

    def tokenizer_path(self) -> Optional[str]:
        """Return local model path from env-var, or ``None``."""
        return os.environ.get(self.tokenizer.env_var)

    # ── factory ──────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path = _DEFAULT_CONFIG_PATH) -> ProjectConfig:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

        root = path.resolve().parent

        return cls(
            project_root=root,
            data=DataConfig(**raw.get("data", {})),
            tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
            score=ScoreConfig(**raw.get("score", {})),
            plotting=PlottingConfig(**raw.get("plotting", {})),
        )


# ── singleton ────────────────────────────────────────────────────────────

_CONFIG: Optional[ProjectConfig] = None
_CONFIG_MTIME: float = 0.0


def get_config(path: Path = _DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Return the cached project config, reloading if the YAML changed."""
    global _CONFIG, _CONFIG_MTIME

    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0

    if _CONFIG is None or mtime != _CONFIG_MTIME:
        _CONFIG = ProjectConfig.from_yaml(path)
        _CONFIG_MTIME = mtime

    return _CONFIG
