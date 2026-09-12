"""Experiment metadata: git commit, FLOPs, metric curves, optimizer snapshot."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arzlm.paths import REPO_ROOT


def git_commit(repo: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_describe(repo: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def estimate_train_flops(n_params: int, n_tokens: int) -> int:
    """6ND transformer training FLOPs (forward + backward matmuls)."""
    return int(6 * n_params * n_tokens)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def perplexity(ce: float | None) -> float | None:
    if ce is None or ce != ce or abs(ce) == float("inf"):
        return None
    try:
        return float(math.exp(ce))
    except OverflowError:
        return float("inf")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
