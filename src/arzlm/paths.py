"""Repository-relative paths."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(os.environ["ARZLM_REPO_ROOT"]).resolve() if os.environ.get("ARZLM_REPO_ROOT") else _DEFAULT_ROOT
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = Path(os.environ["ARZLM_DATA_DIR"]).resolve() if os.environ.get("ARZLM_DATA_DIR") else (REPO_ROOT / "data")
PREPARED_DIR = DATA_DIR / "prepared"
SECURITY_DIR = Path(os.environ["ARZLM_SECURITY_DIR"]).resolve() if os.environ.get("ARZLM_SECURITY_DIR") else (DATA_DIR / "security")
TOKENIZER_DIR = REPO_ROOT / "tokenizer" / "trained"
RUNS_DIR = REPO_ROOT / "runs"
HF_CACHE_DIR = Path(os.environ["HF_HOME"]).resolve() if os.environ.get("HF_HOME") else (REPO_ROOT / "hf-cache")


def configure_hf_home() -> None:
    """Point Hugging Face downloads at this repo's cache, not ~/.cache."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, PREPARED_DIR, SECURITY_DIR, TOKENIZER_DIR.parent, RUNS_DIR, HF_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
