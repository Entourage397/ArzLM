"""Move a project-owned flagged artifact out of the ingest cache without opening it."""

from __future__ import annotations

import json
import os
from pathlib import Path

from arzlm.paths import HF_CACHE_DIR, REPO_ROOT, SECURITY_DIR
from arzlm.training.report import utc_now

USER_HF_CACHE_MARKERS = (".cache/huggingface", "huggingface/hub")


def is_project_owned(path: Path) -> bool:
    path = Path(path).resolve()
    roots = [HF_CACHE_DIR.resolve(), (REPO_ROOT / "data").resolve(), SECURITY_DIR.resolve()]
    return any(path == root or root in path.parents for root in roots)


def looks_like_user_hf_cache(path: Path) -> bool:
    text = str(Path(path).resolve()).replace("\\", "/").lower()
    if "hf-cache" in text and "arzlm" in text:
        return False
    return any(marker in text for marker in USER_HF_CACHE_MARKERS)


def quarantine_project_file(path: Path, *, reason: str) -> Path:
    """Rename a project-owned blob into data/security/quarantine/. Do not open it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if looks_like_user_hf_cache(path) and not is_project_owned(path):
        raise PermissionError(f"refusing to touch user Hugging Face cache: {path}")
    if not is_project_owned(path):
        raise PermissionError(f"refusing to quarantine path outside the project: {path}")
    dest_dir = SECURITY_DIR / "quarantine"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{path.name}.{n}"
        n += 1
    sidecar = {
        "original_path": str(path),
        "reason": reason,
        "quarantined_at": utc_now(),
        "note": "Do not open, deserialize, or execute this file.",
    }
    os.replace(path, dest)
    dest.with_suffix(dest.suffix + ".meta.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return dest
