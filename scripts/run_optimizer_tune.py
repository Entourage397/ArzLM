"""Run the small Muon vs AdamW tuning sweep (2M tokens each, not 50M).

Each trial is a fresh interpreter so CUDA/host RAM from the previous run
is released. In-process sequential training OOMed on WSL after the first 2M trial.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from arzlm.paths import CONFIGS_DIR, REPO_ROOT
from arzlm.training.report import git_commit, write_json

DEFAULT_CONFIGS = (
    CONFIGS_DIR / "tune" / "adamw-2m.yaml",
    CONFIGS_DIR / "tune" / "muon-jordan-lr0.01-2m.yaml",
    CONFIGS_DIR / "tune" / "muon-jordan-lr0.02-2m.yaml",
    CONFIGS_DIR / "tune" / "muon-jordan-lr0.05-2m.yaml",
)


def _trial_summary(config_path: Path, train_result: dict) -> dict:
    return {
        "config": str(config_path),
        "out_dir": train_result.get("out_dir"),
        "optimizer": (train_result.get("optimizer") or {}).get("kind")
        if isinstance(train_result.get("optimizer"), dict)
        else train_result.get("optimizer"),
        "val_loss": train_result.get("val_loss"),
        "val_ppl": train_result.get("val_ppl"),
        "loss": train_result.get("loss"),
        "tokens": train_result.get("tokens"),
        "elapsed_s": train_result.get("elapsed_s"),
        "tokens_per_sec": train_result.get("tokens_per_sec"),
        "peak_allocated_vram_bytes": train_result.get("peak_allocated_vram_bytes"),
        "finite": train_result.get("finite"),
        "max_grad_norm": train_result.get("max_grad_norm"),
        "loss_spike_count": train_result.get("loss_spike_count"),
        "git_commit": train_result.get("git_commit") or git_commit(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", type=Path, default=list(DEFAULT_CONFIGS))
    args = parser.parse_args()
    summary = []
    for path in args.configs:
        path = path if path.is_absolute() else REPO_ROOT / path
        print(f"=== {path} ===")
        completed = subprocess.run(
            [sys.executable, "-m", "arzlm", "train", "--config", str(path)],
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        # out_dir is the YAML `out_dir` field; load it without keeping the model.
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out_dir = Path(raw["out_dir"])
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        result_path = out_dir / "train_result.json"
        train_result = json.loads(result_path.read_text(encoding="utf-8"))
        summary.append(_trial_summary(path, train_result))
        print(json.dumps(summary[-1], indent=2))
    out = REPO_ROOT / "runs" / "tune" / "summary.json"
    write_json(out, {"trials": summary})
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
