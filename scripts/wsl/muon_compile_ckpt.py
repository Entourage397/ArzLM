"""Muon + torch.compile checkpoint save/resume smoke (tiny packed data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from arzlm.paths import PREPARED_DIR, REPO_ROOT, TOKENIZER_DIR
from arzlm.training.config import load_experiment_config
from arzlm.training.loop import run_training
from arzlm.training.report import git_commit, write_json


def _base(out_dir: Path, compile_flag: str, max_steps: int, resume):
    # Reuse the 2M muon recipe but cap steps and point at tiny data.
    exp = load_experiment_config(REPO_ROOT / "configs" / "tune" / "muon-jordan-lr0.05-2m.yaml")
    exp.out_dir = out_dir
    exp.data_dir = PREPARED_DIR / "tiny"
    exp.tokenizer_dir = TOKENIZER_DIR
    exp.compile = compile_flag
    exp.resume = resume
    exp.train.max_steps = max_steps
    exp.train.max_tokens = 10_000_000
    exp.train.save_interval = 10_000
    exp.eval.interval = 10_000
    exp.eval.final_validation = False
    exp.logger_name = "csv"
    return exp


def main() -> int:
    compile_flag = sys.argv[1] if len(sys.argv) > 1 else "true"
    root = REPO_ROOT / "runs" / "wsl" / f"muon-ckpt-{compile_flag}"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    first = _base(root, compile_flag, max_steps=4, resume=False)
    r1 = run_training(first)
    ckpt = first.out_dir / "final" / "lit_model.pth"
    if not ckpt.is_file():
        print("missing checkpoint", ckpt)
        return 1
    second = _base(root, compile_flag, max_steps=6, resume=ckpt)
    r2 = run_training(second)
    report = {
        "git_commit": git_commit(),
        "first_steps": r1["steps"],
        "resume_steps": r2["steps"],
        "first_finite": r1["finite"],
        "resume_finite": r2["finite"],
        "first_loss": r1["loss"],
        "resume_loss": r2["loss"],
        "compile_active": r2.get("compile_active"),
        "ok": bool(r1["finite"] and r2["finite"] and r2["steps"] >= r1["steps"]),
    }
    out = root / "ckpt_smoke.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
