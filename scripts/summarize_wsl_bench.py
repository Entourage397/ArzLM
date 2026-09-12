"""Summarize WSL vs Windows throughput JSON for the migration decision."""

from __future__ import annotations

import json
import math
from pathlib import Path

from arzlm.paths import REPO_ROOT

WINDOWS_BASELINE = REPO_ROOT / "docs" / "baselines" / "baseline-v0" / "windows-benchmark.json"
WSL_RESULTS = REPO_ROOT / "docs" / "wsl" / "results"
THRESHOLD = 0.10
WINDOWS_BEST_TOKS = 19060.0


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(report: dict, label: str) -> list[dict]:
    rows = []
    for trial in report.get("trials") or []:
        if not trial.get("ok"):
            rows.append({"label": label, "micro_batch_size": trial.get("micro_batch_size"), "error": trial.get("error")})
            continue
        rows.append(
            {
                "label": label,
                "mode": trial.get("mode"),
                "micro_batch_size": trial["micro_batch_size"],
                "tokens_per_sec": trial["tokens_per_sec"],
                "step_time_s": trial["step_time_s"],
                "peak_allocated_gb": trial["peak_allocated_vram_bytes"] / 1024**3,
                "peak_reserved_gb": trial["peak_reserved_vram_bytes"] / 1024**3,
                "compile_time_s": trial.get("compile_time_s"),
                "loss": trial.get("training_loss"),
                "sdpa": trial.get("sdpa"),
                "numerical": trial.get("numerical"),
            }
        )
    return rows


def main() -> int:
    windows = _load(WINDOWS_BASELINE)
    reports = []
    for name in ("linux-eager-bf16.json", "linux-compile-bf16.json"):
        path = WSL_RESULTS / name
        if path.exists():
            reports.append((name, _load(path)))
    print(f"Windows baseline best tok/s (frozen): {WINDOWS_BEST_TOKS:.1f}")
    win_best = windows.get("tokens_per_sec") or WINDOWS_BEST_TOKS
    print(f"Windows JSON tokens_per_sec: {win_best:.1f}")
    all_ok = []
    for name, report in reports:
        env = report.get("environment") or {}
        print(f"\n== {name} ==")
        print(
            "pytorch={pytorch} triton={triton} cuda={cuda} sdpa={sdpa}".format(
                pytorch=env.get("pytorch"),
                triton=env.get("triton_version"),
                cuda=env.get("cuda_runtime") or env.get("cuda_version"),
                sdpa=report.get("sdpa_backend"),
            )
        )
        for row in _rows(report, name):
            print(json.dumps(row, indent=2))
            if "tokens_per_sec" in row:
                all_ok.append(row)
    if not all_ok:
        print("No WSL results yet.")
        return 1
    best = max(all_ok, key=lambda r: r["tokens_per_sec"])
    gain = (best["tokens_per_sec"] - WINDOWS_BEST_TOKS) / WINDOWS_BEST_TOKS
    migrate = math.isfinite(gain) and gain >= THRESHOLD and best.get("loss") == best.get("loss")
    print("\nBest WSL trial:", json.dumps(best, indent=2))
    print(f"Gain vs Windows 19,060 tok/s: {100 * gain:.1f}%")
    print(
        "Recommendation: migrate training to WSL."
        if migrate
        else "Recommendation: keep native Windows eager training (no >=10% stable gain)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
