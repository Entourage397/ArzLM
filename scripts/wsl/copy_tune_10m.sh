#!/usr/bin/env bash
set -euo pipefail
DEST="/mnt/c/Users/ASUS/ArzLM/docs/tune"
mkdir -p "$DEST"
for lr in 0.05 0.075; do
  d="/home/asus/arzlm/runs/tune/muon-10m-lr${lr}"
  if [[ ! -f "$d/train_result.json" ]]; then
    echo "missing $d/train_result.json"
    continue
  fi
  tag="wsl-muon-10m-lr${lr}"
  cp "$d/train_result.json" "$DEST/${tag}-train-result.json"
  cp "$d/experiment.json" "$DEST/${tag}-experiment.json"
  cp "$d/metrics.jsonl" "$DEST/${tag}-metrics.jsonl"
done
python3 - <<'PY'
import json
from pathlib import Path
for lr in ["0.05", "0.075"]:
    p = Path(f"/home/asus/arzlm/runs/tune/muon-10m-lr{lr}/train_result.json")
    if not p.exists():
        print(lr, "not finished")
        continue
    r = json.loads(p.read_text())
    q = r.get("qk_health") or {}
    print(
        "10m", lr,
        "val", round(r["val_loss"], 4),
        "train", round(r["loss"], 4),
        "finite", r["finite"],
        "max_gn", r.get("max_grad_norm"),
        "spikes", r.get("loss_spike_count"),
        "qkv_max_abs", q.get("qkv_max_abs"),
        "tok/s", round(r["tokens_per_sec"]),
        "wall", round(r["elapsed_s"], 1),
    )
    vals = []
    m = Path(f"/home/asus/arzlm/runs/tune/muon-10m-lr{lr}/metrics.jsonl")
    for line in m.read_text().splitlines():
        row = json.loads(line)
        if row.get("val_loss") is not None:
            vals.append((row.get("tokens"), round(row["val_loss"], 4), row.get("wall_s")))
    # downsample to ~every 2M plus endpoints for the printed table
    print("  n_val", len(vals), "first", vals[0] if vals else None, "last", vals[-1] if vals else None)
    for tokens_mark in (2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000):
        closest = min(vals, key=lambda t: abs((t[0] or 0) - tokens_mark)) if vals else None
        print("  ~", tokens_mark, closest)
PY
