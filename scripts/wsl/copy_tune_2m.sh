#!/usr/bin/env bash
set -euo pipefail
DEST="/mnt/c/Users/ASUS/ArzLM/docs/tune"
mkdir -p "$DEST"
for lr in 0.05 0.075 0.10; do
  d="/home/asus/arzlm/runs/tune/muon-jordan-lr${lr}-2m"
  tag="wsl-muon-jordan-lr${lr}-2m"
  cp "$d/train_result.json" "$DEST/${tag}-train-result.json"
  cp "$d/experiment.json" "$DEST/${tag}-experiment.json"
  cp "$d/metrics.jsonl" "$DEST/${tag}-metrics.jsonl"
done
python3 - <<'PY'
import json
from pathlib import Path
for lr in ["0.05", "0.075", "0.10"]:
    p = Path(f"/home/asus/arzlm/runs/tune/muon-jordan-lr{lr}-2m/train_result.json")
    r = json.loads(p.read_text())
    q = r.get("qk_health") or {}
    print(
        lr,
        "val", round(r["val_loss"], 4),
        "train", round(r["loss"], 4),
        "finite", r["finite"],
        "max_gn", r.get("max_grad_norm"),
        "spikes", r.get("loss_spike_count"),
        "qkv_rms_max", (q.get("qkv_rms") or {}).get("max"),
        "qkv_max_abs", q.get("qkv_max_abs"),
        "tok/s", round(r["tokens_per_sec"]),
        "wall", round(r["elapsed_s"], 1),
    )
    vals = []
    m = Path(f"/home/asus/arzlm/runs/tune/muon-jordan-lr{lr}-2m/metrics.jsonl")
    for line in m.read_text().splitlines():
        row = json.loads(line)
        if row.get("val_loss") is not None:
            vals.append(
                (
                    row.get("tokens"),
                    round(row["val_loss"], 4),
                    None if row.get("elapsed_s") is None else round(row["elapsed_s"], 1),
                )
            )
    print("  curve", vals)
PY
echo copied
