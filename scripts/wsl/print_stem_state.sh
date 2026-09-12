#!/usr/bin/env bash
set -u
cd /home/asus/arzlm
echo "=== pgrep ==="
pgrep -af "prepare-stem|pack_stem" || echo none
echo "=== nohup tail ==="
tail -n 15 data/prepared/stem-pack-nohup.out 2>/dev/null || echo none
echo "=== log tail ==="
tail -n 8 data/prepared/stem-prep.log 2>/dev/null || echo none
echo "=== general state ==="
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("data/prepared/arzlm-stem-1b-v1/state/general.json")
print("exists", p.is_file())
if p.is_file():
    d = json.loads(p.read_text())
    print({k: d.get(k) for k in ["status","docs_inspected","docs_accepted_train","tokens_train","tokens_val","stream_index"]})
    print("train_shards", len(d.get("train_shards") or []))
root = Path("data/prepared/arzlm-stem-1b-v1")
for split in ("train","val"):
    ddir = root/split/"general"
    if ddir.is_dir():
        bins = list(ddir.glob("shard-*.bin"))
        print(split, "bin_count", len(bins), "bytes", sum(x.stat().st_size for x in bins))
print("du")
PY
du -sh data/prepared/arzlm-stem-1b-v1 2>/dev/null
