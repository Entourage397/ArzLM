#!/usr/bin/env bash
set -u
cd /home/asus/arzlm
echo "=== state ==="
ls -la data/prepared/arzlm-stem-1b-v1/state 2>/dev/null || echo none
echo "=== general state ==="
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("data/prepared/arzlm-stem-1b-v1/state/general.json")
print("exists", p.is_file())
if p.is_file():
    d = json.loads(p.read_text())
    keys = ["status","docs_inspected","docs_accepted_train","docs_accepted_val","tokens_train","tokens_val","stream_index"]
    print({k: d.get(k) for k in keys})
    print("train_shards", len(d.get("train_shards") or []))
PY
echo "=== shards ==="
find data/prepared/arzlm-stem-1b-v1 -type f | head
du -sh data/prepared/arzlm-stem-1b-v1 2>/dev/null
