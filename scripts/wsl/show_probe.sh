#!/usr/bin/env bash
set -euo pipefail
echo "=== sizes ==="
wc -c /tmp/arzlm-stem-probe.json /tmp/arzlm-probe.err || true
echo "=== json probes ==="
/home/asus/arzlm/.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("/tmp/arzlm-stem-probe.json")
if not p.exists() or p.stat().st_size == 0:
    print("empty")
else:
    text = p.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
        probes = data.get("probes", data)
        print(json.dumps(probes, indent=2)[:12000])
    except Exception as exc:
        print("json parse failed", exc)
        print(text[:4000])
PY
echo "=== err tail ==="
tail -100 /tmp/arzlm-probe.err || true
