#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/home/asus/arzlm/data/prepared/50m/meta.json")
print(p.read_text())
PY
