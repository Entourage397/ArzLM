#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
cd /home/asus/arzlm
rsync -a /mnt/c/Users/ASUS/ArzLM/src/arzlm/data/sources.py src/arzlm/data/sources.py
.venv/bin/python - <<'PY'
from collections import Counter
from urllib.request import Request, urlopen
from arzlm.data.sources import pes2o_shard_url, pes2o_shard_names, _iter_gzip_json_lines
import json
url = pes2o_shard_url(pes2o_shard_names()[0])
req = Request(url, headers={"User-Agent": "arzlm-stem/1.0", "Accept-Encoding": "identity"})
c = Counter()
n = 0
with urlopen(req, timeout=120) as resp:
    for blob in _iter_gzip_json_lines(resp):
        n += 1
        row = json.loads(blob.decode("utf-8", "replace"))
        c[str(row.get("source"))] += 1
        if n >= 5000:
            break
print("n", n, dict(c))
PY
