#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
cd /home/asus/arzlm
rsync -a /mnt/c/Users/ASUS/ArzLM/src/arzlm/data/sources.py src/arzlm/data/sources.py
.venv/bin/python - <<'PY'
from urllib.request import Request, urlopen
from arzlm.data.sources import pes2o_shard_url, pes2o_shard_names, _iter_gzip_json_lines
import json
url = pes2o_shard_url(pes2o_shard_names()[0])
print("url", url)
req = Request(url, headers={"User-Agent": "arzlm-stem/1.0", "Accept-Encoding": "identity"})
n = 0
s2orc = 0
s2ag = 0
bad = 0
with urlopen(req, timeout=120) as resp:
    print("ctype", resp.headers.get("Content-Type"), "enc", resp.headers.get("Content-Encoding"), "len", resp.headers.get("Content-Length"))
    for blob in _iter_gzip_json_lines(resp):
        n += 1
        if n <= 3:
            print("line", n, "bytes", len(blob), "start", blob[:80])
        try:
            row = json.loads(blob.decode("utf-8", "replace"))
        except Exception as exc:
            bad += 1
            if n <= 5:
                print("bad", type(exc), exc)
            continue
        src = row.get("source") if isinstance(row, dict) else None
        if src == "s2orc":
            s2orc += 1
        elif src == "s2ag":
            s2ag += 1
        if n >= 200:
            break
print("n", n, "s2orc", s2orc, "s2ag", s2ag, "bad", bad)
PY
