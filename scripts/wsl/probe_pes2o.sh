#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
cd /home/asus/arzlm
rsync -a /mnt/c/Users/ASUS/ArzLM/src/arzlm/data/catalog.py src/arzlm/data/catalog.py
rsync -a /mnt/c/Users/ASUS/ArzLM/src/arzlm/data/sources.py src/arzlm/data/sources.py
.venv/bin/python - <<'PY'
from arzlm.data.network import ByteCounter
from arzlm.data.sources import iter_pes2o_s2orc
c = ByteCounter("s")
d = next(iter_pes2o_s2orc(counter=c, shard_indices=[10], max_docs_per_shard=None))
print("ok", d.extra.get("source"), d.extra.get("shard"), len(d.text), c.snapshot())
print(d.text[:240].replace("\n", " "))
PY
