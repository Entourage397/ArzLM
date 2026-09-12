#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
PY=/home/asus/arzlm/.venv/bin/python
cd /home/asus/arzlm
rsync -a /mnt/c/Users/ASUS/ArzLM/src/arzlm/data/sources.py src/arzlm/data/sources.py

echo "=== peS2o ==="
$PY - <<'PY'
from arzlm.data.network import ByteCounter
from arzlm.data.sources import iter_pes2o_s2orc
c = ByteCounter("s")
d = next(iter_pes2o_s2orc(counter=c, shard_indices=[10], max_docs_per_shard=None))
print(d.dataset_id, d.extra.get("source"), len(d.text), c.snapshot())
print(d.text[:180].replace("\n", " "))
PY

echo "=== stack-edu ==="
$PY - <<'PY'
from arzlm.data.network import ByteCounter
from arzlm.data.sources import iter_stack_edu_language
c = ByteCounter("c")
d = next(iter_stack_edu_language("Python", counter=c, fetch_workers=1))
print(len(d.text), d.native_id, c.snapshot())
PY
