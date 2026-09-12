#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
PY=/home/asus/arzlm/.venv/bin/python
cd /home/asus/arzlm

echo "=== FineWeb-Edu ==="
$PY -c 'from arzlm.data.sources import iter_fineweb_edu; d=next(iter_fineweb_edu()); print(d.dataset_id, len(d.text), d.native_id[:60])'

echo "=== FineMath datasets stream ==="
$PY -c 'from arzlm.data.sources import iter_finemath_4plus; d=next(iter_finemath_4plus()); print(d.dataset_id, len(d.text), d.native_id[:80])' || echo FINE MATH_FAIL

echo "=== peS2o s2orc ==="
$PY -c 'from arzlm.data.network import ByteCounter; from arzlm.data.sources import iter_pes2o_s2orc; c=ByteCounter("s"); d=next(iter_pes2o_s2orc(counter=c, shard_indices=[10], max_docs_per_shard=None)); print(d.dataset_id, d.extra.get("source"), len(d.text), c.snapshot())' || echo PES2O_FAIL

echo "=== Stack-Edu Python + SWH ==="
$PY -c 'from arzlm.data.network import ByteCounter; from arzlm.data.sources import iter_stack_edu_language; c=ByteCounter("c"); d=next(iter_stack_edu_language("Python", counter=c, fetch_workers=1)); print(d.dataset_id, len(d.text), d.native_id, c.snapshot())' || echo CODE_FAIL

echo DONE
