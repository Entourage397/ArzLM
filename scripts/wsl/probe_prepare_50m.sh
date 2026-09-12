#!/usr/bin/env bash
set -euo pipefail
cd /home/asus/arzlm
export PYTHONUNBUFFERED=1
export HF_HOME=/home/asus/arzlm/hf-cache
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo "=== nvidia ==="
nvidia-smi || true
echo "=== probe datasets streaming ==="
.venv/bin/python - <<'PY'
from datasets import load_dataset
print("import ok")
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
print("dataset ok", ds)
row = next(iter(ds))
print("first keys", list(row.keys()))
print("text_len", len(row.get("text") or ""))
print("STREAM_OK")
PY
echo "=== probe tokenizer encode ==="
.venv/bin/python - <<'PY'
from litgpt.tokenizer import Tokenizer
from pathlib import Path
t = Tokenizer(Path("/home/asus/arzlm/tokenizer/trained"))
ids = t.encode("photosynthesis converts light energy", bos=False, eos=False)
print("n_ids", len(ids), "eos", t.eos_id, "vocab", t.vocab_size)
print("TOKENIZER_OK")
PY
echo PROBE_OK
