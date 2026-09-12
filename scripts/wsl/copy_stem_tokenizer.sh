#!/usr/bin/env bash
set -euo pipefail
cd /home/asus/arzlm
mkdir -p /mnt/c/Users/ASUS/ArzLM/tokenizer/stem-v1
rsync -a tokenizer/stem-v1/ /mnt/c/Users/ASUS/ArzLM/tokenizer/stem-v1/
sha256sum tokenizer/stem-v1/tokenizer.json
.venv/bin/python - <<'PY'
from litgpt.tokenizer import Tokenizer
t = Tokenizer("/home/asus/arzlm/tokenizer/stem-v1")
print("vocab", t.vocab_size, "eos", t.eos_id, "bos", t.bos_id)
PY
