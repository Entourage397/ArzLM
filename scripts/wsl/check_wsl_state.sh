#!/usr/bin/env bash
set -euo pipefail
echo "=== processes ==="
pgrep -af 'arzlm|prepare_50m|probe_fineweb|probe_hf' || echo 'none'
echo "=== 50m dir ==="
ls -la /home/asus/arzlm/data/prepared/50m 2>/dev/null || echo 'no 50m dir'
echo "=== tokenizer ==="
ls -l /home/asus/arzlm/tokenizer/trained/tokenizer.json
echo "=== wsl head ==="
git -C /home/asus/arzlm rev-parse HEAD
