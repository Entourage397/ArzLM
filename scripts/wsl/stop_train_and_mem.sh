#!/usr/bin/env bash
set -euo pipefail
pkill -f 'arzlm train' 2>/dev/null || true
pkill -f 'python -m arzlm' 2>/dev/null || true
sleep 3
echo "=== memory ==="
free -h
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "=== python ==="
pgrep -af python || echo none
