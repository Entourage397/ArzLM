#!/usr/bin/env bash
# Sync Windows tree into ~/arzlm without deleting WSL data/venv, then pack 50M once.
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

WIN=/mnt/c/Users/ASUS/ArzLM
LIN=/home/asus/arzlm
cd "${LIN}"

echo "Windows HEAD: $(git -C "${WIN}" rev-parse HEAD)"
echo "WSL HEAD before sync: $(git rev-parse HEAD 2>/dev/null || echo none)"

rsync -a \
  --exclude '.venv/' \
  --exclude 'hf-cache/' \
  --exclude 'runs/' \
  --exclude 'data/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  "${WIN}/" "${LIN}/"

echo "WSL HEAD after sync: $(git rev-parse HEAD)"
echo "=== prepare 50m FineWeb-Edu sample-10BT, reuse tokenizer ==="
.venv/bin/python -m arzlm prepare --preset 50m --source fineweb-edu
echo PREPARE_50M_DONE
