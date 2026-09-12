#!/usr/bin/env bash
# AdamW 50M control. Requires data/prepared/50m. Compile is mandatory.
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export TOKENIZERS_PARALLELISM=false

WIN=/mnt/c/Users/ASUS/ArzLM
LIN=/home/asus/arzlm
cd "${LIN}"

rsync -a \
  --exclude '.venv/' \
  --exclude 'hf-cache/' \
  --exclude 'runs/' \
  --exclude 'data/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  "${WIN}/" "${LIN}/"

if [[ ! -f data/prepared/50m/train.bin || ! -f data/prepared/50m/val.bin ]]; then
  echo "ERROR: data/prepared/50m is incomplete" >&2
  exit 1
fi

# Drop any eager/failed partial run so resume:auto cannot pick it up.
if [[ -d runs/adamw-50m ]]; then
  echo "Removing incomplete runs/adamw-50m"
  rm -rf runs/adamw-50m
fi

echo "HEAD=$(git rev-parse HEAD)"
echo "=== adamw-50m compile, CUDA graphs on, micro-4 ==="
.venv/bin/python -m arzlm train --config configs/adamw-50m.yaml
echo ADAMW_50M_DONE
