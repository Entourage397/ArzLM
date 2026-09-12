#!/usr/bin/env bash
# Muon 50M candidate. Same packed 50m data as AdamW. One GPU job.
# CUDA graphs are disabled automatically for muon_hybrid.
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
if [[ ! -f runs/adamw-50m/train_result.json ]]; then
  echo "ERROR: AdamW control missing; run AdamW first" >&2
  exit 1
fi
if [[ -d runs/muon-50m ]]; then
  echo "Removing incomplete runs/muon-50m"
  rm -rf runs/muon-50m
fi

echo "HEAD=$(git rev-parse HEAD)"
echo "=== muon-50m compile, CUDA graphs off, lr=0.05 wd=0.10, micro-4 ==="
.venv/bin/python -m arzlm train --config configs/muon-50m.yaml
echo MUON_50M_DONE
