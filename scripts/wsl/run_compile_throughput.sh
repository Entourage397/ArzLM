#!/usr/bin/env bash
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
cd /home/asus/arzlm
WIN=/mnt/c/Users/ASUS/ArzLM
DEST="${WIN}/docs/tune"
mkdir -p "${DEST}"
for tag in muon-wd0p05-lr0p075-2m muon-wd0p2-lr0p075-2m; do
  d="runs/tune/${tag}"
  cp "$d/train_result.json" "${DEST}/wsl-${tag}-train-result.json"
  cp "$d/experiment.json" "${DEST}/wsl-${tag}-experiment.json"
  cp "$d/metrics.jsonl" "${DEST}/wsl-${tag}-metrics.jsonl"
done
cp -a "${WIN}/configs/adamw-50m.yaml" "${WIN}/configs/muon-50m.yaml" configs/
echo "=== Muon compile micro-4 train throughput (40 steps, not 50M) ==="
.venv/bin/python -m arzlm train \
  --config configs/tune/muon-jordan-lr0.075-2m.yaml \
  --compile true \
  --micro-batch-size 4 \
  --max-steps 40 \
  --out-dir runs/tune/muon-compile-throughput-mb4
echo "=== AdamW compile micro-4 train throughput (40 steps, not 50M) ==="
.venv/bin/python -m arzlm train \
  --config configs/tune/adamw-2m.yaml \
  --compile true \
  --micro-batch-size 4 \
  --max-steps 40 \
  --out-dir runs/tune/adamw-compile-throughput-mb4
cp runs/tune/muon-compile-throughput-mb4/train_result.json "${DEST}/wsl-muon-compile-throughput-mb4.json"
cp runs/tune/adamw-compile-throughput-mb4/train_result.json "${DEST}/wsl-adamw-compile-throughput-mb4.json"
echo THROUGHPUT_DONE
