#!/usr/bin/env bash
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
cd /home/asus/arzlm
WIN=/mnt/c/Users/ASUS/ArzLM
cp -a "${WIN}/configs/tune/muon-wd0p05-lr0p075-2m.yaml" "${WIN}/configs/tune/muon-wd0p1-lr0p075-2m.yaml" "${WIN}/configs/tune/muon-wd0p2-lr0p075-2m.yaml" configs/tune/
DEST="${WIN}/docs/tune"
mkdir -p "${DEST}"
for lr in 0.05 0.075; do
  d="runs/tune/muon-10m-lr${lr}"
  tag="wsl-muon-10m-lr${lr}"
  cp "$d/train_result.json" "${DEST}/${tag}-train-result.json"
  cp "$d/experiment.json" "${DEST}/${tag}-experiment.json"
  cp "$d/metrics.jsonl" "${DEST}/${tag}-metrics.jsonl"
done
.venv/bin/python -m arzlm train --config configs/tune/muon-wd0p05-lr0p075-2m.yaml
.venv/bin/python -m arzlm train --config configs/tune/muon-wd0p2-lr0p075-2m.yaml
echo WD_DONE
