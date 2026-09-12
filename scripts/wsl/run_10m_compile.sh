#!/usr/bin/env bash
# Sequential compiled 10M Muon LR confirmation. Does not start 50M.
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
cd /home/asus/arzlm
PY=.venv/bin/python
WIN=/mnt/c/Users/ASUS/ArzLM
DEST="${WIN}/docs/tune"

echo "=== 10M compile lr=0.05 ==="
"${PY}" -m arzlm train --config configs/tune/muon-10m-lr0.05-compile.yaml
echo "=== 10M compile lr=0.075 ==="
"${PY}" -m arzlm train --config configs/tune/muon-10m-lr0.075-compile.yaml

mkdir -p "${DEST}"
for tag in muon-10m-lr0.05-compile muon-10m-lr0.075-compile; do
  d="runs/tune/${tag}"
  cp "$d/train_result.json" "${DEST}/wsl-${tag}-train-result.json"
  cp "$d/experiment.json" "${DEST}/wsl-${tag}-experiment.json"
  cp "$d/metrics.jsonl" "${DEST}/wsl-${tag}-metrics.jsonl"
  echo "copied ${tag}"
done
echo 10M_COMPILE_DONE
