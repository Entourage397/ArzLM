#!/usr/bin/env bash
# Muon hybrid: Linux eager vs model torch.compile. Micro-batch 2 and 4 only.
set -euo pipefail

ROOT="${ARZLM_WSL_ROOT:-${HOME}/arzlm}"
cd "${ROOT}"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing ${ROOT}/.venv. Run scripts/wsl/bootstrap.sh first." >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export PYTHONUNBUFFERED=1
PY=".venv/bin/python"
DATA="${ROOT}/data/prepared/tiny"
OUT="${ROOT}/runs/wsl"
mkdir -p "${OUT}"

echo "=== Muon eager BF16 ==="
"${PY}" -m arzlm bench \
  --optimizer muon \
  --data-dir "${DATA}" \
  --compile false \
  --compile-scope model \
  --seq-length 1024 \
  --warmup-steps 8 \
  --measure-steps 20 \
  --microbatches 2,4 \
  --seed 42 \
  --out "${OUT}/linux-muon-eager-bf16.json"

echo "=== Muon model torch.compile BF16 ==="
"${PY}" -m arzlm bench \
  --optimizer muon \
  --data-dir "${DATA}" \
  --compile true \
  --compile-scope model \
  --seq-length 1024 \
  --warmup-steps 20 \
  --measure-steps 20 \
  --microbatches 2,4 \
  --seed 42 \
  --out "${OUT}/linux-muon-compile-bf16.json"

echo "=== Muon train_step torch.compile (micro-2 only, optional) ==="
"${PY}" -m arzlm bench \
  --optimizer muon \
  --data-dir "${DATA}" \
  --compile true \
  --compile-scope train_step \
  --seq-length 1024 \
  --warmup-steps 12 \
  --measure-steps 10 \
  --microbatches 2 \
  --seed 42 \
  --out "${OUT}/linux-muon-compile-train-step-bf16.json" || echo "train_step compile failed (non-default)"

echo "Wrote Muon compile benches under ${OUT}"
