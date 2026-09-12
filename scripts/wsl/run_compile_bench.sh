#!/usr/bin/env bash
# Linux eager vs torch.compile BF16 throughput sweep.
# Same architecture, packed data, seed, and seq_length as the Windows baseline.
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

echo "=== Linux eager BF16 ==="
"${PY}" -m arzlm bench \
  --data-dir "${DATA}" \
  --compile false \
  --seq-length 1024 \
  --warmup-steps 8 \
  --measure-steps 20 \
  --microbatches 1,2,4 \
  --seed 42 \
  --out "${OUT}/linux-eager-bf16.json"

echo "=== Linux torch.compile BF16 ==="
"${PY}" -m arzlm bench \
  --data-dir "${DATA}" \
  --compile true \
  --seq-length 1024 \
  --warmup-steps 20 \
  --measure-steps 20 \
  --microbatches 1,2,4 \
  --seed 42 \
  --out "${OUT}/linux-compile-bf16.json"

echo "Wrote ${OUT}/linux-eager-bf16.json and ${OUT}/linux-compile-bf16.json"
