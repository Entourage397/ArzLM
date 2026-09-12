#!/usr/bin/env bash
# Retry AdamW 50M after a failed first compile. Fresh process, no faulthandler.
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export TOKENIZERS_PARALLELISM=false
# Isolate this run from any corrupted FX/Inductor cache left by prior jobs.
export TORCHINDUCTOR_CACHE_DIR="${HOME}/arzlm/.cache/inductor-50m-adamw"
export TRITON_CACHE_DIR="${HOME}/arzlm/.cache/triton-50m-adamw"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

LIN=/home/asus/arzlm
cd "${LIN}"

pkill -f 'arzlm train' 2>/dev/null || true
pkill -f 'arzlm.data.encode_worker' 2>/dev/null || true
sleep 2
nvidia-smi -L
echo "HEAD=$(git rev-parse HEAD)"
echo "=== retry adamw-50m compile ==="
.venv/bin/python -m arzlm train --config configs/adamw-50m.yaml
echo ADAMW_50M_DONE
