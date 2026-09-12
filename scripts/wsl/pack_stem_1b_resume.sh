#!/usr/bin/env bash
# Resume-safe ArzLM-STEM-1B-v1 pack on WSL. Restarts after segfaults only.
set -u
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false

WIN=/mnt/c/Users/ASUS/ArzLM
LIN=/home/asus/arzlm
LOG="${LIN}/data/prepared/stem-prep.log"
mkdir -p "${LIN}/data/prepared"
cd "${LIN}"

rsync -a --checksum \
  --exclude '.venv/' \
  --exclude 'hf-cache/' \
  --exclude 'runs/' \
  --exclude 'data/' \
  --exclude 'tokenizer/trained/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  --exclude '.cursor/' \
  "${WIN}/" "${LIN}/"

echo "==== $(date -Is) pack-resume HEAD=$(git rev-parse HEAD) ====" | tee -a "${LOG}"
TOK=tokenizer/stem-v1
if [[ ! -f "${TOK}/tokenizer.json" ]]; then
  TOK=tokenizer/trained
fi
echo "Using tokenizer ${TOK}" | tee -a "${LOG}"

MANIFEST="${LIN}/data/prepared/arzlm-stem-1b-v1/manifest.json"
while true; do
  if [[ -f "${MANIFEST}" ]]; then
    echo "PREPARE_STEM_1B_DONE" | tee -a "${LOG}"
    cp -f "${MANIFEST}" "${WIN}/docs/data/arzlm-stem-1b-v1-manifest.json" || true
    break
  fi
  set +e
  .venv/bin/python -m arzlm prepare-stem --preset 1b --source remote --tokenizer-dir "${TOK}" --out-dir data/prepared/arzlm-stem-1b-v1 2>&1 | tee -a "${LOG}"
  code=${PIPESTATUS[0]}
  set -e
  echo "prepare-stem exited ${code} at $(date -Is)" | tee -a "${LOG}"
  if [[ ${code} -eq 0 ]]; then
    continue
  fi
  if [[ ${code} -eq 139 || ${code} -eq 137 || ${code} -eq 143 || ${code} -eq 134 ]]; then
    echo "crash ${code}; resume in 8s" | tee -a "${LOG}"
    sleep 8
    continue
  fi
  echo "non-crash failure ${code}; stopping" | tee -a "${LOG}"
  exit ${code}
done
