#!/usr/bin/env bash
# Long-running STEM tokenizer study + optional 1B pack.
# Usage: run_stem_prep.sh tokenizer|pack|all
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

WIN=/mnt/c/Users/ASUS/ArzLM
LIN=/home/asus/arzlm
PHASE="${1:-tokenizer}"
LOG="${LIN}/data/prepared/stem-prep.log"
mkdir -p "${LIN}/data/prepared"
cd "${LIN}"

rsync -a \
  --exclude '.venv/' \
  --exclude 'hf-cache/' \
  --exclude 'runs/' \
  --exclude 'data/' \
  --exclude 'tokenizer/trained/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  "${WIN}/" "${LIN}/"

echo "==== $(date -Is) PHASE=${PHASE} HEAD=$(git rev-parse HEAD) ====" | tee -a "${LOG}"

if [[ "${PHASE}" == "tokenizer" || "${PHASE}" == "all" ]]; then
  echo "=== STEM tokenizer study ===" | tee -a "${LOG}"
  .venv/bin/python -m arzlm.tokenizer.stem_study 2>&1 | tee -a "${LOG}"
  mkdir -p "${WIN}/docs/data"
  cp -f docs/data/tokenizer-stem-comparison.json "${WIN}/docs/data/tokenizer-stem-comparison.json" || true
  echo TOKENIZER_STUDY_DONE | tee -a "${LOG}"
fi

if [[ "${PHASE}" == "pack" || "${PHASE}" == "all" ]]; then
  TOK=tokenizer/trained
  if [[ -f tokenizer/stem-v1/tokenizer.json && -f docs/data/tokenizer-stem-comparison.json ]]; then
    if .venv/bin/python -c 'import json,sys; p=json.load(open("docs/data/tokenizer-stem-comparison.json")); sys.exit(0 if (p.get("decision") or {}).get("keep_current") is False else 1)'; then
      TOK=tokenizer/stem-v1
    fi
  fi
  echo "=== pack ArzLM-STEM-1B-v1 using ${TOK} ===" | tee -a "${LOG}"
  .venv/bin/python -m arzlm prepare-stem --preset 1b --source remote --tokenizer-dir "${TOK}" --out-dir data/prepared/arzlm-stem-1b-v1 2>&1 | tee -a "${LOG}"
  echo PREPARE_STEM_1B_DONE | tee -a "${LOG}"
  cp -f data/prepared/arzlm-stem-1b-v1/manifest.json "${WIN}/docs/data/arzlm-stem-1b-v1-manifest.json" || true
fi
