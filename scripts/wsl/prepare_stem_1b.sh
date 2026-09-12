#!/usr/bin/env bash
# Sync Windows tree into ~/arzlm (keep WSL data/venv), probe STEM sources,
# run tokenizer study, then pack ArzLM-STEM-1B-v1.
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

WIN=/mnt/c/Users/ASUS/ArzLM
LIN=/home/asus/arzlm
PHASE="${1:-all}"
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

echo "WSL HEAD: $(git rev-parse HEAD)"
echo "PHASE=${PHASE}"

if [[ "${PHASE}" == "probe" || "${PHASE}" == "all" ]]; then
  echo "=== probe official STEM sources ==="
  .venv/bin/python -m arzlm.data.probe_stem | tee /tmp/arzlm-stem-probe.json
fi

if [[ "${PHASE}" == "tokenizer" || "${PHASE}" == "all" ]]; then
  echo "=== STEM tokenizer study (held-out samples; does not pack 1B) ==="
  .venv/bin/python -m arzlm.tokenizer.stem_study
fi

if [[ "${PHASE}" == "pack" || "${PHASE}" == "all" ]]; then
  echo "=== pack ArzLM-STEM-1B-v1 (resume-safe) ==="
  TOK=tokenizer/trained
  if [[ -f tokenizer/stem-v1/tokenizer.json && -f docs/data/tokenizer-stem-comparison.json ]]; then
    if .venv/bin/python -c 'import json,sys; p=json.load(open("docs/data/tokenizer-stem-comparison.json")); sys.exit(0 if (p.get("decision") or {}).get("keep_current") is False else 1)'; then
      TOK=tokenizer/stem-v1
    fi
  fi
  echo "Using tokenizer ${TOK}"
  .venv/bin/python -m arzlm prepare-stem --preset 1b --source remote --tokenizer-dir "${TOK}" --out-dir data/prepared/arzlm-stem-1b-v1
  echo PREPARE_STEM_1B_DONE
fi
