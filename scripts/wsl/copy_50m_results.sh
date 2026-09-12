#!/usr/bin/env bash
# Copy 50M pack hashes + train artifacts to the Windows docs tree.
set -euo pipefail
ROOT="${ARZLM_WSL_ROOT:-${HOME}/arzlm}"
WIN="${ARZLM_WIN_SRC:-/mnt/c/Users/ASUS/ArzLM}"
DEST="${WIN}/docs/experiments/50m-optimizer-comparison"
mkdir -p "${DEST}"

cp -a "${ROOT}/runs/wsl/50m-pack-hashes.json" "${DEST}/50m-pack-hashes.json" 2>/dev/null || true
cp -a "${ROOT}/data/prepared/50m/meta.json" "${DEST}/50m-meta.json"

for run in adamw-50m muon-50m; do
  src="${ROOT}/runs/${run}"
  if [[ -d "${src}" ]]; then
    cp -a "${src}/train_result.json" "${DEST}/${run}-train-result.json" 2>/dev/null || true
    cp -a "${src}/experiment.json" "${DEST}/${run}-experiment.json" 2>/dev/null || true
    cp -a "${src}/metrics.jsonl" "${DEST}/${run}-metrics.jsonl" 2>/dev/null || true
  fi
done
echo "Copied 50M artifacts to ${DEST}"
ls -l "${DEST}"
