#!/usr/bin/env bash
# Copy measured WSL JSON reports back to the Windows working tree (docs only).
set -euo pipefail

ROOT="${ARZLM_WSL_ROOT:-${HOME}/arzlm}"
WIN_SRC="${ARZLM_WIN_SRC:-/mnt/c/Users/ASUS/ArzLM}"
DEST="${WIN_SRC}/docs/wsl/results"
mkdir -p "${DEST}"
cp -a "${ROOT}/runs/wsl/"*.json "${DEST}/" 2>/dev/null || true
for name in muon-ckpt-false muon-ckpt-true; do
  src="${ROOT}/runs/wsl/${name}/ckpt_smoke.json"
  if [[ -f "${src}" ]]; then
    cp -a "${src}" "${DEST}/linux-${name}.json"
  fi
done
echo "Copied WSL results to ${DEST}"
