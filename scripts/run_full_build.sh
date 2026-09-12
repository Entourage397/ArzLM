#!/usr/bin/env bash
# Idempotent ArzLM-100M full build. Safe to re-run. Holds flock so only one
# instance runs. Intended to execute inside tmux session arzlm-full-build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p state/full-build logs/full-build
LOCK="${ROOT}/state/full-build/full-build.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "full-build already running (lock ${LOCK})" >&2
  exit 1
fi
echo $$ > state/full-build/pid
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "missing ${PY}" >&2
  exit 1
fi
echo "full-build start $(date -Is) pwd=${ROOT} pid=$$" | tee -a logs/full-build/orchestrator.log
set +e
"${PY}" -m arzlm.fullbuild --all
rc=$?
set -e
echo "full-build exit ${rc} $(date -Is)" | tee -a logs/full-build/orchestrator.log
exit "${rc}"
