#!/usr/bin/env bash
# Idempotent: start the full-build orchestrator if it is not already running.
# Never starts a second job while the flock is held.
set -euo pipefail
ROOT="/home/asus/arzlm"
cd "${ROOT}"
mkdir -p logs/full-build state/full-build
LOG="${ROOT}/logs/full-build/ensure.log"
if [[ -f state/full-build/BUILD_COMPLETE.json ]]; then
  echo "$(date -Is) BUILD_COMPLETE" | tee -a "${LOG}"
  exit 0
fi
if ! flock -n state/full-build/full-build.lock true; then
  echo "$(date -Is) flock-held" >>"${LOG}"
  exit 0
fi
cmd="cd ${ROOT} && bash scripts/run_full_build.sh; echo ORCH_EXIT:\$?; exec sleep infinity"
if ! tmux has-session -t arzlm-full-build 2>/dev/null; then
  tmux new-session -d -s arzlm-full-build -n full-build2 "${cmd}"
  echo "$(date -Is) started session arzlm-full-build:full-build2" | tee -a "${LOG}"
  exit 0
fi
existing="$(tmux list-windows -t arzlm-full-build -F '#{window_name}' 2>/dev/null || true)"
name="full-build2"
if echo "${existing}" | grep -qx "${name}"; then
  n=3
  name="full-build${n}"
  while echo "${existing}" | grep -qx "${name}"; do
    n=$((n + 1))
    name="full-build${n}"
  done
fi
tmux new-window -t arzlm-full-build -n "${name}" "${cmd}"
echo "$(date -Is) started window ${name}" | tee -a "${LOG}"
