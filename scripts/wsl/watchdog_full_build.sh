#!/usr/bin/env bash
# Keep the full-build orchestrator alive across WSL idle. Does not start a
# second job while the flock is held. Safe to run in a tmux window forever.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
LOG="${ROOT}/logs/full-build/watchdog.log"
mkdir -p logs/full-build state/full-build
echo "$(date -Is) watchdog start pid=$$" >>"${LOG}"
while true; do
  if [[ -f state/full-build/BUILD_COMPLETE.json ]]; then
    echo "$(date -Is) BUILD_COMPLETE present; watchdog exit" >>"${LOG}"
    exit 0
  fi
  if flock -n state/full-build/full-build.lock true; then
    echo "$(date -Is) flock-free; starting orchestrator" >>"${LOG}"
    bash "${ROOT}/scripts/run_full_build.sh" >>"${LOG}" 2>&1 || true
  fi
  sleep 60
done
