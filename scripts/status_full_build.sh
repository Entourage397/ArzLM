#!/usr/bin/env bash
# Status of the crash-safe full build. Does not start or stop training.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
echo "=== ArzLM full-build status ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "repo: ${ROOT}"
echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "HEAD: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "baseline-v0: $(git rev-parse 'baseline-v0^{commit}' 2>/dev/null || echo unknown)"
echo
echo "=== tmux ==="
tmux ls 2>/dev/null | grep -E 'arzlm-full-build|full-build' || echo "no arzlm-full-build session"
echo
echo "=== lock / pid ==="
if [[ -f state/full-build/pid ]]; then
  pid="$(cat state/full-build/pid)"
  echo "recorded_pid: ${pid}"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "pid_alive: yes"
    ps -p "${pid}" -o pid,etime,cmd || true
  else
    echo "pid_alive: no"
  fi
else
  echo "recorded_pid: none"
fi
if command -v lsof >/dev/null 2>&1; then
  lsof state/full-build/full-build.lock 2>/dev/null | head || true
fi
echo
echo "=== state.json ==="
if [[ -f state/full-build/state.json ]]; then
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("state/full-build/state.json")
d = json.loads(p.read_text())
print("current_phase:", d.get("current_phase"))
print("pid:", d.get("pid"))
print("last_error:", d.get("last_error"))
print("started_at:", d.get("started_at"))
print("phases:")
for name, rec in (d.get("phases") or {}).items():
    print(f"  {name:18} {rec.get('status')}")
PY
else
  echo "no state.json yet"
fi
echo
echo "=== BUILD_COMPLETE ==="
if [[ -f state/full-build/BUILD_COMPLETE.json ]]; then
  echo "YES"
  cat state/full-build/BUILD_COMPLETE.json
else
  echo "NO"
fi
echo
echo "=== latest training metrics ==="
for run in runs/muon-stem-sanity-50m runs/arzlm-100m-base-2b runs/arzlm-100m-sft; do
  echo "-- ${run} --"
  if [[ -f "${run}/train_result.json" ]]; then
    .venv/bin/python - "${run}/train_result.json" <<'PY'
import json,sys
d=json.loads(open(sys.argv[1]).read())
keys=("loss","val_loss","tokens","steps","tokens_per_sec","finite")
print({k:d.get(k) for k in keys})
PY
  elif [[ -f "${run}/metrics.jsonl" ]]; then
    tail -n 3 "${run}/metrics.jsonl"
  else
    echo "no metrics"
  fi
  ls -td "${run}"/step-* 2>/dev/null | head -3 || true
done
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
echo
echo "=== disk ==="
df -h "${ROOT}" / /home 2>/dev/null | head
echo
echo "=== last orchestrator log ==="
tail -n 20 logs/full-build/orchestrator.log 2>/dev/null || echo "no orchestrator log"
echo
echo "resume: tmux attach -t arzlm-full-build"
echo "or: bash scripts/run_full_build.sh  (idempotent; flock prevents overlap)"
