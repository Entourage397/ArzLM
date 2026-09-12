#!/usr/bin/env bash
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
cd /home/asus/arzlm
PY=.venv/bin/python
WIN=/mnt/c/Users/ASUS/ArzLM
DEST="${WIN}/docs/tune"
mkdir -p "${DEST}"

for rel in \
  configs/tune/muon-wd0p05-lr0p05-2m.yaml \
  configs/tune/muon-wd0p2-lr0p05-2m.yaml
do
  python3 -c "from pathlib import Path; s=Path('/mnt/c/Users/ASUS/ArzLM')/'${rel}'; d=Path('/home/asus/arzlm')/'${rel}'; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(s.read_bytes().replace(b'\r\n', b'\n'))"
done

# Preserve the one successful compile WD trial if present.
if [[ -f runs/tune/muon-wd0p05-lr0p05-2m-compile/train_result.json ]]; then
  cp runs/tune/muon-wd0p05-lr0p05-2m-compile/train_result.json "${DEST}/wsl-muon-wd0p05-lr0p05-2m-compile-train-result.json"
  cp runs/tune/muon-wd0p05-lr0p05-2m-compile/experiment.json "${DEST}/wsl-muon-wd0p05-lr0p05-2m-compile-experiment.json"
  cp runs/tune/muon-wd0p05-lr0p05-2m-compile/metrics.jsonl "${DEST}/wsl-muon-wd0p05-lr0p05-2m-compile-metrics.jsonl"
  echo copied-compile-wd05
fi

echo "=== muon-wd0p05-lr0p05-2m (eager) ==="
"${PY}" -m arzlm train --config configs/tune/muon-wd0p05-lr0p05-2m.yaml
cp runs/tune/muon-wd0p05-lr0p05-2m/train_result.json "${DEST}/wsl-muon-wd0p05-lr0p05-2m-train-result.json"
cp runs/tune/muon-wd0p05-lr0p05-2m/experiment.json "${DEST}/wsl-muon-wd0p05-lr0p05-2m-experiment.json"
cp runs/tune/muon-wd0p05-lr0p05-2m/metrics.jsonl "${DEST}/wsl-muon-wd0p05-lr0p05-2m-metrics.jsonl"

echo "=== muon-wd0p2-lr0p05-2m (eager) ==="
"${PY}" -m arzlm train --config configs/tune/muon-wd0p2-lr0p05-2m.yaml
cp runs/tune/muon-wd0p2-lr0p05-2m/train_result.json "${DEST}/wsl-muon-wd0p2-lr0p05-2m-train-result.json"
cp runs/tune/muon-wd0p2-lr0p05-2m/experiment.json "${DEST}/wsl-muon-wd0p2-lr0p05-2m-experiment.json"
cp runs/tune/muon-wd0p2-lr0p05-2m/metrics.jsonl "${DEST}/wsl-muon-wd0p2-lr0p05-2m-metrics.jsonl"

echo WD_EAGER_LR05_DONE
