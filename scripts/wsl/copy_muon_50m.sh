#!/usr/bin/env bash
set -euo pipefail
WIN=/mnt/c/Users/ASUS/ArzLM
DEST="${WIN}/docs/experiments/50m-optimizer-comparison"
mkdir -p "${DEST}"
cp -a /home/asus/arzlm/runs/muon-50m/train_result.json "${DEST}/muon-50m-train-result.json"
cp -a /home/asus/arzlm/runs/muon-50m/experiment.json "${DEST}/muon-50m-experiment.json"
cp -a /home/asus/arzlm/runs/muon-50m/metrics.jsonl "${DEST}/muon-50m-metrics.jsonl"
echo copied muon artifacts
ls -l "${DEST}"
