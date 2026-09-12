# baseline-v0

Frozen snapshot of the validated Windows ArzLM-100M baseline.

Do not edit these files after the `baseline-v0` git tag. Later experiments copy configs; they must not rewrite this directory.

| File | Role |
| --- | --- |
| `windows-benchmark.json` | RTX 4070 Laptop micro-batch sweep (best: batch 2, ~19,060 tok/s) |
| `stage2-train-result.json` | 10M-token FineWeb-Edu run |
| `stage1-overfit-train-result.json` | 1M-token overfit sanity run |
| `stage0-smoke-train-result.json` | 5-step smoke |
| `model-arzlm-100m.yaml` | Architecture |
| `train-bench-10m.yaml` | Stage-2 training recipe |
| `stage2-model-config.yaml` | LitGPT config written with the Stage-2 checkpoint |
| `MANIFEST.json` | SHA-256 hashes of the files above |
