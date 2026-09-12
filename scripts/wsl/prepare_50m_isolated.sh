#!/usr/bin/env bash
# Sync Windows tree (no --delete), wipe incomplete 50m pack, pack once with
# isolated FineWeb encoding. Do not regenerate between AdamW and Muon.
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONUNBUFFERED=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export HF_HOME=/home/asus/arzlm/hf-cache
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false

WIN=/mnt/c/Users/ASUS/ArzLM
LIN=/home/asus/arzlm
cd "${LIN}"

echo "Windows HEAD: $(git -C "${WIN}" rev-parse HEAD)"
echo "WSL HEAD before sync: $(git rev-parse HEAD 2>/dev/null || echo none)"

rsync -a \
  --exclude '.venv/' \
  --exclude 'hf-cache/' \
  --exclude 'runs/' \
  --exclude 'data/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  "${WIN}/" "${LIN}/"

echo "WSL HEAD after sync: $(git rev-parse HEAD)"

if [[ ! -f tokenizer/trained/tokenizer.json ]]; then
  echo "ERROR: tokenizer/trained/tokenizer.json missing; refuse to pack 50m" >&2
  exit 1
fi

rm -rf data/prepared/50m
mkdir -p data/prepared/50m runs/wsl

echo "=== prepare 50m FineWeb-Edu sample-10BT, reuse tokenizer, isolated encode ==="
.venv/bin/python -X faulthandler -m arzlm prepare --preset 50m --source fineweb-edu

echo "=== pack hashes ==="
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

root = Path("data/prepared/50m")
tok = Path("tokenizer/trained/tokenizer.json")
meta = json.loads((root / "meta.json").read_text())
report = {
    "train_bin": str(root / "train.bin"),
    "val_bin": str(root / "val.bin"),
    "train_bytes": (root / "train.bin").stat().st_size,
    "val_bytes": (root / "val.bin").stat().st_size,
    "train_sha256": sha256(root / "train.bin"),
    "val_sha256": sha256(root / "val.bin"),
    "tokenizer_sha256": sha256(tok),
    "meta": {
        "preset": meta.get("preset"),
        "source": meta.get("source"),
        "seed": meta.get("seed"),
        "splits": meta.get("splits"),
        "n_documents": meta.get("n_documents"),
        "n_skipped_unencodable": meta.get("n_skipped_unencodable"),
        "n_empty_docs": meta.get("n_empty_docs"),
        "encoder_worker_restarts": meta.get("encoder_worker_restarts"),
        "litgpt_vocab_size": meta.get("litgpt_vocab_size"),
        "eos_id": meta.get("eos_id"),
    },
}
(Path("runs/wsl") / "50m-pack-hashes.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
echo PREPARE_50M_DONE
