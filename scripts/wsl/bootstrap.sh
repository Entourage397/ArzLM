#!/usr/bin/env bash
# Bootstrap ArzLM on the WSL2 Linux filesystem with official Linux CUDA PyTorch.
# Does not modify Windows NVIDIA drivers. Do not install unofficial Triton wheels.
set -euo pipefail

WIN_SRC="${ARZLM_WIN_SRC:-/mnt/c/Users/ASUS/ArzLM}"
DEST="${ARZLM_WSL_ROOT:-${HOME}/arzlm}"

if [[ "${DEST}" == /mnt/* ]]; then
  echo "Refusing to install on ${DEST}. The env must live on the Linux filesystem, not /mnt/c." >&2
  exit 1
fi

if [[ ! -d "${WIN_SRC}" ]]; then
  echo "Windows source not found: ${WIN_SRC}" >&2
  exit 1
fi

mkdir -p "${DEST}"
echo "Syncing ${WIN_SRC} -> ${DEST}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude '.venv/' \
    --exclude 'hf-cache/' \
    --exclude 'runs/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '*.egg-info/' \
    "${WIN_SRC}/" "${DEST}/"
else
  tar -C "${WIN_SRC}" --exclude='.venv' --exclude='hf-cache' --exclude='runs' -cf - . | tar -C "${DEST}" -xf -
fi

# Packed FineWeb windows used by the benchmark (gitignored binaries).
mkdir -p "${DEST}/data/prepared"
if [[ -d "${WIN_SRC}/data/prepared/tiny" ]]; then
  rm -rf "${DEST}/data/prepared/tiny"
  cp -a "${WIN_SRC}/data/prepared/tiny" "${DEST}/data/prepared/tiny"
fi
if [[ -d "${WIN_SRC}/data/prepared/10m" ]]; then
  rm -rf "${DEST}/data/prepared/10m"
  cp -a "${WIN_SRC}/data/prepared/10m" "${DEST}/data/prepared/10m"
fi

UV_BIN="${HOME}/.local/bin/uv"
if [[ ! -x "${UV_BIN}" ]]; then
  echo "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}" >&2
  exit 1
fi

cd "${DEST}"
if ! command -v gcc >/dev/null 2>&1; then
  echo "Installing build-essential (Triton/Inductor needs a C compiler on Linux)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential
fi
"${UV_BIN}" python install 3.12
"${UV_BIN}" venv .venv --python 3.12
# Official Linux CUDA 12.8 wheels (same family as the Windows baseline).
"${UV_BIN}" pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
"${UV_BIN}" pip install --python .venv -e ".[dev]"

echo "Environment:"
.venv/bin/python -m arzlm env
echo "WSL bootstrap complete at ${DEST}"
