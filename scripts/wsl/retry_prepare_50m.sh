#!/usr/bin/env bash
set -euo pipefail
cd /home/asus/arzlm
export TORCH_CUDA_ARCH_LIST=8.9
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1
export CC=gcc
export CXX=g++
export HF_HOME=/home/asus/arzlm/hf-cache
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/wsl
echo "HEAD=$(git rev-parse HEAD)"
.venv/bin/python -X faulthandler -m arzlm prepare --preset 50m --source fineweb-edu
echo PREPARE_50M_DONE
