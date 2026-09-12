#!/usr/bin/env bash
set -u
if [[ -n "${HF_TOKEN:-}" ]]; then echo has_hf_token; elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then echo has_hf_file; else echo no_hf_token; fi
nvidia-smi -L 2>/dev/null || echo no-nvidia-smi
df -h /home/asus | tail -1
