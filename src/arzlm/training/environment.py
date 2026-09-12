"""Detect GPU / CUDA / SDPA backend facts for logs and benchmarks."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any

import torch

from arzlm.training.compile import platform_compile_notes, triton_available, triton_version


def sdpa_backend_flags() -> dict[str, Any]:
    flags: dict[str, Any] = {}
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return {"available": False}
    for name in (
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "cudnn_sdp_enabled",
    ):
        fn = getattr(cuda_backends, name, None)
        if callable(fn):
            try:
                flags[name.replace("_enabled", "")] = bool(fn())
            except Exception as exc:
                flags[name] = f"error: {exc}"
    try:
        from torch.nn.attention import SDPBackend

        flags["sdp_backends"] = [backend.name for backend in SDPBackend]
    except Exception:
        flags["sdp_backends"] = []
    return flags


def preferred_sdpa_name() -> str:
    flags = sdpa_backend_flags()
    if flags.get("flash_sdp"):
        return "flash"
    if flags.get("cudnn_sdp"):
        return "cudnn"
    if flags.get("mem_efficient_sdp"):
        return "mem_efficient"
    if flags.get("math_sdp"):
        return "math"
    return "unknown"


def _nvidia_smi_query(query: str) -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.check_output(
            [smi, f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "triton": triton_available(),
        "triton_version": triton_version(),
        "sdpa": sdpa_backend_flags(),
        "preferred_sdpa": preferred_sdpa_name() if cuda else None,
        "compile_notes": platform_compile_notes(),
        "nvidia_driver": _nvidia_smi_query("driver_version"),
        "nvidia_cuda_version": _nvidia_smi_query("cuda_version"),
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": torch.cuda.get_device_capability(0),
                "gpu_total_memory_bytes": int(props.total_memory),
            }
        )
        if hasattr(torch.version, "cuda") and torch.version.cuda:
            info["cuda_version"] = torch.version.cuda
    try:
        from importlib.metadata import version

        info["litgpt"] = version("litgpt")
        info["lightning"] = version("lightning")
    except Exception:
        info["litgpt"] = info.get("litgpt")
        info["lightning"] = None
    return info
