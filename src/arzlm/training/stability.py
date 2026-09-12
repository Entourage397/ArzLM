"""Stability probes for optimizer sweeps (grad norms, Q/K RMS, loss spikes)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def unwrap_module(model: nn.Module) -> nn.Module:
    current = model
    for _ in range(6):
        nxt = getattr(current, "_orig_mod", None) or getattr(current, "module", None) or getattr(
            current, "_forward_module", None
        )
        if nxt is None or nxt is current:
            break
        current = nxt
    return current


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().pow(2).mean().sqrt().item())


def attention_health(model: nn.Module) -> dict[str, Any]:
    """RMS of QKV / QK-Norm weights. Spikes here are a cheap divergence signal."""
    qkv: list[float] = []
    norm_q: list[float] = []
    norm_k: list[float] = []
    max_abs_qkv = 0.0
    for name, param in unwrap_module(model).named_parameters():
        if name.endswith("attn.qkv.weight"):
            qkv.append(_rms(param))
            max_abs_qkv = max(max_abs_qkv, float(param.detach().float().abs().max().item()))
        elif name.endswith("attn.norm_q.weight"):
            norm_q.append(_rms(param))
        elif name.endswith("attn.norm_k.weight"):
            norm_k.append(_rms(param))

    def _agg(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {"mean": sum(values) / len(values), "max": max(values), "min": min(values)}

    return {
        "qkv_rms": _agg(qkv),
        "norm_q_rms": _agg(norm_q),
        "norm_k_rms": _agg(norm_k),
        "qkv_max_abs": max_abs_qkv,
    }


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return float(value.detach().float().mean().item())
        return float(value.detach().float().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
