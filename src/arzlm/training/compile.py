"""Optional torch.compile with measured fallback (no silent degradation)."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CompileResult:
    model: nn.Module
    requested: str
    active: bool
    backend: str | None
    error: str | None


def triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def triton_version() -> str | None:
    try:
        import triton

        return str(getattr(triton, "__version__", "unknown"))
    except Exception:
        return None


def should_attempt_compile(requested: str, device: torch.device) -> tuple[bool, str | None]:
    if requested == "false":
        return False, None
    if requested == "true":
        return True, None
    # auto
    if device.type != "cuda":
        return False, None
    if sys.platform.startswith("win") and not triton_available():
        return False, (
            "skipped: torch.compile requires Triton; measured TritonMissing on this "
            "Windows CUDA build. Use WSL2 + Linux wheels rather than unofficial Triton."
        )
    return True, None


def maybe_compile(
    model: nn.Module,
    requested: str,
    device: torch.device,
    *,
    disable_cudagraphs: bool = False,
) -> CompileResult:
    attempt, skip_reason = should_attempt_compile(requested, device)
    if not attempt:
        return CompileResult(model=model, requested=requested, active=False, backend=None, error=skip_reason)
    try:
        if disable_cudagraphs:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = False
        compiled = torch.compile(model, backend="inductor")
        backend = "inductor" if not disable_cudagraphs else "inductor-no-cudagraphs"
        return CompileResult(
            model=compiled,
            requested=requested,
            active=True,
            backend=backend,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        return CompileResult(
            model=model,
            requested=requested,
            active=False,
            backend=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def dynamo_reset() -> None:
    try:
        torch._dynamo.reset()
    except Exception:
        return


def dynamo_snapshot() -> dict:
    """Best-effort Inductor/Dynamo counters (graph breaks, recompiles)."""
    out: dict = {}
    try:
        from torch._dynamo.utils import counters

        serialized = {}
        for key, value in dict(counters).items():
            if hasattr(value, "items"):
                serialized[str(key)] = {str(k): int(v) if isinstance(v, int) else v for k, v in value.items()}
            else:
                serialized[str(key)] = value
        out["counters"] = serialized
        breaks = serialized.get("graph_break") or serialized.get("graph_breaks")
        if isinstance(breaks, dict):
            out["graph_break_count"] = int(sum(int(v) for v in breaks.values() if isinstance(v, int)))
        recomp = serialized.get("recompiles") or serialized.get("recompile")
        if isinstance(recomp, dict):
            out["recompile_count"] = int(sum(int(v) for v in recomp.values() if isinstance(v, int)))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from torch._dynamo.utils import CompileTimeLogger  # noqa: F401
    except Exception:
        pass
    try:
        metrics = getattr(torch.compiler, "get_compilation_metrics", None)
        if callable(metrics):
            raw = metrics()
            out["compilation_metrics_n"] = len(raw) if raw is not None else 0
    except Exception:
        pass
    return out


def platform_compile_notes() -> dict[str, str | bool | None]:
    return {
        "platform": sys.platform,
        "triton_available": triton_available(),
        "triton_version": triton_version(),
        "recommendation": (
            "Use WSL2 + Linux PyTorch for reliable torch.compile/Inductor "
            "if Windows compile fails (Triton is not installed on this host)."
            if sys.platform.startswith("win") and not triton_available()
            else "native compile stack looks usable"
        ),
    }
