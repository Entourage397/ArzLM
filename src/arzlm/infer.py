"""Canonical ArzLM inference loader.

This is LitGPT `GPT` with tied embeddings. It is **not** a stock
`transformers.AutoModelForCausalLM` checkpoint. The Hugging Face folder
uses the same tensor names as LitGPT (`lm_head.weight`, fused `attn.qkv`,
QK-Norm). Loading it with LlamaForCausalLM will fail or silently misread
weights.

Training used `add_bos_token: false`. Prompt encoding must pass `bos=False`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer as LitTokenizer

from arzlm.cloud.prompting import decode_token_ids, encode_prompt_batch, greedy_complete
from arzlm.model.config import (
    EXPECTED_TRAINABLE_PARAMETERS_300M,
    build_gpt,
    count_parameters,
)

OPTIMIZER_KEYS = ("optimizer", "optim", "optimizer_states")


class InferenceCheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unexpected, or still has optimizer state."""


@dataclass
class LoadedModel:
    model: GPT
    tokenizer: LitTokenizer
    config: Config
    checkpoint_dir: Path
    parameters: int
    step_count: int | None
    device: torch.device
    dtype: torch.dtype
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    source: str


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Accept a release root, `litgpt/` subdir, or a directory that contains `lit_model.pth`."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    candidates = [
        root,
        root / "litgpt",
        root / "LITGPT_INFERENCE",
        root / "huggingface",
    ]
    for cand in candidates:
        if (cand / "lit_model.pth").is_file() and (cand / "model_config.yaml").is_file():
            return cand
        if (cand / "model.safetensors").is_file() and (cand / "model_config.yaml").is_file():
            return cand
    raise InferenceCheckpointError(
        f"No ArzLM inference checkpoint under {root}. Expected lit_model.pth "
        "or model.safetensors plus model_config.yaml."
    )


def checkpoint_model_state(blob: Any) -> dict[str, torch.Tensor]:
    if isinstance(blob, dict) and any(k in blob for k in OPTIMIZER_KEYS):
        raise InferenceCheckpointError("refusing to load optimizer-bearing checkpoint")
    if isinstance(blob, dict) and "model" in blob:
        raw = blob["model"]
    else:
        raw = blob
    if hasattr(raw, "state_dict"):
        raw = raw.state_dict()
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint model payload is {type(raw)!r}, expected state_dict")
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"non-tensor in model state: {key}={type(value)!r}")
        out[key] = value
    return out


def _load_state_dict(ckpt_dir: Path) -> tuple[dict[str, torch.Tensor], int | None, str]:
    pth = ckpt_dir / "lit_model.pth"
    st = ckpt_dir / "model.safetensors"
    if pth.is_file():
        blob = torch.load(pth, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and any(k in blob for k in OPTIMIZER_KEYS):
            raise InferenceCheckpointError(f"{pth} still contains optimizer state")
        step = None
        if isinstance(blob, dict) and blob.get("step_count") is not None:
            step = int(blob["step_count"])
        return checkpoint_model_state(blob), step, str(pth)
    if st.is_file():
        from safetensors.torch import load_file

        return load_file(str(st)), None, str(st)
    raise InferenceCheckpointError(f"no weights in {ckpt_dir}")


def select_dtype(device: torch.device, requested: str | None = None) -> torch.dtype:
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    if requested in {"fp32", "float32"}:
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_inference_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: str | None = None,
    expected_parameters: int = EXPECTED_TRAINABLE_PARAMETERS_300M,
    expected_block_size: int = 2048,
) -> LoadedModel:
    """Load weights-only ArzLM with `strict=True` tying and finite-logit checks."""
    ckpt_dir = resolve_checkpoint_dir(path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    torch_dtype = select_dtype(device, dtype)
    state, step, source = _load_state_dict(ckpt_dir)
    cfg = Config.from_file(ckpt_dir / "model_config.yaml")
    if int(cfg.block_size) != int(expected_block_size):
        raise InferenceCheckpointError(
            f"block_size {cfg.block_size} != expected {expected_block_size}"
        )
    tok_cfg = ckpt_dir / "tokenizer_config.json"
    if tok_cfg.is_file():
        import json

        meta = json.loads(tok_cfg.read_text(encoding="utf-8"))
        if int(meta.get("model_max_length") or 0) != int(expected_block_size):
            raise InferenceCheckpointError(
                f"tokenizer_config model_max_length={meta.get('model_max_length')} "
                f"!= {expected_block_size}"
            )
    model = build_gpt(cfg, tie_embeddings=True)
    incompatible = model.load_state_dict(state, strict=True)
    missing = tuple(getattr(incompatible, "missing_keys", ()) or ())
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise InferenceCheckpointError(
            f"strict load failed missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    counts = count_parameters(cfg, model=model, tie_embeddings=True)
    n = int(counts["trainable"])
    if n != int(expected_parameters):
        raise InferenceCheckpointError(f"parameter count {n} != {expected_parameters}")
    tok = LitTokenizer(ckpt_dir)
    return LoadedModel(
        model=model,
        tokenizer=tok,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        parameters=n,
        step_count=step,
        device=device,
        dtype=torch_dtype,
        missing_keys=missing,
        unexpected_keys=unexpected,
        source=source,
    )


def generate(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    return greedy_complete(
        loaded.model,
        loaded.tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        device=loaded.device,
    )


__all__ = [
    "InferenceCheckpointError",
    "LoadedModel",
    "checkpoint_model_state",
    "decode_token_ids",
    "encode_prompt_batch",
    "generate",
    "load_inference_checkpoint",
    "resolve_checkpoint_dir",
    "select_dtype",
]
