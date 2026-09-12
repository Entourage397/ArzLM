"""ArzLM-100M architecture, expressed as native litgpt.Config fields.

Every key in ARZLM_100M_FIELDS is checked against the installed litgpt.Config
dataclass. Unknown or silently-dropped fields are a hard error.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import num_parameters

ARZLM_100M_NAME = "ArzLM-100M"
ARZLM_300M_NAME = "ArzLM-300M"

# Verified against litgpt 0.5.13 Config (installed). Do not add keys that are
# not fields on that dataclass.
ARZLM_100M_FIELDS: dict[str, Any] = {
    "name": ARZLM_100M_NAME,
    "hf_config": {"org": "ArzLM", "name": ARZLM_100M_NAME},
    "block_size": 1024,
    "n_layer": 12,
    "n_embd": 768,
    "vocab_size": 32000,
    "padding_multiple": 64,
    "norm_class_name": "RMSNorm",
    "norm_eps": 1e-5,
    "norm_qk": True,
    "norm_qk_type": "default",
    "post_attention_norm": False,
    "post_mlp_norm": False,
    "parallel_residual": False,
    "shared_attention_norm": False,
    "n_head": 12,
    "head_size": 64,
    "n_query_groups": 4,
    "attn_bias": False,
    "rotary_percentage": 1.0,
    "rope_base": 10000,
    "rope_condense_ratio": 1,
    "intermediate_size": 2048,
    "bias": False,
    "mlp_class_name": "LLaMAMLP",
    "lm_head_bias": False,
    "scale_embeddings": False,
    "norm_1": True,
    "norm_2": True,
}

TARGET_PARAMETERS = 100_000_000
EXPECTED_TRAINABLE_PARAMETERS = 100_094_208
TARGET_PARAMETERS_300M = 300_000_000
# Instantiated 2026-09-10 against litgpt 0.5.13 GPT + tied embeddings.
EXPECTED_TRAINABLE_PARAMETERS_300M = 303_353_856
PARAMETER_TOLERANCE = 0.05  # 5% around the named target
PARAMETER_TOLERANCE_300M = 0.05

EXPECTED_BY_NAME = {
    ARZLM_100M_NAME: EXPECTED_TRAINABLE_PARAMETERS,
    ARZLM_300M_NAME: EXPECTED_TRAINABLE_PARAMETERS_300M,
}
TARGET_BY_NAME = {
    ARZLM_100M_NAME: TARGET_PARAMETERS,
    ARZLM_300M_NAME: TARGET_PARAMETERS_300M,
}

ARZLM_300M_FIELDS: dict[str, Any] = {
    **ARZLM_100M_FIELDS,
    "name": ARZLM_300M_NAME,
    "hf_config": {"org": "ArzLM", "name": ARZLM_300M_NAME},
    "block_size": 2048,
    "n_layer": 24,
    "n_embd": 1024,
    "n_head": 16,
    "head_size": 64,
    "n_query_groups": 4,
    "intermediate_size": 2816,
}


def _config_field_names() -> set[str]:
    return {f.name for f in fields(Config)}


def validate_config_fields(raw: dict[str, Any]) -> None:
    known = _config_field_names()
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "Unknown litgpt.Config field(s): "
            + ", ".join(unknown)
            + ". Verify against the installed litgpt.config.Config before changing architecture."
        )


def expected_trainable_parameters(config: Config | str) -> int:
    name = config if isinstance(config, str) else config.name
    if name not in EXPECTED_BY_NAME:
        raise RuntimeError(f"Unknown ArzLM architecture {name!r}")
    return EXPECTED_BY_NAME[name]


def build_model_config(overrides: dict[str, Any] | None = None) -> Config:
    name = (overrides or {}).get("name") or ARZLM_100M_NAME
    base = ARZLM_300M_FIELDS if name == ARZLM_300M_NAME else ARZLM_100M_FIELDS
    raw = dict(base)
    if overrides:
        raw.update(overrides)
    validate_config_fields(raw)
    config = Config(**raw)
    if config.n_embd != config.n_head * config.head_size:
        raise ValueError(
            f"n_embd={config.n_embd} != n_head*head_size={config.n_head * config.head_size}"
        )
    if config.n_head % config.n_query_groups != 0:
        raise ValueError("n_head must be divisible by n_query_groups for GQA")
    if config.padded_vocab_size != config.vocab_size:
        raise ValueError(
            f"Expected unpadded vocab {config.vocab_size}, got padded_vocab_size="
            f"{config.padded_vocab_size}. Adjust padding_multiple rather than silently expanding."
        )
    return config


def load_model_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Model config {path} must be a mapping")
    return build_model_config(raw)


def formula_parameter_count(config: Config, tie_embeddings: bool = True) -> int:
    """Closed-form trainable parameter count matching litgpt.GPT + optional tying."""
    v = config.padded_vocab_size
    c = config.n_embd
    l = config.n_layer
    h = config.n_head
    g = config.n_query_groups
    d = config.head_size
    i = config.intermediate_size
    assert v is not None and i is not None and g is not None and d is not None

    embed = v * c
    lm_head = 0 if tie_embeddings else v * c
    qkv = c * (h + 2 * g) * d
    o_proj = h * d * c
    if config.norm_qk:
        if config.norm_qk_type == "default":
            qk_norm = 2 * d
        else:
            qk_norm = h * d + g * d
    else:
        qk_norm = 0
    attn_norm = c if config.norm_1 else 0
    mlp_norm = c if config.norm_2 and not config.shared_attention_norm else 0
    mlp = 3 * c * i
    layer = qkv + o_proj + qk_norm + attn_norm + mlp_norm + mlp
    final_norm = c
    return embed + lm_head + l * layer + final_norm


def build_gpt(config: Config | None = None, *, tie_embeddings: bool = True) -> GPT:
    config = config or build_model_config()
    model = GPT(config)
    if tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    return model


def count_parameters(
    config: Config | None = None,
    *,
    tie_embeddings: bool = True,
    model: GPT | None = None,
) -> dict[str, int]:
    config = config or (model.config if model is not None else build_model_config())
    if model is None:
        with torch.device("cpu"):
            model = build_gpt(config, tie_embeddings=tie_embeddings)
    trainable = num_parameters(model, requires_grad=True)
    total = num_parameters(model, requires_grad=None)
    formula = formula_parameter_count(config, tie_embeddings=tie_embeddings)
    name = config.name
    return {
        "trainable": trainable,
        "total": total,
        "formula": formula,
        "target": TARGET_BY_NAME.get(name, TARGET_PARAMETERS),
        "expected": EXPECTED_BY_NAME.get(name, EXPECTED_TRAINABLE_PARAMETERS),
        "name": name,
    }


def format_parameter_report(counts: dict[str, int]) -> str:
    trainable = counts["trainable"]
    delta = trainable - counts["target"]
    pct = 100.0 * delta / counts["target"]
    label = str(counts.get("name") or ARZLM_100M_NAME)
    return (
        f"{label} trainable parameters: {trainable:,} "
        f"(formula {counts['formula']:,}; target {counts['target']:,}; {pct:+.3f}%)"
    )


def assert_parameter_budget(counts: dict[str, int], *, name: str | None = None) -> None:
    if counts["trainable"] != counts["formula"]:
        raise RuntimeError(
            f"Instantiated parameter count {counts['trainable']:,} != formula {counts['formula']:,}"
        )
    arch = name or str(counts.get("name") or ARZLM_100M_NAME)
    expected = EXPECTED_BY_NAME.get(arch)
    if expected is None:
        raise RuntimeError(f"Unknown ArzLM architecture {arch!r}")
    if counts["trainable"] != expected:
        raise RuntimeError(
            f"Trainable parameter count {counts['trainable']:,} != "
            f"verified {arch} count {expected:,}. "
            "Do not silently change architecture or vocab padding."
        )
    target = TARGET_BY_NAME.get(arch, counts["target"])
    rel = abs(counts["trainable"] - target) / target
    if rel > PARAMETER_TOLERANCE:
        raise RuntimeError(
            f"Parameter count {counts['trainable']:,} is more than "
            f"{PARAMETER_TOLERANCE:.0%} away from {target:,}"
        )
