"""Parameter routing for hybrid Muon + AdamW.

Muon is only appropriate for hidden 2D weight matrices (Keller Jordan;
Moonlight / Kimi K2 keep embeddings, LM head, and norm/scalar params on AdamW).

Tied `lm_head` / `wte` shares one Parameter; it is assigned to AdamW once.
"""

from __future__ import annotations

import re
from typing import Any

import torch.nn as nn

# LitGPT GPT hidden linear weights: attention qkv/proj and SwiGLU MLP.
_MUON_NAME = re.compile(
    r"^transformer\.h\.\d+\.(attn\.(qkv|proj)|mlp\.(fc_1|fc_2|proj))\.weight$"
)


def partition_litgpt_parameters(model: nn.Module) -> dict[str, Any]:
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        key = id(param)
        if key in seen:
            continue
        seen.add(key)
        if param.ndim == 2 and _MUON_NAME.match(name):
            muon.append(param)
            muon_names.append(name)
        else:
            adamw.append(param)
            adamw_names.append(name)

    assigned = {id(p) for p in muon}
    assigned.update(id(p) for p in adamw)
    missing = [name for name, p in model.named_parameters() if id(p) not in assigned]
    overlap = assigned.intersection({id(p) for p in muon}) & {id(p) for p in adamw}
    if missing:
        raise RuntimeError(f"Unassigned parameters: {missing}")
    if overlap:
        raise RuntimeError("Parameter appeared in both Muon and AdamW groups")
    if not muon:
        raise RuntimeError("Muon group is empty; check LitGPT parameter names")
    if any(p.ndim != 2 for p in muon):
        raise RuntimeError("Muon group contains a non-matrix parameter")

    return {
        "muon": muon,
        "adamw": adamw,
        "muon_names": muon_names,
        "adamw_names": adamw_names,
        "muon_n_params": int(sum(p.numel() for p in muon)),
        "adamw_n_params": int(sum(p.numel() for p in adamw)),
    }
